"""Auto-align a .gyroflow IMU stream to an AiM XRK MyChron log.

Sync principle:
    Both streams measure the chassis yawing through corners — the MyChron
    via GPS (channel `GPS_Yaw_Rate` at 25 Hz) and the wheel-mounted Go 3S
    via its IMU (low-frequency component of the gyro projected onto the
    steering-column axis). The two signals differ in scale (the IMU one
    is contaminated by steering input and scaled by cos(column-tilt))
    but their *shape vs time* is the same. Cross-correlating them
    recovers the clock offset, regardless of which camera/logger clock
    drifted or when each started recording.

Output:
    `SyncedXrk` with:
      - `offset_imu_to_xrk_s` — add this to `stream.t` to land on XRK time.
        (Equivalently: XRK time t_xrk corresponds to IMU time
        `t_xrk + offset_imu_to_xrk_s`.)
      - `corr_peak` — peak normalized cross-correlation value. Above ~0.5
        is a confident match; below ~0.2 is likely a bad match.
      - `sign` — +1 if the two signals were correlated as-is, −1 if we
        had to flip the IMU column axis to match. Stored so downstream
        consumers can flip the IMU yaw before integration if needed.

Usage:
    python -m pipeline.sync_xrk wheel.gyroflow session.xrk
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import butter, correlate, sosfiltfilt

if TYPE_CHECKING:
    from pipeline.extract_imu import ImuStream


# ---------------------------------------------------------------------------
# libxrk noise suppression
# ---------------------------------------------------------------------------

class _NullStream:
    def write(self, *a, **k):
        pass

    def flush(self, *a, **k):
        pass


def _quiet_libxrk_warnings() -> None:
    """libxrk prints `Unknown units[...]` lines to stderr on load for some
    channels. They're cosmetic; suppress so logs stay readable."""
    sys.stderr = _NullStream()


# ---------------------------------------------------------------------------
# Tunables (defined up here so they can be used as defaults below)
# ---------------------------------------------------------------------------

SYNC_RESAMPLE_HZ = 25.0       # match GPS_Yaw_Rate native rate
DEFAULT_MAX_LAG_S = 120.0     # search ±2 minutes
DEFAULT_LP_HZ = 0.3           # chassis yaw evolves on lap-scale (~30s); steering
                              # is at 0.3..3 Hz. Below 0.3 Hz isolates chassis.
EDGE_TRIM_S = 5.0             # trim IMU ends — Butterworth ringing + handling
                              # bumps (sticking/removing the camera) dominate
                              # the variance otherwise and pull the correlation.


# ---------------------------------------------------------------------------
# Column axis & chassis-yaw extraction from IMU
# ---------------------------------------------------------------------------

def detect_column_axis(gyro: np.ndarray) -> np.ndarray:
    """PCA of gyro covariance: the principal eigenvector is the wheel
    rotation axis in body frame (i.e. the steering column direction).

    Works for any wheel-mounted camera regardless of how it's angled,
    as long as wheel rotation dominates the gyro signal.
    """
    centred = gyro - gyro.mean(axis=0)
    cov = centred.T @ centred / len(centred)
    _, evecs = np.linalg.eigh(cov)
    n = evecs[:, -1]
    # Canonicalise: prefer +Z-dominant (lens-axis-ish) direction.
    if n[2] < 0:
        n = -n
    return n


def estimate_quiet_bias(gyro: np.ndarray, quiet_rate_thresh_dps: float = 2.0) -> np.ndarray:
    """Per-axis gyro bias from low-rate (≈stationary-ish) samples."""
    mag = np.linalg.norm(gyro, axis=1)
    quiet = mag < np.deg2rad(quiet_rate_thresh_dps)
    if not quiet.any():
        return np.zeros(3)
    return gyro[quiet].mean(axis=0)


def chassis_yaw_from_imu(stream: ImuStream,
                         lowpass_hz: float = DEFAULT_LP_HZ,
                         edge_trim_s: float = EDGE_TRIM_S,
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (t [s], chassis_yaw_rate [rad/s], column_axis, bias).

    chassis_yaw_rate is the low-frequency component of the gyro projected
    onto the column axis. Steering inputs (higher-frequency) are filtered
    out so what remains matches the GPS heading rate.

    `edge_trim_s` trims that many seconds off each end of the recording —
    the LP filter has ringing artifacts at the boundaries, and the very
    start/end usually contain camera-handling transients (sticking the
    magnet onto the wheel, pulling it off) that aren't real chassis motion.
    """
    bias = estimate_quiet_bias(stream.gyro)
    gyro = stream.gyro - bias
    n_col = detect_column_axis(gyro)
    rate_along_col = gyro @ n_col

    fs = stream.sample_rate_hz
    sos = butter(3, lowpass_hz / (fs / 2.0), btype="low", output="sos")
    yaw_lp = sosfiltfilt(sos, rate_along_col)

    trim = int(edge_trim_s * fs)
    return stream.t[trim:-trim], yaw_lp[trim:-trim], n_col, bias


# ---------------------------------------------------------------------------
# XRK loading
# ---------------------------------------------------------------------------

def load_xrk(xrk_path: Path):
    """Wrap libxrk.aim_xrk(). Returns the LogFile object."""
    _quiet_libxrk_warnings()
    import libxrk
    return libxrk.aim_xrk(str(xrk_path))


def chassis_yaw_from_xrk(log) -> tuple[np.ndarray, np.ndarray]:
    """Returns (t [s], yaw_rate [rad/s]) from XRK `GPS_Yaw_Rate` channel.

    The XRK column is in deg/s. Time is in ms from XRK t=0 (logger start).
    """
    tbl = log.channels.get("GPS_Yaw_Rate")
    if tbl is None:
        raise ValueError("XRK has no GPS_Yaw_Rate channel — cannot sync.")
    t_ms = np.asarray(tbl["timecodes"])
    yr_dps = np.asarray(tbl["GPS_Yaw_Rate"])
    return t_ms / 1000.0, np.deg2rad(yr_dps)


# ---------------------------------------------------------------------------
# Cross-correlation
# ---------------------------------------------------------------------------

def _robust_normalise(x: np.ndarray) -> np.ndarray:
    """Zero-median, MAD-scaled. Resistant to spike artifacts that would
    dominate a std-based normalization (e.g. handling bumps when sticking
    the camera on / pulling it off the wheel).
    """
    m = np.median(x)
    s = 1.4826 * np.median(np.abs(x - m)) + 1e-12
    return (x - m) / s


def cross_correlate_offset(t_imu: np.ndarray, yaw_imu: np.ndarray,
                           t_xrk: np.ndarray, yaw_xrk: np.ndarray,
                           max_lag_s: float = DEFAULT_MAX_LAG_S,
                           fs: float = SYNC_RESAMPLE_HZ,
                           ) -> tuple[float, float, int]:
    """Find the time offset that best aligns two yaw-rate signals.

    Returns (offset_imu_to_xrk_s, peak_normalized_corr, sign).
    - offset: add this to IMU `t` to land on XRK time.
    - sign: ±1; if -1, the IMU signal had to be flipped (column axis points
      opposite to GPS-yaw convention). Apply this flip downstream.
    """
    dt = 1.0 / fs

    # Resample IMU onto its own dt grid
    grid_i = np.arange(t_imu[0], t_imu[-1] + dt, dt)
    y_i = np.interp(grid_i, t_imu, yaw_imu)

    # Resample XRK onto its own dt grid (already 25 Hz but make times exact)
    grid_x = np.arange(t_xrk[0], t_xrk[-1] + dt, dt)
    y_x = np.interp(grid_x, t_xrk, yaw_xrk)

    # Robust normalisation; std would be wrecked by occasional spikes.
    y_i = _robust_normalise(y_i)
    y_x = _robust_normalise(y_x)

    # Cross-correlation. correlate(a, b)[k] = sum_n a[n] * b[n - lag],
    # where lag = k - (len(b) - 1).
    # If `lag` is positive, b is shifted RIGHT relative to a, i.e. b's
    # events happen later in time than a's, i.e. a leads b.
    # We're computing in INDEX space. Convert to seconds and to a
    # "what to add to IMU time" offset.
    raw_corr = correlate(y_i, y_x, mode="full")
    # Lag in samples: index 0 means b is shifted left by (len(b)-1) samples.
    lag_samples = np.arange(-len(y_x) + 1, len(y_i))
    # The lag in TIME units, on the IMU's grid, where:
    #   y_i[n] ~ y_x[n - lag]
    # If lag > 0, the matching XRK sample n-lag is EARLIER in XRK time
    # than the matching IMU sample n is in IMU time. Convert to a
    # clock offset later.
    lag_s = lag_samples * dt

    # Normalize for fair comparison across lags (each lag has a different
    # overlap count). Use the symmetric normalization.
    overlap = np.minimum.reduce([
        len(y_i) - np.maximum(lag_samples, 0),
        len(y_x) + np.minimum(lag_samples, 0),
    ])
    # Avoid div-by-zero for tiny overlaps; require ≥1 second of overlap
    valid = overlap >= int(fs * 1.0)
    norm_corr = np.where(valid, raw_corr / np.maximum(overlap, 1), 0.0)

    # Constrain search to ±max_lag_s
    mask = np.abs(lag_s) <= max_lag_s
    norm_corr = np.where(mask, norm_corr, 0.0)

    best = int(np.argmax(np.abs(norm_corr)))
    sign = int(np.sign(norm_corr[best])) or 1
    peak = float(np.abs(norm_corr[best]))
    best_lag_s = float(lag_s[best])

    # Now convert from "lag in cross-correlation index" to "clock offset
    # to add to IMU time so it lands on XRK clock".
    #   y_i(t_imu_grid[n]) ~ y_x(t_xrk_grid[n - lag])
    # Substitute the grid origins:
    #   t_imu_grid[n] = t_imu[0] + n*dt
    #   t_xrk_grid[n - lag] = t_xrk[0] + (n - lag)*dt
    # The IMU sample at IMU-time t_i corresponds to XRK-time
    #   t_x = t_xrk[0] + (n - lag)*dt = t_imu_grid[n] - (t_imu[0] - t_xrk[0] + lag*dt)
    #       = t_i - (t_imu[0] - t_xrk[0] + lag*dt)
    # So:    t_x = t_i + offset, where offset = -(t_imu[0] - t_xrk[0] + lag*dt)
    #                                         =  t_xrk[0] - t_imu[0] - lag*dt
    offset_imu_to_xrk_s = float(t_xrk[0] - t_imu[0] - best_lag_s)

    return offset_imu_to_xrk_s, peak, sign


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

@dataclass
class SyncedXrk:
    """Result of sync_imu_to_xrk."""
    gyroflow_path: Path
    xrk_path: Path
    offset_imu_to_xrk_s: float    # add to IMU t to land on XRK clock
    corr_peak: float              # 0..1, higher = better fit
    sign: int                     # +1 or -1; flip IMU column-axis if -1
    column_axis: np.ndarray       # principal rotation axis in body frame
    gyro_bias: np.ndarray         # per-axis (rad/s) from quiet samples
    xrk_laps: object              # PyArrow table from libxrk
    xrk_log: object               # the full libxrk LogFile, for downstream use


def sync_imu_to_xrk(gyroflow_path: Path, xrk_path: Path,
                    *, max_lag_s: float = DEFAULT_MAX_LAG_S,
                    lowpass_hz: float = DEFAULT_LP_HZ,
                    ) -> SyncedXrk:
    from pipeline.extract_imu import load_gyroflow
    stream = load_gyroflow(gyroflow_path, source="wheel")
    log = load_xrk(xrk_path)

    t_imu, yaw_imu, n_col, bias = chassis_yaw_from_imu(stream, lowpass_hz=lowpass_hz)
    t_xrk, yaw_xrk = chassis_yaw_from_xrk(log)

    offset, peak, sign = cross_correlate_offset(
        t_imu, yaw_imu, t_xrk, yaw_xrk, max_lag_s=max_lag_s,
    )

    return SyncedXrk(
        gyroflow_path=gyroflow_path,
        xrk_path=xrk_path,
        offset_imu_to_xrk_s=offset,
        corr_peak=peak,
        sign=sign,
        column_axis=n_col,
        gyro_bias=bias,
        xrk_laps=log.laps,
        xrk_log=log,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gyroflow", type=Path)
    ap.add_argument("xrk", type=Path)
    ap.add_argument("--max-lag-s", type=float, default=DEFAULT_MAX_LAG_S)
    ap.add_argument("--lowpass-hz", type=float, default=DEFAULT_LP_HZ)
    args = ap.parse_args()

    r = sync_imu_to_xrk(args.gyroflow, args.xrk,
                        max_lag_s=args.max_lag_s, lowpass_hz=args.lowpass_hz)
    print(f"gyroflow:           {r.gyroflow_path}")
    print(f"xrk:                {r.xrk_path}")
    print(f"column axis (body): ({r.column_axis[0]:+.3f}, {r.column_axis[1]:+.3f}, {r.column_axis[2]:+.3f})")
    print(f"gyro bias (dps):    ({np.rad2deg(r.gyro_bias[0]):+.4f}, {np.rad2deg(r.gyro_bias[1]):+.4f}, {np.rad2deg(r.gyro_bias[2]):+.4f})")
    print(f"sign:               {r.sign:+d}  ({'as-is' if r.sign==1 else 'IMU yaw flipped'})")
    print(f"offset (IMU→XRK):   {r.offset_imu_to_xrk_s:+.3f} s")
    print(f"peak |corr|:        {r.corr_peak:.4f}")
    print()
    print(f"XRK laps ({r.xrk_laps.num_rows}):")
    import pyarrow as pa
    for i in range(r.xrk_laps.num_rows):
        n = r.xrk_laps["num"][i].as_py()
        s = r.xrk_laps["start_time"][i].as_py() / 1000
        e = r.xrk_laps["end_time"][i].as_py() / 1000
        print(f"  lap {n}: {s:7.2f}s .. {e:7.2f}s   ({e-s:.2f}s)")


if __name__ == "__main__":
    main()
