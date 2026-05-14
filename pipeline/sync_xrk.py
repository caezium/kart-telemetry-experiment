"""Auto-align a wheel-mounted .gyroflow IMU stream to an AiM XRK MyChron log.

Sync principle
--------------
Both streams independently measure the chassis yawing through corners:

    * the MyChron via GPS-derived heading rate (channel `GPS_Yaw_Rate` @ 25 Hz)
    * the wheel-mounted Go 3S via its IMU: the low-frequency component of the
      gyro projected onto the steering-column axis IS the chassis yaw rate
      (scaled by the cosine of the column tilt, and contaminated by short
      steering inputs at higher frequencies — both removed by low-pass)

The two signals differ in scale but their *shape vs time* is the same.
Cross-correlating them recovers the time offset between the camera clock
and the logger clock — independent of who started when or how badly they
drifted. Empirically the offset can be tens of seconds even when the
device wall-clocks suggest otherwise.

Geometry
--------
For a wheel-mounted camera in any orientation:

    ω_along_column = ω_steering + k · ω_chassis_yaw

where the column axis in body frame is the principal eigenvector of the
gyro covariance matrix (PCA), and k = |column · world_up|.  The same
recording gives us both: PCA finds the column from the gyro, and the
gravity vector from quiet-sample accelerometer averaging gives world_up.

Usage
-----
    python -m pipeline.sync_xrk wheel.gyroflow session.xrk
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import numpy as np
from scipy.signal import butter, correlate, sosfiltfilt

if TYPE_CHECKING:
    from pipeline.extract_imu import ImuStream


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

SYNC_RESAMPLE_HZ = 25.0       # match GPS_Yaw_Rate native rate
DEFAULT_MAX_LAG_S = 120.0     # search ±2 minutes
DEFAULT_LP_HZ = 0.3           # chassis yaw evolves on lap-scale (~30s); steering
                              # is at 0.3..3 Hz. Below 0.3 Hz isolates chassis.
EDGE_TRIM_S = 5.0             # trim IMU ends — Butterworth ringing + handling
                              # bumps (sticking/removing the camera) dominate
                              # the variance otherwise and pull the correlation.
QUIET_RATE_DPS = 2.0          # threshold for "wheel approximately stationary"
                              # samples used to estimate bias and gravity


# ---------------------------------------------------------------------------
# stderr suppression for libxrk's cosmetic warnings
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _silence_stderr() -> Iterator[None]:
    """libxrk emits `Unknown units[...]` lines to stderr at file-load time
    for channels whose units aren't in its table. They're cosmetic; swallow
    them. Restores stderr on exit (unlike a global reassignment).
    """
    old = sys.stderr
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stderr = old


def load_xrk(xrk_path: Path):
    """Wrap libxrk.aim_xrk(). Returns the LogFile object."""
    import libxrk
    with _silence_stderr():
        return libxrk.aim_xrk(str(xrk_path))


# ---------------------------------------------------------------------------
# Body-frame geometry from the IMU
# ---------------------------------------------------------------------------

def detect_column_axis(gyro: np.ndarray) -> np.ndarray:
    """PCA of gyro covariance: the principal eigenvector is the wheel
    rotation axis in body frame (= steering column direction).

    Works for any wheel-mounted camera regardless of how it's angled.
    Sign is canonicalised so the +Z-dominant direction wins; the actual
    sign convention for steering (left/right positive) is then resolved
    by `cross_correlate_offset` returning `sign = ±1`.
    """
    centred = gyro - gyro.mean(axis=0)
    cov = centred.T @ centred / len(centred)
    _, evecs = np.linalg.eigh(cov)
    n = evecs[:, -1]
    if n[2] < 0:
        n = -n
    return n


def estimate_quiet_bias(gyro: np.ndarray,
                        quiet_rate_thresh_dps: float = QUIET_RATE_DPS,
                        ) -> np.ndarray:
    """Per-axis gyro bias from low-rate samples. Even a session with
    ~5% quiet moments gives a clean estimate (typically 23k+ samples
    at 1 kHz × 25s of stillness across the recording).
    """
    mag = np.linalg.norm(gyro, axis=1)
    quiet = mag < np.deg2rad(quiet_rate_thresh_dps)
    if not quiet.any():
        return np.zeros(3)
    return gyro[quiet].mean(axis=0)


def estimate_column_tilt_factor(stream: ImuStream, column_axis: np.ndarray,
                                quiet_rate_dps: float = QUIET_RATE_DPS,
                                ) -> float:
    """Compute k = |column · world_up| from the recording's own gravity.

    The wheel-mounted IMU column projection sees both steering and
    chassis yaw:

        ω_along_column = ω_steering + k · ω_chassis_yaw

    where k is the cosine of the column tilt from vertical, fixed by
    chassis-and-mount geometry. With both projected onto the same
    1D axis, k can in principle be regressed from the data — but
    GPS_Yaw_Rate has measurement noise, which biases OLS slope toward
    zero (errors-in-variables attenuation). Geometry is exact.

    Returns the magnitude; the sign is folded into the sync `sign` field.
    """
    mag = np.linalg.norm(stream.gyro, axis=1)
    quiet = mag < np.deg2rad(quiet_rate_dps)
    if not quiet.any():
        return 0.0
    g_body = stream.accel[quiet].mean(axis=0)
    world_up = -g_body / np.linalg.norm(g_body)
    return float(abs(np.dot(column_axis, world_up)))


def chassis_yaw_from_imu(stream: ImuStream,
                         lowpass_hz: float = DEFAULT_LP_HZ,
                         edge_trim_s: float = EDGE_TRIM_S,
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (t [s], chassis_yaw_rate [rad/s], column_axis, gyro_bias).

    `chassis_yaw_rate` here is the low-pass-filtered column-axis projection
    of the gyro. Steering inputs (faster than `lowpass_hz`) are removed so
    what remains tracks GPS yaw rate.

    `edge_trim_s` trims that many seconds off each end — the LP filter has
    ringing at the boundaries, and the very start/end usually contain
    camera-handling transients that aren't real motion.
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
# XRK side
# ---------------------------------------------------------------------------

def chassis_yaw_from_xrk(log) -> tuple[np.ndarray, np.ndarray]:
    """Returns (t [s], yaw_rate [rad/s]) from XRK `GPS_Yaw_Rate` channel."""
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
    """Zero-median, MAD-scaled. Resistant to outlier spikes that would
    dominate a std-based normalization (e.g. handling bumps when fitting
    or removing the camera).
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
      - peak: |normalized correlation| at best lag, 0..1.
      - sign: ±1; if -1, the IMU signal had to be flipped (PCA-detected
        column axis points opposite to GPS-yaw convention). Downstream
        consumers should multiply IMU column projection by this sign.
    """
    dt = 1.0 / fs

    # Resample each onto its own uniform grid
    grid_i = np.arange(t_imu[0], t_imu[-1] + dt, dt)
    grid_x = np.arange(t_xrk[0], t_xrk[-1] + dt, dt)
    y_i = _robust_normalise(np.interp(grid_i, t_imu, yaw_imu))
    y_x = _robust_normalise(np.interp(grid_x, t_xrk, yaw_xrk))

    # Cross-correlation. correlate(a, b)[k] = sum_n a[n] · b[n - lag]
    raw_corr = correlate(y_i, y_x, mode="full")
    lag_samples = np.arange(-len(y_x) + 1, len(y_i))
    lag_s = lag_samples * dt

    # Per-lag normalisation: divide by overlap count. Require ≥1s overlap.
    overlap = np.minimum.reduce([
        len(y_i) - np.maximum(lag_samples, 0),
        len(y_x) + np.minimum(lag_samples, 0),
    ])
    valid = overlap >= int(fs * 1.0)
    norm_corr = np.where(valid, raw_corr / np.maximum(overlap, 1), 0.0)
    norm_corr = np.where(np.abs(lag_s) <= max_lag_s, norm_corr, 0.0)

    best = int(np.argmax(np.abs(norm_corr)))
    sign = int(np.sign(norm_corr[best])) or 1
    peak = float(np.abs(norm_corr[best]))
    best_lag_s = float(lag_s[best])

    # Translate cross-correlation lag → clock offset on IMU time:
    #   y_i(t_imu_grid[n]) ≈ y_x(t_xrk_grid[n - lag])
    #   → t_x = t_i + (t_xrk[0] - t_imu[0] - lag·dt)
    offset_imu_to_xrk_s = float(t_xrk[0] - t_imu[0] - best_lag_s)
    return offset_imu_to_xrk_s, peak, sign


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

@dataclass
class SyncedXrk:
    """Result of `sync_imu_to_xrk()`. Everything downstream needs to align
    the IMU stream to XRK and extract clean steering.
    """
    gyroflow_path: Path
    xrk_path: Path
    # Sync timing
    offset_imu_to_xrk_s: float    # add to IMU `t` to land on XRK clock
    corr_peak: float              # 0..1, sync quality (>0.5 confident)
    sign: int                     # +1 or -1; multiply IMU column projection by this
    # Body-frame geometry (from the IMU recording)
    column_axis: np.ndarray       # principal rotation axis in body frame (unit)
    column_tilt_factor: float     # k = |column · world_up|, used to subtract
                                  # chassis yaw from the column projection
    gyro_bias: np.ndarray         # per-axis rad/s, from quiet samples
    # Convenience: the parsed XRK log, so callers don't reload it
    xrk_log: object               # libxrk.LogFile

    @property
    def column_tilt_deg(self) -> float:
        """Column tilt from world-vertical (degrees), for reporting."""
        return float(np.rad2deg(np.arccos(self.column_tilt_factor)))


def sync_imu_to_xrk(gyroflow_path: Path, xrk_path: Path,
                    *, max_lag_s: float = DEFAULT_MAX_LAG_S,
                    lowpass_hz: float = DEFAULT_LP_HZ,
                    ) -> SyncedXrk:
    """End-to-end sync. Loads both files, runs cross-correlation, returns
    everything downstream tools need.
    """
    from pipeline.extract_imu import load_gyroflow

    stream = load_gyroflow(gyroflow_path, source="wheel")
    log = load_xrk(xrk_path)

    t_imu, yaw_imu, n_col, bias = chassis_yaw_from_imu(stream, lowpass_hz=lowpass_hz)
    t_xrk, yaw_xrk = chassis_yaw_from_xrk(log)
    offset, peak, sign = cross_correlate_offset(
        t_imu, yaw_imu, t_xrk, yaw_xrk, max_lag_s=max_lag_s,
    )
    tilt = estimate_column_tilt_factor(stream, n_col)

    return SyncedXrk(
        gyroflow_path=gyroflow_path,
        xrk_path=xrk_path,
        offset_imu_to_xrk_s=offset,
        corr_peak=peak,
        sign=sign,
        column_axis=n_col,
        column_tilt_factor=tilt,
        gyro_bias=bias,
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
    print(f"column tilt:        {r.column_tilt_deg:.1f}° from vertical  (k={r.column_tilt_factor:.4f})")
    print(f"gyro bias (dps):    ({np.rad2deg(r.gyro_bias[0]):+.4f}, "
          f"{np.rad2deg(r.gyro_bias[1]):+.4f}, {np.rad2deg(r.gyro_bias[2]):+.4f})")
    print(f"sign:               {r.sign:+d}  ({'as-is' if r.sign==1 else 'IMU yaw flipped'})")
    print(f"offset (IMU→XRK):   {r.offset_imu_to_xrk_s:+.3f} s")
    print(f"peak |corr|:        {r.corr_peak:.4f}")
    if r.corr_peak < 0.5:
        print(f"  WARNING: corr below 0.5 — files may not be from the same session")
    print()
    print(f"XRK laps ({r.xrk_log.laps.num_rows}):")
    for i in range(r.xrk_log.laps.num_rows):
        n = r.xrk_log.laps["num"][i].as_py()
        s = r.xrk_log.laps["start_time"][i].as_py() / 1000
        e = r.xrk_log.laps["end_time"][i].as_py() / 1000
        print(f"  lap {n}: {s:7.2f}s .. {e:7.2f}s   ({e-s:.2f}s)")


if __name__ == "__main__":
    main()
