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

The PCA column axis is sign-corrected (via the cross-correlation sign) and
stored in `SyncedXrk.column_axis` so that `gyro @ column_axis` is already
the chassis-yaw-positive steering rate — downstream code never re-applies
a sign.

Usage
-----
    python -m pipeline.sync_xrk wheel.gyroflow session.xrk
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import warnings
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
GPS_YAW_GLITCH_DPS = 400.0    # |GPS_Yaw_Rate| above this is a GPS fix wobble,
                              # not real kart rotation — clean before use.
MIN_QUIET_SAMPLES = 50        # below this, gravity/bias geometry is unreliable


# ---------------------------------------------------------------------------
# stderr suppression for libxrk's cosmetic warnings
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _silence_stderr() -> Iterator[str]:
    """libxrk emits `Unknown units[...]` lines to stderr at file-load time
    for channels whose units aren't in its table. They're cosmetic; capture
    them so they don't clutter output, but yield the buffer so a caller can
    inspect it if a genuine error was reported. Restores stderr on exit.
    """
    old = sys.stderr
    buf = io.StringIO()
    sys.stderr = buf
    try:
        yield buf
    finally:
        sys.stderr = old


def load_xrk(xrk_path: Path):
    """Wrap libxrk.aim_xrk(). Returns the LogFile object.

    The `import libxrk` is deferred (function-local) so that importing this
    module — e.g. to reuse the cross-correlation helpers in tests — doesn't
    hard-require the optional libxrk dependency until a file is actually read.
    """
    import libxrk
    with _silence_stderr() as captured:
        try:
            return libxrk.aim_xrk(str(xrk_path))
        except Exception:
            # Surface anything libxrk printed to stderr — it's the only place
            # the C layer reports decode failures.
            noise = captured.getvalue().strip()
            if noise:
                print(noise, file=sys.stderr)
            raise


# ---------------------------------------------------------------------------
# Body-frame geometry from the IMU
# ---------------------------------------------------------------------------

def _quiet_mask(gyro: np.ndarray, quiet_rate_dps: float = QUIET_RATE_DPS) -> np.ndarray:
    """Boolean mask of samples where the wheel is ≈stationary."""
    mag = np.linalg.norm(gyro, axis=1)
    return mag < np.deg2rad(quiet_rate_dps)


def detect_column_axis(gyro: np.ndarray) -> np.ndarray:
    """PCA of gyro covariance: the principal eigenvector is the wheel
    rotation axis in body frame (= steering column direction).

    Works for any wheel-mounted camera regardless of how it's angled.
    Sign is canonicalised so the +Z-dominant direction wins; the actual
    sign convention for steering (left/right positive) is then resolved
    by `cross_correlate_offset` returning `sign = ±1`, which `sync_imu_to_xrk`
    folds back into the stored axis.
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

    Returns zeros if there are no quiet samples — callers should check
    the quiet-sample count (see `count_quiet_samples`) to know whether
    the estimate is trustworthy rather than treating zeros as a real bias.
    """
    quiet = _quiet_mask(gyro, quiet_rate_thresh_dps)
    if not quiet.any():
        return np.zeros(3)
    return gyro[quiet].mean(axis=0)


def count_quiet_samples(gyro: np.ndarray,
                        quiet_rate_dps: float = QUIET_RATE_DPS) -> int:
    """How many ≈stationary samples the recording contains. Used to decide
    whether the gravity/bias geometry is trustworthy."""
    return int(_quiet_mask(gyro, quiet_rate_dps).sum())


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

    Returns the magnitude; the sign is folded into the stored column axis.
    Returns NaN if there are no quiet samples — k=NaN propagates loudly
    instead of a silent 0.0 that would disable chassis-yaw subtraction
    while looking like a real horizontal-column geometry.
    """
    quiet = _quiet_mask(stream.gyro, quiet_rate_dps)
    if not quiet.any():
        return float("nan")
    g_body = stream.accel[quiet].mean(axis=0)
    norm = np.linalg.norm(g_body)
    if norm < 1e-6:
        return float("nan")
    world_up = -g_body / norm
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
    camera-handling transients that aren't real motion. The trim is skipped
    (rather than emptying the signal) when it's 0 or would consume the whole
    recording, so short clips and `edge_trim_s=0` still work.
    """
    bias = estimate_quiet_bias(stream.gyro)
    gyro = stream.gyro - bias
    n_col = detect_column_axis(gyro)
    # Elementwise projection (== gyro @ n_col). The matmul form trips a spurious
    # "divide by zero / overflow / invalid in matmul" RuntimeWarning on macOS
    # Accelerate-BLAS for large (N,3)@(3,) products even with finite input;
    # the row-wise multiply-and-sum is identical and silent.
    rate_along_col = (gyro * n_col).sum(axis=1)

    fs = stream.sample_rate_hz
    sos = butter(3, lowpass_hz / (fs / 2.0), btype="low", output="sos")
    yaw_lp = sosfiltfilt(sos, rate_along_col)

    trim = int(edge_trim_s * fs)
    if trim <= 0 or 2 * trim >= len(stream.t):
        return stream.t, yaw_lp, n_col, bias
    return stream.t[trim:-trim], yaw_lp[trim:-trim], n_col, bias


# ---------------------------------------------------------------------------
# XRK side
# ---------------------------------------------------------------------------

def clean_gps_yaw_dps(t_gps: np.ndarray, yr_dps: np.ndarray,
                      glitch_threshold_dps: float = GPS_YAW_GLITCH_DPS,
                      ) -> np.ndarray:
    """Remove GPS fix-instability spikes from a yaw-rate trace (deg/s).

    GPS_Yaw_Rate occasionally reports physically impossible values (we've
    seen 2500°/s) when the fix wobbles, and may contain NaN/inf. Both are
    treated as bad and linearly interpolated over from neighbouring good
    samples. Only the bad samples are replaced (good samples are untouched).

    This is the single source of yaw cleaning, used by BOTH the
    cross-correlation sync and the steering subtraction, so the two never
    disagree about which samples are real.

    If every sample is bad, returns zeros (no usable yaw information).
    """
    yr = np.asarray(yr_dps, dtype=float).copy()
    bad = ~np.isfinite(yr) | (np.abs(yr) > glitch_threshold_dps)
    if bad.all():
        return np.zeros_like(yr)
    if bad.any():
        good = ~bad
        yr[bad] = np.interp(t_gps[bad], t_gps[good], yr[good])
    return yr


def chassis_yaw_from_xrk(log) -> tuple[np.ndarray, np.ndarray]:
    """Returns (t [s], yaw_rate [rad/s]) from XRK `GPS_Yaw_Rate`, glitch-cleaned.

    The cleaning happens here at the source so the cross-correlation sync
    aligns against the same de-spiked signal the steering extraction uses.
    """
    tbl = log.channels.get("GPS_Yaw_Rate")
    if tbl is None:
        raise ValueError("XRK has no GPS_Yaw_Rate channel — cannot sync.")
    t_ms = np.asarray(tbl["timecodes"], dtype=np.float64)
    yr_dps = np.asarray(tbl["GPS_Yaw_Rate"], dtype=np.float64)
    t_s = t_ms / 1000.0
    return t_s, np.deg2rad(clean_gps_yaw_dps(t_s, yr_dps))


# ---------------------------------------------------------------------------
# Cross-correlation
# ---------------------------------------------------------------------------

def _robust_normalise(x: np.ndarray) -> np.ndarray:
    """Zero-median, MAD-scaled. Resistant to outlier spikes that would
    dominate a std-based normalization.

    The scale floor is relative to the signal, not an absolute 1e-12: when
    the signal is flat over >50% of its samples the MAD collapses to 0, so
    we fall back to the std; a truly constant signal returns zeros (it
    carries no alignment information and must not be divided by ~0, which
    would blow a single spike up to ~1e12 and hijack the correlation).
    """
    x = np.asarray(x, dtype=float)
    m = np.median(x)
    centred = x - m
    mad = np.median(np.abs(centred))
    s = 1.4826 * mad
    if s <= 1e-9:                       # MAD collapsed (mostly-flat signal)
        s = float(centred.std())
    if s <= 1e-12:                      # genuinely constant signal
        return np.zeros_like(x)
    return centred / s


def cross_correlate_offset(t_imu: np.ndarray, yaw_imu: np.ndarray,
                           t_xrk: np.ndarray, yaw_xrk: np.ndarray,
                           max_lag_s: float = DEFAULT_MAX_LAG_S,
                           fs: float = SYNC_RESAMPLE_HZ,
                           ) -> tuple[float, float, int]:
    """Find the time offset that best aligns two yaw-rate signals.

    Returns (offset_imu_to_xrk_s, peak_normalized_corr, sign).
      - offset: add this to IMU `t` to land on XRK time.
      - peak: |normalized cross-correlation| at the best lag, genuinely in
        [0, 1] (Pearson-style: divided by the product of the per-lag window
        L2 norms, so the `< 0.5 = weak` gate downstream is calibrated).
      - sign: ±1; if -1, the IMU signal had to be flipped (PCA-detected
        column axis points opposite to GPS-yaw convention).
    """
    dt = 1.0 / fs

    # Resample each onto its own uniform grid
    grid_i = np.arange(t_imu[0], t_imu[-1] + dt, dt)
    grid_x = np.arange(t_xrk[0], t_xrk[-1] + dt, dt)
    y_i = _robust_normalise(np.interp(grid_i, t_imu, yaw_imu))
    y_x = _robust_normalise(np.interp(grid_x, t_xrk, yaw_xrk))

    # Numerator: sliding dot product. correlate(a,b)[k] = Σ a[n]·b[n-lag].
    raw_corr = correlate(y_i, y_x, mode="full")

    # Denominator: per-lag product of the two overlapping windows' L2 norms,
    # computed as sliding sums of squares (correlate with an ones kernel).
    # This makes the result a true normalized cross-correlation, bounded
    # to [-1, 1] by Cauchy-Schwarz — unlike dividing by the overlap count.
    energy_i = correlate(y_i ** 2, np.ones_like(y_x), mode="full")
    energy_x = correlate(np.ones_like(y_i), y_x ** 2, mode="full")
    denom = np.sqrt(np.maximum(energy_i, 0.0) * np.maximum(energy_x, 0.0))
    ncc = np.where(denom > 1e-12, raw_corr / np.maximum(denom, 1e-12), 0.0)

    lag_samples = np.arange(-len(y_x) + 1, len(y_i))
    lag_s = lag_samples * dt

    # Require ≥1s overlap and constrain to ±max_lag_s.
    overlap = np.minimum.reduce([
        len(y_i) - np.maximum(lag_samples, 0),
        len(y_x) + np.minimum(lag_samples, 0),
    ])
    valid = (overlap >= int(fs * 1.0)) & (np.abs(lag_s) <= max_lag_s)
    ncc = np.where(valid, ncc, 0.0)

    best = int(np.argmax(np.abs(ncc)))
    sign = int(np.sign(ncc[best])) or 1
    peak = float(np.abs(ncc[best]))
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
    corr_peak: float              # [0, 1] normalized cross-correlation (>0.5 confident)
    sign: int                     # detected ±1; ALREADY folded into column_axis below
    # Body-frame geometry (from the IMU recording)
    column_axis: np.ndarray       # sign-corrected unit axis; gyro @ column_axis is
                                  # the chassis-yaw-positive steering rate directly
    column_tilt_factor: float     # k = |column · world_up|, used to subtract chassis
                                  # yaw; NaN if geometry could not be estimated
    gyro_bias: np.ndarray         # per-axis rad/s, from quiet samples
    quiet_sample_count: int       # # of ≈stationary samples backing bias/tilt
    # Loaded inputs, so callers don't reload them
    imu_stream: object            # extract_imu.ImuStream (the loaded wheel IMU)
    xrk_log: object               # libxrk.LogFile

    @property
    def column_tilt_deg(self) -> float:
        """Column tilt from world-vertical (degrees), for reporting."""
        return float(np.rad2deg(np.arccos(np.clip(self.column_tilt_factor, -1.0, 1.0))))

    @property
    def geometry_reliable(self) -> bool:
        """True if there were enough quiet samples to trust bias + tilt."""
        return (self.quiet_sample_count >= MIN_QUIET_SAMPLES
                and np.isfinite(self.column_tilt_factor))


def sync_imu_to_xrk(gyroflow_path: Path, xrk_path: Path,
                    *, max_lag_s: float = DEFAULT_MAX_LAG_S,
                    lowpass_hz: float = DEFAULT_LP_HZ,
                    ) -> SyncedXrk:
    """End-to-end sync. Loads both files, runs cross-correlation, returns
    everything downstream tools need (including the loaded IMU stream, so
    callers don't parse the source a second time).

    `gyroflow_path` may be a .gyroflow project file OR a raw Insta360 .mp4 —
    `load_imu` dispatches on the extension.
    """
    from pipeline.extract_imu import load_imu

    stream = load_imu(gyroflow_path, source="wheel")
    log = load_xrk(xrk_path)

    t_imu, yaw_imu, n_col, bias = chassis_yaw_from_imu(stream, lowpass_hz=lowpass_hz)
    t_xrk, yaw_xrk = chassis_yaw_from_xrk(log)
    offset, peak, sign = cross_correlate_offset(
        t_imu, yaw_imu, t_xrk, yaw_xrk, max_lag_s=max_lag_s,
    )

    # Fold the detected sign into the stored axis so consumers never re-apply it.
    column_axis = sign * n_col
    tilt = estimate_column_tilt_factor(stream, column_axis)
    n_quiet = count_quiet_samples(stream.gyro)

    if n_quiet < MIN_QUIET_SAMPLES or not np.isfinite(tilt):
        warnings.warn(
            f"Only {n_quiet} quiet samples in {gyroflow_path.name}; gyro bias "
            f"and column-tilt (k={tilt:.3f}) geometry are unreliable. Steering "
            f"output may retain chassis yaw. Capture a few seconds of stillness "
            f"(wheel centred, kart stopped) for a trustworthy result.",
            stacklevel=2,
        )

    return SyncedXrk(
        gyroflow_path=gyroflow_path,
        xrk_path=xrk_path,
        offset_imu_to_xrk_s=offset,
        corr_peak=peak,
        sign=sign,
        column_axis=column_axis,
        column_tilt_factor=tilt,
        gyro_bias=bias,
        quiet_sample_count=n_quiet,
        imu_stream=stream,
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
    print(f"quiet samples:      {r.quiet_sample_count}"
          + ("" if r.geometry_reliable else "   ⚠ geometry UNRELIABLE — see warning"))
    print(f"sign:               {r.sign:+d}  (folded into column_axis)")
    print(f"offset (IMU→XRK):   {r.offset_imu_to_xrk_s:+.3f} s")
    print(f"peak |corr|:        {r.corr_peak:.4f}"
          + ("   ⚠ WEAK — files may not be from the same session" if r.corr_peak < 0.5 else ""))
    print()
    print(f"XRK laps ({r.xrk_log.laps.num_rows}):")
    for i in range(r.xrk_log.laps.num_rows):
        n = r.xrk_log.laps["num"][i].as_py()
        s = r.xrk_log.laps["start_time"][i].as_py() / 1000
        e = r.xrk_log.laps["end_time"][i].as_py() / 1000
        print(f"  lap {n}: {s:7.2f}s .. {e:7.2f}s   ({e-s:.2f}s)")


if __name__ == "__main__":
    main()
