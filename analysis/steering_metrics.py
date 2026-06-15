"""
Compute steering input metrics from a wheel-mounted Go 3S IMU stream.

⚠ SCOPE / KNOWN LIMITATION
    This module integrates a SINGLE body axis (`STEER_AXIS`, default gz)
    with NO chassis-yaw subtraction. That is only correct for the
    canonical "Mount 1" geometry — camera lens-axis aligned with the
    steering column. For a tilted wheel mount (lens angled at the tire),
    the integrated angle is contaminated by chassis yaw at corner
    frequencies, which a high-pass filter cannot remove (see commit that
    added analysis/per_lap.py).

    The CORRECT path, whenever MyChron/XRK data is available, is
    `analysis.per_lap.steering_from_synced`, which subtracts
    k · GPS_Yaw_Rate using the PCA-detected column axis. This module is
    retained only for the lens-along-column case and for IMU-only
    sessions with no logger. `analyze()` emits a runtime warning so the
    divergence between the two paths is never silent.

The wheel-mount convention used here:
    - Camera lens-axis (z) aligned with the steering rotation axis
    - gyro_z is therefore steering angular velocity, rad/s
    - Positive gz = right turn (driver-perspective clockwise)

Outputs per session:
    - angle, rate, jerk timeseries
    - per-corner segmented metrics (corners detected as continuous regions
      where |angle| > entry threshold)
    - smoothness score (integral of |jerk| / corner duration)
    - correction count (sub-amplitude reversals during a corner)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


STEER_AXIS = "gz"           # change if mount orientation differs
ANGLE_RESET_HOLD_S = 1.5    # seconds of |angle| < threshold to trigger drift reset
ANGLE_RESET_THRESHOLD_RAD = np.deg2rad(5.0)
CORNER_ENTRY_THRESHOLD_RAD = np.deg2rad(15.0)
CORNER_EXIT_THRESHOLD_RAD = np.deg2rad(8.0)
MIN_CORNER_DURATION_S = 0.4


@dataclass
class CornerMetrics:
    t_start: float
    t_end: float
    direction: str            # "left" | "right"
    peak_angle_deg: float
    peak_rate_deg_s: float
    smoothness: float         # lower = smoother
    correction_count: int
    time_to_apex_s: float


def integrate_angle_with_drift_reset(t: np.ndarray, rate: np.ndarray) -> np.ndarray:
    """
    Trapezoidal integration of angular rate, with drift correction:
    whenever the rate stays near zero for ANGLE_RESET_HOLD_S, snap angle to 0.
    """
    angle = np.zeros_like(rate)
    dt = np.diff(t, prepend=t[0])
    near_zero = np.abs(rate) < np.deg2rad(2.0)

    hold_start_idx = None
    for i in range(1, len(t)):
        angle[i] = angle[i - 1] + 0.5 * (rate[i] + rate[i - 1]) * dt[i]
        if near_zero[i]:
            if hold_start_idx is None:
                hold_start_idx = i
            elif t[i] - t[hold_start_idx] >= ANGLE_RESET_HOLD_S and abs(angle[i]) < ANGLE_RESET_THRESHOLD_RAD:
                # snap a smooth taper of the last samples toward zero
                taper_n = min(i - hold_start_idx, 50)
                angle[i - taper_n + 1:i + 1] = np.linspace(angle[i - taper_n], 0.0, taper_n)
                hold_start_idx = i
        else:
            hold_start_idx = None
    return angle


def detect_corners(t: np.ndarray, angle: np.ndarray) -> list[tuple[int, int]]:
    abs_angle = np.abs(angle)
    in_corner = False
    start = 0
    corners: list[tuple[int, int]] = []
    for i, a in enumerate(abs_angle):
        if not in_corner and a > CORNER_ENTRY_THRESHOLD_RAD:
            in_corner = True
            start = i
        elif in_corner and a < CORNER_EXIT_THRESHOLD_RAD:
            in_corner = False
            if t[i] - t[start] >= MIN_CORNER_DURATION_S:
                corners.append((start, i))
    return corners


def count_corrections(rate_segment: np.ndarray, dt: float) -> int:
    """
    A correction is a zero-crossing of the angular rate (sign reversal)
    that survives a low-pass at ~5 Hz. Quick wiggles only — large
    counter-steering reversals are also counted (and that's intentional).
    """
    if len(rate_segment) < 5:
        return 0
    # smooth a touch beyond the loader's filter to drop noise
    window = max(3, int(0.05 / dt))
    kernel = np.ones(window) / window
    smooth = np.convolve(rate_segment, kernel, mode="same")
    signs = np.sign(smooth)
    zero_crossings = np.sum(np.diff(signs) != 0)
    return int(zero_crossings)


def compute_corner_metrics(
    t: np.ndarray,
    angle: np.ndarray,
    rate: np.ndarray,
    jerk: np.ndarray,
    corner_idx: tuple[int, int],
) -> CornerMetrics:
    a, b = corner_idx
    t_seg = t[a:b]
    angle_seg = angle[a:b]
    rate_seg = rate[a:b]
    jerk_seg = jerk[a:b]
    duration = t_seg[-1] - t_seg[0]
    apex_idx = int(np.argmax(np.abs(angle_seg)))

    return CornerMetrics(
        t_start=float(t_seg[0]),
        t_end=float(t_seg[-1]),
        direction="right" if angle_seg[apex_idx] > 0 else "left",
        peak_angle_deg=float(np.rad2deg(angle_seg[apex_idx])),
        peak_rate_deg_s=float(np.rad2deg(np.max(np.abs(rate_seg)))),
        smoothness=float(np.trapezoid(np.abs(jerk_seg), t_seg) / duration),
        correction_count=count_corrections(rate_seg, np.median(np.diff(t_seg))),
        time_to_apex_s=float(t_seg[apex_idx] - t_seg[0]),
    )


def analyze(session_dir: Path) -> pd.DataFrame:
    import warnings
    warnings.warn(
        "steering_metrics integrates a single body axis with no chassis-yaw "
        "subtraction (correct only for a lens-along-column mount). For a tilted "
        "wheel mount or any session with MyChron/XRK data, use "
        "analysis.per_lap instead — the angle here will be contaminated by "
        "chassis yaw at corner frequencies.",
        stacklevel=2,
    )
    wheel_path = session_dir / "extracted" / "wheel_imu.parquet"
    if not wheel_path.exists():
        raise SystemExit(f"{wheel_path} not found; run extract_imu.py first")

    df = pd.read_parquet(wheel_path)
    t = df["t"].to_numpy()
    rate = df[STEER_AXIS].to_numpy()
    angle = integrate_angle_with_drift_reset(t, rate)
    jerk = np.gradient(np.gradient(rate, t), t)

    corners = detect_corners(t, angle)
    print(f"  detected {len(corners)} corners")

    metrics = [
        compute_corner_metrics(t, angle, rate, jerk, c) for c in corners
    ]
    metrics_df = pd.DataFrame([m.__dict__ for m in metrics])

    out = session_dir / "extracted" / "steering_metrics.parquet"
    metrics_df.to_parquet(out)
    print(f"  wrote {out}")

    if not metrics_df.empty:
        print("\n  Summary:")
        print(f"    median smoothness:    {metrics_df['smoothness'].median():.2f} rad/s^3 avg")
        print(f"    median peak angle:    {metrics_df['peak_angle_deg'].abs().median():.1f} deg")
        print(f"    median peak rate:     {metrics_df['peak_rate_deg_s'].median():.1f} deg/s")
        print(f"    avg corrections/corner: {metrics_df['correction_count'].mean():.1f}")

    return metrics_df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session_dir", type=Path)
    args = ap.parse_args()
    analyze(args.session_dir)


if __name__ == "__main__":
    main()
