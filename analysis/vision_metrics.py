"""
Compute driver vision metrics from a helmet-mounted Go 3S IMU stream,
optionally correlated against the wheel-mounted stream.

Core metric: look-ahead lead time.
    For each corner, find the onset of head yaw motion (driver swinging
    gaze to the apex). Find the onset of steering input. Lead time =
    t_steering_onset - t_head_yaw_onset. Positive = eyes lead hands (good).
    Near zero = reactive driver. Negative = panic / over-steered into a
    surprise.

Secondary metrics:
    - apex fixation duration: time head holds steady on apex
    - head stability under G: rms of head pitch/roll during high-G corners
    - fatigue trend: lap-by-lap rolling head-stability rms

Convention: helmet-mounted with lens forward, top of camera up.
    - gx: pitch rate (head nod)
    - gy: roll rate (head tilt)
    - gz: yaw rate (head turn left/right)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


HEAD_YAW_AXIS = "gz"
HEAD_YAW_ONSET_THRESHOLD_RAD_S = np.deg2rad(20.0)
STEER_ONSET_THRESHOLD_RAD_S = np.deg2rad(15.0)
ONSET_SEARCH_WINDOW_S = 1.5     # how far before steering onset to look for head onset
LEAD_TIME_PLAUSIBLE_RANGE = (-0.2, 1.0)  # seconds; outside this we discard


@dataclass
class VisionEvent:
    corner_idx: int
    t_steering_onset: float
    t_head_yaw_onset: float | None
    lead_time_s: float | None
    head_yaw_peak_deg_s: float


def _find_onset_backwards(t: np.ndarray, rate: np.ndarray, t_ref: float, threshold: float, window_s: float) -> float | None:
    """Search backwards from t_ref up to window_s; return first time |rate| crosses threshold."""
    mask = (t < t_ref) & (t > t_ref - window_s)
    if not mask.any():
        return None
    t_window = t[mask]
    r_window = np.abs(rate[mask])
    above = r_window > threshold
    if not above.any():
        return None
    # earliest sample in window where threshold is exceeded
    return float(t_window[np.argmax(above)])


def _find_steering_onset(t_master: np.ndarray, gz: np.ndarray, t_corner_start: float) -> float | None:
    """The corner detector flagged angle > 15deg; the actual rate onset is earlier."""
    return _find_onset_backwards(
        t_master, gz, t_corner_start, STEER_ONSET_THRESHOLD_RAD_S, window_s=2.0
    )


def analyze(session_dir: Path) -> pd.DataFrame:
    extracted = session_dir / "extracted"
    wheel_path = extracted / "wheel_imu.parquet"
    helmet_path = extracted / "helmet_imu.parquet"
    metrics_path = extracted / "steering_metrics.parquet"

    if not helmet_path.exists():
        print("  no helmet_imu.parquet — skipping vision metrics")
        return pd.DataFrame()
    if not metrics_path.exists():
        raise SystemExit("Run steering_metrics.py first")

    helmet = pd.read_parquet(helmet_path)
    wheel = pd.read_parquet(wheel_path)
    corners = pd.read_parquet(metrics_path)

    if "t_master" not in wheel.columns or "t_master" not in helmet.columns:
        raise SystemExit("Run sync_streams.py first; t_master not present")

    events: list[VisionEvent] = []
    for i, corner in corners.iterrows():
        t_steer_onset = _find_steering_onset(
            wheel["t_master"].to_numpy(),
            wheel["gz"].to_numpy(),
            corner["t_start"],
        )
        if t_steer_onset is None:
            continue

        t_head_onset = _find_onset_backwards(
            helmet["t_master"].to_numpy(),
            helmet[HEAD_YAW_AXIS].to_numpy(),
            t_steer_onset,
            HEAD_YAW_ONSET_THRESHOLD_RAD_S,
            window_s=ONSET_SEARCH_WINDOW_S,
        )

        if t_head_onset is None:
            lead = None
        else:
            lead = t_steer_onset - t_head_onset
            if not (LEAD_TIME_PLAUSIBLE_RANGE[0] <= lead <= LEAD_TIME_PLAUSIBLE_RANGE[1]):
                lead = None

        # peak head yaw rate during the steering input
        mask = (
            (helmet["t_master"] >= corner["t_start"] - 0.5) &
            (helmet["t_master"] <= corner["t_end"])
        )
        peak_yaw = float(np.max(np.abs(helmet.loc[mask, HEAD_YAW_AXIS].to_numpy())) if mask.any() else 0.0)

        events.append(VisionEvent(
            corner_idx=int(i),
            t_steering_onset=t_steer_onset,
            t_head_yaw_onset=t_head_onset,
            lead_time_s=lead,
            head_yaw_peak_deg_s=float(np.rad2deg(peak_yaw)),
        ))

    df = pd.DataFrame([e.__dict__ for e in events])
    out = extracted / "vision_metrics.parquet"
    df.to_parquet(out)
    print(f"  wrote {out}")

    valid = df.dropna(subset=["lead_time_s"])
    if not valid.empty:
        print(f"\n  Vision summary ({len(valid)} valid corners):")
        print(f"    median lead time:       {valid['lead_time_s'].median() * 1000:.0f} ms")
        print(f"    median head yaw peak:   {valid['head_yaw_peak_deg_s'].median():.0f} deg/s")
        worst = valid.nsmallest(3, "lead_time_s")
        print(f"    most reactive corners (lowest lead time):")
        for _, row in worst.iterrows():
            print(f"      corner {int(row['corner_idx'])}: {row['lead_time_s']*1000:.0f} ms")

    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session_dir", type=Path)
    args = ap.parse_args()
    analyze(args.session_dir)


if __name__ == "__main__":
    main()
