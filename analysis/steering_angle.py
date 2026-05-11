"""Plot the steering wheel angle from a wheel-mounted .gyroflow recording.

What this measures:
    Assuming the camera is mounted on the steering wheel hub with its lens
    axis aligned with the steering rotation axis (Mount 1 in the project
    brainstorm), the post-orientation Z gyro IS the steering angular
    velocity. Integrating it over time gives steering angle.

    For Insta360 Go 3S, telemetry-parser applies the 'yXZ' axis convention,
    which keeps the lens axis on Z. So `gz` after orientation = steering
    rate, and this works out of the box.

Drift handling:
    The integrator re-zeros itself whenever the rate sits near zero AND
    the integrated angle has wandered near zero — a "wheel straight"
    detection. Honest drift will accumulate over a long stint; the reset
    keeps it bounded.

Output:
    - A two-panel PNG next to the input file (.steering.png): angle on
      top, rate on bottom.
    - Console summary: peak left, peak right, peak rate, time above 30°.

Caveats:
    If the camera was NOT wheel-mounted (helmet, chest, hand-held), the
    integrated "angle" is just rotation about the camera's lens axis and
    has no physical steering meaning.

    python -m analysis.steering_angle /path/to/file.gyroflow
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.steering_metrics import STEER_AXIS, integrate_angle_with_drift_reset
from pipeline.extract_imu import load_gyroflow


_AXIS_IDX = {"gx": 0, "gy": 1, "gz": 2}


def steering_angle(path: Path, source: str = "wheel", max_points: int = 8000) -> tuple[Path, dict]:
    stream = load_gyroflow(path, source=source)
    axis_idx = _AXIS_IDX[STEER_AXIS]
    rate_rps = stream.gyro[:, axis_idx]
    angle_rad = integrate_angle_with_drift_reset(stream.t, rate_rps)

    angle_deg = np.rad2deg(angle_rad)
    rate_dps = np.rad2deg(rate_rps)

    # Decimate for plotting
    n = len(stream.t)
    step = max(1, n // max_points)
    t = stream.t[::step]

    fig, (ax_a, ax_r) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax_a.plot(t, angle_deg[::step], color="tab:blue", lw=0.9)
    ax_a.axhline(0, color="black", lw=0.4)
    ax_a.set_ylabel("steering angle (deg)")
    ax_a.set_title(
        f"{path.name}  —  steering angle from integrated {STEER_AXIS} "
        f"(orientation '{stream.orientation}', {stream.sample_rate_hz:.0f} Hz)"
    )
    ax_a.grid(alpha=0.3)

    ax_r.plot(t, rate_dps[::step], color="tab:orange", lw=0.7)
    ax_r.axhline(0, color="black", lw=0.4)
    ax_r.set_ylabel("steering rate (deg/s)")
    ax_r.set_xlabel("time (s)")
    ax_r.grid(alpha=0.3)

    fig.tight_layout()
    out = path.with_name(path.stem + ".steering.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)

    stats = {
        "peak_left_deg":     float(angle_deg.min()),
        "peak_right_deg":    float(angle_deg.max()),
        "peak_rate_dps":     float(np.abs(rate_dps).max()),
        "time_above_30deg_s": float((np.abs(angle_deg) > 30).sum() / stream.sample_rate_hz),
        "time_above_60deg_s": float((np.abs(angle_deg) > 60).sum() / stream.sample_rate_hz),
    }
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gyroflow_path", type=Path)
    ap.add_argument("--source", default="wheel",
                    help="tag for the ImuStream loader; doesn't affect output")
    args = ap.parse_args()
    if not args.gyroflow_path.exists():
        raise SystemExit(f"Not found: {args.gyroflow_path}")

    out, stats = steering_angle(args.gyroflow_path, source=args.source)
    print(f"wrote {out}")
    print()
    print(f"peak left:        {stats['peak_left_deg']:+7.1f} deg")
    print(f"peak right:       {stats['peak_right_deg']:+7.1f} deg")
    print(f"peak rate:        {stats['peak_rate_dps']:7.0f} deg/s")
    print(f"time above 30°:   {stats['time_above_30deg_s']:7.1f} s")
    print(f"time above 60°:   {stats['time_above_60deg_s']:7.1f} s")


if __name__ == "__main__":
    main()
