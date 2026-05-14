"""Quick steering-angle viewer for a single .gyroflow file. Limited use.

WHEN THIS WORKS:
    Only when the camera is mounted with its lens axis aligned to the
    steering column axis ("Mount 1" — canonical wheel-hub mount, lens
    pointing along the column). In that case body-frame gz IS the
    steering rate, and integrating it gives angle.

WHEN THIS DOES NOT WORK:
    Tilted wheel mounts where the lens isn't along the column (the
    user's "Mount 2" with lens angled forward/down to see the front
    tire). For those mounts:
      - The column axis in body frame is NOT gz; it's wherever PCA
        of the gyro covariance points.
      - The integrated rotation is contaminated by chassis yaw at
        corner frequencies, which a high-pass filter cannot remove.

For tilted mounts, or any case where you have MyChron / XRK data:
    Use `analysis.per_lap` instead. It auto-detects the column axis
    and subtracts chassis yaw using the XRK's GPS yaw rate, giving
    clean per-corner steering input.

Usage:
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
