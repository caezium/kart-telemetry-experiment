"""Quick visual sanity check of a .gyroflow file's IMU stream.

Plots gyro (x/y/z), accel (x/y/z), and combined magnitudes over the
full recording length. Decimates to ~5k points so the PNG stays a
reasonable size at 1 kHz × 500 s. Saves alongside the input file as
`<name>.imu.png`.

    python -m analysis.quicklook /path/to/file.gyroflow

For a fast sniff of any recording — was the camera moving, what's the
rough steering activity look like, is gravity in the expected axis,
any obvious clipping or dropouts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pipeline.extract_imu import load_gyroflow


GRAVITY_MPS2 = 9.80665


def quicklook(path: Path, source: str = "wheel", max_points: int = 5000) -> Path:
    """Render a 3-panel summary plot of a .gyroflow file. Returns output PNG path."""
    stream = load_gyroflow(path, source=source)
    n = len(stream.t)
    step = max(1, n // max_points)

    t = stream.t[::step]
    gyro_dps = np.rad2deg(stream.gyro[::step])
    accel = stream.accel[::step]
    gyro_mag_dps = np.rad2deg(np.linalg.norm(stream.gyro[::step], axis=1))
    accel_mag = np.linalg.norm(stream.accel[::step], axis=1)

    fig, (ax_g, ax_a, ax_m) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Gyro: 3 axes
    for i, label, color in zip(range(3), ("gx", "gy", "gz"),
                                ("tab:blue", "tab:orange", "tab:green")):
        ax_g.plot(t, gyro_dps[:, i], label=label, alpha=0.75, lw=0.7, color=color)
    ax_g.set_ylabel("angular velocity (deg/s)")
    ax_g.legend(loc="upper right", ncol=3)
    ax_g.set_title(
        f"{path.name}  —  gyro post-orientation '{stream.orientation}', "
        f"{stream.sample_rate_hz:.0f} Hz, {n:,} samples, {stream.t[-1]:.1f} s"
    )
    ax_g.grid(alpha=0.3)
    ax_g.axhline(0, color="black", lw=0.4)

    # Accel: 3 axes + gravity reference lines
    for i, label, color in zip(range(3), ("ax", "ay", "az"),
                                ("tab:blue", "tab:orange", "tab:green")):
        ax_a.plot(t, accel[:, i], label=label, alpha=0.75, lw=0.7, color=color)
    ax_a.axhline(GRAVITY_MPS2, color="gray", linestyle=":", lw=0.6, label="±g")
    ax_a.axhline(-GRAVITY_MPS2, color="gray", linestyle=":", lw=0.6)
    ax_a.set_ylabel("acceleration (m/s²)")
    ax_a.legend(loc="upper right", ncol=4)
    ax_a.grid(alpha=0.3)

    # Magnitudes — gyro on left axis, accel on right
    ax_m.plot(t, gyro_mag_dps, color="tab:blue", lw=0.7, label="|gyro|")
    ax_m.set_ylabel("|gyro| (deg/s)", color="tab:blue")
    ax_m.tick_params(axis="y", labelcolor="tab:blue")
    ax_ma = ax_m.twinx()
    ax_ma.plot(t, accel_mag, color="tab:red", lw=0.7, alpha=0.7, label="|accel|")
    ax_ma.axhline(GRAVITY_MPS2, color="gray", linestyle=":", lw=0.6)
    ax_ma.set_ylabel("|accel| (m/s²)", color="tab:red")
    ax_ma.tick_params(axis="y", labelcolor="tab:red")
    ax_m.set_xlabel("time (s)")
    ax_m.grid(alpha=0.3)

    fig.tight_layout()
    out = path.with_name(path.stem + ".imu.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gyroflow_path", type=Path)
    ap.add_argument("--source", default="wheel",
                    help="tag for the ImuStream; doesn't affect plotting")
    args = ap.parse_args()
    if not args.gyroflow_path.exists():
        raise SystemExit(f"Not found: {args.gyroflow_path}")
    out = quicklook(args.gyroflow_path, source=args.source)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
