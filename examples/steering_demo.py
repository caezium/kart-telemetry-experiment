"""Illustrate steering recovery and its high-pass limitation with known inputs.

Run from the repository root: python -m examples.steering_demo
No recordings or external telemetry decoder are used. Geometry, bias and clock
alignment are deliberately supplied, not estimated. This is a signal-processing
illustration, not an end-to-end sensor validation or a physical kart simulator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.per_lap import DEFAULT_STEERING_HP_HZ, steering_from_synced
from pipeline.extract_imu import ImuStream
from pipeline.sync_xrk import SyncedXrk


def demonstrate(out_dir: Path) -> dict:
    fs = 200.0
    t = np.arange(0.0, 100.0, 1.0 / fs)
    # A short sequence of corrections followed by a deliberately long hold.
    envelope = np.exp(-((t - 22.0) / 5.0) ** 4)
    angle = 25.0 * np.sin(2.0 * np.pi * 0.6 * (t - 22.0)) * envelope
    angle += 17.5 * (np.tanh((t - 50.0) / 0.8) - np.tanh((t - 75.0) / 0.8))
    rate = np.gradient(angle, t)
    yaw = 45.0 * np.exp(-((t - 24.0) / 7.0) ** 2)
    yaw -= 30.0 * np.exp(-((t - 62.0) / 10.0) ** 2)

    tilt = np.deg2rad(40.0)
    axis = np.array([np.sin(tilt), 0.0, np.cos(tilt)])
    k = float(np.cos(tilt))
    vertical = np.array([0.0, 0.0, 1.0])
    # Express world vertical in the rotating wheel-camera frame (Rodrigues).
    theta = -np.deg2rad(angle)
    up = (np.cos(theta)[:, None] * vertical
          + np.sin(theta)[:, None] * np.cross(axis, vertical)
          + (1.0 - np.cos(theta))[:, None] * k * axis)
    gyro = np.deg2rad(rate)[:, None] * axis + np.deg2rad(yaw)[:, None] * up
    stream = ImuStream(t=t, gyro=gyro, accel=9.80665 * up,
                       sample_rate_hz=fs, source="wheel", orientation="XYZ")
    # Minimal in-memory equivalent of the logger channel the function reads.
    log = SimpleNamespace(channels={"GPS_Yaw_Rate": {
        "timecodes": t * 1000.0, "GPS_Yaw_Rate": yaw,
    }})
    sync = SyncedXrk(
        gyroflow_path=Path("synthetic.gyroflow"), xrk_path=Path("synthetic.xrk"),
        offset_imu_to_xrk_s=0.0, corr_peak=float("nan"), sign=1,
        column_axis=axis, column_tilt_factor=k, gyro_bias=np.zeros(3),
        quiet_sample_count=0, imu_stream=stream, xrk_log=log,
    )
    # These are the real analysis function and its default drift filter.
    recovered_t, recovered_angle, recovered_rate = steering_from_synced(stream, sync)
    assert np.array_equal(t, recovered_t)
    hold = (t >= 60.0) & (t <= 65.0)
    summary = {
        "kind": "synthetic illustration, not real-world validation",
        "known_inputs": ["column axis", "tilt", "gyro bias", "clock alignment", "chassis yaw"],
        "sample_rate_hz": fs,
        "column_tilt_deg": 40.0,
        "highpass_hz": DEFAULT_STEERING_HP_HZ,
        "steering_rate_rmse_dps": float(np.sqrt(np.mean((recovered_rate - rate) ** 2))),
        "hold_window_s": [60.0, 65.0],
        "hold_input_mean_deg": float(angle[hold].mean()),
        "hold_output_mean_deg": float(recovered_angle[hold].mean()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / "traces.csv", np.column_stack([
        t, angle, recovered_angle, rate, recovered_rate, yaw,
    ]), delimiter=",", header=("time_s,input_angle_deg,output_angle_deg,"
                               "input_rate_dps,output_rate_dps,chassis_yaw_dps"),
               comments="")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    with plt.rc_context({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold"}):
        fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        fig.subplots_adjust(top=0.81, bottom=0.13, hspace=0.48, left=0.09, right=0.97)
        fig.text(0.09, 0.95, "What the steering estimate keeps — and loses",
                 fontsize=19, weight="bold")
        fig.text(0.09, 0.90, "Synthetic motion · Known geometry and alignment · No sensor noise",
                 color="#52606d", fontsize=11)
        projected = np.rad2deg((gyro * axis).sum(axis=1))
        axs[0].plot(t, projected, color="#aab5bf", lw=1.5, label="Gyro column projection")
        axs[0].plot(t, rate, color="#172b4d", lw=2.5, label="Known steering rate")
        axs[0].plot(t, recovered_rate, color="#007f78", lw=1.4,
                    linestyle="--", label="After chassis-yaw subtraction")
        axs[0].set(title="1. Subtraction separates steering rate in the idealized model",
                   ylabel="Rate (°/s)")
        axs[0].legend(loc="upper right", frameon=False, fontsize=9)
        axs[1].axvspan(50, 75, color="#fff1dc", zorder=0)
        axs[1].plot(t, angle, color="#172b4d", lw=2, label="Known wheel angle")
        axs[1].plot(t, recovered_angle, color="#007f78", lw=1.8,
                    label="Integrated + default drift filter")
        axs[1].set(title="2. The drift filter also removes a held steering angle",
                   ylabel="Angle (°)", xlabel="Time (s)")
        axs[1].annotate("35° held input", xy=(61, 35), xytext=(61, 43),
                        ha="center", fontsize=10, color="#855800")
        axs[1].set_ylim(-38, 52)
        axs[1].legend(loc="lower right", frameon=False, fontsize=9)
        for ax in axs:
            ax.set_xlim(0, 100)
            ax.grid(axis="y", color="#e8edf2", linewidth=0.8)
            ax.set_axisbelow(True)
        fig.text(0.09, 0.035,
                 "Illustrates the existing processing function; does not validate camera decoding, axis estimation or track accuracy.",
                 fontsize=9, color="#52606d")
        fig.savefig(out_dir / "steering-demo.png", dpi=160, facecolor="white")
        plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results/demo"))
    args = parser.parse_args()
    print(json.dumps(demonstrate(args.out_dir), indent=2))
    print(f"Wrote steering-demo.png, traces.csv and summary.json to {args.out_dir}")


if __name__ == "__main__":
    main()
