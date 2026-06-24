"""Validate the camera-only steering against a MyChron 6 2T `Steering Angle`.

This is a DEBUG / verification tool, not part of the production signal path. The
deliverable stays camera-only; the XRK steering channel is used solely to check
how well the wheel-cam recovers steering input.

What it does
------------
1. Sync the wheel-cam IMU to the XRK (cross-correlation of chassis yaw) — works
   with a .gyroflow or a Pro-mode .mp4 (see pipeline.extract_imu.load_imu).
2. Build the production camera steering (chassis-yaw-subtracted, integrated) on
   the XRK clock (analysis.per_lap.steering_from_synced).
3. Load the XRK `Steering Angle` channel.
4. Align the two (a small residual lag is found by cross-correlating their rates,
   on top of the coarse clock sync), fit camera ≈ gain·xrk + offset, and report
   correlation (angle and rate), gain, RMS.
5. Per-lap overlay plots.

Interpreting the result
-----------------------
The camera physically measures wheel rotation; the XRK channel's provenance is
uncertain (see analysis.characterize_steering_source — it reads at standstill and
couples only weakly to chassis yaw, i.e. it is NOT a chassis-yaw restatement, but
whether it is a real column sensor or an AiM estimate is unknown). So:
  * high corr + gain≈1  → both agree; strong cross-validation of the camera method
  * high corr + gain≠1  → same shape, different scale (units/steering-ratio) —
                          still validates the *dynamics*, calibrate the scale
  * low corr            → they disagree; the camera (a direct measurement) is the
                          more trustworthy steering source — investigate the XRK.

Usage
-----
    python -m analysis.validate_steering wheel.mp4 session.xrk
    python -m analysis.validate_steering wheel.gyroflow session.xrk --all --out-dir results/val/
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate

from pipeline.extract_imu import ImuStream
from pipeline.sync_xrk import SyncedXrk, sync_imu_to_xrk
from analysis.per_lap import (
    choose_lap_windows,
    steering_from_synced,
)

STEERING_CHANNEL = "Steering Angle"
COMMON_FS = 50.0           # XRK steering native rate
MAX_RESIDUAL_LAG_S = 2.0   # sync is already coarse-aligned; only mop up a small lag


# ---------------------------------------------------------------------------
# XRK steering + alignment helpers
# ---------------------------------------------------------------------------

def xrk_steering(log) -> tuple[np.ndarray, np.ndarray]:
    """(t_seconds, steering_deg) from the XRK `Steering Angle` channel."""
    tbl = log.channels.get(STEERING_CHANNEL)
    if tbl is None or tbl.num_rows == 0:
        raise ValueError(
            f"XRK has no '{STEERING_CHANNEL}' channel — this validation needs a "
            "MyChron 6 2T log. (Old MyChron 5/6 logs do not record steering.)"
        )
    t = np.asarray(tbl["timecodes"], dtype=np.float64) / 1000.0
    v = np.asarray(tbl[STEERING_CHANNEL], dtype=np.float64)
    return t, v


def _resample_common(t_a, a, t_b, b, fs=COMMON_FS):
    """Resample two time series onto a shared uniform grid over their overlap."""
    lo, hi = max(t_a[0], t_b[0]), min(t_a[-1], t_b[-1])
    if hi - lo < 1.0:
        raise ValueError(f"camera and XRK steering overlap is too short ({hi-lo:.2f}s)")
    grid = np.arange(lo, hi, 1.0 / fs)
    return grid, np.interp(grid, t_a, a), np.interp(grid, t_b, b)


def _best_lag(a, b, fs=COMMON_FS, max_lag_s=MAX_RESIDUAL_LAG_S):
    """Residual lag (s) and signed peak corr aligning a to b (zero-mean, unit-var).

    Positive lag means `a` must be shifted later to match `b`.
    """
    an = (a - a.mean()) / (a.std() + 1e-12)
    bn = (b - b.mean()) / (b.std() + 1e-12)
    c = correlate(an, bn, mode="full") / len(an)
    lags = np.arange(-len(an) + 1, len(an)) / fs
    m = np.abs(lags) <= max_lag_s
    idx = np.argmax(np.abs(c[m]))
    return float(lags[m][idx]), float(c[m][idx])


def _fit_gain_offset(cam, xrk):
    """Least-squares cam ≈ gain·xrk + offset; returns (gain, offset, rms_after_fit)."""
    A = np.vstack([xrk, np.ones_like(xrk)]).T
    (gain, offset), *_ = np.linalg.lstsq(A, cam, rcond=None)
    rms = float(np.sqrt(np.mean((cam - (gain * xrk + offset)) ** 2)))
    return float(gain), float(offset), rms


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class SteeringValidation:
    n: int
    overlap_s: float
    residual_lag_s: float       # small lag on top of the coarse clock sync
    corr_angle: float           # signed; |·| near 1 = agreement
    corr_rate: float            # rate-domain corr (robust to integration drift)
    gain: float                 # camera_deg per xrk_deg
    offset_deg: float
    rms_deg: float              # after gain/offset fit
    sign: int

    def summary(self) -> str:
        return (f"overlap {self.overlap_s:.0f}s  |  corr(angle)={self.corr_angle:+.3f}  "
                f"corr(rate)={self.corr_rate:+.3f}  |  cam ≈ {self.gain:+.2f}·xrk "
                f"{self.offset_deg:+.1f}°  RMS={self.rms_deg:.1f}°  "
                f"(residual lag {self.residual_lag_s:+.2f}s)")


def compare_steering(stream: ImuStream, sync: SyncedXrk) -> tuple[
        SteeringValidation, dict]:
    """Compare camera steering to XRK Steering Angle over the whole overlap.

    Returns the scalar validation plus a dict of the aligned series (for plotting):
    {grid, cam_angle, xrk_angle, cam_rate, xrk_rate}.
    """
    t_cam, cam_angle, cam_rate = steering_from_synced(stream, sync)
    t_st, st_angle = xrk_steering(sync.xrk_log)

    grid, cam_a, xrk_a = _resample_common(t_cam, cam_angle, t_st, st_angle)
    # XRK steering rate by finite difference on the common grid
    xrk_r = np.gradient(xrk_a, 1.0 / COMMON_FS)
    _, cam_r, _ = _resample_common(t_cam, cam_rate, t_st, st_angle)

    # Residual lag from the rate signals (sharper than the integrated angle).
    lag_s, _ = _best_lag(cam_r, xrk_r)
    shift = int(round(lag_s * COMMON_FS))
    if shift != 0:                       # apply the residual lag to the camera side
        cam_a = np.roll(cam_a, -shift)
        cam_r = np.roll(cam_r, -shift)
        sl = slice(abs(shift), len(grid) - abs(shift))
        grid, cam_a, xrk_a, cam_r, xrk_r = grid[sl], cam_a[sl], xrk_a[sl], cam_r[sl], xrk_r[sl]

    corr_angle = float(np.corrcoef(cam_a, xrk_a)[0, 1])
    corr_rate = float(np.corrcoef(cam_r, xrk_r)[0, 1])
    sign = -1 if corr_angle < 0 else 1
    gain, offset, rms = _fit_gain_offset(cam_a, sign * xrk_a)

    val = SteeringValidation(
        n=len(grid), overlap_s=float(grid[-1] - grid[0]),
        residual_lag_s=lag_s, corr_angle=corr_angle, corr_rate=corr_rate,
        gain=gain, offset_deg=offset, rms_deg=rms, sign=sign,
    )
    series = {"grid": grid, "cam_angle": cam_a, "xrk_angle": xrk_a,
              "cam_rate": cam_r, "xrk_rate": xrk_r, "gain": gain,
              "offset": offset, "sign": sign}
    return val, series


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_overlay(series: dict, val: SteeringValidation, out_path: Path,
                 *, title: str = "", t_window: tuple[float, float] | None = None):
    """Overlay camera steering vs scaled XRK steering (angle + rate panels)."""
    grid = series["grid"]
    mask = np.ones_like(grid, bool)
    if t_window is not None:
        mask = (grid >= t_window[0]) & (grid < t_window[1])
    t = grid[mask] - grid[mask][0]
    xrk_scaled = series["sign"] * series["gain"] * series["xrk_angle"][mask] + series["offset"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                   constrained_layout=True)
    ax1.plot(t, series["cam_angle"][mask], lw=1.0, color="tab:blue",
             label="camera (wheel-cam IMU)")
    ax1.plot(t, xrk_scaled, lw=1.0, color="tab:red", alpha=0.8,
             label="MyChron Steering Angle (scaled)")
    ax1.axhline(0, color="k", lw=0.4)
    ax1.set_ylabel("steering angle (deg)")
    ax1.set_title(f"{title}\n{val.summary()}", fontsize=10)
    ax1.legend(loc="upper right"); ax1.grid(alpha=0.3)

    ax2.plot(t, series["cam_rate"][mask], lw=0.8, color="tab:blue", label="camera rate")
    ax2.plot(t, series["sign"] * series["xrk_rate"][mask], lw=0.8, color="tab:red",
             alpha=0.7, label="MyChron rate")
    ax2.axhline(0, color="k", lw=0.4)
    ax2.set_ylabel("steering rate (deg/s)"); ax2.set_xlabel("time (s)")
    ax2.legend(loc="upper right"); ax2.grid(alpha=0.3)

    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("imu", type=Path, help=".gyroflow project file or Pro-mode Insta360 .mp4")
    ap.add_argument("xrk", type=Path, help="MyChron 6 2T .xrk (must have a Steering Angle channel)")
    ap.add_argument("--all", action="store_true", help="also emit a per-lap overlay for every lap")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    print("Loading + syncing...")
    sync = sync_imu_to_xrk(args.imu, args.xrk)
    print(f"  sync corr={sync.corr_peak:.3f}  offset={sync.offset_imu_to_xrk_s:+.2f}s  "
          f"k={sync.column_tilt_factor:.3f}"
          + ("   ⚠ WEAK SYNC" if sync.corr_peak < 0.5 else ""))

    try:
        val, series = compare_steering(sync.imu_stream, sync)
    except ValueError as e:
        raise SystemExit(f"\nCannot validate: {e}")
    print("\n=== camera steering vs MyChron Steering Angle ===")
    print(" ", val.summary())
    verdict = ("AGREE — camera method cross-validated" if abs(val.corr_angle) > 0.7 else
               "PARTIAL — same dynamics, check scale/lag" if abs(val.corr_angle) > 0.4 else
               "DISAGREE — trust the camera (direct measurement); investigate the XRK channel")
    print(f"  --> {verdict}")

    out_dir = args.out_dir or args.imu.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_overlay(series, val, out_dir / f"{args.imu.stem}.steering_validation.png",
                 title=f"{args.imu.name} vs {args.xrk.name}")
    print(f"  wrote {out_dir / (args.imu.stem + '.steering_validation.png')}")

    if args.all:
        windows, src = choose_lap_windows(sync)
        print(f"  per-lap overlays ({len(windows)} laps, {src}):")
        for i, (t0, t1) in enumerate(windows):
            plot_overlay(series, val, out_dir / f"{args.imu.stem}.steering_lap{i}.png",
                         title=f"lap {i}", t_window=(t0, t1))


if __name__ == "__main__":
    main()
