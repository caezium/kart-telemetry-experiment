"""Per-lap driver-input view straight from a MyChron 6 2T `Steering Angle`.

This is NOT the camera pipeline — it reads the logger's own steering channel.
It exists because the 6 2T gives real per-lap steering telemetry today, even
when there is no usable wheel-cam gyro for that session (the camera-only path
is still the project's goal; this is the ground-truth side standing on its own).

For each lap it reports peak angle, peak rate, smoothness (jerk RMS), and a
correction count, and renders a 4-panel plot: steering angle, steering rate,
speed, and the GPS track coloured by steering.

Usage:
    python -m analysis.xrk_steering_laps session_6_2t.xrk --all --out-dir results/steer/
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

from pipeline.sync_xrk import load_xrk
from analysis.per_lap import detect_laps_from_gps
from analysis.validate_steering import xrk_steering

STEER_FS = 50.0               # Steering Angle native rate
RATE_LOWPASS_HZ = 6.0         # smooth angle before differentiating (kills 50 Hz noise)
CORRECTION_RATE_DPS = 30.0    # a rate reversal past this counts as a "correction"
MAX_SANE_LAP_S = 120.0


def _chan(log, name):
    tbl = log.channels.get(name)
    if tbl is None or tbl.num_rows == 0:
        return None
    col = [c for c in tbl.column_names if c != "timecodes"][0]
    return (np.asarray(tbl["timecodes"], np.float64) / 1000.0,
            np.asarray(tbl[col], np.float64))


def _steer_rate(angle: np.ndarray, fs: float = STEER_FS) -> np.ndarray:
    """deg/s, from lightly low-passed angle (raw 50 Hz diff is too noisy)."""
    if len(angle) < 12:
        return np.gradient(angle, 1.0 / fs)
    sos = butter(2, RATE_LOWPASS_HZ / (fs / 2), btype="low", output="sos")
    return np.gradient(sosfiltfilt(sos, angle), 1.0 / fs)


def lap_windows(log) -> tuple[list[tuple[float, float]], str]:
    """XRK beacon laps if they look sane (>1 lap, none > 120 s), else GPS."""
    starts = log.laps["start_time"].to_pylist()
    ends = log.laps["end_time"].to_pylist()
    durs = [(e - s) / 1000.0 for s, e in zip(starts, ends)]
    if len(durs) > 1 and max(durs) <= MAX_SANE_LAP_S:
        return [(s / 1000.0, e / 1000.0) for s, e in zip(starts, ends)], "xrk"
    gps = detect_laps_from_gps(log)
    if not gps:
        raise SystemExit("No usable laps (XRK beacons look wrong and GPS detection failed).")
    return gps, "gps"


@dataclass
class SteerLapMetrics:
    lap: int
    duration_s: float
    peak_deg: float
    mean_abs_deg: float
    peak_rate_dps: float
    jerk_rms_dps2: float       # smoothness — lower is smoother
    corrections: int           # rate sign-reversals past the deadband

    def row(self) -> str:
        return (f"  lap {self.lap:2d}: {self.duration_s:6.2f}s  "
                f"peak ±{self.peak_deg:5.1f}°  mean|{self.mean_abs_deg:4.1f}°|  "
                f"peak rate ±{self.peak_rate_dps:5.0f}°/s  "
                f"jerk {self.jerk_rms_dps2:6.0f}°/s²  corrections {self.corrections:3d}")


def lap_metrics(t, ang, rate, lap_idx, t0, t1) -> SteerLapMetrics | None:
    m = (t >= t0) & (t < t1)
    if m.sum() < 10:
        return None
    a, r, tt = ang[m], rate[m], t[m]
    jerk = np.gradient(r, tt)
    # corrections: sign changes of rate where the swing exceeds the deadband
    big = np.abs(r) > CORRECTION_RATE_DPS
    sign = np.sign(r)
    reversals = int(np.sum((np.diff(sign) != 0) & big[1:]))
    return SteerLapMetrics(
        lap=lap_idx, duration_s=float(t1 - t0),
        peak_deg=float(np.abs(a).max()), mean_abs_deg=float(np.abs(a).mean()),
        peak_rate_dps=float(np.abs(r).max()),
        jerk_rms_dps2=float(np.sqrt(np.mean(jerk ** 2))),
        corrections=reversals,
    )


def plot_lap(log, t_st, ang, rate, lap_idx, t0, t1, out_path, *, title=""):
    m = (t_st >= t0) & (t_st < t1)
    t = t_st[m] - t0
    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[2, 1])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[1, 0], sharex=ax_a)
    ax_s = fig.add_subplot(gs[2, 0], sharex=ax_a)
    ax_m = fig.add_subplot(gs[:, 1])

    ax_a.plot(t, ang[m], lw=0.8, color="tab:blue")
    ax_a.axhline(0, color="k", lw=0.4); ax_a.grid(alpha=0.3)
    ax_a.set_ylabel("steering angle (deg)")
    ax_a.set_title(f"Lap {lap_idx} — {t1 - t0:.2f}s  ({title})")

    ax_r.plot(t, rate[m], lw=0.7, color="tab:purple")
    ax_r.axhline(0, color="k", lw=0.4); ax_r.grid(alpha=0.3)
    ax_r.set_ylabel("steering rate (deg/s)")

    sp = _chan(log, "GPS Speed")
    if sp is not None:
        ms = (sp[0] >= t0) & (sp[0] < t1)
        ax_s.plot(sp[0][ms] - t0, sp[1][ms] * 3.6, lw=0.9, color="tab:green")
        ax_s.set_ylabel("GPS speed (km/h)")
    ax_s.set_xlabel("lap time (s)"); ax_s.grid(alpha=0.3)

    lat = _chan(log, "GPS Latitude"); lon = _chan(log, "GPS Longitude")
    if lat is not None and lon is not None:
        ml = (lat[0] >= t0) & (lat[0] < t1)
        la = lat[1][ml]; lo = np.interp(lat[0][ml], lon[0], lon[1])
        st_on_gps = np.interp(lat[0][ml], t_st, ang)
        lim = max(1.0, np.percentile(np.abs(st_on_gps), 98))
        sc = ax_m.scatter(lo, la, c=st_on_gps, s=5, cmap="coolwarm", vmin=-lim, vmax=lim)
        plt.colorbar(sc, ax=ax_m, label="steering (deg)", shrink=0.7)
        ax_m.set_aspect("equal", adjustable="datalim")
        ax_m.set_title("track — colored by steering"); ax_m.grid(alpha=0.3)
    else:
        ax_m.text(0.5, 0.5, "no GPS", ha="center", va="center", transform=ax_m.transAxes)

    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("xrk", type=Path, help="MyChron 6 2T .xrk (needs a Steering Angle channel)")
    ap.add_argument("--lap", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    log = load_xrk(args.xrk)
    try:
        t_st, ang = xrk_steering(log)
    except ValueError as e:
        raise SystemExit(str(e))
    rate = _steer_rate(ang)
    windows, src = lap_windows(log)
    md = log.metadata or {}
    print(f"{args.xrk.name}  {md.get('Log Date','')} {md.get('Log Time','')}  "
          f"{len(windows)} laps ({src})")

    out_dir = args.out_dir or args.xrk.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.lap is not None:
        targets = [args.lap]
    elif args.all:
        targets = list(range(len(windows)))
    else:
        targets = [max(range(len(windows)), key=lambda i: windows[i][1] - windows[i][0])]

    title = f"{md.get('Log Date','')} {md.get('Log Time','')}"
    for i in targets:
        if not (0 <= i < len(windows)):
            raise SystemExit(f"--lap {i} out of range (0..{len(windows)-1})")
        t0, t1 = windows[i]
        m = lap_metrics(t_st, ang, rate, i, t0, t1)
        if m:
            print(m.row())
        plot_lap(log, t_st, ang, rate, i, t0, t1,
                 out_dir / f"{args.xrk.stem}.steer_lap{i}.png", title=title)


if __name__ == "__main__":
    main()
