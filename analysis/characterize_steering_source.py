"""Is the MyChron 6 2T `Steering Angle` a real column sensor or a model?

Karts almost never have a steering-angle potentiometer wired to the logger, so
the 6 2T's `Steering Angle` is very likely a *computed* channel. If it is, it is
not fully independent ground truth for the camera pipeline — knowing how it is
built tells us what a camera-vs-XRK agreement actually proves.

Discriminators (all from the XRK alone):
  1. Bicycle model. For a kart, steer δ ≈ L · yaw_rate / v (L = wheelbase).
     If `Steering Angle` ≈ gain · (yaw_rate / v) with very high R², it is a
     kinematic model of chassis yaw, not a measurement of the driver's hands.
  2. Low-speed behaviour. A r/v model is undefined as v→0 and is gated to ~0 in
     the pits; a real sensor still reads the held wheel angle.
  3. Residual structure. A real sensor captures corrections / counter-steer that
     do not appear in chassis yaw; a model's residual is just noise.

Usage:
    python -m analysis.characterize_steering_source session.xrk [--lap N] [--out-dir d]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pipeline.sync_xrk import clean_gps_yaw_dps, load_xrk

KART_WHEELBASE_M = 1.05      # typical 125cc shifter / X30; gain fit absorbs error
ON_TRACK_MPS = 5.0           # ignore pit crawl for the kinematic correlation
PIT_MPS = 2.0                # "stopped/crawling" threshold for the low-speed test


def _chan(log, name):
    """(t_seconds, values) for an XRK channel, or None if absent/empty."""
    tbl = log.channels.get(name)
    if tbl is None or tbl.num_rows == 0:
        return None
    col = [c for c in tbl.column_names if c != "timecodes"][0]
    t = np.asarray(tbl["timecodes"], dtype=np.float64) / 1000.0
    v = np.asarray(tbl[col], dtype=np.float64)
    return t, v


def characterize(xrk_path: Path, out_dir: Path, lap: int | None = None) -> dict:
    log = load_xrk(xrk_path)
    if "Steering Angle" not in log.channels:
        raise SystemExit(f"{xrk_path.name} has no 'Steering Angle' channel "
                         "(not a MyChron 6 2T log?).")

    t_st, steer = _chan(log, "Steering Angle")
    yr = _chan(log, "GPS_Yaw_Rate")
    sp = _chan(log, "GPS Speed")
    gz = _chan(log, "GyroZ")
    if yr is None or sp is None:
        raise SystemExit("Need GPS_Yaw_Rate and GPS Speed to characterize.")

    # Common 50 Hz grid over the steering span.
    t0, t1 = t_st[0], t_st[-1]
    grid = np.arange(t0, t1, 1 / 50.0)
    s = np.interp(grid, t_st, steer)
    r_dps = clean_gps_yaw_dps(*yr)               # de-spiked yaw rate, deg/s
    r = np.interp(grid, yr[0], r_dps)
    v = np.interp(grid, sp[0], sp[1])            # m/s
    g = np.interp(grid, gz[0], gz[1]) if gz is not None else None

    # 1. Bicycle-model predictor x = yaw_rate[rad/s] / v, regress steer ~ a*x + b
    on = v > ON_TRACK_MPS
    x = np.deg2rad(r) / np.maximum(v, 1e-3)
    A = np.vstack([x[on], np.ones(on.sum())]).T
    (a, b), *_ = np.linalg.lstsq(A, s[on], rcond=None)
    pred = a * x + b
    ss_res = np.sum((s[on] - pred[on]) ** 2)
    ss_tot = np.sum((s[on] - s[on].mean()) ** 2)
    r2_bike = 1 - ss_res / ss_tot
    corr_bike = float(np.corrcoef(s[on], x[on])[0, 1])
    implied_L = abs(a) * np.pi / 180.0 if a else float("nan")  # deg per (rad/s/(m/s)) → m

    # 2. Low-speed behaviour: steering amplitude when essentially stopped
    stopped = v < PIT_MPS
    lowspeed_std = float(np.std(s[stopped])) if stopped.any() else float("nan")
    lowspeed_p95 = float(np.percentile(np.abs(s[stopped]), 95)) if stopped.any() else float("nan")

    # 3. Residual structure vs straightline yaw model
    resid = s - pred
    resid_frac = float(np.std(resid[on]) / (np.std(s[on]) + 1e-9))

    # correlation with own yaw gyro (chassis), not divided by speed
    corr_gz = float(np.corrcoef(s, g)[0, 1]) if g is not None else float("nan")

    reads_at_standstill = np.isfinite(lowspeed_std) and lowspeed_std > 5.0
    if r2_bike > 0.9:
        verdict = ("MODELED from chassis yaw (r/v) — NOT independent of the "
                   "pipeline's k·GPS_Yaw_Rate subtraction; a weak ground truth.")
    elif reads_at_standstill:
        verdict = ("INDEPENDENT of chassis yaw (reads while stopped, weak r/v "
                   "fit). Either a real column sensor or a proprietary AiM "
                   "estimate — XRK alone can't tell; the wheel-cam (which "
                   "physically measures wheel rotation) is the tiebreaker.")
    else:
        verdict = ("AMBIGUOUS — moderate yaw coupling, no standstill signal; "
                   "treat as a soft reference until the camera confirms it.")

    print(f"\n=== Steering-source characterization: {xrk_path.name} ===")
    print(f"  samples on-track: {on.sum()} of {len(grid)}  ({log.metadata.get('Log Date')} {log.metadata.get('Log Time')})")
    print(f"  bicycle-model fit  steer ≈ {a:+.3f}·(yawrate/v) {b:+.2f}")
    print(f"    R²            = {r2_bike:.3f}    corr = {corr_bike:+.3f}")
    print(f"    implied wheelbase·ratio = {implied_L:.2f} m  (kart L≈{KART_WHEELBASE_M} m)")
    print(f"  corr with own GyroZ (chassis yaw) = {corr_gz:+.3f}")
    print(f"  residual fraction (unexplained by r/v) = {resid_frac:.2f}")
    print(f"  low-speed (<{PIT_MPS} m/s) steering: std={lowspeed_std:.2f}°  p95|·|={lowspeed_p95:.2f}°")
    print(f"  --> VERDICT: {verdict}")

    # Plot: scatter + a representative lap trace
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, (axsc, axtr) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axsc.scatter(x[on], s[on], s=2, alpha=0.2)
    xs = np.linspace(np.percentile(x[on], 1), np.percentile(x[on], 99), 50)
    axsc.plot(xs, a * xs + b, "r-", lw=2, label=f"fit R²={r2_bike:.3f}")
    axsc.set_xlabel("yaw_rate / speed  (rad/m)")
    axsc.set_ylabel("Steering Angle (deg)")
    axsc.set_title("measured steering vs kinematic predictor")
    axsc.legend(); axsc.grid(alpha=0.3)

    # representative window: 30 s around the fastest on-track stretch
    i_fast = int(np.argmax(v))
    w = slice(max(0, i_fast - 750), min(len(grid), i_fast + 750))
    tt = grid[w] - grid[w][0]
    axtr.plot(tt, s[w], label="Steering Angle (measured)", lw=1.0)
    axtr.plot(tt, pred[w], label="r/v kinematic model", lw=1.0, alpha=0.8)
    axtr.set_xlabel("time (s)"); axtr.set_ylabel("deg")
    axtr.set_title("measured vs model (30 s @ top speed)")
    axtr.legend(); axtr.grid(alpha=0.3)
    out = out_dir / f"steering_source_{xrk_path.stem}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")

    return {"r2_bike": r2_bike, "corr_gz": corr_gz, "resid_frac": resid_frac,
            "lowspeed_std": lowspeed_std, "implied_L": implied_L, "verdict": verdict}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("xrk", type=Path)
    ap.add_argument("--lap", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("results/steering_source"))
    args = ap.parse_args()
    characterize(args.xrk, args.out_dir, args.lap)


if __name__ == "__main__":
    main()
