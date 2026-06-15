"""Regenerate a SELF-CONTAINED analysis.ipynb.

The implementation (sync + lap detection + steering + plotting) is embedded
directly into notebook cells by reading the module source, so the in-notebook
code is byte-identical to the tested pipeline/sync_xrk.py and
analysis/per_lap.py. The only import kept is the low-level .gyroflow decoder
(load_gyroflow / ImuStream), which is plumbing rather than "the workflow".
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SYNC = (ROOT / "pipeline" / "sync_xrk.py").read_text()
PERLAP = (ROOT / "analysis" / "per_lap.py").read_text()


def slice_between(src: str, start_marker: str, end_marker: str) -> str:
    """Return src from the line starting with start_marker up to (excluding)
    the line starting with end_marker."""
    lines = src.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(start_marker))
    end = next(i for i, ln in enumerate(lines) if ln.startswith(end_marker))
    return "\n".join(lines[start:end]).rstrip() + "\n"


# Implementation bodies: from the first tunable constant to just before main().
SYNC_IMPL = slice_between(SYNC, "SYNC_RESAMPLE_HZ", "def main(")
PERLAP_IMPL = slice_between(PERLAP, "LAP_DETECT_RADIUS_M", "def main(")

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True) or [text]})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True) or [text]})


# ---------------------------------------------------------------------------
md("""
# Kart Telemetry — End-to-End Workflow (self-contained)

Sync a wheel-cam Insta360 Go 3S (`.gyroflow`) to a MyChron logger (`.xrk`),
segment by lap, and produce per-lap steering / speed / track telemetry —
**all in this one notebook**.

The full implementation is inlined below (Part 1), followed by the run on a
real session (Part 2). The only external import is the low-level `.gyroflow`
decoder (`load_gyroflow`) — base91→zlib→CBOR plumbing that isn't part of the
analysis itself. Everything else — sync, geometry, lap detection, steering
extraction, plotting — is defined here and is identical to the project's
test-covered modules.

**Contents**
- **Part 1 — the pipeline**: [Setup](#part-1) · [Sync implementation](#sync-impl) · [Lap + steering implementation](#lap-impl)
- **Part 2 — run it**: [Inputs](#inputs) · [Synchronize](#synchronize) · [Verify](#verify) · [Geometry](#geometry) · [Laps](#laps) · [Per-lap plots](#plots) · [Summary](#summary) · [What we learned](#learned)
""")

# ---------------------------------------------------------------------------
md("""
<a id="part-1"></a>
## Part 1 — The pipeline

### Setup

Imports, the one low-level decoder dependency, and the session file paths.
Point `GYROFLOW` / `XRK` at your own files to re-run on a different session.
""")

code("""\
%matplotlib inline

import contextlib
import io
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, correlate, sosfiltfilt

# The only external piece: the .gyroflow decoder (base91 -> zlib -> CBOR -> IMU).
# It returns an ImuStream with .t, .gyro (rad/s), .accel (m/s^2), .sample_rate_hz.
from pipeline.extract_imu import ImuStream, load_gyroflow

# === EDIT THESE for your session ===========================================
GYROFLOW = Path("/Users/henry/Desktop/PRO_VID_20260511_145850_00_060.gyroflow")
XRK      = Path("/Users/henry/Desktop/xrk/a_a_XTreme_a_0619.xrk")
# ===========================================================================
""")

# ---------------------------------------------------------------------------
md("""
<a id="sync-impl"></a>
### Sync implementation

Both streams independently measure the chassis yawing through corners — the
MyChron via GPS heading rate (`GPS_Yaw_Rate`), the wheel cam via the
low-frequency component of its gyro projected onto the steering-column axis.
Cross-correlating them recovers the clock offset between the two devices
(which drift independently — trusting wall-clocks doesn't work).

Geometry, for a wheel cam at any angle:

$$\\omega_{\\text{along column}} = \\omega_{\\text{steering}} + k \\cdot \\omega_{\\text{chassis yaw}}$$

The column axis is the principal eigenvector of the gyro covariance (PCA);
$k = |\\text{column} \\cdot \\text{world-up}|$ comes from the recording's own
gravity vector. The detected sign is folded into the stored axis so downstream
code never re-applies it. The cell below is the full sync module.
""")

code(SYNC_IMPL)

# ---------------------------------------------------------------------------
md("""
<a id="lap-impl"></a>
### Lap detection + steering extraction

Steering = the column projection minus $k \\cdot$ GPS-yaw, integrated, then a
light high-pass to clean residual drift — restricted to the IMU/XRK time
overlap so no chassis yaw is fabricated outside the GPS span. Laps come from
the XRK beacon markers when they look sane, else GPS return-to-start
detection. The cell below is the full per-lap module.
""")

code(PERLAP_IMPL)

# ---------------------------------------------------------------------------
md("""
<a id="part-2"></a>
## Part 2 — Run it

<a id="inputs"></a>
### Inputs

Load both files and show what's in them. The `.gyroflow` carries the raw IMU
at 1 kHz; the `.xrk` has 30+ channels at varying rates (GPS @ 25 Hz).
""")

code("""\
stream = load_gyroflow(GYROFLOW, source="wheel")
print("IMU stream:")
print(f"  samples:      {len(stream.t):,}")
print(f"  duration:     {stream.t[-1]:.1f} s")
print(f"  sample rate:  {stream.sample_rate_hz:.1f} Hz")
print(f"  orientation:  {stream.orientation}")
print()

log = load_xrk(XRK)   # defined inline above; silences libxrk's cosmetic warnings
m = log.metadata
print("XRK log:")
print(f"  recording:    {m['Log Date']} {m['Log Time']}")
print(f"  driver:       {m['Driver']!r} @ track {m['Venue']!r}")
print(f"  logger model: {m['Logger Model ID']}, ID {m['Logger ID']}")
print(f"  duration:     {log.laps['end_time'][-1].as_py()/1000:.1f} s")
print(f"  channels:     {len(log.channels)}")
print(f"  XRK laps:     {log.laps.num_rows} (beacon-detected — often unreliable; see Laps)")
""")

# ---------------------------------------------------------------------------
md("""
<a id="synchronize"></a>
### Synchronize

Run the cross-correlation. `corr_peak` is a genuine normalized
cross-correlation in [0, 1] (>0.5 is a confident match).
""")

code("""\
sync = sync_imu_to_xrk(GYROFLOW, XRK)

print(f"  offset (add to IMU t):  {sync.offset_imu_to_xrk_s:+8.3f} s")
print(f"  peak |correlation|:     {sync.corr_peak:8.4f}   "
      f"({'STRONG' if sync.corr_peak > 0.5 else 'WEAK — check'})")
print(f"  sign:                   {sync.sign:+8d}   (folded into column_axis)")
print(f"  column axis (body):     ({sync.column_axis[0]:+.3f}, "
      f"{sync.column_axis[1]:+.3f}, {sync.column_axis[2]:+.3f})")
print(f"  column tilt:            {sync.column_tilt_deg:.1f}° from world-vertical")
print(f"  tilt factor k:          {sync.column_tilt_factor:.4f}")
print(f"  quiet samples:          {sync.quiet_sample_count}   "
      f"(geometry reliable: {sync.geometry_reliable})")
print(f"  gyro bias (deg/s):      ({np.rad2deg(sync.gyro_bias[0]):+.4f}, "
      f"{np.rad2deg(sync.gyro_bias[1]):+.4f}, {np.rad2deg(sync.gyro_bias[2]):+.4f})")
""")

# ---------------------------------------------------------------------------
md("""
<a id="verify"></a>
### Verify the sync visually

Both yaw signals before and after alignment — after, they should peak at the
same moments through every corner.
""")

code("""\
t_imu, yaw_imu, _, _ = chassis_yaw_from_imu(stream)
t_xrk, yaw_xrk = chassis_yaw_from_xrk(log)

fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharey=True)
axes[0].plot(t_imu, np.rad2deg(yaw_imu), lw=0.6, label="IMU (column-axis LP)")
axes[0].plot(t_xrk, np.rad2deg(yaw_xrk), lw=0.7, label="XRK GPS_Yaw_Rate", alpha=0.85)
axes[0].set_title("Before sync — each stream on its own clock")
axes[0].set_xlabel("local time (s)"); axes[0].set_ylabel("yaw rate (deg/s)")
axes[0].legend(loc="upper right"); axes[0].grid(alpha=0.3)

# IMU yaw_imu uses the unsigned PCA axis; apply sync.sign here for the overlay.
axes[1].plot(t_imu + sync.offset_imu_to_xrk_s, sync.sign * np.rad2deg(yaw_imu),
             lw=0.6, label=f"IMU (shifted {sync.offset_imu_to_xrk_s:+.2f}s, sign={sync.sign})")
axes[1].plot(t_xrk, np.rad2deg(yaw_xrk), lw=0.7, label="XRK GPS_Yaw_Rate", alpha=0.85)
axes[1].set_xlim(t_xrk[0] - 5, t_xrk[-1] + 5)
axes[1].set_title(f"After sync — both on XRK clock (corr = {sync.corr_peak:.3f})")
axes[1].set_xlabel("XRK time (s)"); axes[1].set_ylabel("yaw rate (deg/s)")
axes[1].legend(loc="upper right"); axes[1].grid(alpha=0.3)
fig.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------------------
md("""
<a id="geometry"></a>
### Body-frame geometry

The column axis (from PCA) and the gravity vector (from quiet samples) give
the tilt factor `k` used to subtract chassis yaw.
""")

code("""\
quiet = np.linalg.norm(stream.gyro, axis=1) < np.deg2rad(QUIET_RATE_DPS)
g_body = stream.accel[quiet].mean(axis=0)
world_up = -g_body / np.linalg.norm(g_body)
print("Body frame:")
print(f"  column axis:  ({sync.column_axis[0]:+.3f}, {sync.column_axis[1]:+.3f}, {sync.column_axis[2]:+.3f})")
print(f"  world up:     ({world_up[0]:+.3f}, {world_up[1]:+.3f}, {world_up[2]:+.3f})")
print(f"  |g_body|:     {np.linalg.norm(g_body):.3f} m/s²  (gravity ≈ 9.807)")
print(f"  k = |col·up| = {sync.column_tilt_factor:.4f}  →  tilt {sync.column_tilt_deg:.1f}° from vertical")
print()
print("Steering recovery:  ω_steer = (gyro · col_axis) − k · GPS_Yaw_Rate")
""")

# ---------------------------------------------------------------------------
md("""
<a id="laps"></a>
### Lap detection

The XRK beacon markers are unreliable in practice (often only fire once or
twice per session). GPS return-to-start detection is the fallback.
""")

code("""\
gps_laps = detect_laps_from_gps(log)
xrk_durations = [(e - s) / 1000.0 for s, e in zip(
    log.laps["start_time"].to_pylist(), log.laps["end_time"].to_pylist())]

print(f"XRK beacon detection: {len(xrk_durations)} laps")
for i, d in enumerate(xrk_durations):
    print(f"  lap {i}: {d:6.2f} s" + ("   ← unrealistic" if d > 120 else ""))
print()
print(f"GPS return-to-start detection: {len(gps_laps)} laps")
for i, (s, e) in enumerate(gps_laps):
    print(f"  lap {i}: {s:7.2f} .. {e:7.2f} s   ({e - s:5.2f} s)")
""")

# ---------------------------------------------------------------------------
md("""
<a id="plots"></a>
### Per-lap plots

Steering is integrated once over the whole recording, then each lap is sliced
out. Per lap: steering angle, kart yaw rate, GPS speed, and the GPS track
colored by speed.
""")

code("""\
lap_windows, source = choose_lap_windows(sync, force_source="auto")
print(f"Using lap source: {source}  ({len(lap_windows)} laps)")
print()

# Compute the full-recording steering integral ONCE; reuse for every lap.
steering = steering_from_synced(stream, sync)

laps_data = []
for i, window in enumerate(lap_windows):
    lap = extract_lap(stream, sync, window, lap_index=i, steering=steering)
    laps_data.append(lap)
    s = lap_summary(lap)
    print(f"  lap {i}: {s['duration_s']:6.2f}s   "
          f"peak steer ±{s['peak_steer_deg']:>4.0f}°   "
          f"peak yaw rate ±{s.get('peak_yaw_rate_dps', float('nan')):>4.0f}°/s   "
          f"peak speed {s.get('peak_speed_kmh', float('nan')):>5.1f} km/h")
""")

code("""\
# Render each lap inline. plot_lap writes a PNG and closes its own figure;
# embed it via IPython.display.Image so the notebook creates no extra figures
# to leak. PNGs go to a temp dir cleaned up at the end.
import tempfile
import shutil
from IPython.display import Image, display

_tmp = Path(tempfile.mkdtemp(prefix="kart_laps_"))
try:
    title = (f"corr={sync.corr_peak:.2f}, "
             f"offset={sync.offset_imu_to_xrk_s:+.2f}s, "
             f"k={sync.column_tilt_factor:.3f}")
    for lap in laps_data:
        out_path = _tmp / f"lap{lap.lap_index}.png"
        plot_lap(lap, out_path, title=title)
        display(Image(filename=str(out_path)))
finally:
    shutil.rmtree(_tmp, ignore_errors=True)
""")

# ---------------------------------------------------------------------------
md("""
<a id="summary"></a>
### Summary table
""")

code("""\
rows = []
for lap in laps_data:
    s = lap_summary(lap)
    rows.append({
        "lap": lap.lap_index,
        "duration_s": round(s["duration_s"], 2),
        "peak_steer_deg": round(s["peak_steer_deg"], 1),
        "peak_yaw_rate_dps": round(s.get("peak_yaw_rate_dps", float("nan")), 0),
        "peak_speed_kmh": round(s.get("peak_speed_kmh", float("nan")), 1),
        "mean_speed_kmh": round(s.get("mean_speed_kmh", float("nan")), 1),
    })
df = pd.DataFrame(rows).set_index("lap")
df
""")

code("""\
fastest = df["duration_s"].idxmin()
print(f"Fastest lap: {fastest}  ({df.loc[fastest, 'duration_s']} s)")
print(f"  peak steer:     ±{df.loc[fastest, 'peak_steer_deg']}°")
print(f"  peak yaw rate:  ±{df.loc[fastest, 'peak_yaw_rate_dps']}°/s")
print(f"  peak speed:     {df.loc[fastest, 'peak_speed_kmh']} km/h")
""")

# ---------------------------------------------------------------------------
md("""
<a id="learned"></a>
### What we learned

- **Go 3S IMU runs at 1 kHz**, not ~200 Hz — great for jerk math.
- **Wall-clock sync fails**; cross-correlation of chassis yaw is reliable
  (here the device clocks disagreed by ~17 s).
- **MyChron beacon lap detection is unreliable**; GPS return-to-start works.
- **`GPS Speed` is m/s** in the XRK, not km/h.
- **Naive integration leaks chassis yaw** at corner frequencies — you must
  subtract `k · GPS_Yaw_Rate`, restricted to the IMU/GPS time overlap (a
  high-pass filter alone cannot separate the two).
- **PCA finds the column axis** in body frame for any wheel-mount tilt.

Point `GYROFLOW` / `XRK` at the top at a different session and re-run.
""")

# ---------------------------------------------------------------------------
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x", "mimetype": "text/x-python",
                          "file_extension": ".py"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
out = ROOT / "analysis.ipynb"
out.write_text(json.dumps(nb, indent=1))
n_impl = SYNC_IMPL.count("\n") + PERLAP_IMPL.count("\n")
print(f"wrote {out}  ({len(cells)} cells, {n_impl} lines of inlined implementation)")
