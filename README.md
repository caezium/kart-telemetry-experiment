# Kart Telemetry Experiment

Action-camera-based data acquisition for competitive 125 cc karting
(KZ, X30 Senior, ROK Shifter). Augments a MyChron with **driver input
quality**, **tire observation**, and **driver vision** data using one
or two Insta360 Go 3S units as wireless IMU + video sensors.

Not a replacement for proper data systems. A sub-$1k addition that
captures channels the MyChron does not: steering smoothness, look-ahead
timing, inside-front lift, slip angle.

**See [`RESULTS.md`](RESULTS.md) for end-to-end validation on a real
session.** Six clean laps recovered from a 9-minute wheel-cam + XRK
recording, with proper sync and steering extraction.

## What this measures

| Channel | Source | MyChron has it? |
|---|---|---|
| Lap time, splits, speed, GPS | MyChron (`.xrk`) | yes |
| Lat/long G, RPM, GPS yaw rate | MyChron (`.xrk`) | yes |
| Steering angle | Wheel-mounted Go 3S IMU (or MyChron 6) | only on 6 |
| Steering rate / jerk / corrections | Wheel-mounted Go 3S IMU | no |
| Inside-front lift duration | Wheel-mounted Go 3S video | no |
| Slip angle | Wheel-mounted Go 3S video | no |
| Driver look-ahead timing | Helmet-mounted Go 3S IMU | no |
| Head stability / fatigue | Helmet-mounted Go 3S IMU | no |
| Driver POV reference | Helmet-mounted Go 3S video | no |
| Breathing rate | Helmet-mounted Go 3S audio | no |

## Hardware

- 1× or 2× Insta360 Go 3S (the small magnetic action cam)
- Wheel hub mount: VHB or magnetic. **Mount 1** = lens-axis along the
  steering column (the easy case). **Mount 2** = camera angled
  forward / down to see the front tire (harder to process, but the
  pipeline handles it — see [RESULTS.md](RESULTS.md))
- Helmet mount: stick-on chin bar mount only — never inside the shell,
  never anywhere that voids cert
- A MyChron 5 or 6 (already on the kart)

## Quickstart

```bash
python3 -m venv .venv --system-site-packages
.venv/bin/pip install -r pipeline/requirements.txt
```

**Per session — the primary path:**

```bash
# 1. Export a .gyroflow project file from the wheel video.
#    In Gyroflow: open the .mp4 → File → Save (creates `*.gyroflow`).
#    This is the only manual step; everything else is automated.

# 2. Copy the .xrk off the MyChron's SD card.

# 3. Run the pipeline.
.venv/bin/python -m analysis.per_lap \
    path/to/wheel.gyroflow \
    path/to/session.xrk \
    --all --out-dir results/my_session/
```

Generates one 4-panel PNG per lap: steering angle, kart yaw rate, GPS
speed, GPS track overlay colored by speed. Prints a stats table to the
console.

Sample output (real session, see [RESULTS.md](RESULTS.md)):

```
Loading + syncing...
  offset (IMU→XRK):   -14.027 s
  peak |corr|:        0.7835
  sign:               -1
  column tilt:        40.6° from vertical (k = 0.7588)

  lap windows: 6 laps (gps)
  lap 0:  54.08s   peak steer ±63°   peak yaw rate ±309°/s   peak speed  91.4 km/h
  lap 1:  44.56s   peak steer ±43°   peak yaw rate ±161°/s   peak speed 107.1 km/h
  lap 2:  44.32s   peak steer ±74°   peak yaw rate ±263°/s   peak speed 108.6 km/h
  lap 3:  44.64s   peak steer ±57°   peak yaw rate ±267°/s   peak speed 108.6 km/h
  lap 4:  43.44s   peak steer ±41°   peak yaw rate ±125°/s   peak speed 106.9 km/h
  lap 5:  49.48s   peak steer ±54°   peak yaw rate ±381°/s   peak speed 108.5 km/h
```

**One-time per Go 3S unit — bench calibration (Phase 0, optional):**

```bash
# Record the bench protocol described in data/calibration/README.md.
# Open the recording in Gyroflow → File → Save (creates a .gyroflow project file).
# Drop it into data/calibration/<UNIT>/original.gyroflow alongside ground_truth.csv.

.venv/bin/python -m pipeline.calibrate data/calibration/<UNIT>/
```

Writes `result.json` with the per-unit gyro bias and pass/fail. The
per-session sync also extracts bias from each recording's own quiet
samples, so the bench calibration isn't strictly required — it's for
provenance.

## How it works (one paragraph each)

**Sync.** Both the wheel-cam IMU and the MyChron measure chassis yaw
during corners. Low-pass-filter the IMU's column-axis projection to
isolate the chassis component, cross-correlate against the XRK's
`GPS_Yaw_Rate`, and the peak gives the clock offset. Robust against
multi-second clock drift between independent loggers.
See [`pipeline/sync_xrk.py`](pipeline/sync_xrk.py).

**Steering extraction.** The wheel-cam gyro projected onto the column
axis is `ω_steering + k · ω_chassis_yaw` where the column comes from
PCA of the gyro covariance and `k = |column · world_up|` from the
recording's own gravity vector. Subtract `k · GPS_Yaw_Rate` (from
XRK), integrate the residual, and you have pure steering input — the
right way to do it. HP-filtering alone leaks chassis yaw at corner
frequencies. See [`analysis/per_lap.py`](analysis/per_lap.py).

**Lap detection.** Prefer the XRK's beacon-crossing markers if they
look sensible (>1 lap, max duration <120 s). Otherwise fall back to
GPS return-to-start (kart back within 25 m of where it first started
moving). In practice the beacon detection often misfires; the GPS
fallback handles real recordings cleanly.

## Project layout

```
kart-telemetry-experiment/
├── README.md             this file
├── RESULTS.md            end-to-end validation on a real session
├── ROADMAP.md            experimental phases, what's next
├── pipeline/
│   ├── extract_imu.py    .gyroflow project file → uniform-rate ImuStream parquet
│   ├── calibrate.py      Phase 0 bench-validation harness, writes result.json
│   ├── sync_xrk.py       wheel-cam IMU ↔ MyChron XRK time alignment + geometry
│   ├── sync_streams.py   (legacy) tap-detect sync for multi-cam sessions
│   └── requirements.txt
├── analysis/
│   ├── per_lap.py        per-lap multi-channel PNG (primary tool)
│   ├── quicklook.py      single-file IMU sniff-test plot
│   ├── steering_angle.py (limited) gz-only steering, Mount 1 only
│   ├── steering_metrics.py per-corner peak / rate / jerk / symmetry
│   ├── vision_metrics.py   head yaw vs steering lead time (Phase 3)
│   └── tire_observation.py CV stub for inside-front lift (Phase 2)
├── results/
│   └── session_20260511_xtreme/  per-lap PNGs from the validation session
├── tests/                pytest suite — 48 tests
└── data/
    ├── sessions/         per-session raw data (gitignored)
    └── calibration/      per-unit bench-test recordings (gitignored)
```

## Design principles

1. **IMU first, video second.** The gyro is the cleanest signal. CV is
   for things only video can answer (tire state, vision direction).
2. **GPS / speed from MyChron only.** The Go 3S has no GPS. Don't
   integrate the accelerometer for displacement — it drifts to
   nonsense in 30 seconds.
3. **One mount, multiple outputs.** A wheel-cam angled at the tire
   gives both steering IMU *and* tire video. Gyroflow's IMU
   stabilization de-rotates the video for free.
4. **Cross-correlation, not clocks.** Independent loggers' wall
   clocks drift seconds-to-tens-of-seconds. Align using a physical
   signal both measure.
5. **Don't trust integrated angle for long.** Re-zero at known
   straight moments. Subtract chassis yaw using independent
   measurements when available.

## Known gaps / open problems

- **Slip angle from video** is technically possible but ill-defined at
  30 fps and with motion blur. Need 100+ fps and clean lighting to be
  useful. Currently a stub.
- **Inside-front lift detection** needs a labeled clip dataset. The
  CV approach (background subtraction + bottom-of-frame ground
  continuity) is sketched but unvalidated.
- **The Go 3S samples IMU at 1 kHz** (empirically), which is great for
  jerk calculations but means a single session is ~500k samples per
  axis. All processing handles this fine, but it does mean Gyroflow
  export takes ~5 s and parquet files are large.
- **MyChron XRK format** is read via [`libxrk`](https://pypi.org/project/libxrk/),
  Scott Smith's reverse-engineered parser. No AiM DLL required.
- **Helmet legality** varies by sanctioning body. Verify your specific
  series before the chin-bar mount goes on.

## Why this exists

The MyChron 6 finally added a steering gyro, but most of the field is
on 5s. A camera-based approach gives 4s/5s drivers the same input data
plus things even the 6 does not have (vision, tire). For a competitive
125 cc driver looking to find the last 0.2 s/lap, the
highest-leverage unmeasured channels are: **driver vision**,
**steering smoothness**, and **inside-front lift behavior**. This
project targets exactly those.

See [`ROADMAP.md`](ROADMAP.md) for phased build plan.
