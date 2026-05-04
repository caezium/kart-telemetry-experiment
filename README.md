# Kart Telemetry Experiment

Action-camera-based data acquisition for competitive 125cc karting (KZ, X30 Senior, ROK Shifter). Augments a MyChron lap timer with **driver input quality**, **tire observation**, and **driver vision** data using one or two Insta360 Go 3S units as wireless IMU + video sensors.

This is not a replacement for proper data systems. It is a sub-$1k addition that captures channels the MyChron does not: steering smoothness, look-ahead timing, inside-front lift, slip angle.

## What this measures

| Channel | Source | MyChron has it? |
|---|---|---|
| Lap time, splits, speed | MyChron | yes |
| Lat/long G, RPM | MyChron | yes |
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
- Wheel hub mount: VHB or magnetic, **on the steering rotation axis** to avoid centripetal contamination of the accelerometer
- Helmet mount: stick-on chin bar mount only — never inside the shell, never anywhere that voids cert
- A MyChron 5 or 6 (already on the kart)

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r pipeline/requirements.txt
```

**One-time per Go 3S unit — bench calibration (Phase 0):**

```bash
# Record the bench protocol described in data/calibration/README.md.
# Open the recording in Gyroflow → File → Save (creates a .gyroflow project file).
# Drop it into data/calibration/<UNIT>/original.gyroflow alongside ground_truth.csv.

python -m pipeline.calibrate data/calibration/<UNIT>/
```

This writes `result.json` with the per-unit gyro bias, the steering axis, and
pass/fail. Subsequent extract runs auto-load the bias for the matching source.

**Per session:**

```bash
# 1. Drop the .insv/.mp4 files into data/sessions/<session_id>/
# 2. In Gyroflow: open each video, save the .gyroflow project file alongside.
#    Filename must contain "wheel" or "helmet" (e.g. wheel-cam.gyroflow).
# 3. Drop the MyChron CSV export alongside as mychron.csv (optional).

python -m pipeline.extract_imu  data/sessions/<session_id>/
python -m pipeline.sync_streams data/sessions/<session_id>/
python -m analysis.steering_metrics data/sessions/<session_id>/
python -m analysis.vision_metrics  data/sessions/<session_id>/
```

The pipeline reads Gyroflow **project files** (`.gyroflow` JSON), not CSV
exports — see [pipeline/extract_imu.py](pipeline/extract_imu.py) for why
(the CSV path tops out at video fps; we need the ~200 Hz raw IMU stream
that lives inside the project file).

## Project layout

```
kart-telemetry-experiment/
├── README.md             this file
├── ROADMAP.md            experimental phases, what's next
├── pipeline/
│   ├── extract_imu.py    .gyroflow project file → uniform-rate ImuStream parquet
│   ├── calibrate.py      Phase 0 bench-validation harness, writes result.json
│   ├── sync_streams.py   Align wheel + helmet + MyChron timelines on a tap
│   └── requirements.txt
├── analysis/
│   ├── steering_metrics.py   angle, rate, jerk, correction count, symmetry
│   ├── vision_metrics.py     head yaw vs steering input lead time
│   └── tire_observation.py   CV stub for inside-front lift detection (Phase 2)
├── tests/                pytest suite — orientation, decode, calibrate, sync, metrics
└── data/
    ├── sessions/         per-session raw + extracted data
    └── calibration/      per-unit bench-test recordings + result.json
```

## Design principles

1. **IMU first, video second.** The gyro is the cleanest signal. CV is for things only video can answer (tire state, vision direction).
2. **No GPS on the Go 3S.** Position and speed always come from MyChron. Don't try to integrate accelerometer for displacement — it'll drift to nonsense in 30 seconds.
3. **One mount, multiple outputs.** A wheel-cam angled at the tire gives both steering IMU *and* tire video. Gyroflow's IMU stabilization de-rotates the video for free.
4. **Sync is the hard part.** Time alignment between three independent recorders (wheel, helmet, MyChron) is the load-bearing infrastructure. A clap/tap at session start gives a hard sync spike in all three.
5. **Drift-aware integration.** Re-zero gyro angle estimates at known straight-ahead moments (start/finish straight). Don't trust any angle integrated for more than a minute without a reset.

## Known gaps / open problems

- **Slip angle from video** is technically possible but ill-defined at 30 fps and motion blur. Need 100+ fps and clean lighting to be useful. Currently a stub.
- **Inside-front lift detection** — needs a labeled clip dataset to train against. The CV approach (background subtraction + bottom-of-frame ground continuity) is sketched but unvalidated.
- **MyChron CSV format** varies by firmware version. The sync module currently assumes Race Studio 3 export schema.
- **Gyroflow `.gyroflow` schema** is undocumented and reconstructed from source ([gyroflow/src/core/lib.rs](https://github.com/gyroflow/gyroflow/blob/master/src/core/lib.rs), [telemetry-parser/src/util.rs](https://github.com/AdrianEddy/telemetry-parser/blob/master/src/util.rs)). Tested against Gyroflow v1.5+ writers. Older project files use a plain JSON `raw_imu` array which the parser also handles, but the path is less rigorously tested.
- **Helmet legality** varies by sanctioning body. Verify your specific series before the chin-bar mount goes on.

## Why this exists

The MyChron 6 finally added a steering gyro, but most of the field is on 5s. A camera-based approach gives 4s/5s drivers the same input data plus things even the 6 does not have (vision, tire). For a competitive 125cc driver looking to find the last 0.2s/lap, the highest-leverage unmeasured channels are: **driver vision**, **steering smoothness**, and **inside-front lift behavior**. This project targets exactly those.

See [ROADMAP.md](ROADMAP.md) for phased build plan.
