# Steering-angle validation — status & method

Goal: prove the **camera-only** steering output (wheel-cam IMU → chassis-yaw-
subtracted steering) against an independent **MyChron 6 2T `Steering Angle`**
reference. The MyChron channel is used *only* to verify/debug — the production
tool stays camera-only.

## TL;DR

- **Camera IMU now reads straight from the Pro-mode `.mp4`** — no manual Gyroflow
  export. Validated to reproduce the `.gyroflow` decode exactly on the 0511 gold
  session (offset −14.03 s, corr 0.658, k = 0.7588, tilt 40.6°).
- **Blocker for the recent footage:** the Go 3S only embeds raw gyro in **Pro
  Video** mode. All the June 18 & June 20 wheel-cam clips were shot in plain
  **Video** mode (in-camera FlowState), which discards the gyro. They are
  unusable for IMU — the samples were never written.
- **No (Pro-camera + 6-2T) pair exists on disk yet**, so the head-to-head
  validation runs on the *next* Pro-mode track day. The harness is built and
  tested, ready to run.
- The 6-2T `Steering Angle` is an **independent** signal (not a chassis-yaw
  restatement) but its provenance (real sensor vs AiM estimate) is undetermined
  from the XRK alone — the camera will be the tiebreaker.

## 1. Camera ingestion from `.mp4` (done)

`pipeline/extract_imu.py`:
- `load_insta360_mp4(path, source)` shells out to `gyro2bb` (telemetry-parser,
  the engine Gyroflow itself uses), parses the betaflight-blackbox CSV, and
  returns the same `ImuStream` the `.gyroflow` path produces. gyro2bb output
  IS the data a `.gyroflow` caches, so the two are equivalent — the back half of
  both loaders is the shared `_finalize_imu_stream`.
- `load_imu(path, source)` dispatches by extension (`.gyroflow` vs video).
- `pipeline.sync_xrk` and `analysis.*` accept either transparently.

Verify a clip has usable gyro:

```bash
gyro2bb -d clip.mp4 | grep is_flowstate_online   # must be false
```

## 2. The FlowState gyro gotcha (recording requirement)

| recording mode | filename | `is_flowstate_online` | raw gyro |
|---|---|---|---|
| Pro Video | `PRO_VID_*.mp4` | `false` (stabilize later in Studio) | ✅ embedded (~1 kHz) |
| Video | `VID_*.mp4` | `true` (stabilized in-camera) | ❌ discarded |

Confirmed across all 71 clips on the Studio: every `PRO_VID_*` had gyro, every
`VID_*` had none. **Always record the wheel-cam in Pro Video mode.** Last
gyro-bearing clip on disk: `PRO_VID_20260607_092516_00_006` (2026-06-07).

## 3. MyChron 6 2T `Steering Angle` — characterization

`analysis/characterize_steering_source.py` (run on `JUN 70_XTreme_a_0040`,
2026-06-20):

- Bicycle-model fit `steer ≈ a·(yaw_rate/v)` is **weak**: R² = 0.04, residual
  98%. So `Steering Angle` is **not** a restatement of chassis yaw — i.e. it is
  *not* circular with the pipeline's `k·GPS_Yaw_Rate` subtraction. Good.
- It **reads at standstill** (<2 m/s: 19° std, 42° p95), which a `r/v` model
  cannot produce.
- It is steering-bandwidth, not noise: 90% of energy below 2.8 Hz (median 0.9 Hz),
  only 3% above 5 Hz.
- But even lag-compensated it couples only **weakly** to lateral-G / yaw-rate
  (~0.24, best lag ≈ −0.22 s).

**Verdict:** an independent, steering-bandwidth signal — either a real column
sensor or a proprietary AiM estimate; the XRK alone can't decide. The wheel-cam
(a *direct physical* measurement of wheel rotation) is the tiebreaker.

![steering source](results/steering_source/steering_source_JUN%20%2070_XTreme_a_0040.png)

## 4. The validation harness (built, tested, awaiting data)

`analysis/validate_steering.py`:

```bash
python -m analysis.validate_steering wheel.mp4 session_6_2t.xrk --all --out-dir results/val/
```

Syncs the wheel-cam to the XRK, builds the production camera steering, aligns it
to `Steering Angle` (residual lag from rate cross-correlation on top of the
coarse clock sync), fits `camera ≈ gain·xrk + offset`, and reports correlation
(angle + rate), gain, RMS, with per-lap overlays.

Interpreting the result:

| outcome | meaning |
|---|---|
| high corr, gain≈1 | both agree — camera method cross-validated |
| high corr, gain≠1 | same dynamics, different scale (units/steering-ratio) — calibrate scale |
| low corr | they disagree — trust the camera (direct measurement); investigate the XRK channel |

## Next step (needs a track day)

Record one session with: wheel-cam in **Pro Video** mode + MyChron 6 2T running.
Then `validate_steering.py <that .mp4> <that .xrk> --all`. That is the first true
camera-vs-ground-truth steering check.
