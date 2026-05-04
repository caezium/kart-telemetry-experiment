# Bench Calibration Protocol

Phase 0 of the [ROADMAP](../../ROADMAP.md). Validates the toolchain end-to-end before any trackside use, and produces per-unit constants the parser needs.

## What this protocol confirms

1. The `.gyroflow` file decodes correctly (base91 → zlib → CBOR → `raw_imu`).
2. After applying the `imu_orientation` transform, rotation about the physical lens axis appears predominantly on a single gyro axis (Z, by convention).
3. Integrated gyro angle matches a known protractor sweep within **±2°**.
4. Per-unit gyro zero-rate bias is small and repeatable (post-bias static RMS <0.5 deg/s).
5. Static drift over 60s after bias subtraction is **<5°**.

If any of these fail, no session data is meaningful. Phase 0 gates Phase 1.

## Bench rig

Minimum:

- Printed 360° protractor wheel, ~A4 size, tick marks every 5°. Print yours or use any free template.
- Flat surface (tile, glass tabletop) the wheel sits on.
- Centering jig: a small dowel or pencil through the wheel's center hole, into a fixed base. Lets the wheel rotate about a fixed point, no slop.
- The Go 3S, mounted **lens-axis vertical** to the wheel — this matches Mount 1 (wheel hub) physically. Use the magnet base or a strip of VHB. Lens up is fine; document the orientation in `notes.md`.
- A pointer or arrow taped to the camera body, projecting out beyond the protractor edge so you can read the angle off the printed scale.

Centering matters more than rig precision. If the camera is off-center, rotation contaminates the accelerometer with centripetal acceleration. The whole point of the on-axis wheel hub mount is to avoid that — the bench rig must replicate it.

## Recording protocol

One recording per Go 3S unit. Settings:

- Resolution / framerate doesn't matter for IMU. Use whatever default. Lowest setting that records is fine.
- Record a clap or sharp tap on the camera body at the start (sync mark, also useful when we get to multi-camera later).

Sequence (target ~3 min total, run a stopwatch):

| t (s)   | Action                                          | Purpose                                       |
| ------- | ----------------------------------------------- | --------------------------------------------- |
| 0–5     | Static at 0°. Hands off the rig.                | Zero-rate bias baseline.                      |
| 5       | Tap the camera body once, sharp.                | Sync mark.                                    |
| 5–35    | Static at 0°. Hands off.                        | 30s for bias estimation.                      |
| 35      | Rotate to **+90°** in ~1s. Single smooth move.  | Step input.                                   |
| 35–37   | Hold at +90°.                                   | Step plateau.                                 |
| 37      | Rotate back to 0° in ~1s.                       | Step input, opposite sign.                    |
| 37–39   | Hold at 0°.                                     | Re-check bias.                                |
| 39      | Rotate to **−90°** in ~1s.                      | Step, negative side.                          |
| 39–41   | Hold at −90°.                                   |                                               |
| 41      | Rotate back to 0° in ~1s.                       |                                               |
| 41–43   | Hold at 0°.                                     |                                               |
| 43      | Rotate to **+90°** in ~1s.                      | Repeat for symmetry / repeatability.          |
| 43–45   | Hold.                                           |                                               |
| 45      | Rotate back to 0° in ~1s.                       |                                               |
| 45–105  | **Static at 0° for 60s.** Hands completely off. | Drift characterization after motion.          |
| 105–110 | Slow continuous sweep 0° → +90° at ~30 deg/s.   | Constant-rate validation (rate channel test). |
| 110–115 | Hold at +90°.                                   |                                               |
| 115–120 | Slow continuous sweep +90° → 0° at ~30 deg/s.   |                                               |

Read the protractor at each "hold" by eye, write down the actual angle to the nearest degree. The IMU result will be compared against these handwritten ground-truth values.

## Files this generates

For each Go 3S unit (call its serial-suffix `<UNIT>`):

```
data/calibration/<UNIT>/
├── original.insv             # raw camera file
├── original.gyroflow         # opened in Gyroflow + saved
├── ground_truth.csv          # handwritten holds: t_seconds, angle_deg
├── notes.md                  # rig photos, lens-up vs lens-down, anything weird
└── result.json               # output of running the parser+validator on this set
```

`result.json` schema (produced by the parser/validator script in Phase 0):

```json
{
  "unit": "<UNIT>",
  "recorded_at": "YYYY-MM-DD",
  "sample_rate_hz": 200.4,
  "imu_orientation": "yXZ",
  "axis_used_for_steering": "gz",
  "axis_separation_db": 18.5,
  "gyro_bias_dps": [0.012, -0.034, 0.005],
  "accel_bias_mps2": [0.02, 0.01, -0.04],
  "static_rms_after_bias_dps": 0.18,
  "step_input_validation": [
    {"target_deg": 90,  "measured_deg": 89.4, "abs_error_deg": 0.6},
    {"target_deg": 0,   "measured_deg": 0.2,  "abs_error_deg": 0.2},
    {"target_deg": -90, "measured_deg": -90.7,"abs_error_deg": 0.7}
  ],
  "drift_60s_deg": 2.1,
  "constant_rate_segment_dps": {"target": 30.0, "measured_median": 29.7, "iqr": 0.8},
  "passed": true
}
```

## Pass criteria

All of:

- `max(abs_error_deg)` across step inputs **≤ 2°**
- `drift_60s_deg` **< 5°**
- `static_rms_after_bias_dps` **< 0.5 deg/s** (sanity: gyro is actually a gyro, not noise)
- `axis_separation_db` **≥ 10 dB** — the steering axis variance during the step phase is at least 10× the next-largest axis. Lower than that suggests the camera was mounted off-axis or the orientation transform is wrong.

## How to run (once parser exists)

```bash
python -m pipeline.calibrate data/calibration/<UNIT>/
```

The calibration script:

1. Loads `original.gyroflow`, decodes `raw_imu`.
2. Applies the `imu_orientation` transform.
3. Detects static segments (low gyro magnitude across all axes for ≥1s).
4. Estimates per-axis gyro bias from the first 30s static segment.
5. Identifies the "steering axis" as the gyro channel with the largest variance during the step-input phase.
6. Integrates that axis (bias-subtracted) and reports value at each handheld plateau.
7. Cross-references against `ground_truth.csv` to compute errors.
8. Writes `result.json`.

Manual handwritten `ground_truth.csv` example:

```csv
t_seconds,angle_deg,note
5,0,sync_tap
36,90,first_step_target
38,0,
40,-90,
42,0,
44,90,
46,0,
107,90,after_slow_sweep
117,0,after_return_sweep
```

Times don't have to be precise; the script aligns by finding the plateaus in the IMU trace and matches them to the closest handwritten entry. The handwritten values *are* the ground truth.

## Per-unit, not per-session

Run this once per Go 3S unit. Re-run if:

- Gyroflow major version changes
- Camera firmware updates
- A bench result drifts noticeably (suspect a bad mount or aging IMU)
- You buy a second unit (helmet cam in Phase 3 — calibrate it independently)
