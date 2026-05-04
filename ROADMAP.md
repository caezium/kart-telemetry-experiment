# Roadmap

Phased build, smallest tracer-bullet first. Each phase ends with a usable artifact.

## Phase 0 — Bench validation (no kart)

Verify the toolchain end-to-end before going trackside.

- [ ] Mount Go 3S on a known-angle test rig (cardboard, protractor)
- [ ] Record video while rotating through a known sweep (e.g., ±90° at fixed rate)
- [ ] Run Gyroflow CSV export
- [ ] Confirm `pipeline/extract_imu.py` reproduces the swept angle within ±2°
- [ ] Confirm drift over a 60s static recording is < 5°

**Exit criterion:** known input → predicted output within tolerance.

## Phase 1 — Wheel mount, single session

One camera, one session, steering channel only.

- [ ] Mount on wheel hub, on rotation axis (verify with bench-test calibration recording first)
- [ ] Record one full session (~15 min)
- [ ] Extract IMU CSV via Gyroflow
- [ ] Run `analysis/steering_metrics.py` to produce per-lap:
  - peak steering angle per corner
  - steering rate distribution
  - jerk integral per corner (smoothness score)
  - correction count per corner
- [ ] Sync to MyChron lap markers manually (clap test)
- [ ] Output: PDF/HTML report with per-corner metrics overlaid on lap times

**Exit criterion:** can answer "which corner had my smoothest input on lap 7?"

## Phase 2 — Wheel cam tire observation

Same mount, lens angled to capture front tire.

- [ ] Calibrate Gyroflow stabilization to keep tire steady in frame
- [ ] Record session with tire visible
- [ ] Manually label inside-front lift events in 5–10 corners
- [ ] Build CV detector (frame differencing on tire-vs-ground boundary)
- [ ] Validate detector against manual labels
- [ ] Output: lift duration per corner, plotted vs lap time

**Exit criterion:** detector agrees with manual labels >85% of the time.

## Phase 3 — Helmet mount

Add second Go 3S, chin-bar mounted.

- [ ] **Verify series regulations first.** Document compliance.
- [ ] Mount on chin bar, lens forward
- [ ] Record session with both wheel-cam and helmet-cam running
- [ ] Extract head IMU separately
- [ ] Compute head yaw rate timeline
- [ ] Cross-correlate head yaw events vs steering input events per corner
- [ ] Compute look-ahead lead time (median ms between head turn onset and steering input onset)

**Exit criterion:** can quantify "I look 0.34s ahead of my hands at Turn 3, and 0.08s at Turn 7" — the latter is the corner to coach.

## Phase 4 — Multi-stream debrief tool

Stitch everything together into a usable debrief artifact.

- [ ] Synced timeline plot: lap time, speed, lat-G (MyChron) + steering angle/rate (wheel) + head yaw (helmet) + driver POV thumbnail track
- [ ] Per-corner cards: input quality score, lift duration, look-ahead time
- [ ] Lap-over-lap fatigue indicator (head stability + steering jerk + breathing rate composite)
- [ ] HTML output a coach can scrub through

**Exit criterion:** a competitive driver and their coach use it for an actual debrief and come away with one specific actionable.

## Phase 5 — Field beta

Hand it to a second driver. See if it survives a stranger.

- [ ] Setup-from-zero docs
- [ ] One-command session ingestion
- [ ] Standardized session metadata schema (track, kart number, tire compound, weather, kart setup snapshot)

**Exit criterion:** another 125cc driver runs a full session through the tool without my help.

## Stretch

- Slip angle from high-frame-rate tire video (needs 1080p120+ and good lighting)
- Tire wear progression from session-to-session
- Setup change A/B comparison view (axle stiffness, seat position, etc.)
- Kart-to-kart input comparison (same track, same conditions, two drivers)
