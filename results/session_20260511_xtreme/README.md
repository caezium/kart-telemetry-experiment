# XTreme session — 2026-05-11

Six laps from a real driving session. Wheel-cam Go 3S IMU synchronized
to MyChron XRK via cross-correlation, then segmented by GPS lap
detection. See [`../../RESULTS.md`](../../RESULTS.md) for full
methodology + numbers.

## Per-lap files

Each `lapN.png` has four panels:

- **Steering angle (deg)** — from the wheel-cam IMU after chassis-yaw
  subtraction. Returns near 0 on straights, flicks at corners.
- **Kart yaw rate (deg/s)** — XRK `GPS_Yaw_Rate`. Peaks during corners.
- **GPS speed (km/h)** — dips on corner entry, peaks on straights.
- **GPS track** — lat/lon, colored by speed.

## Summary

| Lap | Time | Peak steer | Peak yaw rate | Peak speed |
|----:|-----:|-----------:|--------------:|-----------:|
| 0   | 54.08s | ±63° | ±309°/s ¹ | 91.4 km/h |
| 1   | 44.56s | ±43° | ±161°/s | 107.1 km/h |
| 2   | 44.32s | ±74° | ±263°/s | 108.6 km/h |
| 3   | 44.64s | ±57° | ±267°/s | 108.6 km/h |
| **4** | **43.44s** | **±41°** | ±125°/s | 106.9 km/h |
| 5   | 49.48s | ±54° | ±381°/s | 108.5 km/h |

¹ Lap 0's peak yaw rate is contaminated by a GPS fix-acquisition spike;
real cornering yaw rate is ~150°/s.

**Lap 4 was fastest** (43.44 s) with the smallest peak steering — a
smooth, clean lap.

## Reproducing

From the repo root:

```bash
.venv/bin/python -m analysis.per_lap \
    /path/to/PRO_VID_20260511_145850_00_060.gyroflow \
    /path/to/a_a_XTreme_a_0619.xrk \
    --all --out-dir results/session_20260511_xtreme/
```
