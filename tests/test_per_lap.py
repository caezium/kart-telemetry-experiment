"""Regression tests for the per_lap fixes (code-review round)."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.extract_imu import ImuStream
from pipeline.sync_xrk import SyncedXrk
from analysis.per_lap import (
    LapData,
    detect_laps_from_gps,
    extract_lap,
    lap_summary,
    steering_from_synced,
)
from tests.conftest import (
    FakeChannelTable,
    FakeXrkLog,
    make_gps_track_log,
    make_gps_yaw_channel,
)


# ---------------------------------------------------------------------------
# detect_laps_from_gps — away-state reset on a rejected short window
# ---------------------------------------------------------------------------

def _dist_profile():
    """t (s) and distance-from-start (m): a short aborted nose-out (rejected),
    then two real ~22 s laps."""
    fs = 25.0
    t = np.arange(0, 60, 1 / fs)
    dist = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < 4:
            dist[i] = 0
        elif ti < 8:           # nose out to 60 m and back by t=8  (short: rejected)
            dist[i] = 60 * (1 - abs((ti - 6) / 2))
        elif ti < 30:          # real lap 1: out and back, return ~t=30
            dist[i] = 60 * (1 - abs((ti - 19) / 11))
        else:                  # real lap 2: out and back, return ~t=52
            dist[i] = max(0.0, 60 * (1 - abs((ti - 41) / 11)))
    return t, dist


def test_detect_laps_resets_away_on_short_rejection():
    t, dist = _dist_profile()
    log = make_gps_track_log(t, dist)
    laps = detect_laps_from_gps(log)
    assert len(laps) >= 1
    # The first recorded lap must start AFTER the aborted nose-out (~8 s),
    # not from the original reference at t≈0 (the pre-fix bug).
    assert laps[0][0] > 5.0


def test_detect_laps_tolerates_misaligned_lon_channel():
    t, dist = _dist_profile()
    # Longitude channel has a different length than Latitude — must not crash.
    log = make_gps_track_log(t, dist, lon_length_delta=-7)
    laps = detect_laps_from_gps(log)   # would raise on np.hypot length mismatch pre-fix
    assert isinstance(laps, list)


# ---------------------------------------------------------------------------
# Synthetic synced session for steering / extract / summary tests
# ---------------------------------------------------------------------------

def _synced(imu_dur_s=120.0, gps_start_s=10.0, gps_dur_s=60.0, fs=200.0):
    """IMU recording longer than (and offset from) the GPS span, to exercise
    the overlap-restriction in steering_from_synced."""
    n = int(imu_dur_s * fs)
    t = np.arange(n) / fs
    gyro = np.zeros((n, 3))
    gyro[:, 2] = np.deg2rad(20.0) * np.sin(2 * np.pi * 0.2 * t)  # steering wobble
    accel = np.tile([0, 0, 9.80665], (n, 1)).astype(float)
    stream = ImuStream(t=t, gyro=gyro, accel=accel, sample_rate_hz=fs,
                       source="wheel", orientation="XYZ")

    # GPS yaw over [gps_start_s, gps_start_s+gps_dur_s] on the XRK clock.
    gps_fs = 25.0
    t_gps = np.arange(gps_start_s, gps_start_s + gps_dur_s, 1 / gps_fs)
    yaw_dps = 30.0 * np.sin(2 * np.pi * 0.05 * (t_gps - gps_start_s))
    log = FakeXrkLog({"GPS_Yaw_Rate": make_gps_yaw_channel(t_gps, yaw_dps)})

    # offset chosen so IMU t=0 maps to XRK t=5 -> IMU spans [5, 125] on XRK clock,
    # but GPS only covers [10, 70]; the head [5,10) and tail (70,125] are out of range.
    sync = SyncedXrk(
        gyroflow_path=None, xrk_path=None,
        offset_imu_to_xrk_s=5.0, corr_peak=0.9, sign=1,
        column_axis=np.array([0.0, 0.0, 1.0]),
        column_tilt_factor=0.8, gyro_bias=np.zeros(3),
        quiet_sample_count=10_000, imu_stream=stream, xrk_log=log,
    )
    return stream, sync, (gps_start_s, gps_start_s + gps_dur_s)


def test_steering_restricted_to_gps_overlap():
    stream, sync, (g0, g1) = _synced()
    t_xrk, ang, rate = steering_from_synced(stream, sync)
    # No samples outside the GPS span — the lead-in/tail are dropped, not
    # fabricated with a flat-extrapolated yaw.
    assert t_xrk.min() >= g0 - 1e-6
    assert t_xrk.max() <= g1 + 1e-6
    assert np.isfinite(ang).all()
    assert len(t_xrk) == len(ang) == len(rate)


def test_steering_nan_tilt_falls_back_to_no_subtraction():
    stream, sync, _ = _synced()
    sync.column_tilt_factor = float("nan")   # geometry was unreliable
    t_xrk, ang, rate = steering_from_synced(stream, sync)
    assert np.isfinite(ang).all()            # NaN must not poison the integral


# ---------------------------------------------------------------------------
# extract_lap + lap_summary — empty windows must not crash
# ---------------------------------------------------------------------------

def test_lap_summary_empty_steering_returns_nan():
    lap = LapData(
        lap_index=0, t_start=0.0, t_end=1.0, duration=1.0,
        t_imu=np.array([]), steering_angle_deg=np.array([]),
        steering_rate_dps=np.array([]), xrk_channels={}, xrk_t={},
    )
    s = lap_summary(lap)   # pre-fix: ValueError zero-size reduction
    assert np.isnan(s["peak_steer_deg"])


def test_extract_lap_window_outside_imu_is_empty_not_crash():
    stream, sync, (g0, g1) = _synced()
    steering = steering_from_synced(stream, sync)
    # A window entirely outside the steering coverage -> empty, no crash.
    lap = extract_lap(stream, sync, (g1 + 100.0, g1 + 110.0), lap_index=9,
                      steering=steering)
    assert not lap.has_imu
    assert np.isnan(lap_summary(lap)["peak_steer_deg"])


def test_extract_lap_half_open_no_shared_boundary():
    stream, sync, (g0, g1) = _synced()
    steering = steering_from_synced(stream, sync)
    mid = (g0 + g1) / 2
    a = extract_lap(stream, sync, (g0, mid), 0, steering=steering)
    b = extract_lap(stream, sync, (mid, g1), 1, steering=steering)
    # Half-open [start, end): the sample at t==mid belongs to b only, never both.
    if a.t_imu.size and b.t_imu.size:
        assert a.t_imu.max() < mid <= b.t_imu.min() + 1e-9
