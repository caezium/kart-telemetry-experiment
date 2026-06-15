"""Regression tests for the sync_xrk fixes (code-review round)."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.extract_imu import ImuStream
from pipeline.sync_xrk import (
    SyncedXrk,
    chassis_yaw_from_imu,
    chassis_yaw_from_xrk,
    clean_gps_yaw_dps,
    cross_correlate_offset,
    detect_column_axis,
    estimate_column_tilt_factor,
    _robust_normalise,
)
from tests.conftest import FakeXrkLog, make_gps_yaw_channel


def _stream(t, gyro, accel, fs):
    return ImuStream(t=t, gyro=gyro, accel=accel, sample_rate_hz=fs,
                     source="wheel", orientation="XYZ")


# ---------------------------------------------------------------------------
# clean_gps_yaw_dps
# ---------------------------------------------------------------------------

def test_clean_gps_yaw_replaces_spikes_only():
    t = np.linspace(0, 1, 11)
    yr = np.array([1, 2, 3, 9999, 5, 6, 7, 8, 9, 10, 11], dtype=float)
    out = clean_gps_yaw_dps(t, yr, glitch_threshold_dps=400.0)
    # The spike is replaced by interpolation between neighbours (3 and 5 -> 4).
    assert out[3] == pytest.approx(4.0)
    # Good samples are untouched.
    good_idx = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]
    assert np.allclose(out[good_idx], yr[good_idx])
    assert np.isfinite(out).all()


def test_clean_gps_yaw_handles_nan_and_inf():
    t = np.linspace(0, 1, 5)
    yr = np.array([1.0, np.nan, 3.0, np.inf, 5.0])
    out = clean_gps_yaw_dps(t, yr)
    assert np.isfinite(out).all()
    assert out[1] == pytest.approx(2.0)   # between 1 and 3
    assert out[3] == pytest.approx(4.0)   # between 3 and 5


def test_clean_gps_yaw_all_bad_returns_zeros():
    t = np.linspace(0, 1, 5)
    yr = np.full(5, 9999.0)
    out = clean_gps_yaw_dps(t, yr)
    assert np.array_equal(out, np.zeros(5))


# ---------------------------------------------------------------------------
# _robust_normalise — MAD=0 must not explode
# ---------------------------------------------------------------------------

def test_robust_normalise_flat_with_spike_does_not_explode():
    # >50% identical samples -> MAD collapses to 0; must fall back to std,
    # not divide by 1e-12 (which would blow the spike up to ~1e12).
    x = np.zeros(100)
    x[0] = 1.0
    out = _robust_normalise(x)
    assert np.isfinite(out).all()
    assert np.abs(out).max() < 1e3   # was ~1e12 before the fix


def test_robust_normalise_constant_returns_zeros():
    out = _robust_normalise(np.full(50, 7.3))
    assert np.array_equal(out, np.zeros(50))


# ---------------------------------------------------------------------------
# cross_correlate_offset — bounded peak + recovers a known offset
# ---------------------------------------------------------------------------

def _distinctive_yaw(t):
    """Non-periodic yaw shape (irregular Gaussian bumps of varying sign) so the
    cross-correlation has a single unambiguous peak — a real lap's chassis yaw
    is not periodic, unlike a pure sine which aliases at its period."""
    bumps = [(7.0, 2.0, 1.0), (15.0, 1.0, -0.6), (23.0, 3.0, 0.8),
             (34.0, 1.5, -1.0), (42.0, 2.5, 0.5)]
    y = np.zeros_like(t)
    for c, w, a in bumps:
        y += a * np.exp(-((t - c) ** 2) / (2 * w ** 2))
    return y


def test_cross_correlate_recovers_known_offset_and_peak_bounded():
    fs = 25.0
    t = np.arange(0, 60, 1 / fs)
    shape = _distinctive_yaw(t)
    # XRK starts 8 s later in wall-clock: t_xrk = t_imu + 8 for the same event.
    true_offset = 8.0
    offset, peak, sign = cross_correlate_offset(t, shape, t + true_offset, shape, fs=fs)
    assert offset == pytest.approx(true_offset, abs=1 / fs + 1e-6)
    assert 0.0 <= peak <= 1.0 + 1e-9     # genuine normalized correlation
    assert peak > 0.9                    # identical shape -> near 1
    assert sign == 1


def test_cross_correlate_detects_sign_flip():
    fs = 25.0
    t = np.arange(0, 60, 1 / fs)
    shape = _distinctive_yaw(t)
    offset, peak, sign = cross_correlate_offset(t, shape, t, -shape, fs=fs)
    assert sign == -1
    assert peak == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# chassis_yaw_from_imu — edge trim must not empty the signal
# ---------------------------------------------------------------------------

def _flat_stream(duration_s=30.0, fs=200.0):
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    gyro = np.zeros((n, 3))
    gyro[:, 2] = 0.2 * np.sin(2 * np.pi * 0.1 * t)   # some z motion -> PCA picks z
    accel = np.tile([0, 0, 9.80665], (n, 1)).astype(float)
    return _stream(t, gyro, accel, fs)


def test_edge_trim_zero_keeps_full_signal():
    s = _flat_stream()
    t, yaw, axis, bias = chassis_yaw_from_imu(s, edge_trim_s=0.0)
    assert len(t) == len(s.t)      # not the empty [0:-0] slice
    assert len(yaw) == len(s.t)


def test_edge_trim_skipped_when_longer_than_recording():
    s = _flat_stream(duration_s=2.0, fs=200.0)   # 2 s recording
    t, yaw, axis, bias = chassis_yaw_from_imu(s, edge_trim_s=5.0)  # 5 s trim each end
    assert len(t) == len(s.t)      # trim skipped rather than emptying


# ---------------------------------------------------------------------------
# estimate_column_tilt_factor — NaN (not silent 0.0) when no quiet samples
# ---------------------------------------------------------------------------

def test_tilt_factor_nan_without_quiet_samples():
    n = 1000
    t = np.arange(n) / 200.0
    gyro = np.full((n, 3), np.deg2rad(50.0))   # always spinning fast -> no quiet
    accel = np.tile([0, 0, 9.80665], (n, 1)).astype(float)
    s = _stream(t, gyro, accel, 200.0)
    axis = detect_column_axis(gyro)
    k = estimate_column_tilt_factor(s, axis)
    assert np.isnan(k)


# ---------------------------------------------------------------------------
# chassis_yaw_from_xrk — cleans at the source
# ---------------------------------------------------------------------------

def test_chassis_yaw_from_xrk_is_glitch_cleaned():
    t = np.linspace(0, 2, 51)
    yaw = np.full(51, 10.0)
    yaw[25] = 2509.0    # the real-world fix-loss spike
    log = FakeXrkLog({"GPS_Yaw_Rate": make_gps_yaw_channel(t, yaw)})
    _, yr_rad = chassis_yaw_from_xrk(log)
    # The spike must be gone (cleaned at source so cross-correlation never sees it).
    assert np.rad2deg(np.abs(yr_rad)).max() < 400.0
