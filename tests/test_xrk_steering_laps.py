"""Tests for the per-lap steering metrics derived from a MyChron Steering Angle."""
from __future__ import annotations

import numpy as np

from analysis import xrk_steering_laps as X


def test_steer_rate_recovers_constant_slope():
    fs = X.STEER_FS
    t = np.arange(0, 10, 1 / fs)
    slope = 12.0  # deg/s
    ang = slope * t
    rate = X._steer_rate(ang, fs)
    # interior should be ~constant slope (edges may ring slightly)
    assert np.allclose(rate[50:-50], slope, atol=0.5)


def test_lap_metrics_basic():
    fs = X.STEER_FS
    t = np.arange(0, 40, 1 / fs)          # one 40 s "lap"
    ang = 30 * np.sin(2 * np.pi * 0.2 * t)  # ±30°
    rate = X._steer_rate(ang, fs)
    m = X.lap_metrics(t, ang, rate, lap_idx=3, t0=0.0, t1=40.0)
    assert m is not None
    assert m.lap == 3
    assert m.duration_s == 40.0
    assert m.peak_deg == np.float64(np.abs(ang).max()).astype(float) or m.peak_deg > 25
    assert 0 < m.mean_abs_deg < 30
    assert m.peak_rate_dps > 0
    assert m.jerk_rms_dps2 >= 0


def test_lap_metrics_window_slicing():
    fs = X.STEER_FS
    t = np.arange(0, 100, 1 / fs)
    ang = np.where((t >= 30) & (t < 60), 50.0, 0.0)  # big steer only mid-window
    rate = X._steer_rate(ang, fs)
    inside = X.lap_metrics(t, ang, rate, 0, 30.0, 60.0)
    outside = X.lap_metrics(t, ang, rate, 1, 70.0, 90.0)
    assert inside.peak_deg > 40       # sees the 50° plateau
    assert outside.peak_deg < 5       # quiet stretch


def test_lap_metrics_empty_returns_none():
    t = np.arange(0, 10, 1 / X.STEER_FS)
    ang = np.zeros_like(t)
    rate = np.zeros_like(t)
    assert X.lap_metrics(t, ang, rate, 0, 100.0, 110.0) is None  # window past the data


def test_correction_count_increases_with_sawing():
    fs = X.STEER_FS
    t = np.arange(0, 20, 1 / fs)
    calm = 40 * np.sin(2 * np.pi * 0.15 * t)         # slow, few reversals
    sawing = 40 * np.sin(2 * np.pi * 1.5 * t)        # fast back-and-forth
    mc = X.lap_metrics(t, calm, X._steer_rate(calm, fs), 0, 0.0, 20.0)
    ms = X.lap_metrics(t, sawing, X._steer_rate(sawing, fs), 0, 0.0, 20.0)
    assert ms.corrections > mc.corrections
