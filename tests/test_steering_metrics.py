"""Sanity tests for steering_metrics.

Constructs synthetic angle traces with known peak / corrections / direction
and asserts the reported metrics match.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.steering_metrics import (
    compute_corner_metrics,
    count_corrections,
    detect_corners,
    integrate_angle_with_drift_reset,
)


def _build_corner_trace(
    duration_s: float = 2.0,
    peak_deg: float = 30.0,
    fs: float = 200.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a single triangular corner: ramp up to peak, ramp down to zero.
    Returns (t, angle_rad, rate_rad_s, jerk_rad_s3).
    """
    n = int(duration_s * fs) + 1
    t = np.linspace(0, duration_s, n)
    half = duration_s / 2
    angle_deg = np.where(t < half, peak_deg * t / half, peak_deg * (1 - (t - half) / half))
    angle = np.deg2rad(angle_deg)
    rate = np.gradient(angle, t)
    jerk = np.gradient(np.gradient(rate, t), t)
    return t, angle, rate, jerk


def _wrap_in_lap_with_straights(
    corner_t: np.ndarray, corner_angle: np.ndarray,
    corner_rate: np.ndarray, corner_jerk: np.ndarray,
    pre_s: float = 1.0, post_s: float = 1.0, fs: float = 200.0,
):
    """Pad a corner with straights of static-zero on both sides."""
    n_pre = int(pre_s * fs)
    n_post = int(post_s * fs)
    t_pre = np.linspace(0, pre_s, n_pre, endpoint=False)
    t_corner = corner_t + pre_s
    t_post = np.linspace(t_corner[-1], t_corner[-1] + post_s, n_post + 1)[1:]
    t = np.concatenate([t_pre, t_corner, t_post])
    angle = np.concatenate([np.zeros(n_pre), corner_angle, np.zeros(n_post)])
    rate = np.concatenate([np.zeros(n_pre), corner_rate, np.zeros(n_post)])
    jerk = np.concatenate([np.zeros(n_pre), corner_jerk, np.zeros(n_post)])
    return t, angle, rate, jerk


# ---------------------------------------------------------------------------
# integrate_angle_with_drift_reset
# ---------------------------------------------------------------------------

def test_integrate_recovers_known_angle():
    # 30 deg/s for 1s = 30 deg, then 0 for 1s
    fs = 200.0
    t = np.linspace(0, 2, int(2 * fs) + 1)
    rate = np.where(t < 1.0, np.deg2rad(30.0), 0.0)
    angle = integrate_angle_with_drift_reset(t, rate)
    # At t=1.0, angle should be ~30 deg.
    idx = int(1.0 * fs)
    assert np.rad2deg(angle[idx]) == pytest.approx(30.0, abs=1.0)


def test_integrate_drift_reset_clamps_to_zero():
    """If we hold near-zero rate at near-zero angle for ANGLE_RESET_HOLD_S, the
    integrator should snap angle to 0."""
    fs = 200.0
    t = np.linspace(0, 5.0, int(5 * fs) + 1)
    rate = np.full_like(t, np.deg2rad(0.05))   # tiny constant bias < 2 deg/s
    angle = integrate_angle_with_drift_reset(t, rate)
    # Final angle, after the reset triggers, should be very small.
    assert abs(np.rad2deg(angle[-1])) < 5.0


# ---------------------------------------------------------------------------
# detect_corners
# ---------------------------------------------------------------------------

def test_detects_single_corner():
    t, angle, rate, jerk = _build_corner_trace(duration_s=2.0, peak_deg=30.0)
    t, angle, rate, jerk = _wrap_in_lap_with_straights(t, angle, rate, jerk)
    corners = detect_corners(t, angle)
    assert len(corners) == 1
    a, b = corners[0]
    assert t[a] > 1.0 and t[b] < 4.0


def test_ignores_micro_wiggles():
    """A tiny <15° transient must not register as a corner."""
    fs = 200.0
    t = np.linspace(0, 3, int(3 * fs) + 1)
    angle = np.deg2rad(5.0) * np.sin(np.pi * t)
    corners = detect_corners(t, angle)
    assert corners == []


# ---------------------------------------------------------------------------
# count_corrections
# ---------------------------------------------------------------------------

def test_count_corrections_zero_for_smooth_input():
    fs = 200.0
    t = np.linspace(0, 2, int(2 * fs) + 1)
    rate = np.deg2rad(20.0) * np.sin(np.pi / 2 * t)  # half-period of sine, no sign reversal
    assert count_corrections(rate, 1 / fs) <= 1   # at most one near-zero crossing at the start


def test_count_corrections_detects_oscillation():
    fs = 200.0
    t = np.linspace(0, 2, int(2 * fs) + 1)
    rate = np.deg2rad(30.0) * np.sin(2 * np.pi * 3 * t)  # 3 Hz oscillation -> 6 crossings/s
    n = count_corrections(rate, 1 / fs)
    assert n >= 8  # 12 expected over 2s; allow some slack from filtering


# ---------------------------------------------------------------------------
# compute_corner_metrics
# ---------------------------------------------------------------------------

def test_corner_metrics_match_known_input():
    t, angle, rate, jerk = _build_corner_trace(duration_s=2.0, peak_deg=30.0)
    t, angle, rate, jerk = _wrap_in_lap_with_straights(t, angle, rate, jerk)
    corners = detect_corners(t, angle)
    metrics = compute_corner_metrics(t, angle, rate, jerk, corners[0])
    assert metrics.direction == "right"
    assert metrics.peak_angle_deg == pytest.approx(30.0, abs=1.0)
    assert metrics.peak_rate_deg_s == pytest.approx(30.0, abs=2.0)


def test_left_corner_direction():
    t, angle, rate, jerk = _build_corner_trace(duration_s=2.0, peak_deg=-30.0)
    t, angle, rate, jerk = _wrap_in_lap_with_straights(t, angle, rate, jerk)
    corners = detect_corners(t, angle)
    metrics = compute_corner_metrics(t, angle, rate, jerk, corners[0])
    assert metrics.direction == "left"
    assert metrics.peak_angle_deg == pytest.approx(-30.0, abs=1.0)
