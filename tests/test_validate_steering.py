"""Tests for the camera-vs-MyChron steering validation math.

Hermetic: the alignment / gain-fit / correlation logic is exercised on synthetic
signals by monkeypatching the two upstream producers (camera steering and XRK
steering), so no real video/XRK pair is needed.
"""
from __future__ import annotations

import numpy as np
import pytest

from analysis import validate_steering as V


# ---------------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------------

def test_best_lag_recovers_shift():
    fs = 50.0
    t = np.arange(0, 40, 1 / fs)
    base = np.sin(2 * np.pi * 0.5 * t) + 0.3 * np.sin(2 * np.pi * 1.7 * t)
    shift = 15  # samples → 0.30 s
    a = base.copy()
    b = np.roll(base, shift)            # b lags base by `shift`
    lag, corr = V._best_lag(a, b, fs=fs)
    assert abs(abs(lag) - shift / fs) < 1.0 / fs + 1e-9
    assert abs(corr) > 0.9


def test_fit_gain_offset_exact():
    xrk = np.linspace(-40, 70, 500)
    cam = 2.0 * xrk + 3.0
    gain, offset, rms = V._fit_gain_offset(cam, xrk)
    assert gain == pytest.approx(2.0, abs=1e-6)
    assert offset == pytest.approx(3.0, abs=1e-6)
    assert rms == pytest.approx(0.0, abs=1e-6)


def test_resample_common_overlap():
    t_a = np.arange(0, 10, 0.01)
    t_b = np.arange(2, 12, 0.02)
    grid, a, b = V._resample_common(t_a, np.sin(t_a), t_b, np.cos(t_b), fs=50)
    assert grid[0] >= 2.0 and grid[-1] <= 10.0
    assert len(grid) == len(a) == len(b)


def test_resample_common_too_short_raises():
    t_a = np.arange(0, 5, 0.01)
    t_b = np.arange(4.5, 10, 0.01)   # < 1 s overlap
    with pytest.raises(ValueError, match="(?i)overlap"):
        V._resample_common(t_a, t_a, t_b, t_b)


# ---------------------------------------------------------------------------
# xrk_steering channel guard
# ---------------------------------------------------------------------------

class _FakeLog:
    def __init__(self, channels):
        self.channels = channels


def test_xrk_steering_missing_channel_raises():
    with pytest.raises(ValueError, match="Steering Angle"):
        V.xrk_steering(_FakeLog({}))


# ---------------------------------------------------------------------------
# compare_steering end-to-end (monkeypatched producers)
# ---------------------------------------------------------------------------

def _install_fakes(monkeypatch, *, gain, sign=1, noise=0.0, lag_s=0.0):
    """camera steering = true signal; xrk steering = sign*true/gain (+noise, +lag)."""
    fs_cam = 200.0
    t_cam = np.arange(0, 80, 1 / fs_cam)
    rng = np.random.RandomState(0)
    true_angle = (20 * np.sin(2 * np.pi * 0.3 * t_cam)
                  + 8 * np.sin(2 * np.pi * 1.1 * t_cam))
    true_rate = np.gradient(true_angle, 1 / fs_cam)

    def fake_steering_from_synced(stream, sync, **kw):
        return t_cam, true_angle, true_rate

    fs_x = 50.0
    t_x = np.arange(0, 80, 1 / fs_x)
    xrk_angle = sign * np.interp(t_x - lag_s, t_cam, true_angle) / gain
    xrk_angle = xrk_angle + noise * rng.randn(len(t_x))

    def fake_xrk_steering(log):
        return t_x, xrk_angle

    monkeypatch.setattr(V, "steering_from_synced", fake_steering_from_synced)
    monkeypatch.setattr(V, "xrk_steering", fake_xrk_steering)

    class _S:  # stand-in SyncedXrk; compare_steering only forwards it
        xrk_log = object()
    return None, _S()


def test_compare_steering_recovers_gain_and_high_corr(monkeypatch):
    stream, sync = _install_fakes(monkeypatch, gain=2.0, sign=1, noise=0.0)
    val, series = V.compare_steering(stream, sync)
    assert val.corr_angle > 0.98
    assert val.gain == pytest.approx(2.0, rel=0.05)
    assert val.rms_deg < 1.0
    assert val.sign == 1
    assert series["grid"].size > 1000


def test_compare_steering_handles_sign_flip(monkeypatch):
    stream, sync = _install_fakes(monkeypatch, gain=1.5, sign=-1, noise=0.0)
    val, _ = V.compare_steering(stream, sync)
    assert val.sign == -1
    assert val.corr_angle < -0.9
    assert val.gain == pytest.approx(1.5, rel=0.05)


def test_compare_steering_low_corr_on_unrelated(monkeypatch):
    # XRK steering replaced by independent noise → should report weak agreement
    fs_cam = 200.0
    t_cam = np.arange(0, 60, 1 / fs_cam)
    rng = np.random.RandomState(1)
    cam = 20 * np.sin(2 * np.pi * 0.3 * t_cam)
    monkeypatch.setattr(V, "steering_from_synced",
                        lambda s, y, **k: (t_cam, cam, np.gradient(cam, 1 / fs_cam)))
    t_x = np.arange(0, 60, 1 / 50.0)
    monkeypatch.setattr(V, "xrk_steering",
                        lambda log: (t_x, rng.randn(len(t_x))))

    class _S:
        xrk_log = object()
    val, _ = V.compare_steering(None, _S())
    assert abs(val.corr_angle) < 0.3
