"""Tests for tap-based time alignment of wheel/helmet/MyChron streams."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.sync_streams import (
    GRAVITY_MPS2,
    find_tap_in_imu,
    find_tap_in_mychron,
    parse_manual_offsets,
    sync,
)


def _make_imu_df(tap_at: float | None, duration: float = 10.0, fs: float = 200.0,
                 tap_strength_mps2: float = 30.0, rng_seed: int = 0) -> pd.DataFrame:
    """Build a parquet-shaped DataFrame with optional tap impulse."""
    n = int(round(duration * fs)) + 1
    t = np.linspace(0, duration, n)
    rng = np.random.default_rng(rng_seed)
    gyro = rng.normal(0.0, 0.02, size=(n, 3))
    accel = np.tile([0.0, 0.0, GRAVITY_MPS2], (n, 1)) + rng.normal(0.0, 0.05, size=(n, 3))
    if tap_at is not None:
        i = int(round(tap_at * fs))
        # Brief 3-sample impulse on accel x
        accel[i:i + 3, 0] += tap_strength_mps2
        gyro[i:i + 3, 0] += 5.0
    return pd.DataFrame({
        "t": t,
        "gx": gyro[:, 0], "gy": gyro[:, 1], "gz": gyro[:, 2],
        "ax": accel[:, 0], "ay": accel[:, 1], "az": accel[:, 2],
    })


# ---------------------------------------------------------------------------
# find_tap_in_imu
# ---------------------------------------------------------------------------

def test_finds_clear_tap_on_accel():
    df = _make_imu_df(tap_at=4.2)
    t_tap = find_tap_in_imu(df, accel_threshold=8.0, gyro_threshold=4.0)
    assert t_tap is not None
    assert abs(t_tap - 4.2) < 0.05


def test_no_tap_returns_none():
    df = _make_imu_df(tap_at=None)
    t_tap = find_tap_in_imu(df, accel_threshold=8.0, gyro_threshold=4.0)
    assert t_tap is None


def test_tap_outside_search_window_ignored():
    df = _make_imu_df(tap_at=45.0, duration=60.0)
    t_tap = find_tap_in_imu(df, accel_threshold=8.0, gyro_threshold=4.0, window_s=30.0) \
        if "window_s" in find_tap_in_imu.__code__.co_varnames else find_tap_in_imu(
            df, accel_threshold=8.0, gyro_threshold=4.0,
        )
    # Default window is 30s — tap at 45s is filtered out either way.
    assert t_tap is None


# ---------------------------------------------------------------------------
# MyChron
# ---------------------------------------------------------------------------

def test_finds_mychron_tap_in_lat_acc():
    n = 6000
    t = np.linspace(0, 30, n)
    rng = np.random.default_rng(0)
    lat = rng.normal(0.0, 0.05, size=n)
    i = int(round(3.7 * (n / 30.0)))
    lat[i:i + 5] += 0.6
    df = pd.DataFrame({"t": t, "lat_acc": lat, "rpm": 0})
    t_tap = find_tap_in_mychron(df)
    assert t_tap is not None
    assert abs(t_tap - 3.7) < 0.05


def test_mychron_no_lat_acc_returns_none():
    df = pd.DataFrame({"t": np.linspace(0, 10, 100), "rpm": 0})
    assert find_tap_in_mychron(df) is None


# ---------------------------------------------------------------------------
# parse_manual_offsets
# ---------------------------------------------------------------------------

def test_parse_manual_offsets_basic():
    out = parse_manual_offsets("wheel=0,helmet=2.34,mychron=1.05")
    assert out == {"wheel": 0.0, "helmet": 2.34, "mychron": 1.05}


def test_parse_manual_offsets_empty_returns_dict():
    assert parse_manual_offsets("") == {}
    assert parse_manual_offsets(None) == {}


def test_parse_manual_offsets_rejects_garbage():
    with pytest.raises(ValueError):
        parse_manual_offsets("wheel:0")


# ---------------------------------------------------------------------------
# Top-level sync()
# ---------------------------------------------------------------------------

def test_sync_aligns_wheel_and_helmet_on_taps(tmp_path):
    sess = tmp_path / "session"
    extracted = sess / "extracted"
    extracted.mkdir(parents=True)

    # Wheel taps at 5.0s, helmet sees the same physical event ~2.0s later in
    # its own clock (because the helmet camera was started later).
    wheel = _make_imu_df(tap_at=5.0)
    helmet = _make_imu_df(tap_at=3.0)   # helmet started 2.0s after wheel
    wheel.to_parquet(extracted / "wheel_imu.parquet")
    helmet.to_parquet(extracted / "helmet_imu.parquet")

    synced = sync(sess)
    assert synced.sources_used_for_sync["wheel"] == "tap"
    assert synced.sources_used_for_sync["helmet"] == "tap"
    # Master clock = wheel's tap. After alignment, the tap event in both
    # streams maps to t_master == 0.
    wheel_tap_t_master = synced.wheel.iloc[
        np.argmax(np.abs(synced.wheel["ax"]))
    ]["t_master"]
    helmet_tap_t_master = synced.helmet.iloc[
        np.argmax(np.abs(synced.helmet["ax"]))
    ]["t_master"]
    assert abs(wheel_tap_t_master - 0.0) < 0.05
    assert abs(helmet_tap_t_master - 0.0) < 0.05


def test_sync_uses_manual_offset_when_provided(tmp_path):
    sess = tmp_path / "session"
    extracted = sess / "extracted"
    extracted.mkdir(parents=True)
    wheel = _make_imu_df(tap_at=None)   # no detectable tap
    wheel.to_parquet(extracted / "wheel_imu.parquet")

    synced = sync(sess, manual_offsets={"wheel": 1.234})
    assert synced.sources_used_for_sync["wheel"] == "manual"
    assert synced.t0_offsets["wheel"] == pytest.approx(-1.234)


def test_sync_marks_missing_when_no_tap_no_manual(tmp_path):
    sess = tmp_path / "session"
    extracted = sess / "extracted"
    extracted.mkdir(parents=True)
    wheel = _make_imu_df(tap_at=None)
    wheel.to_parquet(extracted / "wheel_imu.parquet")

    synced = sync(sess)
    assert synced.sources_used_for_sync["wheel"] == "missing"
    assert synced.t0_offsets["wheel"] == 0.0
