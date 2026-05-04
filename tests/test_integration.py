"""Cross-module contract test: calibrate.py writes result.json, then
extract_imu.py auto-picks up the per-unit bias for sessions.

This is the only test that exercises the full directory layout described
in the README (data/calibration/<UNIT>/ and data/sessions/<ID>/).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.calibrate import _result_to_jsonable, calibrate_unit
from pipeline.extract_imu import extract_session
from tests.conftest import make_static_trace, make_step_rotation_trace, write_synthetic_gyroflow


def _build_data_layout(tmp_path: Path) -> tuple[Path, Path, tuple[float, float, float]]:
    """Build a tmp `data/` with one calibration unit and one session.

    Returns (calibration_unit_dir, session_dir, true_bias_dps).
    """
    bias = (0.20, -0.05, 0.30)
    data_root = tmp_path / "data"
    cal_unit = data_root / "calibration" / "UNITA"
    session = data_root / "sessions" / "20260101_practice"
    cal_unit.mkdir(parents=True)
    session.mkdir(parents=True)

    # 1. Calibration recording with a known bias.
    cal_samples, _ = make_step_rotation_trace(
        rotations=[
            (10.0, 1.0,  90.0),
            (12.0, 1.0, -90.0),
            (14.0, 1.0, -90.0),
            (16.0, 1.0,  90.0),
        ],
        axis=2, rate_hz=200.0, gyro_bias_dps=bias, noise_dps=0.05,
    )
    last_t = cal_samples[-1]["timestamp_ms"] / 1000.0
    rng = np.random.default_rng(0)
    for ts in np.linspace(last_t + 1 / 200.0, last_t + 20.0, int(20.0 * 200)):
        cal_samples.append({
            "timestamp_ms": float(ts * 1000.0),
            "gyro": [float(bias[k] + rng.normal(0.0, 0.05)) for k in range(3)],
            "accl": [0.0, 0.0, 9.80665],
            "magn": None,
        })
    write_synthetic_gyroflow(cal_unit / "original.gyroflow", cal_samples, imu_orientation="XYZ")
    pd.DataFrame({
        "t_seconds": [11.5, 13.5, 15.5, 17.5],
        "angle_deg": [90.0,  0.0, -90.0,  0.0],
    }).to_csv(cal_unit / "ground_truth.csv", index=False)

    # 2. Session recording with the same bias (same physical camera unit).
    session_samples = make_static_trace(duration_s=5.0, gyro_bias_dps=bias, noise_dps=0.05)
    write_synthetic_gyroflow(session / "wheel.gyroflow", session_samples, imu_orientation="XYZ")

    return cal_unit, session, bias


def test_session_load_picks_up_calibrated_bias(tmp_path):
    cal_unit, session, true_bias = _build_data_layout(tmp_path)

    # Run the calibration script — produces result.json.
    result = calibrate_unit(cal_unit)
    assert result.passed
    (cal_unit / "result.json").write_text(json.dumps(_result_to_jsonable(result), indent=2))

    # Now load the session.
    streams = extract_session(session)
    assert "wheel" in streams
    wheel = streams["wheel"]
    # The static-recording wheel stream should have near-zero gyro mean
    # (bias has been subtracted at load time).
    np.testing.assert_allclose(wheel.gyro.mean(axis=0), np.zeros(3), atol=np.deg2rad(0.1))
    assert wheel.gyro_bias_rps is not None


def test_session_load_skips_failed_calibration(tmp_path):
    """A calibration with passed=False must not be applied."""
    cal_unit, session, _ = _build_data_layout(tmp_path)

    # Write a result.json that explicitly failed.
    (cal_unit / "result.json").write_text(json.dumps({
        "unit": "UNITA",
        "source": "wheel",
        "passed": False,
        "gyro_bias_dps": [99.0, 99.0, 99.0],   # garbage that we don't want applied
    }))

    streams = extract_session(session)
    wheel = streams["wheel"]
    # Bias was NOT applied — the recorded gyro mean still reflects the injected bias.
    assert wheel.gyro_bias_rps is None
    assert np.linalg.norm(wheel.gyro.mean(axis=0)) > np.deg2rad(0.1)


def test_session_load_handles_no_calibration(tmp_path):
    """Missing calibration is non-fatal — load proceeds without bias."""
    data_root = tmp_path / "data"
    session = data_root / "sessions" / "20260101_practice"
    session.mkdir(parents=True)
    samples = make_static_trace(duration_s=2.0)
    write_synthetic_gyroflow(session / "wheel.gyroflow", samples, imu_orientation="XYZ")

    streams = extract_session(session)
    assert "wheel" in streams
    assert streams["wheel"].gyro_bias_rps is None
