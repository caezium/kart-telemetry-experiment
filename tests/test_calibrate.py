"""End-to-end calibration test using a synthetic recording that mimics the
bench protocol described in data/calibration/README.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.calibrate import (
    calibrate_unit,
    estimate_bias,
    find_active_window,
    find_static_windows,
    identify_steering_axis,
)
from tests.conftest import make_step_rotation_trace, write_synthetic_gyroflow


def _bench_recording(tmp_path: Path, bias_dps: tuple[float, float, float] = (0.05, -0.02, 0.10)) -> Path:
    """Build a synthetic recording that exercises the full calibration script.

    Layout (matches data/calibration/README.md, shrunk for test speed):
      0–10 s static  (bias estimation window)
      10–11 s rotate to +90°
      11–12 s hold
      12–13 s rotate back to 0°
      13–14 s hold
      14–15 s rotate to -90°
      15–16 s hold
      16–17 s rotate back to 0°
      17–37 s static (drift window)

    Steering axis is body Z. Orientation 'XYZ' (no transform) so post-orientation
    Z is also Z.
    """
    rotations = [
        (10.0, 1.0,  90.0),
        (12.0, 1.0, -90.0),
        (14.0, 1.0, -90.0),
        (16.0, 1.0,  90.0),
    ]
    samples, _ = make_step_rotation_trace(
        rotations=rotations,
        axis=2,
        rate_hz=200.0,
        gyro_bias_dps=bias_dps,
        noise_dps=0.05,
    )
    # Extend with 20 s of static at the end for drift window.
    last_t = samples[-1]["timestamp_ms"] / 1000.0
    extra_n = int(20.0 * 200)
    extra_t = np.linspace(last_t + 1 / 200.0, last_t + 20.0, extra_n)
    rng = np.random.default_rng(7)
    for ts in extra_t:
        samples.append({
            "timestamp_ms": float(ts * 1000.0),
            "gyro": [
                float(bias_dps[0] + rng.normal(0.0, 0.05)),
                float(bias_dps[1] + rng.normal(0.0, 0.05)),
                float(bias_dps[2] + rng.normal(0.0, 0.05)),
            ],
            "accl": [0.0, 0.0, 9.80665],
            "magn": None,
        })

    unit_dir = tmp_path / "TESTUNIT"
    unit_dir.mkdir()
    write_synthetic_gyroflow(unit_dir / "original.gyroflow", samples, imu_orientation="XYZ")

    # Ground truth: angle observed at the center of each plateau.
    pd.DataFrame({
        "t_seconds": [11.5, 13.5, 15.5, 17.5],
        "angle_deg": [90.0,  0.0, -90.0,  0.0],
        "note": ["after_+90", "back_0", "after_-90", "back_0"],
    }).to_csv(unit_dir / "ground_truth.csv", index=False)

    return unit_dir


# ---------------------------------------------------------------------------
# Unit tests for the helper functions
# ---------------------------------------------------------------------------

def test_find_static_windows_picks_long_quiet_periods():
    t = np.linspace(0, 10, 1001)
    gyro = np.zeros((len(t), 3))
    # noise everywhere, but with a 3s "loud" patch in the middle
    rng = np.random.default_rng(0)
    gyro += rng.normal(0.0, 0.1, size=gyro.shape)
    gyro[400:700, 2] += 30.0  # active patch
    windows = find_static_windows(t, gyro, threshold_dps=1.0, min_duration_s=2.0)
    assert len(windows) >= 2
    # First static run is roughly 0..4s and second is roughly 7..10s.
    durations = [t[b - 1] - t[a] for a, b in windows]
    assert all(d >= 2.0 for d in durations)


def test_estimate_bias_recovers_injected_bias():
    rng = np.random.default_rng(0)
    n = 4000
    t = np.linspace(0, 20.0, n)
    bias = np.array([0.10, -0.20, 0.05])
    gyro = bias + rng.normal(0.0, 0.05, size=(n, 3))
    accel = np.tile([0.0, 0.0, 9.80665], (n, 1))
    windows = find_static_windows(t, gyro)
    g_bias, a_bias_residual, _ = estimate_bias(t, gyro, accel, windows)
    np.testing.assert_allclose(g_bias, bias, atol=0.01)
    # gravity removed from Z
    np.testing.assert_allclose(a_bias_residual, [0.0, 0.0, 0.0], atol=0.05)


def test_identify_steering_axis_picks_dominant_axis():
    rng = np.random.default_rng(0)
    n = 1000
    gyro = rng.normal(0.0, 0.1, size=(n, 3))
    # Add high-variance content on Z (a sinusoid that varies between -30 and +30)
    # so Z has ~ (30^2)/2 = 450 variance and the others have ~0.01 variance.
    gyro[:, 2] += 30.0 * np.sin(np.linspace(0, 4 * np.pi, n))
    bias = np.zeros(3)
    axis, sep_db = identify_steering_axis(gyro, bias, (0, n))
    assert axis == 2
    assert sep_db > 30


# ---------------------------------------------------------------------------
# Full calibration end-to-end
# ---------------------------------------------------------------------------

def test_calibrate_unit_passes_on_clean_synthetic_recording(tmp_path):
    unit_dir = _bench_recording(tmp_path, bias_dps=(0.05, -0.02, 0.10))
    result = calibrate_unit(unit_dir)

    assert result.passed, f"calibration failed: {result.failure_reasons}"
    assert result.axis_used_for_steering == "gz"
    assert result.axis_separation_db >= 10.0
    # Bias recovered within 0.05 deg/s
    np.testing.assert_allclose(result.gyro_bias_dps, [0.05, -0.02, 0.10], atol=0.05)
    # All step plateaus within 2°
    assert max(s.abs_error_deg for s in result.step_input_validation) < 2.0
    # Drift window of ~20s should yield small drift after bias subtraction.
    assert result.drift_60s_deg is not None
    assert result.drift_60s_deg < 5.0


def test_calibrate_unit_writes_result_json(tmp_path, monkeypatch):
    unit_dir = _bench_recording(tmp_path)
    result = calibrate_unit(unit_dir)
    # The CLI does the write; here we exercise the same shape directly.
    from pipeline.calibrate import _result_to_jsonable
    out_path = unit_dir / "result.json"
    out_path.write_text(json.dumps(_result_to_jsonable(result), indent=2))

    loaded = json.loads(out_path.read_text())
    assert loaded["passed"]
    assert loaded["axis_used_for_steering"] == "gz"
    assert isinstance(loaded["gyro_bias_dps"], list)
    assert len(loaded["gyro_bias_dps"]) == 3


def test_calibrate_unit_fails_loudly_on_huge_step_error(tmp_path):
    """Inject a deliberate scaling bug: rotations are 1.5x what ground truth
    claims. Calibration should flag the step error."""
    rotations = [
        (10.0, 1.0,  135.0),  # ground truth says 90 → off by 45°
        (12.0, 1.0, -135.0),
    ]
    samples, _ = make_step_rotation_trace(
        rotations=rotations, axis=2, rate_hz=200.0, noise_dps=0.05,
    )
    last_t = samples[-1]["timestamp_ms"] / 1000.0
    extra_n = int(15.0 * 200)
    rng = np.random.default_rng(0)
    for ts in np.linspace(last_t + 1 / 200.0, last_t + 15.0, extra_n):
        samples.append({
            "timestamp_ms": float(ts * 1000.0),
            "gyro": [float(rng.normal(0.0, 0.05)) for _ in range(3)],
            "accl": [0.0, 0.0, 9.80665],
            "magn": None,
        })
    unit_dir = tmp_path / "BADUNIT"
    unit_dir.mkdir()
    write_synthetic_gyroflow(unit_dir / "original.gyroflow", samples, imu_orientation="XYZ")
    pd.DataFrame({
        "t_seconds": [11.5, 13.5],
        "angle_deg": [90.0,  0.0],
    }).to_csv(unit_dir / "ground_truth.csv", index=False)

    result = calibrate_unit(unit_dir)
    assert not result.passed
    assert any("step input error" in r for r in result.failure_reasons)
