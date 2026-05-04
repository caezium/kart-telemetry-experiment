"""End-to-end tests for the .gyroflow decoder + ImuStream loader.

The fixtures in conftest.py construct synthetic .gyroflow files via the same
encode pipeline (CBOR → zlib → base91) the modern Gyroflow writer uses, so
these tests exercise the real decode path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.extract_imu import (
    decode_gyroflow_file,
    load_gyroflow,
)
from tests.conftest import (
    make_static_trace,
    make_step_rotation_trace,
    write_synthetic_gyroflow,
)


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def test_decode_modern_compressed(synthetic_static_gyroflow):
    meta = decode_gyroflow_file(synthetic_static_gyroflow)
    assert "raw_imu" in meta
    assert meta["imu_orientation"] == "XYZ"
    assert len(meta["raw_imu"]) > 100


def test_decode_legacy_plain_array(synthetic_legacy_static_gyroflow):
    meta = decode_gyroflow_file(synthetic_legacy_static_gyroflow)
    assert "raw_imu" in meta
    assert isinstance(meta["raw_imu"], list)
    assert len(meta["raw_imu"]) > 10


def test_decode_raises_on_missing_data(tmp_path):
    path = tmp_path / "broken.gyroflow"
    path.write_text(json.dumps({"title": "no imu here", "version": 4}))
    with pytest.raises(ValueError, match="Could not locate"):
        decode_gyroflow_file(path)


# ---------------------------------------------------------------------------
# Load — units, orientation, downstream contract
# ---------------------------------------------------------------------------

def test_load_units_are_si(synthetic_static_gyroflow):
    """Gyro must be rad/s; static accel magnitude must be ~9.8 m/s²."""
    stream = load_gyroflow(synthetic_static_gyroflow, source="wheel", apply_lowpass=False)
    # Static recording with small bias in deg/s → small bias in rad/s after conversion.
    # |gyro| should be very small (just bias + noise, in rad/s).
    assert np.median(np.linalg.norm(stream.gyro, axis=1)) < 0.05  # < ~3 deg/s
    # Accel should sit near gravity.
    assert abs(np.median(np.linalg.norm(stream.accel, axis=1)) - 9.80665) < 0.05


def test_load_dataframe_columns_match_contract(synthetic_static_gyroflow):
    """Downstream parquet readers expect t/gx/gy/gz/ax/ay/az."""
    stream = load_gyroflow(synthetic_static_gyroflow, source="wheel")
    df = stream.to_dataframe()
    assert list(df.columns) == ["t", "gx", "gy", "gz", "ax", "ay", "az"]


def test_load_t_starts_at_zero_and_is_uniform(synthetic_static_gyroflow):
    stream = load_gyroflow(synthetic_static_gyroflow, source="wheel")
    assert stream.t[0] == 0.0
    dt = np.diff(stream.t)
    assert np.allclose(dt, dt[0], rtol=1e-9)
    assert 50 < stream.sample_rate_hz < 1000


def test_load_orientation_propagates(tmp_path):
    """The orientation string is recorded on the ImuStream for traceability."""
    samples = make_static_trace(duration_s=1.0)
    path = tmp_path / "wheel.gyroflow"
    write_synthetic_gyroflow(path, samples, imu_orientation="yXZ")
    stream = load_gyroflow(path, source="wheel")
    assert stream.orientation == "yXZ"


def test_load_applies_orientation_to_gyro(tmp_path):
    """A pure rotation about the camera's body Y axis, with orientation 'yXZ',
    must show up on the post-orientation X axis (and negated)."""
    # Inject 50 deg/s on body Y for 1s, embedded in a 2s recording.
    samples, _ = make_step_rotation_trace(
        rotations=[(0.5, 1.0, 50.0)],   # 50 deg/s for 1s = 50° total
        axis=1,                          # body Y
        rate_hz=200.0,
        noise_dps=0.0,
    )
    path = tmp_path / "wheel.gyroflow"
    write_synthetic_gyroflow(path, samples, imu_orientation="yXZ")
    stream = load_gyroflow(path, source="wheel", apply_lowpass=False)
    # During the pulse, post-orientation gx should be ~-50 deg/s (= -0.873 rad/s),
    # gy and gz should be ~0.
    pulse_mask = (stream.t > 0.6) & (stream.t < 1.4)
    gx_pulse = stream.gyro[pulse_mask, 0]
    gy_pulse = stream.gyro[pulse_mask, 1]
    gz_pulse = stream.gyro[pulse_mask, 2]
    assert np.median(gx_pulse) == pytest.approx(-np.deg2rad(50.0), abs=0.05)
    assert np.allclose(gy_pulse, 0.0, atol=0.05)
    assert np.allclose(gz_pulse, 0.0, atol=0.05)


def test_load_applies_gyro_bias(tmp_path):
    """If a gyro_bias_dps is provided at load time, it gets subtracted."""
    bias = (1.0, 2.0, 3.0)
    samples = make_static_trace(
        duration_s=2.0,
        gyro_bias_dps=bias,
        noise_dps=0.0,
    )
    path = tmp_path / "wheel.gyroflow"
    write_synthetic_gyroflow(path, samples, imu_orientation="XYZ")
    # Without bias correction: gyro mean reflects the bias (in rad/s).
    s_uncorr = load_gyroflow(path, source="wheel", apply_lowpass=False)
    np.testing.assert_allclose(s_uncorr.gyro.mean(axis=0), np.deg2rad(bias), atol=1e-3)
    # With bias correction: gyro mean is ~zero.
    s_corr = load_gyroflow(
        path, source="wheel",
        gyro_bias_dps=np.array(bias),
        apply_lowpass=False,
    )
    np.testing.assert_allclose(s_corr.gyro.mean(axis=0), np.zeros(3), atol=1e-3)


def test_load_handles_null_gyro_or_accl_samples(tmp_path):
    """raw_imu may have null gyro/accl at edges — those samples must be dropped."""
    samples = make_static_trace(duration_s=1.0)
    samples[0]["gyro"] = None
    samples[-1]["accl"] = None
    path = tmp_path / "wheel.gyroflow"
    write_synthetic_gyroflow(path, samples)
    stream = load_gyroflow(path, source="wheel")
    # Should still load successfully with most samples present.
    assert len(stream.t) > 100
