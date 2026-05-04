"""Verifies apply_orientation against Gyroflow's `orient()` source convention.

Reference: gyroflow/src/core/gyro_source/imu_transforms.rs:73-83.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.extract_imu import apply_orientation


def _v(x, y, z):
    return np.array([[x, y, z]], dtype=np.float64)


def test_identity_xyz_is_noop():
    g = _v(1, 2, 3)
    a = _v(0, 0, 9.8)
    g2, a2 = apply_orientation(g, a, "XYZ")
    np.testing.assert_array_equal(g, g2)
    np.testing.assert_array_equal(a, a2)


def test_yxz_insta360_go3s_convention():
    """For orientation 'yXZ':  new_x = -old_y,  new_y = +old_x,  new_z = +old_z."""
    g = _v(1.0, 2.0, 3.0)
    a = _v(0.0, 9.8, 0.0)
    g2, a2 = apply_orientation(g, a, "yXZ")
    np.testing.assert_allclose(g2, _v(-2.0, 1.0, 3.0))
    np.testing.assert_allclose(a2, _v(-9.8, 0.0, 0.0))


def test_full_negation():
    """'xyz' negates every axis."""
    g = _v(1.0, 2.0, 3.0)
    a = _v(4.0, 5.0, 6.0)
    g2, a2 = apply_orientation(g, a, "xyz")
    np.testing.assert_allclose(g2, _v(-1.0, -2.0, -3.0))
    np.testing.assert_allclose(a2, _v(-4.0, -5.0, -6.0))


@pytest.mark.parametrize("orientation", ["YZX", "ZXY", "yzx", "ZyX"])
def test_permutation_preserves_norm(orientation):
    """Any permutation (with sign flips) preserves vector magnitude."""
    rng = np.random.default_rng(0)
    g = rng.normal(size=(50, 3))
    a = rng.normal(size=(50, 3))
    g2, a2 = apply_orientation(g, a, orientation)
    np.testing.assert_allclose(np.linalg.norm(g2, axis=1), np.linalg.norm(g, axis=1))
    np.testing.assert_allclose(np.linalg.norm(a2, axis=1), np.linalg.norm(a, axis=1))


def test_rejects_bad_length():
    g = _v(1, 2, 3)
    a = _v(0, 0, 9.8)
    with pytest.raises(ValueError, match="3 chars"):
        apply_orientation(g, a, "XY")


def test_rejects_bad_character():
    g = _v(1, 2, 3)
    a = _v(0, 0, 9.8)
    with pytest.raises(ValueError, match="Invalid orientation"):
        apply_orientation(g, a, "XQZ")


def test_round_trip_via_inverse():
    """For pure permutations (no sign flips), applying twice in the right
    order returns to identity. Using 'YZX' which is the inverse of 'ZXY'."""
    rng = np.random.default_rng(1)
    g = rng.normal(size=(10, 3))
    a = rng.normal(size=(10, 3))
    g_once, a_once = apply_orientation(g, a, "YZX")
    g_twice, a_twice = apply_orientation(g_once, a_once, "ZXY")
    np.testing.assert_allclose(g_twice, g)
    np.testing.assert_allclose(a_twice, a)
