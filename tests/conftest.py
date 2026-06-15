"""Test fixtures: synthetic .gyroflow file builder + canned IMU traces.

The fixture builder mirrors the modern Gyroflow v4 writer's encode pipeline
(CBOR → zlib → base91), so the parser is exercised against the real wire
format rather than a simplified mock.
"""

from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

# Make `pipeline.*` and `analysis.*` importable when pytest is run from repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Encode side — mirrors gyroflow/src/core/util.rs:20-68
# ---------------------------------------------------------------------------

def encode_file_metadata(meta: dict) -> str:
    """CBOR → zlib → base91 — the exact pipeline a v4 writer uses."""
    import base91
    import cbor2
    blob = cbor2.dumps(meta)
    compressed = zlib.compress(blob, level=9)
    encoded = base91.encode(compressed)
    return encoded if isinstance(encoded, str) else encoded.decode("ascii")


def write_synthetic_gyroflow(
    path: Path,
    raw_imu: list[dict],
    imu_orientation: str = "XYZ",
    legacy_plain_array: bool = False,
) -> None:
    """Write a .gyroflow file containing the given raw_imu samples.

    By default uses the modern compressed `gyro_source.file_metadata` path.
    Set `legacy_plain_array=True` to use the legacy JSON-array fallback so
    the parser's legacy code path is exercised too.
    """
    if legacy_plain_array:
        proj = {
            "title": "synthetic test file",
            "version": 4,
            "gyro_source": {
                "raw_imu": raw_imu,
                "imu_orientation": imu_orientation,
            },
        }
    else:
        meta = {"raw_imu": raw_imu, "imu_orientation": imu_orientation}
        proj = {
            "title": "synthetic test file",
            "version": 4,
            "gyro_source": {"file_metadata": encode_file_metadata(meta)},
        }
    with path.open("w") as f:
        json.dump(proj, f)


# ---------------------------------------------------------------------------
# Synthetic IMU traces
# ---------------------------------------------------------------------------

def make_static_trace(
    duration_s: float = 1.0,
    rate_hz: float = 200.0,
    gyro_bias_dps: tuple[float, float, float] = (0.0, 0.0, 0.0),
    accel_at_rest_mps2: tuple[float, float, float] = (0.0, 0.0, 9.80665),
    rng_seed: int = 0,
    noise_dps: float = 0.05,
) -> list[dict]:
    """A stationary trace with optional bias + Gaussian noise on gyro."""
    rng = np.random.default_rng(rng_seed)
    n = int(round(duration_s * rate_hz)) + 1
    t_ms = np.linspace(0.0, duration_s * 1000.0, n)
    samples = []
    for i, ts in enumerate(t_ms):
        gyro = [
            float(gyro_bias_dps[k] + rng.normal(0.0, noise_dps)) for k in range(3)
        ]
        samples.append({
            "timestamp_ms": float(ts),
            "gyro": gyro,
            "accl": list(accel_at_rest_mps2),
            "magn": None,
        })
    return samples


def make_step_rotation_trace(
    rotations: list[tuple[float, float, float]],
    rate_hz: float = 200.0,
    axis: int = 2,
    gyro_bias_dps: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rng_seed: int = 1,
    noise_dps: float = 0.05,
) -> tuple[list[dict], list[float]]:
    """Generate a trace consisting of step rotations.

    Each rotation is (t_start_s, duration_s, total_angle_deg). Between
    rotations the camera is static at the latest accumulated angle. The
    rotation is a triangular pulse around the named gyro axis (so the
    integrated angle over the pulse equals total_angle_deg).

    Returns (samples, plateau_times) — plateau_times are the centers of
    each static segment after a rotation (for ground-truth checks).
    """
    rng = np.random.default_rng(rng_seed)
    if not rotations:
        raise ValueError("rotations must be non-empty")

    end_t = max(t + d for t, d, _ in rotations) + 2.0
    n = int(round(end_t * rate_hz)) + 1
    t_s = np.linspace(0.0, end_t, n)

    rate = np.zeros((n, 3), dtype=np.float64)
    plateau_times: list[float] = []
    for t_start, dur, angle_deg in rotations:
        # Constant rate over the duration that integrates to angle_deg.
        in_pulse = (t_s >= t_start) & (t_s < t_start + dur)
        rate[in_pulse, axis] = angle_deg / dur
        plateau_times.append(float(t_start + dur + 0.5))  # 0.5s after each pulse ends

    rate += rng.normal(0.0, noise_dps, size=rate.shape)
    rate += np.asarray(gyro_bias_dps).reshape(1, 3)

    samples = []
    for i, ts in enumerate(t_s):
        samples.append({
            "timestamp_ms": float(ts * 1000.0),
            "gyro": [float(rate[i, 0]), float(rate[i, 1]), float(rate[i, 2])],
            "accl": [0.0, 0.0, 9.80665],
            "magn": None,
        })
    return samples, plateau_times


# ---------------------------------------------------------------------------
# Fake XRK log — mimics the libxrk.LogFile surface the code actually uses
# ---------------------------------------------------------------------------

class FakeChannelTable:
    """Mimics a pyarrow table: indexable by column name, has num_rows and
    column_names. Enough for np.asarray(tbl["col"]) and the slicing code."""

    def __init__(self, columns: dict):
        self._cols = {k: np.asarray(v) for k, v in columns.items()}

    @property
    def num_rows(self) -> int:
        return len(next(iter(self._cols.values()))) if self._cols else 0

    @property
    def column_names(self) -> list:
        return list(self._cols.keys())

    def __getitem__(self, key):
        return self._cols[key]


class FakeLapsTable:
    """Mimics the pyarrow laps table: columns num/start_time/end_time, each
    indexable to an object with .as_py()."""

    class _Cell:
        def __init__(self, v):
            self._v = v

        def as_py(self):
            return self._v

    class _Col:
        def __init__(self, vals):
            self._vals = list(vals)

        def __getitem__(self, i):
            return FakeLapsTable._Cell(self._vals[i])

        def to_pylist(self):
            return list(self._vals)

    def __init__(self, nums, starts_ms, ends_ms):
        self._cols = {"num": list(nums), "start_time": list(starts_ms),
                      "end_time": list(ends_ms)}
        self.num_rows = len(nums)

    def __getitem__(self, key):
        return FakeLapsTable._Col(self._cols[key])


class FakeXrkLog:
    """Mimics libxrk.LogFile: .channels dict + .laps table + .metadata."""

    def __init__(self, channels: dict, laps: FakeLapsTable | None = None,
                 metadata: dict | None = None):
        self.channels = channels
        self.laps = laps or FakeLapsTable([], [], [])
        self.metadata = metadata or {}


def make_gps_yaw_channel(t_s: np.ndarray, yaw_dps: np.ndarray) -> FakeChannelTable:
    return FakeChannelTable({
        "timecodes": np.asarray(t_s) * 1000.0,
        "GPS_Yaw_Rate": np.asarray(yaw_dps),
    })


def make_gps_track_log(
    t_s: np.ndarray,
    dist_m: np.ndarray,
    speed_mps: float | np.ndarray = 20.0,
    yaw_dps: np.ndarray | None = None,
    lon_length_delta: int = 0,
) -> FakeXrkLog:
    """Build a FakeXrkLog with GPS Latitude/Longitude/Speed (+optional yaw).

    `dist_m` is the desired distance-from-first-sample profile; it's encoded
    as a pure-latitude offset (lon held constant) so the module's
    equirectangular distance recovers it. `lon_length_delta` lets a test
    make the Longitude channel a different length than Latitude (to exercise
    the misalignment-tolerant interp path).
    """
    t_s = np.asarray(t_s, dtype=float)
    dist_m = np.asarray(dist_m, dtype=float)
    deg_per_m = 1.0 / (6_371_000.0 * np.pi / 180.0)
    lat = dist_m * deg_per_m
    lon = np.zeros_like(lat)
    speed = np.full_like(t_s, float(speed_mps)) if np.isscalar(speed_mps) else np.asarray(speed_mps, float)

    chans = {
        "GPS Latitude": FakeChannelTable({"timecodes": t_s * 1000.0, "GPS Latitude": lat}),
        "GPS Speed": FakeChannelTable({"timecodes": t_s * 1000.0, "GPS Speed": speed}),
    }
    if lon_length_delta:
        n = len(t_s) + lon_length_delta
        t_lon = np.linspace(t_s[0], t_s[-1], n)
        lon_resampled = np.interp(t_lon, t_s, lon)
        chans["GPS Longitude"] = FakeChannelTable(
            {"timecodes": t_lon * 1000.0, "GPS Longitude": lon_resampled})
    else:
        chans["GPS Longitude"] = FakeChannelTable(
            {"timecodes": t_s * 1000.0, "GPS Longitude": lon})
    if yaw_dps is not None:
        chans["GPS_Yaw_Rate"] = make_gps_yaw_channel(t_s, yaw_dps)
    return FakeXrkLog(chans)


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_static_gyroflow(tmp_path) -> Path:
    """A ~30-second static recording at 200 Hz with small bias and noise."""
    samples = make_static_trace(
        duration_s=30.0,
        gyro_bias_dps=(0.05, -0.10, 0.02),
    )
    path = tmp_path / "wheel.gyroflow"
    write_synthetic_gyroflow(path, samples, imu_orientation="XYZ")
    return path


@pytest.fixture
def synthetic_legacy_static_gyroflow(tmp_path) -> Path:
    samples = make_static_trace(duration_s=2.0)
    path = tmp_path / "wheel.gyroflow"
    write_synthetic_gyroflow(path, samples, legacy_plain_array=True)
    return path
