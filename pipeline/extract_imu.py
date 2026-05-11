"""
Extract clean IMU streams from Gyroflow project files (.gyroflow JSON).

Why .gyroflow JSON and not the CSV export:
    Gyroflow's CSV export emits one sample per video frame (~30 Hz) for raw
    gyro/accl. The per-IMU-sample data (1 kHz on a Go 3S, empirically) is
    only available inside the .gyroflow project file, in the
    `gyro_source.file_metadata` blob. Steering jerk (2nd derivative of
    angle) is the metric this project cares about most — useless at 30 Hz,
    excellent at 1 kHz.

Decode pipeline (modern v4 writer):
    proj["gyro_source"]["file_metadata"]   # base91 string
        -> base91 decode
        -> zlib decompress
        -> CBOR loads
        -> dict { "raw_imu": [...], "imu_orientation": "yXZ", ... }

Each `raw_imu` entry: {timestamp_ms, gyro:[x,y,z]|None, accl:[x,y,z]|None,
magn:[x,y,z]|None}. Values are pre-orientation (camera body frame).
Gyroflow forces "XYZ" identity at parse time and stores the camera's real
orientation hint separately. We apply that transform here so downstream
analysis sees post-orientation, body-aligned data — matching what
Gyroflow's runtime sees.

Source convention (what downstream analysis modules assume):
    - gyro in rad/s (we convert from Gyroflow's deg/s)
    - accel in m/s²
    - axes after orientation: x/y/z in the camera's "natural" body frame
    - parquet columns: t, gx, gy, gz, ax, ay, az
"""

from __future__ import annotations

import argparse
import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


GYRO_LOWPASS_HZ = 30.0   # mechanical steering inputs do not exceed ~10-15 Hz
ACCEL_LOWPASS_HZ = 20.0
DEFAULT_TARGET_RATE_HZ = 1000.0   # Insta360 Go 3S; only used as a fallback
                                  # if a recording has no inferable rate.
RESAMPLE_TOLERANCE = 0.05   # if non-uniformity exceeds 5% of median dt, resample


@dataclass
class ImuStream:
    t: np.ndarray         # seconds, starts at 0
    gyro: np.ndarray      # (N, 3) rad/s, axes (x, y, z) post-orientation
    accel: np.ndarray     # (N, 3) m/s²
    sample_rate_hz: float
    source: str           # "wheel" | "helmet" | other tag
    orientation: str      # the imu_orientation string applied (e.g. "yXZ")
    video_path: str | None = None
    gyro_bias_rps: np.ndarray | None = None   # if calibration was applied at load time

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "t": self.t,
            "gx": self.gyro[:, 0], "gy": self.gyro[:, 1], "gz": self.gyro[:, 2],
            "ax": self.accel[:, 0], "ay": self.accel[:, 1], "az": self.accel[:, 2],
        })


# ---------------------------------------------------------------------------
# .gyroflow file decode
# ---------------------------------------------------------------------------

def decode_gyroflow_file(path: Path) -> dict[str, Any]:
    """Load a .gyroflow project file and return the FileMetadata dict.

    Handles both the modern compressed path (base91 → zlib → CBOR) and the
    legacy plain-array path where `raw_imu` is a top-level JSON array.
    """
    try:
        import base91  # type: ignore
        import cbor2   # type: ignore
    except ImportError as e:
        raise ImportError(
            "Decoding .gyroflow files requires `base91` and `cbor2`. "
            "Install with: pip install -r pipeline/requirements.txt"
        ) from e

    with path.open() as f:
        proj = json.load(f)

    gyro_source = proj.get("gyro_source", {})
    file_meta = gyro_source.get("file_metadata")

    if isinstance(file_meta, str):
        # Modern path: base91 string -> zlib -> CBOR
        decoded = base91.decode(file_meta)
        if isinstance(decoded, list):
            decoded = bytes(decoded)
        decompressed = zlib.decompress(decoded)
        meta = cbor2.loads(decompressed)
        if not isinstance(meta, dict):
            raise ValueError(f"CBOR payload in {path} is not a dict")
        return meta

    # Legacy fallback: raw_imu as a plain JSON array, possibly inside gyro_source
    raw = gyro_source.get("raw_imu", proj.get("raw_imu"))
    if isinstance(raw, list):
        return {
            "raw_imu": raw,
            "imu_orientation": (
                gyro_source.get("imu_orientation") or proj.get("imu_orientation")
            ),
        }

    raise ValueError(
        f"Could not locate IMU data in {path}. Expected either "
        f"gyro_source.file_metadata (compressed) or a raw_imu array."
    )


# ---------------------------------------------------------------------------
# Orientation transform
# ---------------------------------------------------------------------------

# Mirrors gyroflow/src/core/gyro_source/imu_transforms.rs:73-83
# and telemetry-parser/src/tags_impl.rs:253-263.
_ORIENT_AXIS = {
    "X": (0, 1.0), "x": (0, -1.0),
    "Y": (1, 1.0), "y": (1, -1.0),
    "Z": (2, 1.0), "z": (2, -1.0),
}


def apply_orientation(
    gyro: np.ndarray,
    accel: np.ndarray,
    orientation: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Gyroflow's `imu_orientation` convention.

    Each of the three characters selects a source axis for the new x/y/z
    output, with lowercase indicating that the source axis is negated.

    Example: "yXZ" produces:
        new_x = -old_y
        new_y =  old_x
        new_z =  old_z

    "XYZ" is identity (no-op).
    """
    if not orientation:
        return gyro, accel
    if len(orientation) != 3:
        raise ValueError(f"Invalid imu_orientation: {orientation!r} (expected 3 chars)")

    cols = []
    for ch in orientation:
        if ch not in _ORIENT_AXIS:
            raise ValueError(f"Invalid orientation character {ch!r} in {orientation!r}")
        cols.append(_ORIENT_AXIS[ch])

    out_gyro = np.empty_like(gyro)
    out_accel = np.empty_like(accel)
    for new_idx, (src_idx, sign) in enumerate(cols):
        out_gyro[:, new_idx] = sign * gyro[:, src_idx]
        out_accel[:, new_idx] = sign * accel[:, src_idx]
    return out_gyro, out_accel


# ---------------------------------------------------------------------------
# Sample-array assembly + resampling
# ---------------------------------------------------------------------------

def _samples_to_arrays(
    samples: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert raw_imu list to (t_seconds, gyro_dps, accel_mps2) arrays.

    Drops samples where either gyro or accl is null/missing — interpolation
    upstream may have left gaps at edges.
    """
    if not samples:
        raise ValueError("raw_imu is empty")

    ts: list[float] = []
    gs: list[list[float]] = []
    as_: list[list[float]] = []
    for s in samples:
        g = s.get("gyro")
        a = s.get("accl")
        if g is None or a is None:
            continue
        ts.append(float(s["timestamp_ms"]))
        gs.append([float(g[0]), float(g[1]), float(g[2])])
        as_.append([float(a[0]), float(a[1]), float(a[2])])

    if not ts:
        raise ValueError("No samples with both gyro and accl present")

    t_ms = np.asarray(ts, dtype=np.float64)
    gyro = np.asarray(gs, dtype=np.float64)
    accel = np.asarray(as_, dtype=np.float64)

    order = np.argsort(t_ms)
    return t_ms[order] / 1000.0, gyro[order], accel[order]


def _is_uniform(t: np.ndarray) -> bool:
    if len(t) < 3:
        return True
    dt = np.diff(t)
    median_dt = np.median(dt)
    if median_dt <= 0:
        return False
    return float(np.max(np.abs(dt - median_dt)) / median_dt) <= RESAMPLE_TOLERANCE


def _resample_uniform(
    t: np.ndarray,
    gyro: np.ndarray,
    accel: np.ndarray,
    target_fs: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Resample to a uniform grid via linear interpolation."""
    if target_fs is None:
        target_fs = float(1.0 / np.median(np.diff(t)))

    duration = float(t[-1] - t[0])
    n = max(2, int(round(duration * target_fs)) + 1)
    t_new = np.linspace(0.0, duration, n)

    g_new = np.column_stack([np.interp(t_new, t - t[0], gyro[:, i]) for i in range(3)])
    a_new = np.column_stack([np.interp(t_new, t - t[0], accel[:, i]) for i in range(3)])
    return t_new, g_new, a_new, target_fs


def _butter_lowpass(cutoff_hz: float, fs: float, order: int = 4):
    return butter(order, cutoff_hz / (fs / 2.0), btype="low", output="sos")


def _filter(stream: np.ndarray, cutoff_hz: float, fs: float) -> np.ndarray:
    if cutoff_hz >= fs / 2:
        return stream
    sos = _butter_lowpass(cutoff_hz, fs)
    return sosfiltfilt(sos, stream, axis=0)


# ---------------------------------------------------------------------------
# Public load function
# ---------------------------------------------------------------------------

def load_gyroflow(
    path: Path,
    source: str,
    video_path: Path | None = None,
    gyro_bias_dps: np.ndarray | None = None,
    target_rate_hz: float | None = None,
    apply_lowpass: bool = True,
) -> ImuStream:
    """Load a .gyroflow file as a ready-to-use ImuStream.

    Steps:
        1. Decode the compressed FileMetadata blob.
        2. Build numpy arrays from raw_imu (deg/s, m/s², pre-orientation).
        3. Apply imu_orientation.
        4. Resample to a uniform grid if non-uniform.
        5. Convert gyro deg/s -> rad/s.
        6. Subtract per-axis gyro bias if provided.
        7. Optional low-pass to remove sensor noise above mechanical bandwidth.

    Args:
        path: .gyroflow file
        source: tag like "wheel" or "helmet" — propagated for downstream consumers
        video_path: optional pointer to the source mp4/insv
        gyro_bias_dps: per-axis bias in deg/s **post-orientation**; subtract from raw
        target_rate_hz: resample target; defaults to inferred sample rate
        apply_lowpass: if False, skip the Butterworth filter (useful for tests)
    """
    meta = decode_gyroflow_file(path)
    samples = meta.get("raw_imu") or []
    orientation = meta.get("imu_orientation") or "XYZ"

    t_s, gyro_dps, accel = _samples_to_arrays(samples)
    gyro_dps, accel = apply_orientation(gyro_dps, accel, orientation)

    # gyro_bias_dps is provided in post-orientation axes (matches what calibration
    # measures and what Gyroflow stores in IMUTransforms.gyro_bias).
    bias_rps = None
    if gyro_bias_dps is not None:
        bias_rps = np.deg2rad(np.asarray(gyro_bias_dps, dtype=np.float64))

    if not _is_uniform(t_s):
        t_s, gyro_dps, accel, fs = _resample_uniform(t_s, gyro_dps, accel, target_rate_hz)
    else:
        t_s = t_s - t_s[0]
        fs = float(1.0 / np.median(np.diff(t_s))) if len(t_s) > 1 else DEFAULT_TARGET_RATE_HZ

    if not (10.0 < fs < 5000.0):
        raise ValueError(f"Implausible sample rate {fs:.1f} Hz from {path}")

    gyro = np.deg2rad(gyro_dps)
    if bias_rps is not None:
        gyro = gyro - bias_rps

    if apply_lowpass:
        gyro = _filter(gyro, GYRO_LOWPASS_HZ, fs)
        accel = _filter(accel, ACCEL_LOWPASS_HZ, fs)

    return ImuStream(
        t=t_s, gyro=gyro, accel=accel,
        sample_rate_hz=fs, source=source,
        orientation=orientation,
        video_path=str(video_path) if video_path else None,
        gyro_bias_rps=bias_rps,
    )


# ---------------------------------------------------------------------------
# Session-level glue
# ---------------------------------------------------------------------------

def detect_source_from_filename(name: str) -> str:
    n = name.lower()
    if "wheel" in n or "steering" in n:
        return "wheel"
    if "helmet" in n or "head" in n:
        return "helmet"
    return "unknown"


def _load_calibration_bias(session_dir: Path, source: str) -> np.ndarray | None:
    """Look up gyro_bias_dps from a calibration result.json adjacent to the session.

    Convention: data/calibration/<unit>/result.json. The session metadata may
    name the unit; fall back to scanning all calibrations and matching by
    `source` (wheel/helmet).
    """
    cal_root = session_dir.parent.parent / "calibration"
    if not cal_root.is_dir():
        return None

    candidates = sorted(cal_root.glob("*/result.json"))
    for cand in candidates:
        try:
            with cand.open() as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("source") == source and data.get("passed"):
            bias = data.get("gyro_bias_dps")
            if bias is not None:
                return np.asarray(bias, dtype=np.float64)
    return None


def extract_session(session_dir: Path) -> dict[str, ImuStream]:
    streams: dict[str, ImuStream] = {}
    candidates = sorted(session_dir.glob("*.gyroflow"))
    if not candidates:
        # Helpful diagnostic for the most common mistake.
        if list(session_dir.glob("*.csv")):
            print("  hint: this pipeline reads .gyroflow project files, not Gyroflow CSV exports.")
            print("        In Gyroflow: open the video, then File → Export project file.")
    for path in candidates:
        source = detect_source_from_filename(path.name)
        if source == "unknown":
            print(f"  skipping {path.name} (rename to include 'wheel' or 'helmet')")
            continue
        if source in streams:
            raise ValueError(f"Duplicate {source} stream: {path.name}")
        bias = _load_calibration_bias(session_dir, source)
        streams[source] = load_gyroflow(path, source=source, gyro_bias_dps=bias)
        s = streams[source]
        bias_note = " (bias-corrected)" if bias is not None else ""
        print(f"  loaded {source}: {len(s.t)} samples @ {s.sample_rate_hz:.1f} Hz, "
              f"orientation={s.orientation}{bias_note}")
    return streams


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session_dir", type=Path)
    args = ap.parse_args()

    if not args.session_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.session_dir}")

    print(f"Extracting IMU streams from {args.session_dir}")
    streams = extract_session(args.session_dir)
    if not streams:
        raise SystemExit(
            "No streams found. Place .gyroflow project files in the session directory, "
            "with filenames containing 'wheel' or 'helmet'."
        )

    out_dir = args.session_dir / "extracted"
    out_dir.mkdir(exist_ok=True)
    for source, stream in streams.items():
        out_path = out_dir / f"{source}_imu.parquet"
        stream.to_dataframe().to_parquet(out_path)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
