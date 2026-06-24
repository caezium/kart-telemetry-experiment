"""Tests for the direct Insta360 .mp4 → IMU reader (gyro2bb path).

Hermetic: no real video or gyro2bb binary needed. We synthesise the exact CSV
gyro2bb emits and monkeypatch the subprocess call, so the parsing, the
zero-sample FlowState guard, the non-finite filtering, and the dispatch logic
are all exercised without external dependencies.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from pipeline import extract_imu as E


# gyro2bb CSV preamble (abbreviated) + data header. Mirrors the real layout:
# metadata key/value rows, then the column header, then numeric rows.
_PREAMBLE = (
    '"Product","Blackbox flight data recorder by Nicholas Sherlock"\n'
    '"camera_type","Insta360 GO 3S"\n'
    '"gyro_cfg_info",{"acc_range":32,"gyro_range":2000}\n'
    '"loopIteration","time","gyroADC[0]","gyroADC[1]","gyroADC[2]",'
    '"accSmooth[0]","accSmooth[1]","accSmooth[2]"\n'
)


def _make_csv(n: int = 2000, fs: float = 1000.0, *, bad_row: bool = False,
              t0_us: float = -1234.0) -> str:
    """Build a synthetic gyro2bb CSV with n samples at fs Hz."""
    dt_us = 1e6 / fs
    lines = [_PREAMBLE.rstrip("\n")]
    for i in range(n):
        t = t0_us + i * dt_us
        gx, gy, gz = 5.0 * np.sin(i / 50), 1.0, -0.5
        ax, ay, az = -6600.0, -730.0, -18700.0  # ~1g in raw counts on one axis
        lines.append(f"{i},{t},{gx},{gy},{gz},{ax},{ay},{az}")
    if bad_row:
        # a malformed/ragged row that genfromtxt turns into NaNs
        lines.append("999999,,,,,,,")
    return "\n".join(lines) + "\n"


def _write_sidecar(video: Path, content: str) -> Path:
    csv = E._gyro2bb_csv_path(video)
    csv.write_text(content)
    return csv


# ---------------------------------------------------------------------------
# _parse_gyro2bb_csv
# ---------------------------------------------------------------------------

def test_parse_basic(tmp_path):
    csv = tmp_path / "clip.mp4.csv"
    csv.write_text(_make_csv(n=2000, fs=1000.0))
    t, gyro_dps, accel = E._parse_gyro2bb_csv(csv)
    assert len(t) == 2000
    assert gyro_dps.shape == (2000, 3)
    assert accel.shape == (2000, 3)
    # t returned in seconds, monotonic, ~1 ms spacing
    assert np.all(np.diff(t) > 0)
    assert np.isclose(np.median(np.diff(t)), 1e-3, rtol=1e-3)
    assert np.isfinite(gyro_dps).all() and np.isfinite(accel).all()


def test_parse_drops_nonfinite_rows(tmp_path):
    csv = tmp_path / "clip.mp4.csv"
    csv.write_text(_make_csv(n=500, bad_row=True))
    t, gyro_dps, accel = E._parse_gyro2bb_csv(csv)
    assert len(t) == 500  # the ragged NaN row is dropped, not kept
    assert np.isfinite(gyro_dps).all()


def test_parse_zero_samples_raises_flowstate_hint(tmp_path):
    """A clip recorded in plain Video mode yields header-only output."""
    csv = tmp_path / "clip.mp4.csv"
    csv.write_text(_PREAMBLE)  # header, no data rows
    with pytest.raises(ValueError, match="(?i)pro video|flowstate|0 imu samples"):
        E._parse_gyro2bb_csv(csv)


def test_parse_no_header_raises(tmp_path):
    csv = tmp_path / "clip.mp4.csv"
    csv.write_text('"camera_type","Insta360 GO 3S"\n')  # no loopIteration header
    with pytest.raises(ValueError, match="(?i)no data header"):
        E._parse_gyro2bb_csv(csv)


# ---------------------------------------------------------------------------
# load_insta360_mp4 (subprocess + finalize), monkeypatched
# ---------------------------------------------------------------------------

def _patch_gyro2bb(monkeypatch, csv_content: str):
    monkeypatch.setattr(E, "_find_gyro2bb", lambda explicit=None: "fake-gyro2bb")

    def fake_run(cmd, *a, **kw):
        video = Path(cmd[1])
        _write_sidecar(video, csv_content)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(E.subprocess, "run", fake_run)


def test_load_mp4_produces_stream(tmp_path, monkeypatch):
    _patch_gyro2bb(monkeypatch, _make_csv(n=3000, fs=1000.0))
    video = tmp_path / "PRO_VID_test.mp4"
    video.write_bytes(b"\x00")  # presence only; gyro2bb is faked
    stream = E.load_insta360_mp4(video, source="wheel", apply_lowpass=False)
    assert stream.source == "wheel"
    assert 950 < stream.sample_rate_hz < 1050
    assert stream.gyro.shape[0] == stream.t.shape[0] == 3000
    assert np.isclose(stream.t[0], 0.0)         # zero-based time
    # gyro converted to rad/s: 5 deg/s peak → ~0.087 rad/s
    assert np.rad2deg(np.abs(stream.gyro).max()) < 10.0
    # sidecar CSV cleaned up by default
    assert not E._gyro2bb_csv_path(video).exists()


def test_load_mp4_keep_csv(tmp_path, monkeypatch):
    _patch_gyro2bb(monkeypatch, _make_csv(n=1500))
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")
    E.load_insta360_mp4(video, source="wheel", keep_csv=True)
    assert E._gyro2bb_csv_path(video).exists()


def test_load_mp4_zero_samples_cleans_up_and_raises(tmp_path, monkeypatch):
    _patch_gyro2bb(monkeypatch, _PREAMBLE)
    video = tmp_path / "VID_flowstate.mp4"
    video.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="(?i)pro video|flowstate"):
        E.load_insta360_mp4(video, source="wheel")
    # even on failure, the sidecar is removed (it was not pre-existing)
    assert not E._gyro2bb_csv_path(video).exists()


def test_load_mp4_subprocess_failure_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "_find_gyro2bb", lambda explicit=None: "fake-gyro2bb")
    monkeypatch.setattr(E.subprocess, "run",
                        lambda cmd, *a, **kw: subprocess.CompletedProcess(cmd, 1, "", "boom"))
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")
    with pytest.raises(RuntimeError, match="(?i)gyro2bb failed"):
        E.load_insta360_mp4(video, source="wheel")


# ---------------------------------------------------------------------------
# load_imu dispatch
# ---------------------------------------------------------------------------

def test_load_imu_dispatches_mp4(tmp_path, monkeypatch):
    called = {}
    monkeypatch.setattr(E, "load_insta360_mp4",
                        lambda p, s, **kw: called.setdefault("mp4", (p, s)))
    E.load_imu(tmp_path / "clip.mp4", "wheel")
    assert "mp4" in called


def test_load_imu_dispatches_gyroflow(tmp_path, monkeypatch):
    called = {}
    monkeypatch.setattr(E, "load_gyroflow",
                        lambda p, s, **kw: called.setdefault("gf", (p, s)))
    E.load_imu(tmp_path / "proj.gyroflow", "wheel")
    assert "gf" in called


def test_load_imu_rejects_unknown_extension(tmp_path):
    with pytest.raises(ValueError, match="(?i)unsupported"):
        E.load_imu(tmp_path / "data.txt", "wheel")
