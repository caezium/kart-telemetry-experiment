"""
Time-align wheel IMU, helmet IMU, and MyChron logs onto a single clock.

Sync strategies, in order of preference:

1. **Hard sync mark.** A sharp tap on the steering wheel (or the kart body)
   at the start of the session creates an impulse visible in:
       - wheel accel (direct impact + transient)
       - wheel gyro (sympathetic rotation)
       - helmet accel (vibration through chassis → seat → spine → helmet)
       - MyChron lateral-G (small but detectable on the chassis IMU)
   We use accel-deviation-from-gravity as the primary detector and gyro
   magnitude as a backup. The first sample exceeding threshold within
   the first 30 s is taken as t=0 of the master clock.

2. **Manual override.** Pass `--manual-offset wheel=0,helmet=2.34,mychron=1.05`
   if the tap is missing or noisy.

3. **Cross-correlation fallback** (planned, not implemented). Cross-correlate
   bias-corrected wheel-accel-Z against MyChron lat_acc over the first
   ~30 s of driving to recover offset.

MyChron CSV: tested against Race Studio 3 exports (metadata-prefix rows,
"Time" header, lat_acc / lon_acc / gps_speed columns). Other firmware
versions may need a different parser; flagged in README known gaps.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


GRAVITY_MPS2 = 9.80665
SYNC_SEARCH_WINDOW_S = 30.0

# Thresholds for tap detection. Tuned conservatively so a real session-start
# tap is detected without firing on incidental vibration.
WHEEL_ACCEL_THRESHOLD_MPS2 = 8.0    # |a|−g spikes well above this on a sharp tap
HELMET_ACCEL_THRESHOLD_MPS2 = 4.0   # damped through chassis+seat+spine
WHEEL_GYRO_THRESHOLD_RAD_S = 4.0    # used only as a tiebreaker
MYCHRON_LATG_THRESHOLD_G = 0.3      # chassis IMU sees a smaller spike


@dataclass
class SyncedSession:
    """Aligned session. `t0_offsets[source]` is the value added to that
    stream's local `t` to put it on the master clock (where the tap = 0).
    """
    wheel: pd.DataFrame | None
    helmet: pd.DataFrame | None
    mychron: pd.DataFrame | None
    t0_offsets: dict[str, float]
    sources_used_for_sync: dict[str, str]   # source -> "tap" | "manual" | "missing"


# ---------------------------------------------------------------------------
# Tap detection
# ---------------------------------------------------------------------------

def _peak_above(t: np.ndarray, signal: np.ndarray, threshold: float, window_s: float) -> float | None:
    mask = t < window_s
    if not mask.any():
        return None
    seg = signal[mask]
    idx = int(np.argmax(seg))
    if seg[idx] < threshold:
        return None
    return float(t[mask][idx])


def find_tap_in_imu(
    df: pd.DataFrame,
    accel_threshold: float,
    gyro_threshold: float,
    window_s: float = SYNC_SEARCH_WINDOW_S,
) -> float | None:
    """Find a sharp tap in an ImuStream parquet (columns t, gx/y/z, ax/y/z).

    Primary: |accel| − g exceeds accel_threshold (m/s²).
    Backup: |gyro| exceeds gyro_threshold (rad/s).
    """
    t = df["t"].to_numpy()
    accel = df[["ax", "ay", "az"]].to_numpy()
    gyro = df[["gx", "gy", "gz"]].to_numpy()

    accel_dev = np.abs(np.linalg.norm(accel, axis=1) - GRAVITY_MPS2)
    t_a = _peak_above(t, accel_dev, accel_threshold, window_s)
    if t_a is not None:
        return t_a

    gyro_mag = np.linalg.norm(gyro, axis=1)
    return _peak_above(t, gyro_mag, gyro_threshold, window_s)


def find_tap_in_mychron(df: pd.DataFrame, window_s: float = SYNC_SEARCH_WINDOW_S) -> float | None:
    """MyChron exports lat_acc in g. A tap on a stationary kart shows up as
    a brief excursion above ~0.3 g, well above ambient idle vibration.
    """
    if "lat_acc" not in df.columns or "t" not in df.columns:
        return None
    t = df["t"].to_numpy()
    lat = np.abs(df["lat_acc"].to_numpy())
    return _peak_above(t, lat, MYCHRON_LATG_THRESHOLD_G, window_s)


# ---------------------------------------------------------------------------
# MyChron loader
# ---------------------------------------------------------------------------

def load_mychron(path: Path) -> pd.DataFrame:
    """Race Studio 3 exports prefix the data with metadata rows. Sniff for
    the row whose first cell is 'Time' and read from there.
    """
    with path.open() as f:
        lines = f.readlines()
    header_idx = None
    for i, line in enumerate(lines):
        cells = [c.strip().strip('"') for c in line.split(",")]
        if cells and cells[0].lower() == "time":
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(
            f"Could not find 'Time' header in {path}. "
            f"This loader expects a Race Studio 3 CSV export."
        )

    df = pd.read_csv(path, skiprows=header_idx)
    df.columns = [c.strip().strip('"').lower().replace(" ", "_") for c in df.columns]
    if "time" not in df.columns:
        raise ValueError(f"MyChron export missing 'time' column: {list(df.columns)}")
    return df.rename(columns={"time": "t"})


# ---------------------------------------------------------------------------
# Manual-offset parsing
# ---------------------------------------------------------------------------

def parse_manual_offsets(spec: str | None) -> dict[str, float]:
    """Parse a "wheel=0,helmet=2.34,mychron=1.05" style string."""
    if not spec:
        return {}
    out: dict[str, float] = {}
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(f"Bad --manual-offset entry {piece!r}; expected source=seconds")
        k, v = piece.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def sync(
    session_dir: Path,
    manual_offsets: dict[str, float] | None = None,
) -> SyncedSession:
    extracted = session_dir / "extracted"
    if not extracted.is_dir():
        raise SystemExit(f"Run extract_imu.py first; {extracted} not found")

    manual = manual_offsets or {}

    wheel_path = extracted / "wheel_imu.parquet"
    helmet_path = extracted / "helmet_imu.parquet"
    mychron_candidates = list(session_dir.glob("mychron*.csv")) + list(session_dir.glob("MyChron*.csv"))

    wheel = pd.read_parquet(wheel_path) if wheel_path.exists() else None
    helmet = pd.read_parquet(helmet_path) if helmet_path.exists() else None
    mychron = load_mychron(mychron_candidates[0]) if mychron_candidates else None

    offsets: dict[str, float] = {}
    used: dict[str, str] = {}

    # Helper to resolve a stream's t=0: manual override wins, else tap, else
    # mark as "missing" and offset 0.0 (downstream consumers can guard).
    def resolve(name: str, tap_t: float | None) -> tuple[float, str]:
        if name in manual:
            return -manual[name], "manual"
        if tap_t is not None:
            return -tap_t, "tap"
        return 0.0, "missing"

    if wheel is not None:
        tap = find_tap_in_imu(
            wheel,
            accel_threshold=WHEEL_ACCEL_THRESHOLD_MPS2,
            gyro_threshold=WHEEL_GYRO_THRESHOLD_RAD_S,
        )
        offsets["wheel"], used["wheel"] = resolve("wheel", tap)
        print(f"  wheel sync: {used['wheel']} (offset {offsets['wheel']:+.3f}s)")

    if helmet is not None:
        tap = find_tap_in_imu(
            helmet,
            accel_threshold=HELMET_ACCEL_THRESHOLD_MPS2,
            gyro_threshold=WHEEL_GYRO_THRESHOLD_RAD_S * 0.6,
        )
        offsets["helmet"], used["helmet"] = resolve("helmet", tap)
        print(f"  helmet sync: {used['helmet']} (offset {offsets['helmet']:+.3f}s)")

    if mychron is not None:
        tap = find_tap_in_mychron(mychron)
        offsets["mychron"], used["mychron"] = resolve("mychron", tap)
        print(f"  mychron sync: {used['mychron']} (offset {offsets['mychron']:+.3f}s)")

    # Apply offsets to a t_master column on each frame.
    for source, df in [("wheel", wheel), ("helmet", helmet), ("mychron", mychron)]:
        if df is not None and source in offsets:
            df["t_master"] = df["t"] + offsets[source]

    return SyncedSession(wheel=wheel, helmet=helmet, mychron=mychron,
                         t0_offsets=offsets, sources_used_for_sync=used)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session_dir", type=Path)
    ap.add_argument("--manual-offset", type=str, default=None,
                    help='Override auto-detected offsets. e.g. "wheel=0,helmet=2.34"')
    args = ap.parse_args()

    manual = parse_manual_offsets(args.manual_offset)
    synced = sync(args.session_dir, manual_offsets=manual)

    out_dir = args.session_dir / "extracted"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "synced.parquet"

    frames = []
    for source in ("wheel", "helmet", "mychron"):
        df = getattr(synced, source)
        if df is None:
            continue
        df = df.copy()
        df["source"] = source
        frames.append(df)
    if not frames:
        raise SystemExit("Nothing to sync.")
    pd.concat(frames, ignore_index=True).to_parquet(out)
    print(f"  wrote {out}")
    print(f"  sources: {synced.sources_used_for_sync}")
    print(f"  offsets: {synced.t0_offsets}")


if __name__ == "__main__":
    main()
