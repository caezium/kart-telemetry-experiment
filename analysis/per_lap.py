"""Per-lap visualization: wheel-cam IMU + MyChron, on one synchronized clock.

Pipeline
--------
    1. Sync wheel.gyroflow ↔ session.xrk via cross-correlation of chassis
       yaw (pipeline.sync_xrk).
    2. Get clean steering input by subtracting (k · GPS_Yaw_Rate) from the
       IMU's column-axis projection — k = column-tilt factor from the same
       sync step. HP-filter only mops up residual drift; the heavy lifting
       is the chassis-yaw subtraction.
    3. Segment by lap. The XRK's built-in beacon lap markers are unreliable
       in practice (sometimes only fire once or twice per session, fusing
       many physical laps into one). Fall back to GPS-based detection:
       a lap is complete when the kart returns to within 25 m of the
       starting position after having left it.
    4. Per-lap 4-panel plot: steering angle, kart yaw rate, GPS speed,
       GPS track overlay colored by speed.

Usage
-----
    python -m analysis.per_lap wheel.gyroflow session.xrk
    python -m analysis.per_lap wheel.gyroflow session.xrk --all
    python -m analysis.per_lap wheel.gyroflow session.xrk --lap 4
    python -m analysis.per_lap wheel.gyroflow session.xrk --out-dir results/

Without --lap or --all, the longest detected lap is plotted (usually a
representative complete lap, not the partial out- or in-lap).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

from pipeline.extract_imu import ImuStream
from pipeline.sync_xrk import (
    GPS_YAW_GLITCH_DPS,
    SyncedXrk,
    clean_gps_yaw_dps,
    sync_imu_to_xrk,
)


# ---------------------------------------------------------------------------
# Lap detection from GPS
# ---------------------------------------------------------------------------

LAP_DETECT_RADIUS_M = 25.0       # "near start" threshold
LAP_DETECT_MIN_DURATION_S = 15.0  # discard suspiciously short "laps"
GPS_MOVING_THRESHOLD_MPS = 5.0 / 3.6  # 5 km/h, ~minimum to consider "on track"


def _gps_distance_to_ref_m(lat: np.ndarray, lon: np.ndarray,
                           lat_ref: float, lon_ref: float) -> np.ndarray:
    """Equirectangular small-region approximation. Accurate to <1% for any
    kart track (always < 1 km across)."""
    R = 6_371_000.0
    dlat = np.deg2rad(lat - lat_ref)
    dlon = np.deg2rad(lon - lon_ref) * np.cos(np.deg2rad(lat_ref))
    return R * np.hypot(dlat, dlon)


def detect_laps_from_gps(log, radius_m: float = LAP_DETECT_RADIUS_M,
                         ) -> list[tuple[float, float]]:
    """Return [(start_time_s, end_time_s), ...] per detected lap.

    Algorithm: take the first moving sample (GPS_Speed > 5 km/h) as the
    reference. Walk the trace; whenever the kart returns within `radius_m`
    of the reference AFTER having left it, declare a lap.
    """
    if "GPS Latitude" not in log.channels or "GPS Longitude" not in log.channels:
        return []
    lat_tbl = log.channels["GPS Latitude"]
    lon_tbl = log.channels["GPS Longitude"]
    t = np.asarray(lat_tbl["timecodes"], dtype=np.float64) / 1000.0
    lat = np.asarray(lat_tbl["GPS Latitude"], dtype=np.float64)
    # Longitude is a separate channel table with its own (possibly different
    # length / slightly misaligned) timecodes — align it onto the lat grid
    # rather than assuming index-for-index correspondence.
    t_lon = np.asarray(lon_tbl["timecodes"], dtype=np.float64) / 1000.0
    lon = np.interp(t, t_lon, np.asarray(lon_tbl["GPS Longitude"], dtype=np.float64))

    if "GPS Speed" in log.channels:
        sp_tbl = log.channels["GPS Speed"]
        t_sp = np.asarray(sp_tbl["timecodes"], dtype=np.float64) / 1000.0
        spd = np.asarray(sp_tbl["GPS Speed"], dtype=np.float64)  # m/s
        spd_on_gps = np.interp(t, t_sp, spd)
    else:
        spd_on_gps = np.full_like(t, 10.0)  # assume moving if no speed channel

    moving = np.where(spd_on_gps > GPS_MOVING_THRESHOLD_MPS)[0]
    if len(moving) < 2:
        return []
    i_ref = moving[0]
    dist = _gps_distance_to_ref_m(lat, lon, lat[i_ref], lon[i_ref])

    laps: list[tuple[float, float]] = []
    lap_start_t = t[i_ref]
    away = False
    for i in range(i_ref + 1, len(t)):
        if dist[i] > radius_m * 2:
            away = True
        elif away and dist[i] < radius_m:
            lap_end_t = t[i]
            # Always re-arm and restart timing on a return-to-start, whether or
            # not the window was long enough to count. Leaving `away` True on a
            # rejected short window (e.g. an aborted launch / nose-out) would
            # measure the next real lap from a stale start time.
            if lap_end_t - lap_start_t >= LAP_DETECT_MIN_DURATION_S:
                laps.append((lap_start_t, lap_end_t))
            lap_start_t = lap_end_t
            away = False
    return laps


def choose_lap_windows(sync: SyncedXrk, force_source: str = "auto"
                       ) -> tuple[list[tuple[float, float]], str]:
    """Decide whether to use XRK beacon laps or GPS-detected laps.

    `force_source` ∈ {"auto", "xrk", "gps"}. In auto mode, fall back to
    GPS detection if the XRK has ≤2 lap markers OR any lap >120s (real
    kart laps are 30–90s; a long one indicates missed beacon crossings).

    Returns (lap_windows, source_used) where source_used is "xrk" or "gps".
    """
    log = sync.xrk_log
    xrk_starts = log.laps["start_time"].to_pylist()
    xrk_ends = log.laps["end_time"].to_pylist()
    xrk_durations = [(e - s) / 1000.0 for s, e in zip(xrk_starts, xrk_ends)]

    use_gps = (
        force_source == "gps"
        or (force_source == "auto"
            and (len(xrk_durations) <= 2 or (xrk_durations and max(xrk_durations) > 120)))
    )
    if use_gps:
        gps = detect_laps_from_gps(log)
        if not gps:
            raise SystemExit(
                "GPS lap detection found nothing usable. "
                "Try --lap-source xrk to use the XRK markers anyway."
            )
        return gps, "gps"
    return [(s / 1000.0, e / 1000.0) for s, e in zip(xrk_starts, xrk_ends)], "xrk"


# ---------------------------------------------------------------------------
# Steering extraction (chassis-yaw subtracted)
# ---------------------------------------------------------------------------

DEFAULT_STEERING_HP_HZ = 0.05    # residual drift removal after subtraction


def _hp_filter(angle_deg: np.ndarray, fs: float, hp_hz: float) -> np.ndarray:
    """Zero-phase high-pass. Falls back to mean-removal for clips too short
    for sosfiltfilt's padding requirement (instead of raising)."""
    sos = butter(3, hp_hz / (fs / 2), btype="high", output="sos")
    try:
        return sosfiltfilt(sos, angle_deg)
    except ValueError:
        return angle_deg - angle_deg.mean() if angle_deg.size else angle_deg


def steering_from_synced(stream: ImuStream, sync: SyncedXrk,
                         hp_hz: float = DEFAULT_STEERING_HP_HZ,
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (t_xrk [s], steering_angle [deg], steering_rate [deg/s])
    over the IMU/XRK time OVERLAP, on the XRK clock.

    Method:
        ω_along_col = (gyro - bias) · column_axis     # sign already folded in
        ω_steering  = ω_along_col − k · GPS_Yaw_Rate(interpolated)
        angle       = trapezoidal integral of ω_steering, HP-filtered

    Only the portion of the IMU recording that overlaps the GPS time span is
    returned. np.interp would otherwise flat-extrapolate GPS_Yaw_Rate (hold
    the first/last value) across the IMU lead-in/tail, fabricating chassis
    yaw and biasing the integrated angle of the laps adjacent to the span
    edges. Lap windows always fall inside the GPS span, so nothing real is lost.

    HP at `hp_hz` is a small safety net for residual integration drift.
    """
    # 1. IMU column projection on XRK clock (sign already folded into the axis)
    gyro = stream.gyro - sync.gyro_bias
    rate_along_col = gyro @ sync.column_axis            # rad/s
    t_imu_xrk = stream.t + sync.offset_imu_to_xrk_s

    # 2. GPS yaw rate from XRK, already glitch-cleaned at the source by
    #    chassis_yaw_from_xrk; re-clean here defensively (cheap, idempotent)
    #    in case a caller passes a hand-built log.
    yr_tbl = sync.xrk_log.channels["GPS_Yaw_Rate"]
    t_gps = np.asarray(yr_tbl["timecodes"], dtype=np.float64) / 1000.0
    yr_dps = clean_gps_yaw_dps(t_gps, np.asarray(yr_tbl["GPS_Yaw_Rate"], dtype=np.float64))

    # 3. Map GPS yaw onto the IMU clock, but ONLY within the GPS-covered span.
    yr_on_imu = np.interp(t_imu_xrk, t_gps, yr_dps, left=np.nan, right=np.nan)
    in_range = np.isfinite(yr_on_imu)
    if not in_range.any():
        raise ValueError(
            "IMU and XRK time ranges do not overlap after sync "
            f"(offset {sync.offset_imu_to_xrk_s:+.1f}s); cannot extract steering."
        )

    # Restrict everything to the contiguous in-range slice. The slice is
    # contiguous because t_imu_xrk is monotonic and the GPS span is an interval.
    t_imu_xrk = t_imu_xrk[in_range]
    t_local = stream.t[in_range]
    rate_along_col = rate_along_col[in_range]
    yr_rps = np.deg2rad(yr_on_imu[in_range])

    # 4. Subtract k · GPS yaw. If geometry was unreliable (k=NaN, no quiet
    #    samples) fall back to no subtraction rather than poisoning everything
    #    with NaN — sync_imu_to_xrk already warned the user in that case.
    k = sync.column_tilt_factor if np.isfinite(sync.column_tilt_factor) else 0.0
    steer_rate_rps = rate_along_col - k * yr_rps

    # 5. Trapezoidal integral on the (monotonic) in-range time base.
    angle_rad = np.zeros_like(steer_rate_rps)
    angle_rad[1:] = np.cumsum(0.5 * (steer_rate_rps[1:] + steer_rate_rps[:-1])
                              * np.diff(t_local))
    angle_deg = _hp_filter(np.rad2deg(angle_rad), stream.sample_rate_hz, hp_hz)

    return t_imu_xrk, angle_deg, np.rad2deg(steer_rate_rps)


# ---------------------------------------------------------------------------
# Per-lap slicing
# ---------------------------------------------------------------------------

@dataclass
class LapData:
    """One lap's worth of synchronized telemetry. Times are XRK-clock seconds."""
    lap_index: int                          # 0-based index into the lap list
    t_start: float
    t_end: float
    duration: float
    # IMU side, sliced to the lap window
    t_imu: np.ndarray
    steering_angle_deg: np.ndarray
    steering_rate_dps: np.ndarray
    # XRK side, one entry per channel that has data in this window
    xrk_channels: dict[str, np.ndarray]
    xrk_t: dict[str, np.ndarray]

    @property
    def label(self) -> str:
        return f"lap {self.lap_index}"

    @property
    def has_imu(self) -> bool:
        return self.steering_angle_deg.size > 0


def extract_lap(stream: ImuStream, sync: SyncedXrk,
                lap_window: tuple[float, float], lap_index: int,
                *, steering: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
                ) -> LapData:
    """Slice one lap out of the synchronized session.

    `steering` is the (t_xrk, angle_deg, rate_dps) tuple from
    `steering_from_synced`. Pass it in to compute the (expensive,
    full-recording) steering integral ONCE and reuse it across every lap;
    if omitted it is computed here (convenient for one-off single-lap use).
    """
    t_start, t_end = lap_window
    t_start_ms = int(t_start * 1000)
    t_end_ms = int(t_end * 1000)

    if steering is None:
        steering = steering_from_synced(stream, sync)
    t_imu_xrk, ang_deg, rate_dps = steering

    # Half-open [t_start, t_end) so adjacent laps don't share a boundary sample.
    mask = (t_imu_xrk >= t_start) & (t_imu_xrk < t_end)

    xrk_ch: dict[str, np.ndarray] = {}
    xrk_t: dict[str, np.ndarray] = {}
    for name, tbl in sync.xrk_log.channels.items():
        if tbl.num_rows == 0:
            continue
        # Cast to int64 so a chunked-arrow object dtype can't break comparisons,
        # and use searchsorted (timecodes are monotonic) instead of a full mask.
        t_ms = np.asarray(tbl["timecodes"], dtype=np.int64)
        lo = int(np.searchsorted(t_ms, t_start_ms, side="left"))
        hi = int(np.searchsorted(t_ms, t_end_ms, side="left"))  # half-open
        if hi <= lo:
            continue
        col = [c for c in tbl.column_names if c != "timecodes"][0]
        xrk_ch[name] = np.asarray(tbl[col])[lo:hi]
        xrk_t[name] = t_ms[lo:hi] / 1000.0

    return LapData(
        lap_index=lap_index,
        t_start=t_start, t_end=t_end, duration=t_end - t_start,
        t_imu=t_imu_xrk[mask],
        steering_angle_deg=ang_deg[mask],
        steering_rate_dps=rate_dps[mask],
        xrk_channels=xrk_ch,
        xrk_t=xrk_t,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_lap(lap: LapData, out_path: Path, *, title: str = "") -> None:
    """Render a 4-panel summary PNG for one lap.

    Panels:
        - top-left   : steering angle (deg) vs lap-relative time
        - middle-left: kart yaw rate (deg/s) from XRK GPS_Yaw_Rate
        - bottom-left: GPS speed (km/h)
        - right      : GPS track lat/lon overlay, colored by speed
    """
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    try:
        gs = fig.add_gridspec(3, 2, width_ratios=[2, 1])
        ax_steer = fig.add_subplot(gs[0, 0])
        ax_yaw = fig.add_subplot(gs[1, 0], sharex=ax_steer)
        ax_speed = fig.add_subplot(gs[2, 0], sharex=ax_steer)
        ax_map = fig.add_subplot(gs[:, 1])

        t0 = lap.t_start

        # Steering angle
        ax_steer.plot(lap.t_imu - t0, lap.steering_angle_deg, lw=0.8, color="tab:blue")
        ax_steer.axhline(0, color="black", lw=0.4)
        ax_steer.set_ylabel("steering angle (deg)")
        ax_steer.set_title(f"Lap {lap.lap_index}  —  {lap.duration:.2f}s  ({title})")
        ax_steer.grid(alpha=0.3)

        # Kart yaw rate
        if "GPS_Yaw_Rate" in lap.xrk_channels:
            ax_yaw.plot(lap.xrk_t["GPS_Yaw_Rate"] - t0,
                        lap.xrk_channels["GPS_Yaw_Rate"],
                        lw=0.9, color="tab:orange")
        ax_yaw.axhline(0, color="black", lw=0.4)
        ax_yaw.set_ylabel("kart yaw rate (deg/s)")
        ax_yaw.grid(alpha=0.3)

        # GPS speed (m/s in raw XRK → km/h for display)
        if "GPS Speed" in lap.xrk_channels:
            ax_speed.plot(lap.xrk_t["GPS Speed"] - t0,
                          lap.xrk_channels["GPS Speed"] * 3.6,
                          lw=0.9, color="tab:green")
            ax_speed.set_ylabel("GPS speed (km/h)")
        ax_speed.set_xlabel("lap time (s)")
        ax_speed.grid(alpha=0.3)

        # Track map
        if "GPS Latitude" in lap.xrk_channels and "GPS Longitude" in lap.xrk_channels:
            t_lat = lap.xrk_t["GPS Latitude"]
            lat = lap.xrk_channels["GPS Latitude"]
            # Align longitude onto the latitude samples (independent channels).
            lon = np.interp(t_lat, lap.xrk_t["GPS Longitude"],
                            lap.xrk_channels["GPS Longitude"])
            if "GPS Speed" in lap.xrk_channels:
                sp_kmh = np.interp(t_lat, lap.xrk_t["GPS Speed"],
                                   lap.xrk_channels["GPS Speed"]) * 3.6
                sc = ax_map.scatter(lon, lat, c=sp_kmh, s=4, cmap="viridis")
                plt.colorbar(sc, ax=ax_map, label="speed (km/h)", shrink=0.7)
            else:
                ax_map.plot(lon, lat, lw=1.0)
            ax_map.set_xlabel("longitude")
            ax_map.set_ylabel("latitude")
            ax_map.set_title("track (GPS)")
            ax_map.set_aspect("equal", adjustable="datalim")
            ax_map.grid(alpha=0.3)
        else:
            ax_map.text(0.5, 0.5, "no GPS", ha="center", va="center",
                        transform=ax_map.transAxes)

        fig.savefig(out_path, dpi=110, bbox_inches="tight")
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def lap_summary(lap: LapData) -> dict[str, float]:
    """One-liner per-lap stats. Used for the console table and RESULTS.md.

    Guards every reduction against an empty array — a lap window with no
    overlapping IMU samples (or no GPS in range) yields NaN rather than a
    `zero-size array to reduction` crash that would abort an `--all` run.
    """
    steer = lap.steering_angle_deg
    out: dict[str, float] = {
        "duration_s": lap.duration,
        "peak_steer_deg": float(np.abs(steer).max()) if steer.size else float("nan"),
    }
    if "GPS_Yaw_Rate" in lap.xrk_channels:
        yr = lap.xrk_channels["GPS_Yaw_Rate"]
        clean = yr[np.abs(yr) <= GPS_YAW_GLITCH_DPS]
        out["peak_yaw_rate_dps"] = float(np.abs(clean).max()) if clean.size else float("nan")
    if "GPS Speed" in lap.xrk_channels and lap.xrk_channels["GPS Speed"].size:
        sp = lap.xrk_channels["GPS Speed"]
        out["peak_speed_kmh"] = float(sp.max() * 3.6)
        out["mean_speed_kmh"] = float(sp.mean() * 3.6)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gyroflow", type=Path)
    ap.add_argument("xrk", type=Path)
    ap.add_argument("--lap", type=int, default=None,
                    help="Plot just this lap index. Default: the longest one.")
    ap.add_argument("--all", action="store_true",
                    help="Plot every detected lap.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory (default: next to the .gyroflow).")
    ap.add_argument("--lap-source", choices=("auto", "xrk", "gps"), default="auto",
                    help="Lap-window source. `auto` uses XRK beacon markers if "
                         "they look sensible (>1 lap, no lap >120s), else GPS.")
    args = ap.parse_args()

    print("Loading + syncing...")
    sync = sync_imu_to_xrk(args.gyroflow, args.xrk)
    print(f"  offset (IMU→XRK):   {sync.offset_imu_to_xrk_s:+.3f} s")
    print(f"  peak |corr|:        {sync.corr_peak:.4f}"
          + ("   ⚠ WEAK — sanity-check the result" if sync.corr_peak < 0.5 else ""))
    print(f"  sign:               {sync.sign:+d} (folded into column axis)")
    print(f"  column tilt:        {sync.column_tilt_deg:.1f}° from vertical "
          f"(k = {sync.column_tilt_factor:.4f})")
    if not sync.geometry_reliable:
        print(f"  ⚠ geometry UNRELIABLE: only {sync.quiet_sample_count} quiet samples; "
              f"steering may retain chassis yaw.")
    print()

    # Reuse the stream loaded inside sync (no second parse of the .gyroflow).
    stream = sync.imu_stream
    out_dir = args.out_dir or args.gyroflow.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    lap_windows, source = choose_lap_windows(sync, force_source=args.lap_source)
    if not lap_windows:
        raise SystemExit("No laps detected — nothing to plot.")
    print(f"  lap windows: {len(lap_windows)} laps ({source})")

    # Decide which lap indices to plot, with bounds checking on --lap.
    if args.lap is not None:
        if not (0 <= args.lap < len(lap_windows)):
            raise SystemExit(
                f"--lap {args.lap} out of range; {len(lap_windows)} laps detected "
                f"(valid indices 0..{len(lap_windows) - 1})."
            )
        targets = [args.lap]
    elif args.all:
        targets = list(range(len(lap_windows)))
    else:
        targets = [max(range(len(lap_windows)),
                       key=lambda i: lap_windows[i][1] - lap_windows[i][0])]

    title = (f"sync corr={sync.corr_peak:.2f}, "
             f"offset={sync.offset_imu_to_xrk_s:+.2f}s, "
             f"k={sync.column_tilt_factor:.3f}")

    # Compute the full-recording steering integral ONCE, reuse for every lap.
    steering = steering_from_synced(stream, sync)

    for i in targets:
        lap = extract_lap(stream, sync, lap_windows[i], lap_index=i, steering=steering)
        out = out_dir / f"{args.gyroflow.stem}.lap{i}.png"
        plot_lap(lap, out, title=title)
        s = lap_summary(lap)
        print(f"  lap {i}: {s['duration_s']:6.2f}s   "
              f"peak steer ±{s['peak_steer_deg']:>5.0f}°   "
              f"peak yaw rate ±{s.get('peak_yaw_rate_dps', float('nan')):>5.0f}°/s   "
              f"peak speed {s.get('peak_speed_kmh', float('nan')):>5.1f} km/h   "
              f"→ {out.name}")


if __name__ == "__main__":
    main()
