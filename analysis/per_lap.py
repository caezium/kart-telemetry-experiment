"""Produce a per-lap visualization combining the wheel-cam IMU with MyChron.

Workflow:
    1. Cross-correlate IMU chassis-yaw vs XRK GPS_Yaw_Rate to find the
       clock offset (via pipeline.sync_xrk).
    2. For each XRK lap (from the beacon-crossing markers), extract the
       overlapping window of IMU data and the lap's slice of every XRK
       channel.
    3. Compute steering wheel angle from the IMU using the PCA column
       axis + high-pass detrend (the technique developed in
       analysis/steering_angle.py — but for wheel-mounted-with-tilt
       cameras, not the canonical lens-along-column mount).
    4. Plot a 4-panel summary per lap: steering angle, kart yaw rate,
       GPS speed, GPS track (lat/lon overlay).

Usage:
    python -m analysis.per_lap <wheel.gyroflow> <session.xrk>
        [--lap N | --all]
        [--out-dir DIR]

If --lap is omitted, the longest lap in the XRK is plotted (usually
the first complete one).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
from scipy.signal import butter, sosfiltfilt

from pipeline.extract_imu import ImuStream, load_gyroflow
from pipeline.sync_xrk import (
    EDGE_TRIM_S,
    detect_column_axis,
    estimate_quiet_bias,
    load_xrk,
    sync_imu_to_xrk,
)


# ---------------------------------------------------------------------------
# GPS-based lap detection
# ---------------------------------------------------------------------------
#
# The XRK's built-in beacon-crossing lap detector is unreliable for the data
# we've seen — it can fire only once or twice over a whole session, fusing
# many physical laps into one "lap". So we re-detect from the GPS track
# itself: a lap is complete when the kart returns to within a small radius
# of the start point AFTER having left it.

LAP_DETECT_RADIUS_M = 25.0      # "near start" threshold
LAP_DETECT_MIN_DURATION_S = 15.0  # discard suspiciously short "laps"
LAP_DETECT_WARMUP_S = 5.0       # ignore "returns" within this many seconds
                                # of the first start-passage (still warming up)


def _gps_distance_to_ref_m(lat: np.ndarray, lon: np.ndarray,
                           lat_ref: float, lon_ref: float) -> np.ndarray:
    """Equirectangular small-region approximation. Good to <1% for tracks
    spanning <1 km, which kart tracks always are.
    """
    R = 6_371_000.0
    dlat = np.deg2rad(lat - lat_ref)
    dlon = np.deg2rad(lon - lon_ref) * np.cos(np.deg2rad(lat_ref))
    return R * np.hypot(dlat, dlon)


def detect_laps_from_gps(log, radius_m: float = LAP_DETECT_RADIUS_M
                         ) -> list[tuple[float, float]]:
    """Return list of (start_time_s, end_time_s) per detected lap, on XRK clock.

    Algorithm: take the start position to be the first sample where
    GPS_Speed > 5 km/h (i.e. once moving, after the initial fix lock).
    Walk through the trace; each time we come within `radius_m` of the
    start position AFTER having left the start radius (with a velocity-based
    debounce), declare a lap.
    """
    if "GPS Latitude" not in log.channels or "GPS Longitude" not in log.channels:
        return []
    t = np.asarray(log.channels["GPS Latitude"]["timecodes"]) / 1000.0
    lat = np.asarray(log.channels["GPS Latitude"]["GPS Latitude"])
    lon = np.asarray(log.channels["GPS Longitude"]["GPS Longitude"])
    if "GPS Speed" in log.channels:
        t_sp = np.asarray(log.channels["GPS Speed"]["timecodes"]) / 1000.0
        spd = np.asarray(log.channels["GPS Speed"]["GPS Speed"])
        spd_on_gps = np.interp(t, t_sp, spd)
    else:
        spd_on_gps = np.full_like(t, 10.0)  # assume moving if no speed

    # Find first moving sample as reference point
    moving = np.where(spd_on_gps > 5.0)[0]
    if len(moving) < 2:
        return []
    i_ref = moving[0]
    lat_ref = lat[i_ref]
    lon_ref = lon[i_ref]
    t_ref = t[i_ref]

    dist = _gps_distance_to_ref_m(lat, lon, lat_ref, lon_ref)

    # State machine: track whether we've left the start radius
    laps: list[tuple[float, float]] = []
    lap_start_t = t_ref
    away = False
    for i in range(i_ref + 1, len(t)):
        if dist[i] > radius_m * 2:
            away = True
        elif away and dist[i] < radius_m:
            # Lap complete
            lap_end_t = t[i]
            if lap_end_t - lap_start_t >= LAP_DETECT_MIN_DURATION_S:
                laps.append((lap_start_t, lap_end_t))
                lap_start_t = lap_end_t
                away = False
    return laps


@dataclass
class LapData:
    """Per-lap merged data, all on XRK clock."""
    lap_num: int
    t_start: float           # XRK time (s)
    t_end: float
    duration: float
    # IMU side (resampled within the lap window)
    t_imu: np.ndarray        # XRK time (s)
    steering_angle_deg: np.ndarray   # detrended, returns to ~0 between corners
    steering_rate_dps: np.ndarray
    # XRK side
    xrk_channels: dict[str, np.ndarray]   # channel name -> values
    xrk_t: dict[str, np.ndarray]          # channel name -> time array (s)


def steering_angle_from_imu(stream: ImuStream, column_axis: np.ndarray,
                            bias: np.ndarray, sign: int,
                            hp_hz: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (t [s], steering_angle [deg], steering_rate [deg/s]).

    Math: project gyro onto column axis (sign-corrected), then high-pass
    the cumulative integral to subtract the chassis-yaw leakage. What
    remains is the wheel-relative-to-chassis rotation = steering input.
    """
    gyro = stream.gyro - bias
    rate = sign * (gyro @ column_axis)             # rad/s along column
    rate_dps = np.rad2deg(rate)

    # Cumulative integral via trapezoidal
    angle = np.zeros_like(rate)
    angle[1:] = np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(stream.t))
    angle_deg = np.rad2deg(angle)

    # High-pass the angle to remove the chassis-yaw drift
    fs = stream.sample_rate_hz
    sos = butter(3, hp_hz / (fs / 2), btype="high", output="sos")
    angle_deg = sosfiltfilt(sos, angle_deg)
    return stream.t, angle_deg, rate_dps


def extract_lap(stream: ImuStream, log, sync_offset_s: float,
                column_axis: np.ndarray, bias: np.ndarray, sign: int,
                lap_num: int,
                lap_windows: list[tuple[float, float]] | None = None,
                ) -> LapData:
    """Extract one lap's data, all on XRK clock.

    If `lap_windows` is provided, lap_num indexes into it (GPS-detected laps).
    Otherwise it indexes into the XRK's built-in laps table (beacon-detected).
    """
    if lap_windows is not None:
        if lap_num < 0 or lap_num >= len(lap_windows):
            raise ValueError(f"Lap {lap_num} out of range (have {len(lap_windows)})")
        t_start, t_end = lap_windows[lap_num]
    else:
        laps = log.laps.to_pydict()
        if lap_num not in laps["num"]:
            raise ValueError(f"Lap {lap_num} not in XRK (have {laps['num']})")
        i = laps["num"].index(lap_num)
        t_start = laps["start_time"][i] / 1000.0
        t_end = laps["end_time"][i] / 1000.0
    t_start_ms = int(t_start * 1000)
    t_end_ms = int(t_end * 1000)

    # IMU: compute steering, then slice to this lap
    t_imu_local, ang_deg, rate_dps = steering_angle_from_imu(
        stream, column_axis, bias, sign,
    )
    t_imu_xrk = t_imu_local + sync_offset_s
    mask = (t_imu_xrk >= t_start) & (t_imu_xrk <= t_end)

    # XRK channels: slice each by its own timecodes
    xrk_ch: dict[str, np.ndarray] = {}
    xrk_t: dict[str, np.ndarray] = {}
    for name, tbl in log.channels.items():
        if tbl.num_rows == 0:
            continue
        t_ms = np.asarray(tbl["timecodes"])
        col = [c for c in tbl.column_names if c != "timecodes"][0]
        vals = np.asarray(tbl[col])
        m = (t_ms >= t_start_ms) & (t_ms <= t_end_ms)
        if m.any():
            xrk_ch[name] = vals[m]
            xrk_t[name] = t_ms[m] / 1000.0

    return LapData(
        lap_num=lap_num,
        t_start=t_start,
        t_end=t_end,
        duration=t_end - t_start,
        t_imu=t_imu_xrk[mask],
        steering_angle_deg=ang_deg[mask],
        steering_rate_dps=rate_dps[mask],
        xrk_channels=xrk_ch,
        xrk_t=xrk_t,
    )


def plot_lap(lap: LapData, out_path: Path, title_suffix: str = "") -> None:
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[2, 1])
    ax_steer = fig.add_subplot(gs[0, 0])
    ax_yaw = fig.add_subplot(gs[1, 0], sharex=ax_steer)
    ax_speed = fig.add_subplot(gs[2, 0], sharex=ax_steer)
    ax_map = fig.add_subplot(gs[:, 1])

    # Use lap-relative time on the x-axis (0..duration)
    t0 = lap.t_start
    t_imu_rel = lap.t_imu - t0

    # Steering angle
    ax_steer.plot(t_imu_rel, lap.steering_angle_deg, lw=0.8, color="tab:blue")
    ax_steer.axhline(0, color="black", lw=0.4)
    ax_steer.set_ylabel("steering angle (deg)")
    ax_steer.set_title(
        f"Lap {lap.lap_num}  —  {lap.duration:.2f}s  ({title_suffix})"
    )
    ax_steer.grid(alpha=0.3)

    # Yaw rate (XRK)
    if "GPS_Yaw_Rate" in lap.xrk_channels:
        ax_yaw.plot(lap.xrk_t["GPS_Yaw_Rate"] - t0, lap.xrk_channels["GPS_Yaw_Rate"],
                    lw=0.9, color="tab:orange")
    ax_yaw.axhline(0, color="black", lw=0.4)
    ax_yaw.set_ylabel("kart yaw rate (deg/s)\n(XRK)")
    ax_yaw.grid(alpha=0.3)

    # Speed
    if "GPS Speed" in lap.xrk_channels:
        ax_speed.plot(lap.xrk_t["GPS Speed"] - t0, lap.xrk_channels["GPS Speed"],
                      lw=0.9, color="tab:green")
        ax_speed.set_ylabel("GPS speed (km/h)")
    ax_speed.set_xlabel("lap time (s)")
    ax_speed.grid(alpha=0.3)

    # Track map (GPS lat/lon, colored by speed)
    if ("GPS Latitude" in lap.xrk_channels
        and "GPS Longitude" in lap.xrk_channels):
        lat = lap.xrk_channels["GPS Latitude"]
        lon = lap.xrk_channels["GPS Longitude"]
        if "GPS Speed" in lap.xrk_channels:
            # Interp speed onto GPS samples
            t_gps = lap.xrk_t["GPS Latitude"]
            t_sp = lap.xrk_t["GPS Speed"]
            sp_lat = np.interp(t_gps, t_sp, lap.xrk_channels["GPS Speed"])
            sc = ax_map.scatter(lon, lat, c=sp_lat, s=4, cmap="viridis")
            plt.colorbar(sc, ax=ax_map, label="speed (km/h)", shrink=0.7)
        else:
            ax_map.plot(lon, lat, lw=1.0)
        ax_map.set_xlabel("longitude")
        ax_map.set_ylabel("latitude")
        ax_map.set_title("track (GPS)")
        ax_map.set_aspect("equal", adjustable="datalim")
        ax_map.grid(alpha=0.3)
    else:
        ax_map.text(0.5, 0.5, "no GPS", ha="center", va="center")

    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gyroflow", type=Path)
    ap.add_argument("xrk", type=Path)
    ap.add_argument("--lap", type=int, default=None,
                    help="Lap number to plot. Default: the longest one.")
    ap.add_argument("--all", action="store_true",
                    help="Plot every lap.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory (default: next to the .gyroflow).")
    ap.add_argument("--lap-source", choices=("auto", "xrk", "gps"), default="auto",
                    help="auto: use XRK beacon markers if they give >1 lap "
                         "of plausible duration, else fall back to GPS-based "
                         "detection. xrk: force XRK markers. gps: force "
                         "GPS-based detection.")
    args = ap.parse_args()

    print(f"Loading + syncing...")
    sync = sync_imu_to_xrk(args.gyroflow, args.xrk)
    print(f"  offset (IMU→XRK):   {sync.offset_imu_to_xrk_s:+.3f} s")
    print(f"  peak |corr|:        {sync.corr_peak:.4f}")
    print(f"  sign:               {sync.sign:+d}")
    print()

    if sync.corr_peak < 0.5:
        print(f"  WARNING: weak correlation ({sync.corr_peak:.2f}). "
              f"The IMU and XRK may not be from the same session. "
              f"Sanity-check the per-lap plot before trusting timing.")
        print()

    stream = load_gyroflow(args.gyroflow, source="wheel")
    log = sync.xrk_log
    out_dir = args.out_dir or args.gyroflow.parent

    # Decide lap source: XRK beacon markers vs GPS-based fallback
    xrk_durations = [(e - s) / 1000.0 for s, e in zip(
        log.laps["start_time"].to_pylist(), log.laps["end_time"].to_pylist()
    )]
    use_gps = (
        args.lap_source == "gps"
        or (args.lap_source == "auto"
            and (len(xrk_durations) <= 2 or max(xrk_durations) > 120))
    )

    if use_gps:
        gps_laps = detect_laps_from_gps(log)
        if not gps_laps:
            raise SystemExit(
                "GPS-based lap detection found nothing usable. "
                "Try --lap-source xrk to use the XRK markers anyway."
            )
        lap_windows = gps_laps
        print(f"  using GPS-based lap detection: {len(lap_windows)} laps")
        print(f"  (XRK beacon markers gave {len(xrk_durations)} laps, "
              f"max duration {max(xrk_durations):.1f}s — too few/long for real laps)")
        targets = list(range(len(lap_windows)))
    else:
        lap_windows = None
        xrk_nums = log.laps["num"].to_pylist()
        targets = xrk_nums
        print(f"  using XRK beacon markers: {len(xrk_nums)} laps")

    if args.lap is not None:
        targets = [args.lap]
    elif not args.all:
        # Pick the longest lap (likely a real complete one, not the
        # partial out/in lap at either end).
        if lap_windows is not None:
            ix = max(range(len(lap_windows)),
                     key=lambda i: lap_windows[i][1] - lap_windows[i][0])
            targets = [ix]
        else:
            full_laps = [(n, d) for n, d in zip(targets, xrk_durations) if d >= 30]
            if not full_laps:
                full_laps = list(zip(targets, xrk_durations))
            targets = [max(full_laps, key=lambda x: x[1])[0]]

    for lap_num in targets:
        lap = extract_lap(
            stream, log, sync.offset_imu_to_xrk_s,
            sync.column_axis, sync.gyro_bias, sync.sign,
            lap_num, lap_windows=lap_windows,
        )
        out = out_dir / f"{args.gyroflow.stem}.lap{lap_num}.png"
        plot_lap(
            lap, out,
            title_suffix=f"sync corr={sync.corr_peak:.2f}, "
                         f"offset={sync.offset_imu_to_xrk_s:+.2f}s",
        )
        peak_steer = float(np.abs(lap.steering_angle_deg).max())
        peak_yaw = (float(np.abs(lap.xrk_channels["GPS_Yaw_Rate"]).max())
                    if "GPS_Yaw_Rate" in lap.xrk_channels else float("nan"))
        peak_speed = (float(lap.xrk_channels["GPS Speed"].max())
                      if "GPS Speed" in lap.xrk_channels else float("nan"))
        print(f"  lap {lap_num}: {lap.duration:6.2f}s   "
              f"peak steer ±{peak_steer:>5.0f}°   "
              f"peak yaw rate ±{peak_yaw:>5.0f}°/s   "
              f"peak speed {peak_speed:>5.1f} km/h   "
              f"→ {out.name}")


if __name__ == "__main__":
    main()
