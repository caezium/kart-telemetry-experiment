"""
Phase 0 bench calibration — validates the toolchain end-to-end and produces
per-unit constants the rest of the pipeline consumes.

Inputs (per Go 3S unit):

    data/calibration/<UNIT>/
    ├── original.gyroflow      Gyroflow project file from a bench recording
    ├── ground_truth.csv       handwritten plateaus: t_seconds,angle_deg[,note]
    ├── notes.md               (optional) rig description, photos, anomalies

Outputs:

    data/calibration/<UNIT>/result.json

The script:

    1. Decodes the .gyroflow file and applies imu_orientation (matching what
       the runtime parser does), so the gyro axes here are the same axes
       analysis modules will see during a real session.
    2. Estimates per-axis gyro bias from the longest sustained-static window
       in the first ~40 s.
    3. Identifies which gyro axis carries the steering rotation (highest
       variance during the active rotation phase) and computes how cleanly
       it dominates (axis_separation_db).
    4. Bias-corrects and integrates that axis to get an angle timeseries.
    5. Samples the integrated angle at each ground_truth timestamp and
       reports per-step error.
    6. Estimates 60s drift from a long static segment after the rotations.
    7. Writes a result.json that satisfies the schema in
       data/calibration/README.md.

See data/calibration/README.md for the recording protocol and pass criteria.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.extract_imu import (
    apply_orientation,
    decode_gyroflow_file,
    _samples_to_arrays,
    _is_uniform,
    _resample_uniform,
)


STATIC_RATE_THRESHOLD_DPS = 1.0     # below this, axis is "still"
STATIC_MIN_DURATION_S = 5.0          # require at least this long of stillness
STEP_RATE_THRESHOLD_DPS = 5.0        # active above this on the chosen axis
PASS_STEP_ERROR_DEG = 2.0
PASS_DRIFT_DEG_60S = 5.0
PASS_STATIC_RMS_DPS = 0.5
PASS_AXIS_SEPARATION_DB = 10.0


@dataclass
class StepResult:
    target_deg: float
    measured_deg: float
    abs_error_deg: float
    t_seconds: float
    note: str = ""


@dataclass
class CalibrationResult:
    unit: str
    recorded_at: str
    sample_rate_hz: float
    imu_orientation: str
    axis_used_for_steering: str
    axis_separation_db: float
    gyro_bias_dps: list[float]
    accel_bias_mps2: list[float]
    static_rms_after_bias_dps: float
    step_input_validation: list[StepResult]
    drift_60s_deg: float
    constant_rate_segment_dps: dict[str, float]
    passed: bool
    failure_reasons: list[str] = field(default_factory=list)
    source: str = "wheel"


# ---------------------------------------------------------------------------
# Static-window detection and bias estimation
# ---------------------------------------------------------------------------

def find_static_windows(
    t: np.ndarray,
    gyro_dps: np.ndarray,
    threshold_dps: float = STATIC_RATE_THRESHOLD_DPS,
    min_duration_s: float = STATIC_MIN_DURATION_S,
) -> list[tuple[int, int]]:
    """Return [(i_start, i_end_exclusive), ...] for windows of low gyro magnitude."""
    mag = np.linalg.norm(gyro_dps, axis=1)
    quiet = mag < threshold_dps
    windows: list[tuple[int, int]] = []
    i = 0
    n = len(t)
    while i < n:
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j < n and quiet[j]:
            j += 1
        if t[j - 1] - t[i] >= min_duration_s:
            windows.append((i, j))
        i = j
    return windows


def estimate_bias(
    t: np.ndarray,
    gyro_dps: np.ndarray,
    accel_mps2: np.ndarray,
    static_windows: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Use the **first** sustained-static window for bias estimation.

    The protocol mandates a hands-off period at the start of the recording,
    before any rotations. Taking the first window (rather than the longest)
    leaves the longer post-rotation window free to characterize drift, and
    avoids contaminating bias with any residual motion from active phases.

    Returns (gyro_bias_dps, accel_bias_residual, (i_start, i_end)) so
    callers can annotate which window was used.
    """
    if not static_windows:
        raise ValueError(
            "No static windows long enough for bias estimation. "
            "Did the recording start with a hands-off period?"
        )
    i_start, i_end = static_windows[0]
    gyro_bias = gyro_dps[i_start:i_end].mean(axis=0)
    accel_bias = accel_mps2[i_start:i_end].mean(axis=0)
    # Subtract gravity from accel bias if it dominates the largest axis. The
    # accel "bias" is not a true offset — at rest it should read +/- 9.8 on
    # one axis. We report the residual after removing gravity from that axis.
    g_axis = int(np.argmax(np.abs(accel_bias)))
    if abs(accel_bias[g_axis]) > 5.0:
        accel_bias[g_axis] -= np.sign(accel_bias[g_axis]) * 9.80665
    return gyro_bias, accel_bias, (i_start, i_end)


# ---------------------------------------------------------------------------
# Steering-axis identification
# ---------------------------------------------------------------------------

def identify_steering_axis(
    gyro_dps: np.ndarray,
    bias_dps: np.ndarray,
    active_window: tuple[int, int],
) -> tuple[int, float]:
    """Return (axis_index, separation_db) — the gyro axis with highest variance
    during the rotation phase, and how cleanly it dominates the others.

    separation_db = 10 * log10(var_max / max(var_others)).
    """
    a, b = active_window
    seg = gyro_dps[a:b] - bias_dps  # bias-corrected
    var = seg.var(axis=0)
    max_axis = int(np.argmax(var))
    others = np.delete(var, max_axis)
    if others.max() <= 0:
        return max_axis, float("inf")
    sep = 10.0 * float(np.log10(var[max_axis] / others.max()))
    return max_axis, sep


def find_active_window(
    t: np.ndarray,
    gyro_dps: np.ndarray,
    bias_dps: np.ndarray,
) -> tuple[int, int]:
    """The "active" rotation phase is the contiguous span where any axis
    crosses the step threshold. We pick the union of all motion samples plus
    a small margin.
    """
    bias_corrected = gyro_dps - bias_dps
    mag = np.linalg.norm(bias_corrected, axis=1)
    active = mag > STEP_RATE_THRESHOLD_DPS
    if not active.any():
        raise ValueError(
            "Could not find an active rotation phase. The recording may not "
            "include the step-input section of the protocol."
        )
    idxs = np.where(active)[0]
    return int(idxs[0]), int(idxs[-1] + 1)


# ---------------------------------------------------------------------------
# Integration + ground-truth comparison
# ---------------------------------------------------------------------------

def integrate_angle(t: np.ndarray, rate_dps: np.ndarray) -> np.ndarray:
    """Trapezoidal integration of angular rate. No drift correction here —
    we *want* to see the raw drift in the calibration so we can characterize
    it.
    """
    angle = np.zeros_like(rate_dps)
    if len(t) < 2:
        return angle
    dt = np.diff(t)
    inc = 0.5 * (rate_dps[1:] + rate_dps[:-1]) * dt
    angle[1:] = np.cumsum(inc)
    return angle


def sample_at_times(
    t: np.ndarray,
    angle_deg: np.ndarray,
    times: np.ndarray,
    window_s: float = 0.3,
) -> np.ndarray:
    """For each query time, return the median angle in a small window around
    it — robust against single-sample noise on the plateau.
    """
    out = np.empty_like(times, dtype=np.float64)
    for k, tq in enumerate(times):
        mask = np.abs(t - tq) <= window_s / 2
        if not mask.any():
            j = int(np.argmin(np.abs(t - tq)))
            out[k] = angle_deg[j]
        else:
            out[k] = float(np.median(angle_deg[mask]))
    return out


# ---------------------------------------------------------------------------
# Drift and constant-rate segments
# ---------------------------------------------------------------------------

def measure_drift_60s(
    t: np.ndarray,
    angle_deg: np.ndarray,
    static_windows: list[tuple[int, int]],
    bias_window: tuple[int, int],
) -> float:
    """Find the longest static window AFTER the bias window and report the
    angle excursion across (up to) the first 60 s of it.
    """
    later = [w for w in static_windows if w[0] >= bias_window[1]]
    if not later:
        return float("nan")
    a, b = max(later, key=lambda w: w[1] - w[0])
    t0 = t[a]
    mask = (t >= t0) & (t <= t0 + 60.0) & (np.arange(len(t)) >= a) & (np.arange(len(t)) < b)
    if not mask.any():
        return float("nan")
    seg = angle_deg[mask]
    return float(seg.max() - seg.min())


def measure_constant_rate(
    t: np.ndarray,
    rate_dps: np.ndarray,
    target_dps: float = 30.0,
    tolerance_dps: float = 15.0,
) -> dict[str, float]:
    """Find samples where bias-corrected rate sits near `target_dps` for at
    least 1.5 s. Report median and IQR. Returns NaNs if not found.
    """
    near = np.abs(np.abs(rate_dps) - target_dps) < tolerance_dps
    if near.sum() < 3:
        return {"target": target_dps, "measured_median": float("nan"), "iqr": float("nan")}

    # Find the longest contiguous run.
    diffs = np.diff(near.astype(int))
    starts = np.where(diffs == 1)[0] + 1
    ends = np.where(diffs == -1)[0] + 1
    if near[0]:
        starts = np.concatenate(([0], starts))
    if near[-1]:
        ends = np.concatenate((ends, [len(near)]))
    if not len(starts) or not len(ends):
        return {"target": target_dps, "measured_median": float("nan"), "iqr": float("nan")}
    runs = list(zip(starts, ends))
    a, b = max(runs, key=lambda r: t[r[1] - 1] - t[r[0]])
    if t[b - 1] - t[a] < 1.5:
        return {"target": target_dps, "measured_median": float("nan"), "iqr": float("nan")}

    seg = np.abs(rate_dps[a:b])
    q1, q3 = np.percentile(seg, [25, 75])
    return {
        "target": target_dps,
        "measured_median": float(np.median(seg)),
        "iqr": float(q3 - q1),
    }


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def _load_ground_truth(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    needed = {"t_seconds", "angle_deg"}
    if not needed.issubset(df.columns):
        raise ValueError(
            f"{path} must have columns t_seconds, angle_deg (got {list(df.columns)})"
        )
    if "note" not in df.columns:
        df["note"] = ""
    return df.sort_values("t_seconds").reset_index(drop=True)


def calibrate_unit(unit_dir: Path, source: str = "wheel") -> CalibrationResult:
    gyroflow_path = unit_dir / "original.gyroflow"
    if not gyroflow_path.exists():
        # any .gyroflow in the dir
        candidates = sorted(unit_dir.glob("*.gyroflow"))
        if not candidates:
            raise FileNotFoundError(f"No .gyroflow file in {unit_dir}")
        gyroflow_path = candidates[0]

    gt_path = unit_dir / "ground_truth.csv"
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing {gt_path}")

    meta = decode_gyroflow_file(gyroflow_path)
    samples = meta.get("raw_imu") or []
    orientation = meta.get("imu_orientation") or "XYZ"

    t_s, gyro_dps, accel = _samples_to_arrays(samples)
    gyro_dps, accel = apply_orientation(gyro_dps, accel, orientation)

    if not _is_uniform(t_s):
        t_s, gyro_dps, accel, fs = _resample_uniform(t_s, gyro_dps, accel)
    else:
        t_s = t_s - t_s[0]
        fs = float(1.0 / np.median(np.diff(t_s)))

    static_windows = find_static_windows(t_s, gyro_dps)
    if not static_windows:
        raise ValueError(
            "No static windows detected. The recording must start with a "
            "hands-off static period (≥5 s) for bias estimation."
        )

    gyro_bias, accel_bias_residual, bias_window = estimate_bias(
        t_s, gyro_dps, accel, static_windows
    )

    active_window = find_active_window(t_s, gyro_dps, gyro_bias)
    axis_idx, axis_sep_db = identify_steering_axis(gyro_dps, gyro_bias, active_window)

    rate_dps_corrected = gyro_dps[:, axis_idx] - gyro_bias[axis_idx]
    angle_deg = integrate_angle(t_s, rate_dps_corrected)

    gt = _load_ground_truth(gt_path)
    measured = sample_at_times(t_s, angle_deg, gt["t_seconds"].to_numpy())
    step_results: list[StepResult] = []
    for (_, row), m in zip(gt.iterrows(), measured):
        target = float(row["angle_deg"])
        step_results.append(StepResult(
            target_deg=target,
            measured_deg=float(m),
            abs_error_deg=float(abs(m - target)),
            t_seconds=float(row["t_seconds"]),
            note=str(row.get("note", "")),
        ))

    drift = measure_drift_60s(t_s, angle_deg, static_windows, bias_window)
    rate_seg = measure_constant_rate(t_s, rate_dps_corrected)

    # Static RMS post-bias from the same window we used to estimate bias.
    a, b = bias_window
    static_rms_after = float(np.sqrt(
        np.mean((gyro_dps[a:b, axis_idx] - gyro_bias[axis_idx]) ** 2)
    ))

    failure_reasons: list[str] = []
    if step_results and max(s.abs_error_deg for s in step_results) > PASS_STEP_ERROR_DEG:
        worst = max(step_results, key=lambda s: s.abs_error_deg)
        failure_reasons.append(
            f"step input error {worst.abs_error_deg:.2f}° at t={worst.t_seconds:.1f}s "
            f"(target {worst.target_deg:.0f}°) exceeds tolerance ±{PASS_STEP_ERROR_DEG}°"
        )
    if not np.isnan(drift) and drift > PASS_DRIFT_DEG_60S:
        failure_reasons.append(
            f"60s drift {drift:.2f}° exceeds tolerance {PASS_DRIFT_DEG_60S}°"
        )
    if static_rms_after > PASS_STATIC_RMS_DPS:
        failure_reasons.append(
            f"post-bias static RMS {static_rms_after:.2f} deg/s exceeds {PASS_STATIC_RMS_DPS}"
        )
    if axis_sep_db < PASS_AXIS_SEPARATION_DB:
        failure_reasons.append(
            f"axis separation {axis_sep_db:.1f} dB below {PASS_AXIS_SEPARATION_DB} dB; "
            f"camera may not be on-axis with the steering rotation"
        )

    return CalibrationResult(
        unit=unit_dir.name,
        recorded_at=dt.date.today().isoformat(),
        sample_rate_hz=float(fs),
        imu_orientation=orientation,
        axis_used_for_steering=("gx", "gy", "gz")[axis_idx],
        axis_separation_db=float(axis_sep_db),
        gyro_bias_dps=[float(x) for x in gyro_bias],
        accel_bias_mps2=[float(x) for x in accel_bias_residual],
        static_rms_after_bias_dps=static_rms_after,
        step_input_validation=step_results,
        drift_60s_deg=float(drift) if not np.isnan(drift) else float("nan"),
        constant_rate_segment_dps=rate_seg,
        passed=not failure_reasons,
        failure_reasons=failure_reasons,
        source=source,
    )


def _result_to_jsonable(result: CalibrationResult) -> dict:
    d = asdict(result)
    # Replace NaN with None for JSON correctness.
    if isinstance(d.get("drift_60s_deg"), float) and np.isnan(d["drift_60s_deg"]):
        d["drift_60s_deg"] = None
    rs = d.get("constant_rate_segment_dps", {})
    for k, v in list(rs.items()):
        if isinstance(v, float) and np.isnan(v):
            rs[k] = None
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("unit_dir", type=Path,
                    help="data/calibration/<UNIT>/ — must contain original.gyroflow and ground_truth.csv")
    ap.add_argument("--source", choices=("wheel", "helmet"), default="wheel",
                    help="which mount this calibration is for (used by extract_imu)")
    args = ap.parse_args()

    if not args.unit_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.unit_dir}")

    print(f"Calibrating {args.unit_dir.name} ({args.source})")
    result = calibrate_unit(args.unit_dir, source=args.source)

    out = args.unit_dir / "result.json"
    with out.open("w") as f:
        json.dump(_result_to_jsonable(result), f, indent=2)

    status = "PASS" if result.passed else "FAIL"
    print(f"\n  {status}")
    print(f"  sample rate:        {result.sample_rate_hz:.1f} Hz")
    print(f"  steering axis:      {result.axis_used_for_steering} "
          f"(separation {result.axis_separation_db:.1f} dB)")
    print(f"  gyro bias (deg/s):  {result.gyro_bias_dps}")
    print(f"  static RMS:         {result.static_rms_after_bias_dps:.3f} deg/s")
    if result.step_input_validation:
        worst = max(result.step_input_validation, key=lambda s: s.abs_error_deg)
        print(f"  worst step error:   {worst.abs_error_deg:.2f}° "
              f"(target {worst.target_deg:.0f}°, t={worst.t_seconds:.1f}s)")
    if not np.isnan(result.drift_60s_deg) if result.drift_60s_deg is not None else False:
        pass  # NaN handled above
    if result.drift_60s_deg is not None and not (isinstance(result.drift_60s_deg, float)
                                                  and np.isnan(result.drift_60s_deg)):
        print(f"  drift over 60s:     {result.drift_60s_deg:.2f}°")
    if result.failure_reasons:
        print("  reasons for failure:")
        for r in result.failure_reasons:
            print(f"    - {r}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
