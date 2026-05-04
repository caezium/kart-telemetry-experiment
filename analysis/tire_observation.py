"""
Tire observation from a wheel-mounted Go 3S video, lens angled to keep the
front tire in frame.

Status: STUB / sketch. Inside-front-lift detection is the first target,
because it has a clean visual signature (the tire's bottom edge separating
from the ground) and high tuning value for kart setup.

Approach:
    1. Load Gyroflow-stabilized output (wheel-cam de-rotated to world frame).
       The stabilization uses the same IMU we already extracted, so the
       tire stays roughly fixed in the frame across the session.
    2. Define a mask polygon around the tire's expected location.
    3. For each frame, find the bottom edge of the tire (gradient + Hough
       line, or a learned segmenter — start with classical CV).
    4. Estimate the gap between tire-bottom and ground reference.
    5. Threshold + temporal smoothing → lift events.

Validation: hand-label 5-10 corners across a session. Train threshold
parameters against the labels until detector precision/recall > 85%.

This file currently contains the data-flow scaffolding only; the CV
pipeline is left for Phase 2 of ROADMAP.md.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class LiftEvent:
    t_start: float
    t_end: float
    duration_s: float
    peak_lift_px: float


def detect_lift_events(stabilized_video_path: Path, tire_roi: tuple[int, int, int, int]) -> list[LiftEvent]:
    """
    Args:
        stabilized_video_path: Gyroflow-exported MP4 with IMU stabilization applied.
        tire_roi: (x, y, w, h) bounding the tire in the de-rotated frame.

    Returns:
        list of LiftEvent

    NOT IMPLEMENTED. See module docstring for approach.
    """
    raise NotImplementedError(
        "Tire CV pipeline scheduled for Phase 2. "
        "See ROADMAP.md and the module docstring for the planned approach."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session_dir", type=Path)
    args = ap.parse_args()
    print(f"Tire observation is a Phase 2 deliverable. Session dir: {args.session_dir}")
    print("See ROADMAP.md for the build plan.")


if __name__ == "__main__":
    main()
