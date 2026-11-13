# Steering validation: evidence and limits

**Final status for the concluded steering experiment:** the real-data pipeline
produced estimates, but independent steering-angle accuracy was not established.
There is no remaining track-capture commitment.

## Two different questions

The May 11 session shows that the camera and logger can be decoded, aligned and
turned into per-lap output. Its logger provides GPS yaw for chassis subtraction.
That is a processing demonstration, not a comparison against known steering.

A separate comparison was proposed against the MyChron 6 2T `Steering Angle`
channel. The experiment did not complete a matching Pro Video camera/6 2T
reference capture. The channel's physical origin and calibration were also not
established. It must not be treated as ground truth merely because it is named
`Steering Angle`.

## Evidence retained

| Evidence | What it supports | What it does not establish |
|---|---|---|
| [Historical lap plots](results/session_20260511_xtreme/) | The real-data processing path ran | Absolute steering accuracy or driving improvement |
| [Synthetic example](examples/steering_demo.py) | Subtraction, integration and filtering with known inputs | Automatic geometry, time alignment or real sensor accuracy |
| [Regression tests](tests/) | Behavior under the fixtures' assumptions | Performance on unseen track recordings |
| [XRK characterization](analysis/characterize_steering_source.py) | Examining the logged channel's dynamics | Sensor identity, calibration or independence from every possible model |

The historical characterization found a weak fit to a simple yaw-rate/speed
model and nonzero variation at low speed. Those observations argue against that
particular model; they do not prove a calibrated physical steering sensor.
The low-speed subset was below 2 m/s, not an independently observed stationary
wheel sweep.

## Signal-processing limits

The published real-session path uses camera IMU **and logger GPS yaw**. It is
not a validated camera-only measurement. Its column estimate depends on which
motion dominates gyro covariance; its gravity estimate needs quiet samples.
Chassis pitch/roll, mounting movement, yaw-reference errors and timing errors
can contaminate the recovered steering rate.

Integration accumulates residual errors. The default 0.05 Hz high-pass filter
reduces drift but also removes slowly varying and sustained steering. Therefore
the plotted angle is a band-limited estimate, not an absolute wheel position.
The synthetic example deliberately includes a held angle to expose this.

Synchronization correlation measures agreement between the signals used to find
the clock offset. It is not a steering-accuracy score, and fitting lag, gain and
offset afterward must not be reported as raw sensor accuracy.

## Retained comparison tool

For independent follow-up with suitable recordings:

```bash
python -m analysis.validate_steering wheel.gyroflow session_6_2t.xrk \
  --all --out-dir results/validation
```

The tool aligns the streams, compares the logger-assisted camera estimate with
the XRK channel, and reports angle/rate correlation, fitted gain and offset,
residual lag, and RMS **after** the gain/offset fit.

Agreement would justify further investigation of calibration and shared error
sources. Disagreement would require checking both sources, the model and timing.
Neither source wins by assumption. The CLI reports waveform agreement without
claiming that correlation establishes physical accuracy.

## Recording-format finding

In the Go 3S files examined during development, `PRO_VID_` recordings retained
raw gyro while ordinary `VID_` recordings with in-camera stabilization did not.
Filename and metadata flags are useful hints; an actual IMU decode is the
decisive check. Re-encoded footage may have lost the original telemetry.

Original camera and logger files are not included in this public case study.
Nothing in the closeout authorizes deleting original recordings.
