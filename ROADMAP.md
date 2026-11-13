# Steering experiment closeout

**Concluded · 13 November 2026**

The steering experiment is complete as an exploratory engineering case study.
It is not complete as a calibrated steering instrument or a demonstrated
coaching tool. No additional track sessions or feature development are planned
for this experiment.

## What is retained

The [README](README.md) presents the question, method and outcome. The
[real-session results](RESULTS.md) and [executed notebook](analysis.ipynb)
preserve the historical analysis. A [synthetic example](examples/steering_demo.py)
lets a reader inspect the core signal processing without private recordings.
The implementation and its existing tests remain available for study.

## Where the original scope ended

| Workstream | Disposition |
|---|---|
| Camera IMU ingestion and logger synchronization | Implemented; historical real-session output retained |
| Per-lap steering estimates and plots | Implemented; absolute accuracy remains unverified |
| Known-angle bench calibration | Harness exists; physical calibration not established by this case study |
| Independent steering-reference comparison | Harness exists; a matched reference capture was not completed |
| Tire observation, helmet look-ahead, multi-stream coaching and field beta | Removed from this experiment's active scope |

The original backlog is available in Git history. Its unchecked tasks are not
remaining commitments. There is no pending request to collect more track data,
perform a clap test, or build a ghost comparison.

## Why stop here

The intended outcome was practical improvement in driving. The experiment
produced useful engineering work, but it did not establish an actionable
coaching benefit. Continuing would require a separate validation effort for
sensor accuracy, mounting stability and interpretation.

If someone independently continues the method, the first question is measurement
validity: compare against a calibrated steering reference, including long
held angles, synchronization error and drift. The [validation notes](STEERING_VALIDATION.md)
explain why correlation or a plausible plot alone cannot settle that question.
