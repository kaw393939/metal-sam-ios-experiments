# Sprint 08 — Profiling, GPU Counters, and Regression Gates

## Goal
Make performance and accuracy regressions impossible to miss.

## Why
Transformer performance work is brittle: one extra sync or allocation can wipe out 2× gains.

## Platform levers
Metal provides GPU counters and counter sample buffers for runtime metrics.
Reference: https://developer.apple.com/documentation/metal (see “GPU counters and counter sample buffers”).

## Deliverables
1. A standard benchmark harness:
   - p50/p90 latency
   - peak memory
   - GPU counters snapshots
2. A correctness harness:
   - per-module numerical checks
   - end-to-end mask checks on your asset suite
3. CI-friendly “budget tests”:
   - fail if latency > threshold
   - fail if allocations occur in hot loop

## Tasks
- Add instrumentation macros:
  - time spans per stage (encoder/neck/prompt/decoder/tracker)
  - record command buffer completion times
- Add GPU counter capture hooks in dev builds.
- Create a minimal “golden set”:
  - small images/videos with stored expected outputs

## Acceptance criteria
- You can answer in 5 minutes:
  - what stage dominates time
  - whether you’re compute or bandwidth bound
  - whether a change changed accuracy

## Risks
- Counter capture adds overhead; keep it dev-only.

## Notes
This sprint should run in parallel with others, but it’s worth doing as a dedicated push.