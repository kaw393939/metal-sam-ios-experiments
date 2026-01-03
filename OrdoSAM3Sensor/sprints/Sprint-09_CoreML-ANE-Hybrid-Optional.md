# Sprint 09 — Core ML / ANE Hybrid (Optional, if it Wins)

## Goal
If (and only if) it improves latency/energy, selectively offload subgraphs to Core ML / ANE while keeping the rest as Metal.

## Why
Core ML can leverage CPU/GPU/Neural Engine automatically. Newer APIs expose compute devices and compute plans.
Reference: https://developer.apple.com/documentation/coreml

## Deliverables
1. A proof-of-value prototype:
   - Offload one heavy component (e.g., MLP or QKV GEMM blocks) to Core ML.
   - Measure end-to-end impact vs pure Metal/MPSGraph.
2. Device control:
   - Use compute policies / compute devices to steer execution.

## Tasks
- Identify candidates:
  - large GEMMs where Core ML can hit ANE fast paths.
- Create model packages:
  - use `.mlpackage` for modular deploy.
- Integrate with Swift concurrency:
  - ensure Core ML execution doesn’t reintroduce blocking on the critical path.

## Acceptance criteria
- Must beat the pure path in either:
  - latency, or
  - energy, or
  - memory footprint

## Risks
- Core ML can be a black box; if it adds overhead or forces format conversions, it may lose.

## Notes
Because you want zero back-compat, you can require devices/OS versions where ANE behavior is best and drop everything else.