# Sprint 01 — Async Execution + Graph Compilation (Kill `graph.run` Stalls)

## Goal
Eliminate global stalls caused by synchronous `MPSGraph.run(...)` and frequent `waitUntilCompleted()`. Establish a single, consistent, safe async execution model.

## Why this is first
Right now, multiple call sites fall back to `graph.run(...)` and/or block on command buffers (examples show ~seconds of stall). Until graph execution is reliably **encoded into your own command buffers**, you can’t pipeline encoder/decoder/tracker and you can’t reason about latency.

## Key platform leverage
MPSGraph is designed to compile graphs to executables and run them efficiently. The docs explicitly call out:
- Compile graph into an executable.
- Serialize/load executables (`.mpsgraphpackage`).
- Use execution descriptors to control scheduling/synchronization.

Reference: https://developer.apple.com/documentation/metalperformanceshadersgraph

## Current hot spots (from repo audit)
- Multiple `graph.run(...)` call sites across encoder, attention, token pruning, mask decoder.
- Multiple `waitUntilCompleted()` barriers across RoPE, prompt encoder, geometry encoder, weights loader.

## Deliverables
1. A unified execution utility that:
   - Builds graphs once.
   - Compiles to `MPSGraphExecutable`.
   - Executes via encoding on a provided `MTLCommandBuffer` (no implicit internal command buffers).
   - Supports completion handlers + shared events for scheduling.
2. Refactor the top offenders to use the executable path.
3. A regression test that detects reintroduction of blocking sync points.

## Tasks
- Replace per-module “run or encode” forks with a single implementation:
  - Standardize feed creation (`MPSGraphTensorData`) and output export.
  - Always use one `MTLCommandQueue` and caller-owned `MTLCommandBuffer`.
- Add safe synchronization:
  - Prefer `MTLSharedEvent` / `MPSGraphExecutionDescriptor` callbacks over `waitUntilCompleted()`.
  - Only block at the *API boundary* if the public API requires a synchronous result.
- Investigate and fix the crash you saw with async encode:
  - Typical failure modes: lifetime of `MPSGraphTensorData`, reusing placeholders incorrectly, exporting NDArrays after the command buffer is released, or mixing `graph.run` with caller-owned command buffers.
- Add an option for reduced precision fast math where acceptable:
  - MPSGraph provides reduced-precision fast-math hints (non-guaranteed). Use only after establishing correctness gates.

## Acceptance criteria
- Encoder path:
  - No `graph.run(...)` in the encoder/neck path.
  - No `waitUntilCompleted()` in the steady-state hot loop.
  - Measured end-to-end encoder latency is stable (low variance across runs).
- Correctness:
  - Existing Swift tests still pass.
  - Reference intermediate comparison remains within existing tolerances.

## Risks
- MPSGraph graph compilation may produce different numerics vs eager run; lock down tolerances per module.
- If ANE scheduling is unpredictable, you may need explicit compute policies (see Sprint 09).

## Notes / next
Once this sprint lands, you can start *pipelining*: frame N encoder overlaps with frame N-1 decoder/tracker, which is the first big step toward real-time video PCS.