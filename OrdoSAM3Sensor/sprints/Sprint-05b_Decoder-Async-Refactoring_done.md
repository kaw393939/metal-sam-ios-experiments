# Sprint 05b — Decoder Async Refactoring

## Goal
Eliminate blocking sync points in MaskDecoder by converting to async CompiledGraphCache pattern, enabling interactive decoder performance.

## Why
Sprint 05 identified that MaskDecoder.forward() has blocking calls (graph.run() and waitUntilCompleted()) that prevent interactive prompt performance. This is Sprint 01 technical debt.

## Current State
- MaskDecoder.forward() builds graph and executes synchronously
- Line 334: `graph.run()` blocks until GPU completes
- Line 351: `waitUntilCompleted()` blocks export
- Prevents overlapping decoder operations
- Interactive prompts feel slow

## Deliverables
1. MaskDecoder.buildGraph() method (graph construction only)
2. MaskDecoder.addFeeds() helper method
3. SAM3Predictor.predict() updated to use CompiledGraphCache
4. All blocking calls removed
5. Tests verify async execution

## Tasks
- Split forward() into buildGraph() + execution
- Add addFeeds() helper for feed preparation
- Update SAM3Predictor.predict() to use CompiledGraphCache pattern
- Remove graph.run() blocking call
- Remove waitUntilCompleted() blocking call
- Verify decoder correctness maintained
- Measure decoder latency improvement

## Acceptance Criteria
- No blocking calls in decoder hot path
- Decoder latency < 100ms (target)
- Mask IoU matches reference
- Interactive prompts feel instantaneous

## Risks
- Complex refactoring (200+ lines)
- Multiple file coordination
- Potential regressions

## Notes
Follow SAM3Encoder pattern from Sprint 01. Estimated effort: 2-3 hours.

## Reference
See Sources/Sam3Sensor/MaskDecoder_ASYNC_TODO.md for detailed implementation guide.
