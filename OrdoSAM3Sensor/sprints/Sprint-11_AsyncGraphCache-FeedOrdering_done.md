# Sprint 11 — Async Graph Cache + Deterministic Feed Ordering (P0)

**Goal**: Make graph execution and caching correct and performant under iteration by eliminating hidden sync points and ensuring deterministic executable feed ordering.

**Primary risks addressed**: CPU-side blocking from `graph.run`, incorrect results/crashes from feed ordering mismatch, lock contention during compile.

---

## Scope

### 6) Remove hidden sync points; prefer executable execution

**Problem**: `graph.run(...)` is usually blocking and kills throughput.

**Tasks**:
- Prefer compiled `MPSGraphExecutable` for all hot-path graphs.
- Avoid per-iteration graph construction.
- Ensure execution does not block on CPU (no waits in the inner loop).

**Acceptance**:
- Hot-path ops (LN/Attention/MLP/Neck) do not call `graph.run` in steady state.

---

### 7) Deterministic executable feed ordering

**Problem**: If feeds are passed via dictionary, executable expects a specific ordered input array.

**Tasks**:
- Cache `executable.feedTensors` (or equivalent ordering) alongside the executable.
- At runtime, build `[MPSGraphTensorData]` in that exact order.
- Add assert/logging for missing feed tensors.

**Acceptance**:
- No silent wrong answers due to swapped feeds.
- Repeat runs are stable (no intermittent correctness failures).

---

### 8) Compile outside locks; insert inside locks

**Problem**: Holding cache locks during compile increases contention and slows warm-up.

**Tasks**:
- Use double-checked pattern:
  1. fast check under lock
  2. compile without lock
  3. re-check + insert under lock

**Acceptance**:
- Cache lock is not held while building graphs or compiling executables.

---

## Gates

### Correctness gate
- Add a stress test that runs encoder N times and checks:
  - stable output shape
  - no crashes
  - deterministic results for fixed weights + fixed input

### Performance gate
- Compare `testEncoderPerformance` before/after:
  - warm-up time should drop (compile once)
  - steady-state ms/frame improves

---

## Exit Criteria

- All hot-path graphs use executables in steady state.
- Feed ordering is deterministic and validated.
- Cache locking is lightweight.
