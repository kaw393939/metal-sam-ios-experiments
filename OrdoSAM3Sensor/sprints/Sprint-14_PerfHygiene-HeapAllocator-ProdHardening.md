# Sprint 14 — Performance Hygiene + Heap Allocator + Production Hardening (P2)

**Goal**: Remove avoidable overhead, stabilize runtime behavior, and make memory allocation deterministic and fast.

---

## Scope

### 15) Remove per-block `Date()` / verbose logging from hot loops

**Tasks**:
- Ensure no `Date()` creation or verbose logs in per-block loops.
- Keep logging behind flags and ensure the default is silent.

**Acceptance**:
- Hot loop has zero timestamp calls in steady state.

---

### 16) Replace `fatalError` with structured failures in production paths

**Tasks**:
- Keep `fatalError` in tests.
- In runtime/production APIs, prefer `throws` / error returns.

**Acceptance**:
- No user-triggerable `fatalError` in production entry points.

---

### 17) Make heap-backed buffer allocator real

**Tasks**:
- Ensure private buffers come from heap/pool consistently.
- Recycling is deterministic (no growth without bound).
- Add metrics/counters for allocations vs reuses.

**Acceptance**:
- Significant reduction in per-frame allocations.
- Stable memory behavior across many iterations.

---

## Gates

- Benchmark gate:
  - run encoder perf for N frames and report:
    - median/p90 ms
    - allocations/frame
    - peak heap usage

---

## Exit Criteria

- Logging overhead removed from hot path.
- Production code avoids fatal termination.
- Heap allocator reduces alloc churn and stabilizes perf.
