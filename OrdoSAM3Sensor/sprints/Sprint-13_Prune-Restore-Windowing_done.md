# Sprint 13 — Token Pruning + Restore + Windowing Rules (P1)

**Goal**: Make pruning/restore deterministic and dtype/layout safe, and ensure windowed-vs-global attention rules are consistent pre/post prune.

---

## Scope

### 13) Pruner dtype-safe + layout-safe + deterministic

**Tasks**:
- Ensure prune/gather/scatter preserves:
  - deterministic token ordering
  - dtype (F16 stays F16; RoPE stays F32)
  - correct shapes
- Add a round-trip test:
  - `restoreSpatial(prune(x)) ≈ x` for controlled fixtures / identity-like setup

**Acceptance**:
- Round-trip passes under tolerance.
- No dtype reinterpretation bugs.

---

### 14) Windowed attention rules post-prune

**Rule**:
- If pruned, force global attention (as current behavior), but ensure all execution paths respect it:
  - unfused block
  - fused block

**Acceptance**:
- No accidental windowing after prune.

---

## Gates

- Add tests:
  - prune/restore round-trip
  - pruned path uses global attention (assert via config or instrumentation)

---

## Exit Criteria

- Pruning is stable, deterministic, and dtype-correct.
- Windowed/global rules are enforced consistently.
