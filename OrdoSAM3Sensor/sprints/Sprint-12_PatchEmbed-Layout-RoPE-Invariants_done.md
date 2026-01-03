# Sprint 12 — PatchEmbed + Layout + RoPE Invariants (P1)

**Goal**: Fix IO/layout invariants so the model’s tensor contracts are explicit and correct (token-major vs NHWC), and patch embedding + RoPE match the checkpoint and the graph consumers.

---

## Scope

### 9) PatchEmbedding output texture format + storage

**Tasks**:
- Do not inherit output pixel format from input texture.
- Pick a known-good format (typically `.rgba16Float`) and enforce it.
- Prefer `.private` storage for GPU locality (later heap-backed).

**Acceptance**:
- Patch embedding output always has consistent format and intended storage mode.

---

### 10) PatchEmbedding weight ownership via data source

**Problem**: raw-pointer weight passing is fragile.

**Tasks**:
- Use `MPSCNNConvolutionDataSource` (existing `PatchEmbedDataSource`) for weight/bias.
- Support float16 cleanly.

**Acceptance**:
- No raw-pointer lifetime hazards.
- Patch embed weights can be swapped/reloaded safely.

---

### 11) Make seq↔spatial layout explicit

**Problem**: Transformer operates in `[B,S,C]`, neck wants `[B,H,W,C]`.

**Tasks**:
- Choose canonical memory layout for “features buffer”.
- Add explicit reshape/reinterpret steps where layout changes.
- Ensure neck’s feed tensor shape matches the buffer layout.

**Acceptance**:
- No “assume it matches” code paths.
- Tests verify reshapes are consistent.

---

### 12) RoPE buffer format and generator match consumers

**Tasks**:
- Ensure RoPE generator emits **exact** layout required by attention:
  - dtype: float32
  - shape: `[S_eff, DPH/2, 2]`
- Remove/fix fallback sizing that hardcodes old dims.

**Acceptance**:
- RoPE generator + attention feed shapes match exactly for global/windowed.

---

## Gates

- Add tests:
  - patch embed output shape/format
  - seq↔spatial reshape roundtrip
  - rope shape contract tests for global + windowed

---

## Exit Criteria

- Patch embed format/storage is stable.
- Layout invariants are explicit.
- RoPE generator is contract-correct.
