# Sprint 10 — Encoder Config + Precision + Kernel Contracts (P0)

**Goal**: Remove correctness blockers by making the encoder match the real SAM3 checkpoint configuration and enforcing consistent dtype/kernel contracts end-to-end.

**Primary risks addressed**: architecture mismatch, precision mismatch, kernel signature mismatch, silent OOB / wrong dispatch sizing.

---

## Scope

### 1) Single source of truth config

Create a single `SAM3EncoderConfig` (or similar) that is the only place encoder topology and IO sizes live.

**Target checkpoint constants (per `WeightMapper`)**:
- `embedDim = 1024`
- `numHeads = 16`
- `numBlocks = 32`
- `patchSize = 14`
- `inputSize = 1008`

**Must cover**:
- `embedDim` (e.g. 1024)
- `numHeads` (e.g. 16)
- `numBlocks` (e.g. 32)
- `patchSize` (e.g. 14)
- `inputSize` (e.g. 1008)
- `mlpRatio` (if checkpoint implies)
- global/windowed attention block schedule
- RoPE params (`dimPerHead`, rope shape conventions)
- pruning parameters (`keepK`, prune block index)

**Update all of these to derive from config**:
- `ViTEncoder`
- `PatchEmbedding`
- `TransformerBlock`
- `RoPE` generator + buffers
- `TokenPruner`
- `Neck`/`NeckLayer` (input/output dims and reshape assumptions)

**Acceptance**:
- Weight loading keys + tensor shapes line up with the checkpoint without adapter hacks.
- No hard-coded `dim=768`, `patchSize=16`, `numBlocks=24` in the encoder hot path.

---

### 2) End-to-end half precision contract (F16)

Make the F16 path internally consistent across:
- texture → buffer flatten
- transformer blocks
- residual add
- neck

**Tasks**:
- Implement `texture_to_buffer_flat_half` (or `half2/half4` packed variant) and select it when `useHalfPrecision=1`.
- Fix buffer sizing everywhere to use `bytesPerElement` instead of `MemoryLayout<Float>.stride`.
- Ensure LN/MLP/Attention feed + output dtypes match (no reinterpretation / “treat half as float”).

**Acceptance**:
- When `useHalfPrecision=true`, no allocations sized as float32 for activations.
- No compute kernels taking `device float*` when output buffers are half.

---

### 3) Unify residual-add kernel contract

Pick one contract and update all call sites.

**Option A (recommended): out-of-place**
- `add_residual(a, b, out, count)` for both float and half
- bounds-safe (kernel checks `i < count`)

**Option B: in-place**
- `add_inplace(a, b, count)` but still pass `count` and bounds check

**Must update**:
- pos embed add
- block residuals (both residuals inside transformer)
- any other add kernels using inconsistent signatures

**Acceptance**:
- Exactly one residual-add entry point per dtype.
- All Swift call sites match the chosen kernel signature.

---

### 4) Fix element counts for float16

Remove `a.length / 4` style element-count logic.

**Rule**:
- `count = buffer.length / bytesPerElement`

**Acceptance**:
- All dispatch sizing uses bytesPerElement (F16 safe).

---

### 5) Fused TransformerBlock as correctness-preserving fusion

Implement fused block with the exact sequence:

`LN1 → QKV → RoPE → attention → proj → residual → LN2 → MLP → residual`

**Approach**:
- Reuse the existing attention graph structure (QKV split + RoPE) as the reference.
- Fusion must be behind a flag (e.g. `SAM3_FUSED_BLOCK=1`) until parity is proven.

**Acceptance**:
- For fixed-seed inputs, fused output matches unfused within tolerance:
  - F32: max error / MSE under tight threshold
  - F16: relaxed threshold but stable

---

## Gates

### Correctness gate
- Add tests:
  - `testFusedBlockMatchesUnfused_F32`
  - `testFusedBlockMatchesUnfused_F16`

### Performance gate
- Measure `testEncoderPerformance` before/after.
- Target: meaningful reduction in `blocks_*` timing without regressions elsewhere.

---

## Exit Criteria

- Encoder topology derives from config and matches checkpoint.
- F16 is coherent end-to-end.
- Residual contract is unified and dtype-safe.
- Fused block passes parity tests (even if left behind a flag).

---

## Minimal acceptance checklist (must pass)

- Encoder config matches checkpoint constants (1024/16/32/14/1008).
- Half precision is consistent: flatten → blocks → neck agree on dtype; no float-stride sizing when F16 enabled.
- Residual add uses one kernel contract everywhere (bounds-safe; passes `count`).
- Fused block matches unfused numerically (test + tolerance), then optimize.
