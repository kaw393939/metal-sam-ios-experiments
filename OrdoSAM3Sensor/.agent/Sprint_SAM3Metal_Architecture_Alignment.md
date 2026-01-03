# SAM3Metal — Sprint: Architecture Alignment & Real Weight Validation

**Created**: 2025-12-30  
**Status**: Planning  
**Prime Directive**: Tests are the spec. Real weights must produce valid segmentation.

---

## Purpose

Align the SAM3Metal Swift implementation with the actual SAM3 architecture from Meta's checkpoint, enabling real weight loading and validated inference.

## Problem Statement

The current Swift implementation was built with **assumed dimensions** that don't match the real SAM3 checkpoint:

| Component | Current Swift | Real SAM3 Weights |
|-----------|--------------|-------------------|
| Vision embedDim | 768 | **1024** |
| MLP hidden | 3072 | **4736** |
| Attention | Separate Q,K,V | **Fused QKV** |
| Weight keys | `norm1.weight` | `backbone.vision_backbone.trunk.blocks.0.norm1.weight` |

## Reference Materials

- **Paper**: [Sensors/papers/sam3.txt](file:///Users/kwilliams/Projects/ordo/Sensors/papers/sam3.txt)
- **Python Implementation**: [Sensors/research/sam3/sam3/model/](file:///Users/kwilliams/Projects/ordo/Sensors/research/sam3/sam3/model/)
- **Weights**: [SAM3Metal/Resources/sam3_weights.npz](file:///Users/kwilliams/Projects/ordo/Sensors/SAM3Metal/Resources/sam3_weights.npz) (1102 tensors)
- **Test Assets**: `Sensors/research/sam3/assets/images/` (groceries.jpg, truck.jpg, test_image.jpg)

---

## Gates (Definition of Done)

### Gate A — Weight Loading
**Done when**:
- NPZ weights load successfully into Swift `ModelWeights` dictionary
- All 1102 weight tensors accessible by key
- Unit test verifies correct shapes for critical weights

### Gate B — Architecture Dimensions Match
**Done when**:
- ViTEncoder uses `embedDim=1024`, `mlpHidden=4736`
- Attention layers support fused QKV weights (3072 x 1024)
- Weight key mapping transforms `backbone.vision_backbone.trunk.blocks.N.X` → component format
- Unit tests verify tensor shapes match PyTorch reference

### Gate C — Encoder Produces Non-Zero Embeddings
**Done when**:
- Loading real weights into ViTEncoder succeeds
- Forward pass on test image produces valid embeddings
- Integration test compares output stats to PyTorch reference (mean, std, range)

### Gate D — End-to-End Segmentation
**Done when**:
- Full pipeline (Encoder → Decoder) runs with real weights
- Test image "truck.jpg" produces reasonable mask for "truck" prompt
- IoU > 0.5 against PyTorch reference mask
- E2E test saves visual output for verification

---

## Test Specification

### Unit Tests

#### Test 1 — NPZ Weight Loading
**File**: `SAM3MetalTests/WeightLoadingTests.swift`
**Asserts**:
- `ModelLoader.load(url:)` returns 1102 weight entries
- Key `backbone.vision_backbone.trunk.blocks.0.norm1.weight` exists with shape [1024]
- Key `backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight` exists with shape [3072, 1024]

#### Test 2 — Weight Key Mapping
**File**: `SAM3MetalTests/WeightKeyMappingTests.swift`
**Asserts**:
- `WeightMapper.encoderKey(block: 0, layer: "norm1.weight")` returns correct full path
- Mapper handles all layer types: norm, attn, mlp

#### Test 3 — ViTEncoder Dimensions
**File**: `SAM3MetalTests/ViTEncoderDimensionTests.swift`
**Asserts**:
- `ViTEncoder(embedDim: 1024)` initializes without error
- Output buffer size matches `[1, 64, 64, 1024]` for 1024px input
- MLP hidden dim is 4736 (4.625 × embedDim)

#### Test 4 — Attention Fused QKV
**File**: `SAM3MetalTests/AttentionQKVTests.swift`  
**Asserts**:
- `MPSGraphAttention` accepts fused QKV weight [3*embedDim, embedDim]
- Split into Q, K, V produces correct shapes [embedDim, embedDim] each
- Output matches unfused attention numerically

### Integration Tests

#### Test 5 — Encoder with Real Weights
**File**: `SAM3MetalTests/EncoderRealWeightsTests.swift`
**Asserts**:
- Load weights into ViTEncoder succeeds
- Forward pass on `truck.jpg` returns non-NaN embeddings
- Embedding statistics (mean, std) within expected range

#### Test 6 — Decoder with Real Weights
**File**: `SAM3MetalTests/DecoderRealWeightsTests.swift`
**Asserts**:
- Decoder loads `segmentation_head.*` weights
- Forward pass with dummy embeddings produces mask output
- Mask contains non-zero values

### E2E Tests

#### Test 7 — Full Pipeline on Truck Image
**File**: `SAM3MetalTests/E2E_TruckSegmentationTests.swift`
**Asserts**:
- Load `truck.jpg` from assets
- Run inference with prompt "truck"
- Output mask IoU > 0.5 against reference
**Evidence**:
- Save output mask to `.ordo-artifacts/tests/E2E_TruckSegmentation/<run-id>/mask.png`
- Save IoU score to `metrics.json`

#### Test 8 — Full Pipeline on Groceries Image  
**File**: `SAM3MetalTests/E2E_GroceriesSegmentationTests.swift`
**Asserts**:
- Load `groceries.jpg` from assets
- Run inference with prompt "banana"
- Output masks contain multiple instances
**Evidence**:
- Save annotated image to artifacts

---

## Implementation Phases

### Phase 1: Weight Infrastructure (Gate A)
1. [x] Add NPZ loading to ModelLoader
2. [ ] Create `WeightMapper` class for key transformation
3. [ ] Add weight shape validation utilities
4. [ ] Write unit tests 1-2

### Phase 2: Architecture Alignment (Gate B)
1. [ ] Update ViTEncoder to support configurable embedDim
2. [ ] Refactor `MPSGraphAttention` for fused QKV
3. [ ] Update MLP to use correct hidden dimension
4. [ ] Update all layer norm shapes
5. [ ] Write unit tests 3-4

### Phase 3: Real Weight Loading (Gate C)
1. [ ] Implement weight loading into ViTEncoder blocks
2. [ ] Add weight validation (shape checks on load)
3. [ ] Write integration test 5

### Phase 4: End-to-End Validation (Gate D)
1. [ ] Implement decoder weight loading
2. [ ] Create reference data from PyTorch
3. [ ] Write E2E tests 7-8
4. [ ] Document results and metrics

---

## How to Run

```bash
# Unit tests (fast, no weights required)
cd Sensors/SAM3Metal
swift test --filter WeightLoadingTests
swift test --filter ViTEncoderDimensionTests

# Integration tests (requires weights)
swift test --filter EncoderRealWeightsTests

# E2E tests (full validation)
swift test --filter E2E_

# All tests
swift test
swift test -c release
```

---

## Success Criteria

| Gate | Metric | Target |
|------|--------|--------|
| A | Weights loaded | 1102/1102 |
| B | Shape mismatches | 0 |
| C | Encoder output | Non-zero, valid range |
| D | IoU on test images | > 0.5 |
