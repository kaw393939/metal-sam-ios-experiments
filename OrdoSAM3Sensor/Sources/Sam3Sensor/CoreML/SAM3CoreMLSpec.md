# SAM3 Core ML ANE Conversion Specification

## Goal
Convert SAM3 Encoder Blocks 0-8 to Core ML for **Apple Neural Engine (ANE)** execution on M3 Air, achieving 2-3x speedup with thermal efficiency.

## Architecture Breakdown

### Input
- **Shape**: `[1, 5184, 1024]` (batch=1, tokens=72×72, embed_dim=1024)
- **Type**: FP16 (will be quantized to INT8 for ANE)
- **Source**: Output from patch embedding (Metal GPU)

### Blocks 0-8 (Each block contains)
1. **Layer Norm 1**
   - Input: `[1, 5184, 1024]`
   - Params: `norm1.weight`, `norm1.bias` (1024 each)
   
2. **Attention (QKV Fused)**
   - QKV Linear: `[1024, 3×1024]` → `[5184, 3072]`
   - Reshape to Q, K, V: each `[1, 16, 5184, 64]` (heads=16, dim=64)
   - RoPE applied (from Metal, precomputed)
   - Scaled Dot Product Attention
   - Output Linear: `[1024, 1024]`
   
3. **Residual Add**

4. **Layer Norm 2**

5. **MLP**
   - FC1: `[1024, 4736]` with GELU
   - FC2: `[4736, 1024]`
   
6. **Residual Add**

### Output
- **Shape**: `[1, 5184, 1024]`
- **Type**: FP16
- **Destination**: Handoff to Metal GPU for blocks 9-31

## Core ML Conversion Strategy

### Approach: PyTorch → Core ML Direct
1. **Extract Weights**: Load blocks 0-8 from safetensors
2. **Build PyTorch Module**: Create a `SAM3EarlyBlocks` nn.Module
3. **Trace with Example Input**: Use `torch.jit.trace`
4. **Convert to Core ML**: Use `coremltools.convert`
5. **Quantize to INT8**: Use `coremltools.optimize.coreml.OpLinearQuantizer`

### Key Optimizations for ANE
- **INT8 Quantization**: W8A8 (weights + activations)
- **Compute Units**: `.cpuAndNeuralEngine` (prefer ANE)
- **Precision**: `MLCOMPUTEPrecision.Float16` for non-quantized ops
- **Batch Size**: Fixed at 1 (no dynamic shapes for ANE)

## Integration with Metal Pipeline

### Handoff Point 1: Metal → Core ML
- **After**: Patch embedding + positional encoding (Metal)
- **Format**: `[1, 5184, 1024]` FP16 `MTLBuffer`
- **Mechanism**: Copy to `MLMultiArray` (minimize copy overhead)

### Handoff Point 2: Core ML → Metal  
- **Before**: Blocks 9-31 (Metal)
- **Format**: `[1, 5184, 1024]` FP16 `MTLBuffer`
- **Mechanism**: Use `MTLSharedEvent` for GPU-ANE synchronization

## Expected Performance (M3 Air)

| Component | Current (GPU) | Target (ANE) | Speedup |
|:----------|:-------------|:-------------|:--------|
| Blocks 0-8 | ~45ms | **15-20ms** | 2.2-3x |
| Handoff Overhead | 0ms | ~2-3ms | - |
| **Net Gain** | 45ms | 17-23ms | **~2x** |

## Implementation Files

1. `CoreML/SAM3EarlyBlocks.py` - PyTorch export script
2. `CoreML/SAM3ANEConverter.swift` - Core ML loader

3. `SAM3HybridEncoder.swift` - Orchestrates ANE + Metal GPU
4. `Tests/SAM3ANETests.swift` - Validation tests
