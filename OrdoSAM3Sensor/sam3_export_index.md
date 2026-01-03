# SAM3 Complete Source Export Index

**Generated**: Fri Jan  2 11:17:12 EST 2026

## Files Included

- **AttentionKernels.swift** (      69 lines)
- **Benchmarking.swift** (     126 lines)
- **CheckInterface.swift** (       7 lines)
- **GeometryEncoder.swift** (     467 lines)
- **IoUMetrics.swift** (      96 lines)
- **MPSAttention.swift** (     259 lines)
- **MPSGraphAttention.swift** (     193 lines)
- **MaskDecoder.swift** (     579 lines)
- **MemoryBank.swift** (      62 lines)
- **ModelLoader.swift** (     161 lines)
- **MultiObjectMemoryBank.swift** (      83 lines)
- **OptimizationInfrastructure.swift** (     351 lines)
- **PromptEncoder.swift** (     462 lines)
- **RoPE.swift** (     127 lines)
- **SAM3Encoder.swift** (     807 lines)
- **SAM3EncoderConfig.swift** (      64 lines)
- **SAM3MemoryAttention.swift** (     324 lines)
- **SAM3MemoryEncoder.swift** (      84 lines)
- **SAM3MetalPipeline.swift** (     178 lines)
- **SAM3Neck.swift** (     271 lines)
- **SAM3Predictor.swift** (     524 lines)
- **SAM3Tracker.swift** (     176 lines)
- **Sam3Debug.swift** (      45 lines)
- **Sam3Log.swift** (      26 lines)
- **TokenPruner.swift** (     197 lines)
- **TwoWayTransformer.swift** (     829 lines)
- **ViTEncoder.swift** (    1512 lines)
- **WeightMapper.swift** (     157 lines)
- **WeightsLoader.swift** (     276 lines)

## Summary
- **Total Files**: 29
- **Total Lines**: 8512
- **Export File**: `sam3_complete_export.swift`

## File Descriptions

### Core Components
- **ViTEncoder.swift**: Vision Transformer encoder with fused attention blocks
- **MaskDecoder.swift**: Mask prediction decoder with upsampling
- **SAM3Predictor.swift**: Main API for image encoding and mask prediction
- **SAM3Encoder.swift**: High-level encoder wrapper

### Attention & Transformer
- **MPSAttention.swift**: Flash attention implementation
- **MPSGraphAttention.swift**: Graph-based attention
- **TwoWayTransformer.swift**: Bidirectional transformer for decoder
- **AttentionKernels.swift**: Metal compute kernels for attention

### Optimization Infrastructure
- **OptimizationInfrastructure.swift**: CompiledGraphCache, BufferAllocator
- **TokenPruner.swift**: Dynamic token pruning for efficiency
- **RoPE.swift**: Rotary position embeddings

### Memory & Tracking
- **SAM3Tracker.swift**: Multi-object tracking
- **MemoryBank.swift**: Frame memory for temporal coherence
- **MultiObjectMemoryBank.swift**: Multi-object memory management
- **SAM3MemoryAttention.swift**: Memory attention mechanism
- **SAM3MemoryEncoder.swift**: Memory encoding

### Utilities
- **WeightsLoader.swift**: Binary weights loading
- **WeightMapper.swift**: Weight key mapping
- **GeometryEncoder.swift**: Position encoding
- **PromptEncoder.swift**: Point/box prompt encoding
- **Sam3Log.swift**: Logging utilities
