# SAM3Metal Testing Guide

## Overview

Complete validation suite for SAM3Metal implementation, ensuring correctness and performance.

## Test Structure

### 1. **Unit Tests** (Component-Level)
- `RoPETests.swift` - Position encoding kernel validation
- Validates Metal kernel correctness vs PyTorch
- MSE tolerance: < 1e-5

### 2. **Integration Tests** (End-to-End)
- `EncoderValidationTests.swift` - Full encoder pipeline
- Compares Metal output vs PyTorch reference
- MSE tolerance: < 1e-3

### 3. **Performance Benchmarks**
- Measures actual FPS and latency
- Compares against baseline (8.5s) and target (170ms)
- Reports speedup multiplier

## Running Tests

### Quick Test (Swift only)
```bash
cd SAM3Metal
swift test
```

Note: For Mac App Store distribution, the shipped product must not rely on Python.
Python is used only offline to generate reference artifacts (ground truth) and is not part of the Swift package targets.

For performance and App Store compliance, runtime should load packed weights (`.wts`) generated offline.
Use `scripts/python_extraction/convert_npz_to_wts.py` to convert an exported `.npz` checkpoint into a `.wts` file.

### Full Validation Suite
```bash
./scripts/validate_sam3metal.sh
```

This will:
1. Generate PyTorch reference outputs
2. Convert weights to binary format
3. Build SAM3Metal package
4. Run all tests
5. Generate performance report

## Test Files

```
SAM3Metal/Tests/SAM3MetalTests/
├── RoPETests.swift                # Metal RoPE validation
└── EncoderValidationTests.swift   # Full encoder validation

SAM3Metal/scripts/python_extraction/
├── generate_encoder_reference.py  # PyTorch ground truth (offline)
└── RoPETests.py                   # PyTorch RoPE reference (offline)
```

## Expected Results

### Correctness
- **Patch Embedding**: MSE < 1e-3
- **Transformer Block**: MSE < 1e-3
- **Full Encoder**: MSE < 1e-3

### Performance (1024x1024 input)
- **Baseline**: 8,500ms (Core ML CPU)
- **Minimum**: 4,250ms (2x speedup)
- **Target**: 170ms (50x speedup)
- **Best Case**: <100ms (85x speedup)

## Interpreting Results

### Success Criteria
✅ All tests pass  
✅ MSE < tolerance  
✅ Performance > 2x baseline  

### Warning Signs
⚠️ MSE > 1e-2 → Weight loading issue  
⚠️ Performance < baseline → Profile needed  
⚠️ Crashes → Check Metal compatibility  

## Debugging Failed Tests

### High MSE (>1e-2)
1. Verify weights loaded correctly
2. Check tensor shapes match PyTorch
3. Validate data type conversions (Float16/32)

### Poor Performance
1. Run Instruments Metal System Trace
2. Check ANE utilization %
3. Profile hotspots (likely MPS operations)
4. Verify threadgroup sizes

### Crashes
1. Check Metal feature set compatibility
2. Verify buffer sizes
3. Validate texture formats

## Continuous Testing

During development:
```bash
# Watch mode (rerun on file change)
swift test --filter EncoderValidationTests

# Specific test
swift test --filter testEncoderPerformance

# With verbose output
swift test -v
```

## Next Steps After Validation

1. ✅ **Tests Pass** → Proceed to Decoder implementation
2. ⚠️ **MSE Issues** → Debug weight loading
3. ⚠️ **Perf Issues** → Profile and optimize

## Reference Data

All reference files stored in `/tmp/`:
- `encoder_reference.npz` - Full encoder ground truth
- `patch_embed_reference.npz` - Patch embedding
- `transformer_block_reference.npz` - Single block
- `rope_reference.npz` - RoPE kernel

## Performance Tracking

| Date | Latency | Speedup | Notes |
|------|---------|---------|-------|
| Target | 170ms | 50x | Design goal |
| TBD | TBD | TBD | First benchmark |
