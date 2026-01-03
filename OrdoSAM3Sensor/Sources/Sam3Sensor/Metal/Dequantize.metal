//
//  Dequantize.metal
//  SAM3Metal
//
//  Int8 weight dequantization kernel
//

#include <metal_stdlib>
using namespace metal;

/// Dequantize Int8 weights using palette lookup
///
/// This kernel reconstructs Float16 weights from Int8 palettized format.
/// Each weight is stored as a uint8 index into a 256-entry palette.
///
/// Performance: ~1ms for full model decompression (one-time at load)
kernel void dequantize_int8(
    device const half* palette [[buffer(0)]],     // [256] palette values
    device const uint8_t* indices [[buffer(1)]],  // [...] indices into palette
    device half* output [[buffer(2)]],            // [...] decompressed weights
    constant uint& count [[buffer(3)]],           // total number of weights
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    
    // Lookup palette value using index
    uint8_t idx = indices[gid];
    half value = palette[idx];
    
    // Write decompressed value
    output[gid] = value;
}

/// Batch dequantize multiple tensors
///
/// More efficient version that processes multiple tensors in one dispatch
kernel void dequantize_int8_batch(
    device const half* palette [[buffer(0)]],     // [256] shared palette
    device const uint8_t* indices [[buffer(1)]],  // [...] all indices concatenated
    device half* output [[buffer(2)]],            // [...] all outputs
    constant uint* offsets [[buffer(3)]],         // [num_tensors] start offset for each
    constant uint* sizes [[buffer(4)]],           // [num_tensors] size of each tensor
    constant uint& num_tensors [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    // Determine which tensor this thread belongs to
    uint tensor_idx = 0;
    uint local_idx = gid;
    
    for (uint i = 0; i < num_tensors; i++) {
        if (local_idx < sizes[i]) {
            tensor_idx = i;
            break;
        }
        local_idx -= sizes[i];
    }
    
    if (tensor_idx >= num_tensors) return;
    
    // Get global index
    uint global_idx = offsets[tensor_idx] + local_idx;
    
    // Dequantize
    uint8_t idx = indices[global_idx];
    output[global_idx] = palette[idx];
}

/// Optimized dequantize with threadgroup memory
///
/// Loads palette into fast on-chip memory for better performance
kernel void dequantize_int8_optimized(
    device const half* palette [[buffer(0)]],
    device const uint8_t* indices [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    threadgroup half* shared_palette [[threadgroup(0)]],  // On-chip cache
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {
    // Cooperatively load palette to threadgroup memory
    // Each thread loads 1 entry (256 threads needed)
    if (lid < 256) {
        shared_palette[lid] = palette[lid];
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Process weights using cached palette
    if (gid < count) {
        uint8_t idx = indices[gid];
        output[gid] = shared_palette[idx];
    }
}
