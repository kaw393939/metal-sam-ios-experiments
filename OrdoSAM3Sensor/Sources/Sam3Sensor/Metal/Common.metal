//
//  Common.metal
//  SAM3Metal
//
//  Shared utilities for all Metal kernels
//

#include <metal_stdlib>
using namespace metal;

/// Element-wise addition (for residual connections)
/// Contract: out = a + b, with explicit element count.
kernel void add_residual_half(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    output[gid] = a[gid] + b[gid];
}

/// Element-wise addition (for residual connections)
/// Contract: out = a + b, with explicit element count.
kernel void add_residual_float(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    output[gid] = a[gid] + b[gid];
}

/// GELU activation
/// GELU(x) = x * Φ(x) where Φ is Gaussian CDF
/// Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
inline half gelu(half x) {
    const half sqrt_2_over_pi = 0.7978845608h;
    const half coeff = 0.044715h;
    
    half x_cubed = x * x * x;
    half inner = sqrt_2_over_pi * (x + coeff * x_cubed);
    return half(0.5) * x * (half(1.0) + tanh(inner));
}

kernel void apply_gelu(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    output[gid] = gelu(input[gid]);
}

/// Layer Norm kernel
/// Normalize over last dimension: (x - mean) / sqrt(var + eps) * gamma + beta
kernel void layer_norm(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    device const half* gamma [[buffer(2)]],
    device const half* beta [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& dim [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= seq_len) return;
    
    // Compute mean
    float sum = 0.0;
    uint base = gid * dim;
    for (uint i = 0; i < dim; i++) {
        sum += float(input[base + i]);
    }
    float mean = sum / float(dim);
    
    // Compute variance
    float var_sum = 0.0;
    for (uint i = 0; i < dim; i++) {
        float diff = float(input[base + i]) - mean;
        var_sum += diff * diff;
    }
    float variance = var_sum / float(dim);
    float std_dev = sqrt(variance + eps);
    
    // Normalize
    for (uint i = 0; i < dim; i++) {
        float normalized = (float(input[base + i]) - mean) / std_dev;
        output[base + i] = half(normalized * float(gamma[i]) + float(beta[i]));
    }
}

/// Softmax kernel
/// Numerically stable: subtract max before exp
kernel void softmax(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& seq_len [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= seq_len) return;
    
    uint base = gid * dim;
    
    // Find max
    float max_val = float(input[base]);
    for (uint i = 1; i < dim; i++) {
        max_val = max(max_val, float(input[base + i]));
    }
    
    // Compute exp and sum
    float sum = 0.0;
    for (uint i = 0; i < dim; i++) {
        float exp_val = exp(float(input[base + i]) - max_val);
        output[base + i] = half(exp_val);
        sum += exp_val;
    }
    
    // Normalize
    for (uint i = 0; i < dim; i++) {
        output[base + i] = half(float(output[base + i]) / sum);
    }
}

/// Transpose 2D matrix
/// Input: [M, N], Output: [N, M]
kernel void transpose_2d(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& M [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= N || gid.y >= M) return;
    
    uint in_idx = gid.y * N + gid.x;   // [M, N]
    uint out_idx = gid.x * M + gid.y;  // [N, M]
    
    output[out_idx] = input[in_idx];
}
