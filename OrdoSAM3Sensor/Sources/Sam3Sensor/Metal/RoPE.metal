//
//  RoPE.metal
//  SAM3Metal
//
//  Rotary Position Embedding (RoPE) for Vision Transformer
//  Based on SAM3's axial position encoding
//

#include <metal_stdlib>
using namespace metal;

struct RoPEParams {
    uint num_heads;
    uint dim_per_head;
    uint height;
    uint width;
    float theta;  // Rotation base (default: 10000.0)
};

/// Compute RoPE frequencies for 2D spatial positions
/// Generates complex frequencies for rotary embeddings
kernel void compute_rope_freqs_2d(
    device float2* freqs_cis [[buffer(0)]],        // Output: [H*W, dim/2] complex frequencies
    constant RoPEParams& params [[buffer(1)]],
    uint pos_idx [[thread_position_in_grid]]
) {
    if (pos_idx >= params.height * params.width) return;
    
    uint h = pos_idx / params.width;
    uint w = pos_idx % params.width;
    
    uint dim = params.dim_per_head;
    uint half_dim = dim / 4; // Number of complex pairs per axis (16 for dim=64)
    
    // Compute frequencies for this spatial position
    for (uint i = 0; i < half_dim; i++) {
        // Frequency calculation: 1.0 / (theta^(2i/half_dim))
        // Wait, SAM3 RoPE axial freq: 1.0 / (theta^(2i/half_dim))
        float freq_x = 1.0 / pow(params.theta, float(2 * i) / float(half_dim * 2)); // Logic check: standard freq
        float freq_y = freq_x;
        
        // Position-dependent angles
        float angle_x = float(w) * freq_x;
        float angle_y = float(h) * freq_y;
        
        // Store X frequencies in the first half of the head's complex pairs
        freqs_cis[pos_idx * (dim / 2) + i] = float2(cos(angle_x), sin(angle_x));
        
        // Store Y frequencies in the second half
        freqs_cis[pos_idx * (dim / 2) + half_dim + i] = float2(cos(angle_y), sin(angle_y));
    }
}

/// Apply RoPE to query and key tensors
/// Performs complex multiplication: (a + ib) * (c + id) = (ac - bd) + i(ad + bc)
kernel void apply_rope_2d(
    device float* q [[buffer(0)]],                   // Query [H*W, num_heads, dim_per_head]
    device float* k [[buffer(1)]],                   // Key [H*W, num_heads, dim_per_head]
    device const float2* freqs_cis [[buffer(2)]],   // Frequencies [H*W, dim_per_head/2]
    constant RoPEParams& params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]          // (head, token, dim/2)
) {
    uint head_idx = gid.x;
    uint token_idx = gid.y;
    uint dim_pair_idx = gid.z;
    
    if (head_idx >= params.num_heads || 
        token_idx >= params.height * params.width ||
        dim_pair_idx >= params.dim_per_head / 2) {
        return;
    }
    
    uint dim = params.dim_per_head;
    
    // Get base indices
    uint q_base = token_idx * params.num_heads * dim + head_idx * dim;
    uint k_base = token_idx * params.num_heads * dim + head_idx * dim;
    uint freq_idx = token_idx * (dim / 2) + dim_pair_idx;
    
    // Load query pair (treating adjacent dims as real/imag)
    uint q_idx = q_base + dim_pair_idx * 2;
    float q_real = q[q_idx];
    float q_imag = q[q_idx + 1];
    
    // Load key pair
    uint k_idx = k_base + dim_pair_idx * 2;
    float k_real = k[k_idx];
    float k_imag = k[k_idx + 1];
    
    // Load rotation (complex number)
    float2 rotation = freqs_cis[freq_idx];
    float cos_val = rotation.x;
    float sin_val = rotation.y;
    
    // Apply rotation via complex multiplication
    // (a + ib) * (cos + i*sin) = (a*cos - b*sin) + i(a*sin + b*cos)
    float q_out_real = q_real * cos_val - q_imag * sin_val;
    float q_out_imag = q_real * sin_val + q_imag * cos_val;
    
    float k_out_real = k_real * cos_val - k_imag * sin_val;
    float k_out_imag = k_real * sin_val + k_imag * cos_val;
    
    // Write back
    q[q_idx] = q_out_real;
    q[q_idx + 1] = q_out_imag;
    k[k_idx] = k_out_real;
    k[k_idx + 1] = k_out_imag;
}

/// Optimized version for batch processing
kernel void apply_rope_2d_batch(
    device float* qkv [[buffer(0)]],                 // Combined QKV [H*W, 3*num_heads*dim]
    device const float2* freqs_cis [[buffer(1)]],
    constant RoPEParams& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint head_idx = gid.x;
    uint token_idx = gid.y;
    uint dim_pair_idx = gid.z;
    
    if (head_idx >= params.num_heads || 
        token_idx >= params.height * params.width ||
        dim_pair_idx >= params.dim_per_head / 2) {
        return;
    }
    
    uint dim = params.dim_per_head;
    uint total_dim = 3 * params.num_heads * dim;  // Q, K, V concatenated
    
    // Q and K offsets (V doesn't get RoPE)
    uint q_base = token_idx * total_dim + head_idx * dim;
    uint k_base = token_idx * total_dim + params.num_heads * dim + head_idx * dim;
    uint freq_idx = token_idx * (dim / 2) + dim_pair_idx;
    
    // Load frequency
    float2 rotation = freqs_cis[freq_idx];
    float cos_val = rotation.x;
    float sin_val = rotation.y;
    
    // Apply to Q
    uint q_idx = q_base + dim_pair_idx * 2;
    float q_real = qkv[q_idx];
    float q_imag = qkv[q_idx + 1];
    qkv[q_idx] = q_real * cos_val - q_imag * sin_val;
    qkv[q_idx + 1] = q_real * sin_val + q_imag * cos_val;
    
    // Apply to K
    uint k_idx = k_base + dim_pair_idx * 2;
    float k_real = qkv[k_idx];
    float k_imag = qkv[k_idx + 1];
    qkv[k_idx] = k_real * cos_val - k_imag * sin_val;
    qkv[k_idx + 1] = k_real * sin_val + k_imag * cos_val;
}
