//
//  ViTUtils.metal
//  SAM3Metal
//
//  Utilities for converting between Texture (CNN) and Buffer (Transformer) formats
//

#include <metal_stdlib>
using namespace metal;

/// Flattens a 2D Array Texture (RGBA) into a linear Buffer (Float)
/// Texture: [Width, Height] with Slices. Each pixel is 4 channels.
/// Buffer: [Batch*Width*Height, Dim] where Dim = Slices * 4.
/// Used to convert Patch Embedding output to Transformer Input.
kernel void texture_to_buffer_flat(
    texture2d_array<float, access::read> input_texture [[texture(0)]],
    device float* output_buffer [[buffer(0)]],
    constant uint& dim [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // Grid: [Width, Height, 1]
    uint w = gid.x;
    uint h = gid.y;
    
    if (w >= input_texture.get_width() || h >= input_texture.get_height()) return;
    
    uint num_slices = input_texture.get_array_size();
    
    // Calculate base index in buffer
    // Sequence index = h * Width + w
    // Buffer layout: [Token0(Dim), Token1(Dim)...]
    uint token_idx = h * input_texture.get_width() + w;
    uint buffer_start = token_idx * dim;
    
    // Loop through slices (each slice has 4 channels: RGBA)
    for (uint s = 0; s < num_slices; s++) {
        float4 color = input_texture.read(uint2(w, h), s);
        
        // Write 4 channels to buffer
        uint ch_base = s * 4;
        
        // Ensure we don't write past dim (if dim is not multiple of 4, though usually it is 768)
        if (ch_base < dim) output_buffer[buffer_start + ch_base]     = color.r;
        if (ch_base + 1 < dim) output_buffer[buffer_start + ch_base + 1] = color.g;
        if (ch_base + 2 < dim) output_buffer[buffer_start + ch_base + 2] = color.b;
        if (ch_base + 3 < dim) output_buffer[buffer_start + ch_base + 3] = color.a;
    }
}

/// Flattens a 2D Array Texture (RGBA16F) into a linear Buffer (half)
/// Texture: [Width, Height] with Slices. Each pixel is 4 channels.
/// Buffer: [Batch*Width*Height, Dim] where Dim = Slices * 4.
kernel void texture_to_buffer_flat_half(
    texture2d_array<half, access::read> input_texture [[texture(0)]],
    device half* output_buffer [[buffer(0)]],
    constant uint& dim [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint w = gid.x;
    uint h = gid.y;

    if (w >= input_texture.get_width() || h >= input_texture.get_height()) return;

    uint num_slices = input_texture.get_array_size();
    uint token_idx = h * input_texture.get_width() + w;
    uint buffer_start = token_idx * dim;

    for (uint s = 0; s < num_slices; s++) {
        half4 color = input_texture.read(uint2(w, h), s);
        uint ch_base = s * 4;
        if (ch_base < dim) output_buffer[buffer_start + ch_base] = color.r;
        if (ch_base + 1 < dim) output_buffer[buffer_start + ch_base + 1] = color.g;
        if (ch_base + 2 < dim) output_buffer[buffer_start + ch_base + 2] = color.b;
        if (ch_base + 3 < dim) output_buffer[buffer_start + ch_base + 3] = color.a;
    }
}
