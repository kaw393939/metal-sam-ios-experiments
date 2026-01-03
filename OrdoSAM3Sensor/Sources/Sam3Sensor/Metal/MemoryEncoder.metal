#include <metal_stdlib>
using namespace metal;

// SAM 3 Memory Fusion Kernel
// Fuses (Image Embedding + Mask Embedding into RGBA16Float Texture Array)
//
// Inputs:
// - image_embeddings: Float32 Buffer [64, 64, 256] (Layout: NHWC or HWC)
// - mask_embeddings: Float32 Buffer [64, 64, 256]
// - out_texture: Texture2DArray RGBA16Float [64, 64, 64 slices]
//
// Grid: (64, 64, 64)

kernel void fuse_memory(
    device const float* image_embeddings [[ buffer(0) ]],
    device const float* mask_embeddings [[ buffer(1) ]],
    texture2d_array<half, access::write> out_texture [[ texture(0) ]],
    uint3 gid [[ thread_position_in_grid ]]
) {
    uint x = gid.x; // 0..71
    uint y = gid.y; // 0..71
    uint slice = gid.z; // 0..63
    
    if (x >= 64 || y >= 64 || slice >= 64) return;
    
    // Calculate Buffer Index
    // Shape: [64, 64, 256]
    // Stride Y = 64 * 256
    // Stride X = 256
    // Stride Slice = 4 (RGBA)
    uint base_idx = y * (64 * 256) + x * 256 + slice * 4;
    
    // Read 4 values
    float4 img_val = float4(
        image_embeddings[base_idx + 0],
        image_embeddings[base_idx + 1],
        image_embeddings[base_idx + 2],
        image_embeddings[base_idx + 3]
    );
    
    float4 mask_val = float4(
        mask_embeddings[base_idx + 0],
        mask_embeddings[base_idx + 1],
        mask_embeddings[base_idx + 2],
        mask_embeddings[base_idx + 3]
    );
    
    // Fuse (Add)
    float4 sum = img_val + mask_val;
    
    // Write (Cast to half automatically by texture type)
    out_texture.write(half4(sum), uint2(x, y), slice);
}
