#!/usr/bin/env python3
"""
Extract comprehensive weights needed for all tests
"""
import numpy as np
import json

def extract_all_test_weights(npz_path, output_path):
    weights = np.load(npz_path)
    
    # Extract weights needed for tests
    test_keys = [
        # Patch embedding
        'backbone.vision_backbone.trunk.patch_embed.proj.weight',
        'backbone.vision_backbone.trunk.patch_embed.proj.bias',
        
        # Position embedding
        'backbone.vision_backbone.trunk.pos_embed',
        
        # First transformer block (for testing)
        'backbone.vision_backbone.trunk.blocks.0.norm1.weight',
        'backbone.vision_backbone.trunk.blocks.0.norm1.bias',
        'backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight',
        'backbone.vision_backbone.trunk.blocks.0.attn.qkv.bias',
        'backbone.vision_backbone.trunk.blocks.0.attn.proj.weight',
        'backbone.vision_backbone.trunk.blocks.0.attn.proj.bias',
    ]
    
    output = {}
    
    for key in test_keys:
        if key in weights.files:
            array = weights[key]
            if array.dtype != np.float32:
                array = array.astype(np.float32)
            
            output[key] = {
                'shape': list(array.shape),
                'dtype': str(array.dtype),
                'data': array.flatten().tolist()
            }
            print(f"✅ {key}: {array.shape}")
        else:
            print(f"⚠️  Missing: {key}")
    
    with open(output_path, 'w') as f:
        json.dump(output, f)
    
    print(f"\n✅ Saved {len(output)} test weights to {output_path}")
    
    # Calculate total size
    total_params = sum(weights[k].size for k in test_keys if k in weights.files)
    print(f"📊 Total parameters: {total_params:,}")
    print(f"📊 Memory: {total_params * 4 / 1024 / 1024:.1f} MB")

if __name__ == '__main__':
    extract_all_test_weights(
        '/Users/kwilliams/Projects/ordo/SAM3Metal/Resources/sam3_weights.npz',
        '/Users/kwilliams/Projects/ordo/SAM3Metal/Resources/sam3_weights.json'
    )
