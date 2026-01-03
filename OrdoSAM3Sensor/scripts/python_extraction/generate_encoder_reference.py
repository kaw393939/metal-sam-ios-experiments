#!/usr/bin/env python3
"""
Generate PyTorch reference outputs for ViT Encoder validation

This creates ground truth outputs from SAM3's PyTorch implementation
to validate Metal kernel correctness.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "experiment/sam3"))

from unittest.mock import MagicMock

# Mock dependencies
sys.modules["_lzma"] = MagicMock()
sys.modules["lzma"] = MagicMock()
sys.modules["pycocotools"] = MagicMock()
sys.modules["pycocotools.mask"] = MagicMock()

if 'triton' not in sys.modules:
    sys.modules['triton'] = MagicMock()
    sys.modules['triton.language'] = MagicMock()
    sys.modules['triton.backends'] = MagicMock()
    sys.modules['triton.backends.compiler'] = MagicMock()
    sys.modules['triton.compiler'] = MagicMock()
    sys.modules['triton.compiler.compiler'] = MagicMock()

import torch
import numpy as np
from sam3.model_builder import build_sam3_image_model


def generate_encoder_reference():
    """Generate reference outputs for encoder validation"""
    
    print("=" * 60)
    print("SAM3 Encoder Reference Generator")
    print("=" * 60)
    
    # Build model
    print("\nLoading SAM3 model...")
    model = build_sam3_image_model(
        checkpoint_path=None,
        load_from_HF=False,
        device='cpu'
    )
    model.eval()
    
    # Create test input
    print("Creating test input (1024x1024)...")
    batch = 1
    channels = 3
    height = 1024
    width = 1024
    
    test_image = torch.randn(batch, channels, height, width)
    
    # Run encoder
    print("Running encoder...")
    with torch.no_grad():
        # Get vision backbone
        backbone = model.backbone.vision_backbone
        
        # Forward through encoder
        sam3_features, sam3_pos, sam2_features, sam2_pos = backbone.forward(test_image)
        
        # Extract features (usually first scale)
        encoder_output = sam3_features[0] if isinstance(sam3_features, list) else sam3_features
        
    print(f"\nEncoder output shape: {encoder_output.shape}")
    print(f"Output range: [{encoder_output.min():.4f}, {encoder_output.max():.4f}]")
    print(f"Output mean: {encoder_output.mean():.4f}, std: {encoder_output.std():.4f}")
    
    # Save reference
    reference = {
        'input': test_image.numpy(),
        'output': encoder_output.numpy(),
        'output_shape': list(encoder_output.shape)
    }
    
    output_path = '/tmp/encoder_reference.npz'
    np.savez(output_path, **reference)
    print(f"\n✅ Saved reference to {output_path}")
    
    return reference


def test_patch_embedding():
    """Test patch embedding layer specifically"""
    
    print("\n" + "=" * 60)
    print("Testing Patch Embedding")
    print("=" * 60)
    
    # Build model
    model = build_sam3_image_model(
        checkpoint_path=None,
        load_from_HF=False,
        device='cpu'
    )
    model.eval()
    
    # Test input
    test_image = torch.randn(1, 3, 1024, 1024)
    
    # Run patch embed
    with torch.no_grad():
        backbone = model.backbone.vision_backbone
        if hasattr(backbone, 'trunk') and hasattr(backbone.trunk, 'patch_embed'):
            patch_embed = backbone.trunk.patch_embed
            patches = patch_embed(test_image)
            
            print(f"\nInput: {test_image.shape}")
            print(f"Patches: {patches.shape}")
            print(f"Expected: [1, 768, 64, 64] or similar")
            
            # Save
            np.savez('/tmp/patch_embed_reference.npz',
                    input=test_image.numpy(),
                    output=patches.numpy())
            print("✅ Saved patch embedding reference")


def test_transformer_block():
    """Test single transformer block"""
    
    print("\n" + "=" * 60)
    print("Testing Transformer Block")
    print("=" * 60)
    
    # Build model
    model = build_sam3_image_model(
        checkpoint_path=None,
        load_from_HF=False,
        device='cpu'
    )
    model.eval()
    
    # Create input (simulating patch embeddings)
    seq_len = 64 * 64  # 64x64 patches
    dim = 768
    test_input = torch.randn(1, seq_len, dim)
    
    # Run first transformer block
    with torch.no_grad():
        backbone = model.backbone.vision_backbone
        if hasattr(backbone, 'trunk') and hasattr(backbone.trunk, 'blocks'):
            block = backbone.trunk.blocks[0]
            output = block(test_input)
            
            print(f"\nInput: {test_input.shape}")
            print(f"Output: {output.shape}")
            print(f"Output stats: mean={output.mean():.4f}, std={output.std():.4f}")
            
            # Save
            np.savez('/tmp/transformer_block_reference.npz',
                    input=test_input.numpy(),
                    output=output.numpy())
            print("✅ Saved transformer block reference")


def main():
    # Test components
    test_patch_embedding()
    test_transformer_block()
    
    # Full encoder
    generate_encoder_reference()
    
    print("\n" + "=" * 60)
    print("✅ All references generated!")
    print("=" * 60)
    print("\nReference files created:")
    print("  - /tmp/encoder_reference.npz")
    print("  - /tmp/patch_embed_reference.npz")
    print("  - /tmp/transformer_block_reference.npz")
    print("\nNext steps:")
    print("  1. Load references in Swift tests")
    print("  2. Run Metal kernels with same inputs")
    print("  3. Compare outputs (MSE < 1e-3)")


if __name__ == "__main__":
    main()
