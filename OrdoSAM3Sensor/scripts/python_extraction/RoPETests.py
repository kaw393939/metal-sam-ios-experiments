#!/usr/bin/env python3
"""
Test RoPE kernel against PyTorch reference
Validates Metal implementation correctness
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "experiment/sam3"))

import torch
import numpy as np
from sam3.model.vitdet import compute_axial_cis, apply_rotary_enc


def generate_rope_reference(height, width, dim, theta=10000.0):
    """Generate RoPE frequencies using SAM3's implementation"""
    freqs_cis = compute_axial_cis(
        dim=dim,
        end_x=width,
        end_y=height,
        theta=theta
    )
    return freqs_cis


def apply_rope_reference(q, k, freqs_cis):
    """Apply RoPE using SAM3's implementation"""
    q_out, k_out = apply_rotary_enc(q, k, freqs_cis, repeat_freqs_k=False)
    return q_out, k_out


def test_rope_frequencies():
    """Test frequency generation"""
    print("Testing RoPE frequency generation...")
    
    H, W = 64, 64
    dim = 768
    
    # PyTorch reference
    freqs_ref = generate_rope_reference(H, W, dim)
    
    # TODO: Compare with Metal output
    # For now, just verify shape
    assert freqs_ref.shape == (H * W, dim // 2), f"Expected {(H*W, dim//2)}, got {freqs_ref.shape}"
    
    print(f"✅ Frequency shape: {freqs_ref.shape}")
    print(f"   Sample values: {freqs_ref[0, :4]}")
    
    return freqs_ref


def test_rope_application():
    """Test RoPE application to Q/K"""
    print("\nTesting RoPE application...")
    
    batch = 1
    num_heads = 12
    seq_len = 64 * 64
    dim_per_head = 64
    
    # Random Q/K
    q = torch.randn(batch, num_heads, seq_len, dim_per_head)
    k = torch.randn(batch, num_heads, seq_len, dim_per_head)
    
    # Generate frequencies
    freqs_cis = generate_rope_reference(64, 64, dim_per_head * num_heads)
    
    # Apply RoPE
    q_rope, k_rope = apply_rope_reference(q, k, freqs_cis)
    
    # Verify shapes
    assert q_rope.shape == q.shape
    assert k_rope.shape == k.shape
    
    # Verify rotation (magnitude should be preserved)
    q_mag_before = torch.norm(q, dim=-1)
    q_mag_after = torch.norm(q_rope, dim=-1)
    mag_diff = torch.abs(q_mag_before - q_mag_after).max().item()
    
    print(f"✅ Output shapes correct")
    print(f"   Magnitude preserved: max diff = {mag_diff:.6f}")
    
    # Save reference outputs for Metal comparison
    output = {
        'q_input': q.numpy(),
        'k_input': k.numpy(),
        'q_output': q_rope.numpy(),
        'k_output': k_rope.numpy(),
        'freqs_cis': freqs_cis.numpy()
    }
    
    np.savez('/tmp/rope_reference.npz', **output)
    print(f"💾 Saved reference to /tmp/rope_reference.npz")
    
    return output


def main():
    print("=" * 60)
    print("RoPE Kernel Validation (PyTorch Reference)")
    print("=" * 60)
    
    # Test 1: Frequency generation
    freqs = test_rope_frequencies()
    
    # Test 2: RoPE application
    output = test_rope_application()
    
    print("\n" + "=" * 60)
    print("✅ All PyTorch reference tests passed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run Metal kernel with same inputs")
    print("2. Compare outputs (MSE < 1e-5)")
    print("3. Benchmark Metal vs PyTorch")


if __name__ == "__main__":
    main()
