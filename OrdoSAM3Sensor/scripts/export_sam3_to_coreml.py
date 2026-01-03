#!/usr/bin/env python3
"""
SAM3 Early Blocks Core ML Export Script

Converts SAM3 Encoder Blocks 0-8 to Core ML for ANE execution.

Requirements:
  pip install torch coremltools safetensors numpy

Usage:
  python3 export_sam3_to_coreml.py --weights sam2.1_hiera_tiny.metal_fp16.safetensors --output SAM3EarlyBlocks.mlpackage
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
from safetensors import safe_open
import numpy as np
from pathlib import Path


class SAM3TransformerBlock(nn.Module):
    """Single SAM3 Transformer Block matching Metal implementation"""
    
    def __init__(self, embed_dim=1024, num_heads=16, mlp_hidden=4736):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Layer Norm 1
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Attention (QKV fused)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)
        
        # Layer Norm 2
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP
        self.mlp_fc1 = nn.Linear(embed_dim, mlp_hidden, bias=True)
        self.mlp_fc2 = nn.Linear(mlp_hidden, embed_dim, bias=True)
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, embed_dim]
        Returns:
            [batch, seq_len, embed_dim]
        """
        # Attention block
        shortcut = x
        x = self.norm1(x)
        
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, N, head_dim]
        
        # Scaled dot-product attention (no RoPE here - applied in Metal)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        x = self.proj(x)
        x = shortcut + x
        
        # MLP block
        shortcut = x
        x = self.norm2(x)
        x = self.mlp_fc1(x)
        x = F.gelu(x)
        x = self.mlp_fc2(x)
        x = shortcut + x
        
        return x


class SAM3EarlyBlocks(nn.Module):
    """SAM3 Encoder Blocks 0-8 for ANE deployment"""
    
    def __init__(self, num_blocks=9, embed_dim=1024, num_heads=16, mlp_hidden=4736):
        super().__init__()
        self.blocks = nn.ModuleList([
            SAM3TransformerBlock(embed_dim, num_heads, mlp_hidden)
            for _ in range(num_blocks)
        ])
    
    def forward(self, x):
        """
        Args:
            x: [1, 5184, 1024] - tokens from patch embedding
        Returns:
            [1, 5184, 1024] - processed features
        """
        for block in self.blocks:
            x = block(x)
        return x


def load_weights_from_checkpoint(model, weights_path, num_blocks=9):
    """Load weights from SAM3 checkpoint (supports .safetensors or .pt)"""
    print(f"Loading weights from {weights_path}...")
    
    weights_path = Path(weights_path)
    
    if weights_path.suffix == '.safetensors':
        # Original safetensors loading
        with safe_open(str(weights_path), framework="pt", device="cpu") as f:
            prefix = "detector.backbone.vision_backbone.trunk.blocks"
            
            for block_idx in range(num_blocks):
                block_prefix = f"{prefix}.{block_idx}"
                block = model.blocks[block_idx]
                
                # Load LayerNorm 1
                block.norm1.weight.data = f.get_tensor(f"{block_prefix}.norm1.weight")
                block.norm1.bias.data = f.get_tensor(f"{block_prefix}.norm1.bias")
                
                # Load QKV (fused)
                block.qkv.weight.data = f.get_tensor(f"{block_prefix}.attn.qkv.weight")
                block.qkv.bias.data = f.get_tensor(f"{block_prefix}.attn.qkv.bias")
                
                # Load Attention Projection
                block.proj.weight.data = f.get_tensor(f"{block_prefix}.attn.proj.weight")
                block.proj.bias.data = f.get_tensor(f"{block_prefix}.attn.proj.bias")
                
                # Load LayerNorm 2
                block.norm2.weight.data = f.get_tensor(f"{block_prefix}.norm2.weight")
                block.norm2.bias.data = f.get_tensor(f"{block_prefix}.norm2.bias")
                
                # Load MLP
                block.mlp_fc1.weight.data = f.get_tensor(f"{block_prefix}.mlp.fc1.weight")
                block.mlp_fc1.bias.data = f.get_tensor(f"{block_prefix}.mlp.fc1.bias")
                block.mlp_fc2.weight.data = f.get_tensor(f"{block_prefix}.mlp.fc2.weight")
                block.mlp_fc2.bias.data = f.get_tensor(f"{block_prefix}.mlp.fc2.bias")
                
                print(f"  ✓ Loaded block {block_idx}")
    
    elif weights_path.suffix == '.pt':
        # PyTorch checkpoint loading
        checkpoint = torch.load(str(weights_path), map_location='cpu')
        
        # Extract state dict (handle different checkpoint formats)
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        prefix = "detector.backbone.vision_backbone.trunk.blocks"
        
        for block_idx in range(num_blocks):
            block_prefix = f"{prefix}.{block_idx}"
            block = model.blocks[block_idx]
            
            # Load LayerNorm 1
            block.norm1.weight.data = state_dict[f"{block_prefix}.norm1.weight"]
            block.norm1.bias.data = state_dict[f"{block_prefix}.norm1.bias"]
            
            # Load QKV (fused)
            block.qkv.weight.data = state_dict[f"{block_prefix}.attn.qkv.weight"]
            block.qkv.bias.data = state_dict[f"{block_prefix}.attn.qkv.bias"]
            
            # Load Attention Projection
            block.proj.weight.data = state_dict[f"{block_prefix}.attn.proj.weight"]
            block.proj.bias.data = state_dict[f"{block_prefix}.attn.proj.bias"]
            
            # Load LayerNorm 2
            block.norm2.weight.data = state_dict[f"{block_prefix}.norm2.weight"]
            block.norm2.bias.data = state_dict[f"{block_prefix}.norm2.bias"]
            
            # Load MLP
            block.mlp_fc1.weight.data = state_dict[f"{block_prefix}.mlp.fc1.weight"]
            block.mlp_fc1.bias.data = state_dict[f"{block_prefix}.mlp.fc1.bias"]
            block.mlp_fc2.weight.data = state_dict[f"{block_prefix}.mlp.fc2.weight"]
            block.mlp_fc2.bias.data = state_dict[f"{block_prefix}.mlp.fc2.bias"]
            
            print(f"  ✓ Loaded block {block_idx}")
    
    else:
        raise ValueError(f"Unsupported weights format: {weights_path.suffix}. Expected .safetensors or .pt")
    
    print("Weights loaded successfully!")
    return model


def export_to_coreml(model, output_path, quantize_int8=True):
    """Export PyTorch model to Core ML with ANE optimization"""
    
    # Example input: [batch=1, seq_len=5184, embed_dim=1024]
    example_input = torch.randn(1, 5184, 1024)
    
    print("Tracing model...")
    model.eval()
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)
    
    print("Converting to Core ML...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input", shape=(1, 5184, 1024))],
        outputs=[ct.TensorType(name="output")],
        compute_units=ct.ComputeUnit.CPU_AND_NE,  # Enable ANE
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram"
    )
    
    if quantize_int8:
        print("Applying INT8 quantization for ANE...")
        from coremltools.optimize.coreml import OpLinearQuantizerConfig, linear_quantize_weights
        
        config = OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int8",
            granularity="per_channel"
        )
        mlmodel = linear_quantize_weights(mlmodel, config)
        print("  ✓ Quantization complete")
    
    # Save
    output_path = Path(output_path)
    mlmodel.save(str(output_path))
    print(f"\n✅ Core ML model saved to: {output_path}")
    print(f"   Size: {sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / (1024**2):.2f} MB")
    
    return mlmodel


def main():
    parser = argparse.ArgumentParser(description="Export SAM3 Early Blocks to Core ML")
    parser.add_argument("--weights", type=str, required=True, help="Path to SAM3 safetensors weights")
    parser.add_argument("--output", type=str, default="SAM3EarlyBlocks.mlpackage", help="Output Core ML package path")
    parser.add_argument("--num-blocks", type=int, default=9, help="Number of blocks to export (default: 9)")
    parser.add_argument("--no-quantize", action="store_true", help="Skip INT8 quantization")
    
    args = parser.parse_args()
    
    # Create model
    print(f"Creating SAM3 Early Blocks model ({args.num_blocks} blocks)...")
    model = SAM3EarlyBlocks(num_blocks=args.num_blocks)
    
    # Load weights
    model = load_weights_from_checkpoint(model, args.weights, args.num_blocks)
    
    # Export to Core ML
    export_to_coreml(model, args.output, quantize_int8=not args.no_quantize)
    
    print("\n🎉 Export complete! Next steps:")
    print(f"   1. Integrate {args.output} into Xcode project")
    print("   2. Test ANE performance with SAM3HybridEncoder.swift")
    print("   3. Benchmark end-to-end latency")


if __name__ == "__main__":
    main()
