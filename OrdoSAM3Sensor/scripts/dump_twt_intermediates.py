#!/usr/bin/env python3
"""
Generate intermediate outputs from SAM3 TwoWayTransformer for debugging.
This script loads the same inputs used by MaskDecoderIsolationTests and
dumps the output after each transformer block to binary files for comparison.
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/Users/kwilliams/Projects/ordo/Sensors/research/sam3')

from sam3.build_sam import build_sam3_vit_h

def load_bin(path):
    """Load float32 binary file."""
    return np.fromfile(path, dtype=np.float32)

def save_bin(arr, path):
    """Save numpy array as float32 binary."""
    arr.astype(np.float32).tofile(path)
    print(f"Saved: {path} shape={arr.shape}")

def main():
    root = "/Users/kwilliams/Projects/ordo/Sensors/"
    
    # Load reference inputs
    embed = load_bin(root + "debug_img_embed.bin")  # [1, 4096, 256]
    pos = load_bin(root + "debug_pos_enc.bin")      # [1, 4096, 256]
    sparse = load_bin(root + "debug_sparse.bin")    # [1, N, 256]
    
    print(f"Loaded embed: {embed.shape}, pos: {pos.shape}, sparse: {sparse.shape}")
    
    # Reshape
    embed_t = torch.from_numpy(embed.reshape(1, 4096, 256))
    pos_t = torch.from_numpy(pos.reshape(1, 4096, 256))
    
    # Sparse shape: Depends on how many tokens. Let's compute from size
    sparse_count = len(sparse) // 256
    sparse_t = torch.from_numpy(sparse.reshape(1, sparse_count, 256))
    print(f"Sparse tokens: {sparse_count}")
    
    # Load model
    ckpt_path = "/Users/kwilliams/Projects/ordo/Sensors/research/sam3/checkpoints/sam3.pt"
    model = build_sam3_vit_h(checkpoint=ckpt_path)
    model.eval()
    
    # Access the MaskDecoder's TwoWayTransformer
    transformer = model.tracker.sam_mask_decoder.transformer
    
    print(f"Transformer depth: {transformer.depth}")
    print(f"Transformer layers: {len(transformer.layers)}")
    
    # Manually run the transformer step-by-step to capture intermediates
    with torch.no_grad():
        queries = sparse_t.clone()  # [B, N, 256]
        keys = embed_t.clone()      # [B, 4096, 256]
        
        query_pe = sparse_t  # In SAM, point_embedding is used as both content and PE
        key_pe = pos_t
        
        print(f"Initial queries: {queries.shape}, keys: {keys.shape}")
        
        # Each block
        for i, layer in enumerate(transformer.layers):
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=query_pe,
                key_pe=key_pe
            )
            
            # Dump intermediate
            save_bin(queries.numpy().flatten(), root + f"debug_twt_block{i}_queries.bin")
            save_bin(keys.numpy().flatten(), root + f"debug_twt_block{i}_keys.bin")
            
            print(f"Block {i}: queries sum={queries.sum().item():.4f}, keys sum={keys.sum().item():.4f}")
        
        # Final Attention (Token -> Image)
        q = queries + query_pe
        k = keys + key_pe
        attn_out = transformer.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        
        save_bin(queries.numpy().flatten(), root + "debug_twt_final_attn_queries.bin")
        print(f"Final Attn: queries sum={queries.sum().item():.4f}")
        
        # Final Norm
        queries = transformer.norm_final_attn(queries)
        
        save_bin(queries.numpy().flatten(), root + "debug_twt_final_norm_queries.bin")
        print(f"Final Norm: queries sum={queries.sum().item():.4f}")
    
    print("\nDone! Intermediate outputs saved.")

if __name__ == "__main__":
    main()
