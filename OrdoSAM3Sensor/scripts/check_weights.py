#!/usr/bin/env python3
"""
Check if patch_embed bias exists
"""
import numpy as np

weights = np.load('/Users/kwilliams/Projects/ordo/SAM3Metal/Resources/sam3_weights.npz')

print("All keys with 'patch':")
for k in weights.files:
    if 'patch' in k.lower():
        print(f"  - {k}: {weights[k].shape}")

print("\nAll keys with 'pos':")
for k in weights.files:
    if 'pos' in k.lower():
        print(f"  - {k}: {weights[k].shape}")
