#!/usr/bin/env python3
"""
Extract weights from NPZ to JSON for Swift loading
"""
import numpy as np
import json
import sys

def extract_npz_to_json(npz_path, output_path):
    """Extract all weights from NPZ and save as JSON with base64 encoded arrays"""
    
    weights = np.load(npz_path)
    
    output = {}
    
    for key in weights.files:
        array = weights[key]
        
        # Convert to float32 if needed
        if array.dtype != np.float32:
            array = array.astype(np.float32)
        
        # Store as list for JSON
        output[key] = {
            'shape': list(array.shape),
            'dtype': str(array.dtype),
            'data': array.flatten().tolist()
        }
        
        print(f"Extracted: {key} {array.shape}")
    
    with open(output_path, 'w') as f:
        json.dump(output, f)
    
    print(f"\n✅ Saved {len(output)} weights to {output_path}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python extract_weights_json.py <input.npz> <output.json>")
        sys.exit(1)
    
    extract_npz_to_json(sys.argv[1], sys.argv[2])
