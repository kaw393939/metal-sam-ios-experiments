import numpy as np
import sys

def list_keys(path):
    print(f"Loading {path}...")
    try:
        d = np.load(path)
        keys = sorted(d.files)
        print(f"Total keys: {len(keys)}")
        print("First 20 keys:")
        for k in keys[:20]:
            print(k)
            
        # Check for backbone keys
        bb_keys = [k for k in keys if "backbone" in k]
        print(f"\nBackbone keys: {len(bb_keys)}")
        if bb_keys:
            print("First 20 backbone keys:")
            for k in bb_keys[:20]:
                print(k)
    except Exception as e:
        print(e)

list_keys(sys.argv[1])
