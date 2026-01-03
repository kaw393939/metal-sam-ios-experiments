import numpy as np
import sys
import os

def check_weights(path):
    print(f"Checking {path}...")
    if not os.path.exists(path):
        print("File not found!")
        return
        
    try:
        d = np.load(path)
        keys = d.files
        
        # Check for SAM MaskDecoder keys
        sam_keys = [k for k in keys if k.startswith('sam_mask_decoder')]
        
        print(f"Total keys: {len(keys)}")
        print(f"SAM MaskDecoder keys found: {len(sam_keys)}")
        
        if len(sam_keys) > 0:
            print("\nSUCCESS: File contains SAM MaskDecoder weights!")
            print("First 5 keys:")
            for k in sam_keys[:5]:
                print(f" - {k}")
        else:
            print("\nFAILURE: No 'sam_mask_decoder' keys found.")
            print("This appears to be a detection-only checkpoint.")
            
    except Exception as e:
        print(f"Error loading file: {e}")

if __name__ == "__main__":
    path = "sam3_weights.npz"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    check_weights(path)
