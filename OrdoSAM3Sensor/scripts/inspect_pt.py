import torch
import sys

def inspect_pt(path):
    print(f"Inspecting {path}...")
    try:
        ckpt = torch.load(path, map_location='cpu')
        if isinstance(ckpt, dict) and 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt
            
        keys = list(state_dict.keys())
        print(f"Total keys: {len(keys)}")
        
        sam_keys = [k for k in keys if 'sam_mask_decoder' in k]
        print(f"SAM MaskDecoder keys found: {len(sam_keys)}")
        
        if len(sam_keys) > 0:
            print("First 5 SAM keys:")
            for k in sam_keys[:5]:
                print(f" - {k}")
        else:
            print("No 'sam_mask_decoder' keys found.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_pt(sys.argv[1])
