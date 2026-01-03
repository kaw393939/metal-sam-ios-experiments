import torch
import numpy as np
import sys
import os

def convert_weights(pt_path, npz_path):
    print(f"Loading {pt_path}...")
    try:
        # Load PyTorch checkpoint
        # Map location cpu to avoid CUDA errors on Mac if generic
        ckpt = torch.load(pt_path, map_location='cpu')
        
        # Check if it's a dict with 'model' or just the state dict
        if isinstance(ckpt, dict) and 'model' in ckpt:
            print("Found 'model' key in checkpoint.")
            state_dict = ckpt['model']
        else:
            state_dict = ckpt
            
        # AGGRESSIVE SPACE MANAGEMENT: Delete the input file now that it's loaded!
        print(f"Removing input file {pt_path} to free disk space for output...")
        del ckpt 
        import gc; gc.collect() # Try to free memory
        try:
            os.remove(pt_path)
            print("Input file removed.")
        except Exception as e:
            print(f"Warning: Could not remove input file: {e}")
            
        print(f"Found {len(state_dict)} keys. Converting to numpy...")
        
        # Convert to numpy dict, filtering out language backbone to save space
        numpy_dict = {}
        for k, v in state_dict.items():
            if 'language_backbone' in k:
                continue
                
            if isinstance(v, torch.Tensor):
                numpy_dict[k] = v.detach().cpu().numpy()
            else:
                print(f"Skipping non-tensor key: {k}")
                
        # Save as compressed NPZ
        print(f"Saving to {npz_path}...")
        np.savez_compressed(npz_path, **numpy_dict)
        print("Conversion complete!")
        
        # Verify
        print("\nVerifying keys...")
        sam_keys = [k for k in numpy_dict.keys() if k.startswith('sam_mask_decoder')]
        if sam_keys:
            print(f"SUCCESS: Found {len(sam_keys)} SAM tracking keys!")
        else:
            print("WARNING: No 'sam_mask_decoder' keys found. Check if this is the correct checkpoint.")
            
    except Exception as e:
        print(f"Error converting: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 convert_pt_to_npz.py <input.pt> [output.npz]")
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace('.pt', '.npz').replace('.pth', '.npz')
        convert_weights(input_path, output_path)
