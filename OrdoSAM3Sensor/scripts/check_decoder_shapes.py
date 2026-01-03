import numpy as np
import sys

def check_shapes(path):
    print(f"Loading {path}...")
    try:
        d = np.load(path)
        keys = sorted(d.files)
        
        # Check MaskDecoder weights
        decoder_keys = [k for k in keys if "sam_mask_decoder" in k]
        print(f"Found {len(decoder_keys)} decoder keys.")
        
        for k in decoder_keys:
            # We are interested in weights that might be 256x256
            # q_proj, k_proj, v_proj, out_proj
            if "proj.weight" in k or "mlp.fc" in k:
                shape = d[k].shape
                size = d[k].size * 4 # Assuming float32
                print(f"{k}: shape={shape}, size={size} bytes")
                
                # Check for 256x256 (65536 floats)
                if shape == (256, 256):
                    print(f"  -> MATCHES [256, 256]")
                elif d[k].size == 65536:
                     print(f"  -> SIZE MATCHES 64KB floats")
                     
    except Exception as e:
        print(e)

check_shapes(sys.argv[1])
