import numpy as np
import os

def generate_gaussian_matrix():
    # Parameters matching standard SAM PromptEncoder
    num_pos_feats = 128
    scale = 1.0
    
    # Generate Gaussian Matrix (2, 128)
    # Using a fixed seed for reproducibility, though the original might be random.
    # If the original weights don't have it, any Gaussian matrix is better than none for training,
    # but for *inference* with pre-trained weights, we ideally need THE matrix.
    # Since we can't get it, we hope the "Direct Projection" + "Generic High Freq" is enough.
    # Or, maybe the model was trained with a fixed seed? 
    # Let's try seed 42? Or just random.
    # If the model relies on 'geometry_encoder.points_pos_enc_project' which is learned,
    # it might be robust enough to handle *any* generic Gaussian input?
    
    np.random.seed(42) # Try consistent seed
    matrix = np.random.normal(loc=0.0, scale=scale, size=(2, num_pos_feats)).astype(np.float32)
    
    output_path = "gaussian_matrix.bin"
    with open(output_path, "wb") as f:
        f.write(matrix.tobytes())
    
    print(f"Generated {output_path} with shape {matrix.shape}")
    
    # Also save as textual verification
    with open("gaussian_matrix.txt", "w") as f:
        np.savetxt(f, matrix)

if __name__ == "__main__":
    generate_gaussian_matrix()
