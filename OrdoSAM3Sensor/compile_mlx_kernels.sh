#!/bin/bash
set -e

# MLX Build Script for Manual Metadata Generation

MLX_CHECKOUT=".build/checkouts/mlx-swift/Source/Cmlx"
MLX_ROOT="$MLX_CHECKOUT/mlx"
GENERATED_ROOT="$MLX_CHECKOUT/mlx-generated"
OUTPUT_DIR=".build/arm64-apple-macosx/debug"

echo "Compiling MLX Metal Kernels..."

# Create a temp dir for air files
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

INCLUDES="-I $MLX_ROOT -I $MLX_CHECKOUT/mlx-c -I $GENERATED_ROOT/metal -I $MLX_ROOT/mlx/backend/metal/kernels"

# Find and compile all .metal files (excluding examples and steel kernels for speed)
find "$MLX_CHECKOUT" -name "*.metal" -not -path "*examples*" -not -path "*steel*" | while read -r file; do
    filename=$(basename "$file")
    name="${filename%.*}"
    # Handling potential duplicates or nested paths by using hash or unique prefix could be safer
    # But for now assume names are unique enough or flattening is okay (metallib handles it?)
    # Actually metallib takes list of files.
    
    echo "Compiling $filename..."
    xcrun -sdk macosx metal -c "$file" $INCLUDES -o "$TEMP_DIR/$name.air"
done

echo "Linking default.metallib..."
xcrun -sdk macosx metallib "$TEMP_DIR"/*.air -o "$OUTPUT_DIR/default.metallib"

echo "Done. Created $OUTPUT_DIR/default.metallib"
