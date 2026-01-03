// Quick MLX API test to verify syntax
import MLX

func testMLXAPI() {
    // Test array creation
    let ones = MLXArray.ones([2, 3], type: Float16.self)
    let zeros = MLXArray.zeros([2, 3], type: Float16.self)
    
    // Test reshape
    let reshaped = ones.reshaped([6])
    
    // Test operations
    let sum = ones + zeros
    let product = ones * 2.0
    
    print("MLX test passed")
}
