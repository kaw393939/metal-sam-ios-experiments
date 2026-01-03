//
//  MLXLayerNorm.swift
//  Sam3Sensor - MLX Hybrid Migration (CORRECTED API)
//

import Foundation
import Metal
import MLX
import MLXNN
import MLXRandom

/// LayerNorm using MLX for Apple Silicon optimization
@available(macOS 15.0, *)
public class MLXLayerNorm {
    private let normalizedShape: [Int]
    private let eps: Float
    private let device: MTLDevice
    
    // Learnable parameters
    private var gamma: MLXArray
    private var beta: MLXArray
    
    public init(normalizedShape: [Int], eps: Float = 1e-6, device: MTLDevice) {
        self.normalizedShape = normalizedShape
        self.eps = eps
        self.device = device
        
        // Initialize with ones and zeros
        let size = normalizedShape.reduce(1, *)
        self.gamma = MLXArray.ones([size], type: Float16.self)
        self.beta = MLXArray.zeros([size], type: Float16.self)
    }
    
    /// Load weights from MLXArray
    public func loadWeights(gamma: MLXArray, beta: MLXArray) throws {
        // Validate shapes
        guard gamma.shape == normalizedShape, beta.shape == normalizedShape else {
             throw SAM3Error.weightsNotLoaded("MLXLayerNorm: shape mismatch. Expected \(normalizedShape), got gamma:\(gamma.shape) beta:\(beta.shape)")
        }
        
        self.gamma = gamma.asType(DType.float16)
        self.beta = beta.asType(DType.float16)
    }
    
    /// Randomly initialize weights for synthetic benchmarking
    public func randomInitialize() {
        let size = normalizedShape.reduce(1, *)
        // Gamma ~ 1.0, Beta ~ 0.0
        self.gamma = MLXRandom.uniform(low: 0.9, high: 1.1, [size]).asType(DType.float16)
        self.beta = MLXRandom.uniform(low: -0.1, high: 0.1, [size]).asType(DType.float16)
    }
    
    /// Forward pass: LayerNorm(x)
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Compute mean and variance over last dimension
        let mean = MLX.mean(x, axes: [-1], keepDims: true)
        let variance = MLX.variance(x, axes: [-1], keepDims: true, ddof: 0)
        
        // Normalize: (x - mean) / sqrt(var + eps)
        let normalized = (x - mean) / MLX.sqrt(variance + eps)
        
        // Scale and shift: gamma * norm + beta
        return gamma * normalized + beta
    }
}
