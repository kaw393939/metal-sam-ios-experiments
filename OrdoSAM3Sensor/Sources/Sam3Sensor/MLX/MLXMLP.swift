//
//  MLXMLP.swift
//  Sam3Sensor - MLX Hybrid Migration
//
//  MLX-based 2-layer MLP with GELU activation
//  Replaces MPSGraph MLP for transformer blocks
//

import Foundation
import Metal
import MLX
import MLXNN
import MLXRandom

/// Two-layer MLP using MLX
@available(macOS 15.0, *)
public class MLXMLP {
    private let inputDim: Int
    private let hiddenDim: Int
    private let outputDim: Int
    private let device: MTLDevice
    
    // Weights
    private var fc1W: MLXArray  // [inputDim, hiddenDim]
    private var fc1B: MLXArray  // [hiddenDim]
    private var fc2W: MLXArray  // [hiddenDim, outputDim]
    private var fc2B: MLXArray  // [outputDim]
    
    public init(inputDim: Int, hiddenDim: Int, outputDim: Int, device: MTLDevice) {
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.outputDim = outputDim
        self.device = device
        
        // Initialize with zeros (will be loaded from weights)
        self.fc1W = MLXArray.zeros([inputDim, hiddenDim], type: Float16.self)
        self.fc1B = MLXArray.zeros([hiddenDim], type: Float16.self)
        self.fc2W = MLXArray.zeros([hiddenDim, outputDim], type: Float16.self)
        self.fc2B = MLXArray.zeros([outputDim], type: Float16.self)
    }
    
    /// Load weights from MLXArray
    public func loadWeights(fc1W: MLXArray, fc1B: MLXArray, fc2W: MLXArray, fc2B: MLXArray) throws {
        // FC1: [hiddenDim, inputDim] (PyTorch format) → transpose to [inputDim, hiddenDim]
        self.fc1W = fc1W.T.asType(DType.float16)
        self.fc1B = fc1B.asType(DType.float16)
        
        // FC2: [outputDim, hiddenDim] → transpose to [hiddenDim, outputDim]
        self.fc2W = fc2W.T.asType(DType.float16)
        self.fc2B = fc2B.asType(DType.float16)
    }
    
    /// Randomly initialize weights for synthetic benchmarking
    public func randomInitialize() {
        self.fc1W = MLXRandom.uniform(low: -0.1, high: 0.1, [inputDim, hiddenDim]).asType(DType.float16)
        self.fc1B = MLXRandom.uniform(low: -0.1, high: 0.1, [hiddenDim]).asType(DType.float16)
        self.fc2W = MLXRandom.uniform(low: -0.1, high: 0.1, [hiddenDim, outputDim]).asType(DType.float16)
        self.fc2B = MLXRandom.uniform(low: -0.1, high: 0.1, [outputDim]).asType(DType.float16)
    }
    
    /// Forward pass: FC2(GELU(FC1(x)))
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // FC1: x @ w1 + b1
        var h = MLX.matmul(x, fc1W) + fc1B
        
        // GELU activation
        h = gelu(h)
        
        // FC2: h @ w2 + b2
        h = MLX.matmul(h, fc2W) + fc2B
        
        return h
    }
    
    /// Helper: Convert MTLBuffer to MLXArray
    private func convertMTLBufferToMLX(_ buffer: MTLBuffer, shape: [Int], dtype: DType) throws -> MLXArray {
        let ptr = buffer.contents()
        let count = shape.reduce(1, *)
        
        switch dtype {
        case .float16:
            let data = UnsafeBufferPointer<Float16>(start: ptr.assumingMemoryBound(to: Float16.self), count: count)
            let array = MLXArray(Array(data))
            return array.reshaped(shape)
        case .float32:
            let data = UnsafeBufferPointer<Float>(start: ptr.assumingMemoryBound(to: Float.self), count: count)
            let array = MLXArray(Array(data))
            return array.reshaped(shape)
        default:
            throw SAM3Error.invalidInput("Unsupported dtype: \(dtype)")
        }
    }
}
