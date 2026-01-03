//
//  MLXConversion.swift  
//  Sam3Sensor - MLX Hybrid Migration (CORRECTED API)
//

import Foundation
import Metal
import MLX

/// Conversion utilities for MTLBuffer ↔ MLXArray
@available(macOS 15.0, *)
public extension MTLBuffer {
    /// Convert MTLBuffer to MLXArray
    func toMLXArray(shape: [Int], dtype: DType = .float16) -> MLXArray {
        let ptr = self.contents()
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
            fatalError("Unsupported dtype: \(dtype)")
        }
    }
}

@available(macOS 15.0, *)
public extension MLXArray {
    /// Convert MLXArray to MTLBuffer
    func toMTLBuffer(device: MTLDevice) -> MTLBuffer? {
        let flatArray = self.flattened()
        let count = flatArray.size
        
        // Convert to Float16 for now (SAM3 uses F16)
        let f16Array = flatArray.asType(Float16.self)
        let data = f16Array.asArray(Float16.self)
        
        return device.makeBuffer(bytes: data, length: data.count * 2, options: .storageModeShared)
    }
    
    /// Get shape as array of Ints
    var shapeArray: [Int] {
        return self.shape.map { $0 }
    }
}
