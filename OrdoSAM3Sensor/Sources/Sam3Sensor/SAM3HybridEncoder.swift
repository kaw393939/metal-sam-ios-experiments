//
//  SAM3HybridEncoder.swift
//  Sam3Sensor
//
//  ANE+GPU Hybrid Encoder for SAM3 - MVP Version
//  Validates Core ML/ANE inference for blocks 0-8
//

import Foundation
import Metal
import CoreML

@available(macOS 14.0, *)
public final class SAM3HybridEncoder {
    
    private let device: MTLDevice
    private let coreMLModel: MLModel
    
    // Configuration
    private let embedDim = 1024
    private let spatialTokens = 5184 // 72 x 72
    
    public init(device: MTLDevice, modelURL: URL) throws {
        self.device = device
        
        // Load Core ML model for blocks 0-8
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine // Prefer ANE
        self.coreMLModel = try MLModel(contentsOf: modelURL, configuration: config)
        
        print("✅ SAM3HybridEncoder: Loaded Core ML model targeting ANE")
    }
    
    /// Run ANE inference on blocks 0-8
    /// - Parameter input: MLMultiArray [1, 5184, 1024] FP16
    /// - Returns: MLMultiArray [1, 5184, 1024] FP16
    public func runANEInference(input: MLMultiArray) throws -> MLMultiArray {
        let inputDict: [String: Any] = ["input": input]
        let output = try coreMLModel.prediction(from: MLDictionaryFeatureProvider(dictionary: inputDict))
        
        guard let outputArray = output.featureValue(for: "output")?.multiArrayValue else {
            throw SAM3Error.executionFailed("Failed to get ANE output")
        }
        
        return outputArray
    }
    
    // MARK: - Helper Methods
    
    /// Convert MTLBuffer to MLMultiArray
    public func convertBufferToMLMultiArray(
        buffer: MTLBuffer,
        shape: [Int],
        dataType: MLMultiArrayDataType
    ) throws -> MLMultiArray {
        
        let mlArray = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: dataType)
        
        let elementCount = shape.reduce(1, *)
        let bytesPerElement = dataType == .float16 ? 2 : 4
        
        // Copy from MTLBuffer to MLMultiArray
        let bufferPointer = buffer.contents().bindMemory(to: UInt8.self, capacity: elementCount * bytesPerElement)
        let mlPointer = mlArray.dataPointer.bindMemory(to: UInt8.self, capacity: elementCount * bytesPerElement)
        
        memcpy(mlPointer, bufferPointer, elementCount * bytesPerElement)
        
        return mlArray
    }
    
    /// Convert MLMultiArray to MTLBuffer
    public func convertMLMultiArrayToBuffer(
        mlArray: MLMultiArray,
        device: MTLDevice,
        label: String
    ) throws -> MTLBuffer {
        
        let elementCount = mlArray.shape.map { $0.intValue }.reduce(1, *)
        let dataType = mlArray.dataType
        let bytesPerElement = dataType == .float16 ? 2 : 4
        let totalBytes = elementCount * bytesPerElement
        
        guard let buffer = device.makeBuffer(
            length: totalBytes,
            options: .storageModeShared
        ) else {
            throw SAM3Error.bufferAllocationFailed(label)
        }
        
        // Copy from MLMultiArray to MTLBuffer
        let mlPointer = mlArray.dataPointer.bindMemory(to: UInt8.self, capacity: totalBytes)
        let bufferPointer = buffer.contents().bindMemory(to: UInt8.self, capacity: totalBytes)
        
        memcpy(bufferPointer, mlPointer, totalBytes)
        
        return buffer
    }
}
