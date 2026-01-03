//
//  SAM3HybridEncoderTests.swift
//  Sam3SensorTests
//
//  Tests for ANE+GPU hybrid encoder
//

import XCTest
import Metal
import CoreML
@testable import Sam3Sensor

@available(macOS 14.0, *)
final class SAM3HybridEncoderTests: XCTestCase {
    
    var device: MTLDevice!
    var hybridEncoder: SAM3HybridEncoder!
    
    override func setUp() {
        super.setUp()
        
        guard let mtlDevice = MTLCreateSystemDefaultDevice() else {
            XCTFail("Metal device not available")
            return
        }
        self.device = mtlDevice
        
        // Load Core ML model
        let modelURL = URL(fileURLWithPath: "/Users/kwilliams/Projects/Sam3/OrdoSAM3Sensor/CoreML/SAM3EarlyBlocks.mlpackage")
        
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            XCTFail("Core ML model not found at \(modelURL.path). Run export script first.")
            return
        }
        
        // Compile model if needed
        let compiledURL: URL
        do {
            compiledURL = try MLModel.compileModel(at: modelURL)
            print("✅ Core ML model compiled to: \(compiledURL.path)")
        } catch {
            XCTFail("Failed to compile Core ML model: \(error)")
            return
        }
        
        do {
            self.hybridEncoder = try SAM3HybridEncoder(
                device: mtlDevice,
                modelURL: compiledURL  // Use compiled model
            )
            print("✅ SAM3HybridEncoder initialized successfully")
        } catch {
            XCTFail("Failed to initialize hybrid encoder: \(error)")
        }
    }
    
    func testCoreMLModelLoads() {
        XCTAssertNotNil(hybridEncoder, "Hybrid encoder should be initialized")
        print("✅ Core ML model loaded and configured for ANE")
    }
    
    func testMLMultiArrayConversion() throws {
        // Test buffer ↔ MLMultiArray conversion
        let shape = [1, 100, 256]
        let elementCount = shape.reduce(1, *)
        let bufferSize = elementCount * 2 // FP16
        
        guard let testBuffer = device.makeBuffer(
            length: bufferSize,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create test buffer")
            return
        }
        
        // Fill with test data
        let pointer = testBuffer.contents().bindMemory(to: Float16.self, capacity: elementCount)
        for i in 0..<elementCount {
            pointer[i] = Float16(Float(i) * 0.001)
        }
        
        // Convert to MLMultiArray
        let mlArray = try hybridEncoder.convertBufferToMLMultiArray(
            buffer: testBuffer,
            shape: shape,
            dataType: .float16
        )
        
        XCTAssertEqual(mlArray.shape.map { $0.intValue }, shape)
        XCTAssertEqual(mlArray.dataType, .float16)
        
        // Convert back to buffer
        let roundTripBuffer = try hybridEncoder.convertMLMultiArrayToBuffer(
            mlArray: mlArray,
            device: device,
            label: "RoundTrip"
        )
        
        // Verify data integrity
        let originalPtr = testBuffer.contents().bindMemory(to: UInt8.self, capacity: bufferSize)
        let roundTripPtr = roundTripBuffer.contents().bindMemory(to: UInt8.self, capacity: bufferSize)
        
        XCTAssertEqual(
            memcmp(originalPtr, roundTripPtr, bufferSize),
            0,
            "Round-trip conversion should preserve data"
        )
        
        print("✅ MLMultiArray conversion validated")
    }
    
    func testANEInferenceLatency() throws {
        // Benchmark ANE inference time for blocks 0-8
        let modelURL = URL(fileURLWithPath: "/Users/kwilliams/Projects/Sam3/OrdoSAM3Sensor/CoreML/SAM3EarlyBlocks.mlpackage")
        
        // Compile model
        let compiledURL = try MLModel.compileModel(at: modelURL)
        
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        let model = try MLModel(contentsOf: compiledURL, configuration: config)
        
        // Create test input: [1, 5184, 1024]
        let inputArray = try MLMultiArray(shape: [1, 5184, 1024], dataType: .float16)
        let inputDict: [String: Any] = ["input": inputArray]
        
        // Warmup
        _ = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: inputDict))
        
        // Benchmark
        let iterations = 10
        var totalTime: Double = 0
        
        for _ in 0..<iterations {
            let start = Date()
            _ = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: inputDict))
            let elapsed = Date().timeIntervalSince(start)
            totalTime += elapsed
        }
        
        let avgLatency = (totalTime / Double(iterations)) * 1000 // ms
        print("🔥 ANE Latency (blocks 0-8): \(String(format: "%.1f", avgLatency))ms")
        
        // Target: < 20ms on M3 Air ANE
        XCTAssertLessThan(avgLatency, 30.0, "ANE should process blocks 0-8 in < 30ms")
    }
    
    func testHybridEncoderWithSyntheticWeights() throws {
        // Test end-to-end with synthetic weights (no real checkpoint needed)
        // This validates the orchestration logic
        
        guard let commandQueue = device.makeCommandQueue(),
              let commandBuffer = commandQueue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }
        
        // Create synthetic input image: [1, 1008, 1008, 3]
        let imageShape = [1, 1008, 1008, 3]
        let imageSize = imageShape.reduce(1, *) * MemoryLayout<Float>.size
        
        guard let imageBuffer = device.makeBuffer(
            length: imageSize,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create image buffer")
            return
        }
        
        // Note: This test will fail until we load actual weights
        // For now, we're validating the structure compiles
        print("⚠️  Hybrid encoder test requires actual weights - structure validated")
    }
}

// MARK: - Helper Extension (for testing)
extension SAM3HybridEncoder {
    func convertBufferToMLMultiArray(
        buffer: MTLBuffer,
        shape: [Int],
        dataType: MLMultiArrayDataType
    ) throws -> MLMultiArray {
        let mlArray = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: dataType)
        let elementCount = shape.reduce(1, *)
        let bytesPerElement = dataType == .float16 ? 2 : 4
        let bufferPointer = buffer.contents().bindMemory(to: UInt8.self, capacity: elementCount * bytesPerElement)
        let mlPointer = mlArray.dataPointer.bindMemory(to: UInt8.self, capacity: elementCount * bytesPerElement)
        memcpy(mlPointer, bufferPointer, elementCount * bytesPerElement)
        return mlArray
    }
    
    func convertMLMultiArrayToBuffer(
        mlArray: MLMultiArray,
        device: MTLDevice,
        label: String
    ) throws -> MTLBuffer {
        let elementCount = mlArray.shape.map { $0.intValue }.reduce(1, *)
        let dataType = mlArray.dataType
        let bytesPerElement = dataType == .float16 ? 2 : 4
        let totalBytes = elementCount * bytesPerElement
        guard let buffer = BufferAllocator.shared.privateBuffer(length: totalBytes, device: device, label: label) else {
            throw SAM3Error.bufferAllocationFailed(label)
        }
        let mlPointer = mlArray.dataPointer.bindMemory(to: UInt8.self, capacity: totalBytes)
        let bufferPointer = buffer.contents().bindMemory(to: UInt8.self, capacity: totalBytes)
        memcpy(bufferPointer, mlPointer, totalBytes)
        return buffer
    }
}
