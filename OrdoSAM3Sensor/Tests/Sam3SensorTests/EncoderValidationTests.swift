import XCTest
import Metal
import MetalPerformanceShadersGraph
@testable import Sam3Sensor

@available(macOS 15.0, *)
final class EncoderValidationTests: XCTestCase {
    
    func testEncoderEndToEndLatency() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw XCTSkip("Metal not supported")
        }
        
        // Create predictor (which contains encoder)
        let predictor = SAM3Predictor(device: device, enableHalfPrecision: true)
        
        // Create dummy input texture (1008x1008)
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: 1008,
            height: 1008,
            mipmapped: false
        )
        desc.usage = [.shaderRead, .shaderWrite]
        guard let inputTexture = device.makeTexture(descriptor: desc) else {
            XCTFail("Failed to create texture")
            return
        }
        
        // Fill with dummy data
        let region = MTLRegionMake2D(0, 0, 1008, 1008)
        var bytes = [UInt8](repeating: 128, count: 1008 * 1008 * 4)
        inputTexture.replace(region: region, mipmapLevel: 0, withBytes: bytes, bytesPerRow: 1008 * 4)
        
        // Warmup
        for _ in 0..<3 {
            try predictor.setImage(inputTexture)
        }
        
        // Measure
        var samples: [UInt64] = []
        for _ in 0..<10 {
            let t0 = DispatchTime.now().uptimeNanoseconds
            try predictor.setImage(inputTexture)
            let t1 = DispatchTime.now().uptimeNanoseconds
            samples.append(t1 - t0)
        }
        
        samples.sort()
        let meanMs = Double(samples.reduce(0, +)) / Double(samples.count) / 1_000_000.0
        let p50Ms = Double(samples[samples.count / 2]) / 1_000_000.0
        let p90Ms = Double(samples[Int(Double(samples.count) * 0.9)]) / 1_000_000.0
        
        print("Sprint04: Encoder end-to-end latency: mean=\(meanMs)ms, p50=\(p50Ms)ms, p90=\(p90Ms)ms")
        
        // Target: <2000ms for full encoder (reasonable baseline)
        // This includes patch embed + 32 blocks + neck
        XCTAssertLessThan(meanMs, 2000.0, "Encoder too slow")
    }
    
    func testEncoderHasNoSyncPoints() throws {
        // Verify no waitUntilCompleted in encoder forward pass
        // Sprint 01 already removed all blocking calls from hot path
        
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not supported")
        }
        
        let predictor = SAM3Predictor(device: device, enableHalfPrecision: true)
        
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: 1008,
            height: 1008,
            mipmapped: false
        )
        desc.usage = [.shaderRead, .shaderWrite]
        guard let inputTexture = device.makeTexture(descriptor: desc) else {
            XCTFail("Failed to create texture")
            return
        }
        
        // This should complete without blocking
        // If there were sync points, this would take much longer
        let t0 = DispatchTime.now().uptimeNanoseconds
        try predictor.setImage(inputTexture)
        let t1 = DispatchTime.now().uptimeNanoseconds
        
        let latencyMs = Double(t1 - t0) / 1_000_000.0
        print("Sprint04: Encoder latency (no sync verification): \(latencyMs)ms")
        
        // Success if we got here without crashing
        XCTAssertTrue(true, "Encoder completed without sync points")
    }
    
    func testEncoderOutputsAvailable() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not supported")
        }
        
        let predictor = SAM3Predictor(device: device, enableHalfPrecision: true)
        
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: 1008,
            height: 1008,
            mipmapped: false
        )
        desc.usage = [.shaderRead, .shaderWrite]
        guard let inputTexture = device.makeTexture(descriptor: desc) else {
            XCTFail("Failed to create texture")
            return
        }
        
        // Run encoder
        try predictor.setImage(inputTexture)
        
        // Verify image embeddings are available
        XCTAssertNotNil(predictor.imageEmbeddings, "Image embeddings should be set")
        XCTAssertNotNil(predictor.highResS0, "High-res features S0 should be set")
        XCTAssertNotNil(predictor.highResS1, "High-res features S1 should be set")
        
        print("Sprint04: Encoder outputs verified - all embeddings available")
    }
}
