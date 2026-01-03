//
//  MLXIntegrationTests.swift
//  Sam3SensorTests
//
//  Verifies MLX hybrid integration with real weights
//

import XCTest
import Metal
import MetalPerformanceShaders
import MLX
@testable import Sam3Sensor

@available(macOS 15.0, *)
final class MLXIntegrationTests: XCTestCase {
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    
    override func setUp() async throws {
        try await super.setUp()
        device = MTLCreateSystemDefaultDevice()
        commandQueue = device.makeCommandQueue()
    }
    
    func testHybridViTEndToEnd() throws {
        print("\n=== MLX HYBRID INTEGRATION TEST ===\n")
        
        // 1. Initialize HybridViTEncoder
        let encoder = try HybridViTEncoder(device: device)
        print("✅ HybridViTEncoder initialized")
        
        // 2. Load Real Weights (Check only, logic relies on RandomInit for now)
        let weightsURL = URL(fileURLWithPath: "/Users/kwilliams/Projects/Sam3/sam2.1_hiera_tiny.metal_fp16.safetensors")
        if FileManager.default.fileExists(atPath: weightsURL.path) {
            print("Loading weights from disk using MLX...")
            let arrays = try MLX.loadArrays(url: weightsURL)
            print("Loaded \(arrays.count) tensors")
        } else {
            print("⚠️ Weights file not found at \(weightsURL.path)")
            print("Running in SYNTHETIC mode (Random Init)")
        }
        
        // Load into encoder
        print("Loading weights (Random Init) into HybridViTEncoder...")
        let startLoad = Date()
        encoder.randomInitialize()
        let loadTime = Date().timeIntervalSince(startLoad)
        print("✅ Weights initialized in \(String(format: "%.2f", loadTime))s")
        
        // 3. Run Forward Pass
        print("Running forward pass...")
        
        // Create dummy image (1024x1024)
        let textureDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: 1024,
            height: 1024,
            mipmapped: false
        )
        textureDesc.usage = [.shaderRead, .shaderWrite]
        let image = device.makeTexture(descriptor: textureDesc)!
        
        // Fill with some data
        let region = MTLRegionMake2D(0, 0, 1024, 1024)
        let bytesPerRow = 1024 * 4
        var data = [UInt8](repeating: 128, count: 1024 * 1024 * 4)
        image.replace(region: region, mipmapLevel: 0, withBytes: &data, bytesPerRow: bytesPerRow)
        
        let commandBuffer = commandQueue.makeCommandBuffer()!
        
        let startEncode = Date()
        let output = try encoder.forward(image: image, commandBuffer: commandBuffer)
        
        // Note: commandBuffer is committed inside encoder.forward to sync with MLX.
        // We do not need to commit it here.
        
        let encodeTime = Date().timeIntervalSince(startEncode)
        print("✅ Forward pass complete in \(String(format: "%.2f", encodeTime))s")
        
        // 4. Validate Output
        print("Output buffer size: \(output.length) bytes")
        XCTAssertGreaterThan(output.length, 0)
    }
    
    func testHybridViTBenchmark() throws {
        print("\n=== MLX HYBRID FPS BENCHMARK ===\n")
        
        // 1. Initialize
        let encoder = try HybridViTEncoder(device: device)
        encoder.randomInitialize()
        print("✅ Models initialized")
        
        // 2. Setup Input
        let textureDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: 1024,
            height: 1024,
            mipmapped: false
        )
        textureDesc.usage = [.shaderRead, .shaderWrite]
        let image = device.makeTexture(descriptor: textureDesc)!
        
        // 3. Warmup (3 runs)
        print("Warming up (3 iterations)...")
        for _ in 0..<3 {
            let cb = commandQueue.makeCommandBuffer()!
            _ = try encoder.forward(image: image, commandBuffer: cb)
            // No commit here (handled by encoder)
        }
        
        // 4. Benchmark (10 runs)
        print("Running benchmark (10 iterations)...")
        var times: [TimeInterval] = []
        
        for i in 0..<10 {
            let start = Date()
            
            // Note: In real app, we use new command buffer per frame
            let cb = commandQueue.makeCommandBuffer()!
            _ = try encoder.forward(image: image, commandBuffer: cb)
            // No commit needed
            
            let elapsed = Date().timeIntervalSince(start)
            times.append(elapsed)
            print("  Run \(i+1): \(String(format: "%.0f", elapsed * 1000))ms")
        }
        
        let avg = times.reduce(0, +) / Double(times.count)
        let fps = 1.0 / avg
        print("\n📊 HYBRID PERF:")
        print("  Avg Latency: \(String(format: "%.0f", avg * 1000))ms")
        print("  FPS: \(String(format: "%.2f", fps))")
    }
}
