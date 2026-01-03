
import XCTest
import Metal
import MetalPerformanceShaders
@testable import Sam3Sensor

final class EndToEndBenchmarkTests: XCTestCase {
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    
    override func setUpWithError() throws {
        try super.setUpWithError()
        guard #available(macOS 15.0, *) else {
            throw XCTSkip("Requires macOS 15.0+ (ViTEncoder/MPSGraph)")
        }
        guard let d = MTLCreateSystemDefaultDevice() else {
            XCTFail("Metal device not supported")
            return
        }
        device = d
        commandQueue = device.makeCommandQueue()
    }
    
    func testViTEncoderPerformance() throws {
        guard #available(macOS 15.0, *) else {
            throw XCTSkip("Requires macOS 15.0+ (ViTEncoder/MPSGraph)")
        }

        // This is a performance gate, not a correctness test.
        // Opt-in to avoid failing on debug builds / thermal throttling / background load.
        if ProcessInfo.processInfo.environment["SAM3_RUN_PERF_TESTS"] != "1" {
            throw XCTSkip("Set SAM3_RUN_PERF_TESTS=1 to enable performance benchmarks")
        }
        // Target: 170ms per frame (approx 6 FPS)
        // Previous Baseline: ~1189ms
        // Optimization Goal: 50x speedup from initial 8.5s (Python) -> 170ms (Metal)
        
        print("\n🚀 Starting End-to-End Performance Benchmark...")
        
        // 1. Initialize Encoder (Float16 by default)
        // 1. Initialize Encoder (Float16 by default)
        let fullEncoder = try ViTEncoder(device: device, numBlocks: 24)
        let multiplier = 1.0
        
        // 2. Create Input
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float,
            width: 1024,
            height: 1024,
            mipmapped: false
        )
        desc.usage = [.shaderRead, .shaderWrite]
        let inputTexture = device.makeTexture(descriptor: desc)!
        
        // 3. Warmup
        print("Warmup...")
        let warmupCommandBuffer = commandQueue.makeCommandBuffer()!
        _ = try? fullEncoder.forward(image: inputTexture, commandBuffer: warmupCommandBuffer)
        warmupCommandBuffer.commit()
        warmupCommandBuffer.waitUntilCompleted()
        
        // 4. Benchmark Loop
        let iterations = 2
        print("Benchmarking \(iterations) frames...")
        
        let start = CFAbsoluteTimeGetCurrent()
        
        for _ in 0..<iterations {
            let commandBuffer = commandQueue.makeCommandBuffer()!
            _ = try? fullEncoder.forward(image: inputTexture, commandBuffer: commandBuffer)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        
        let end = CFAbsoluteTimeGetCurrent()
        let totalTime = end - start
        let avgTime = (totalTime / Double(iterations)) * multiplier // Estimate for 24 blocks
        let ms = avgTime * 1000
        
        print("\n📊 RESULT: ViTEncoder (24 blocks) Average Time: \(String(format: "%.2f", ms)) ms")
        
        // Success Criteria
        if ms <= 170 { // 170ms target
            print("✅ SUCCESS: Target Met (<170ms)")
        } else {
            print("⚠️ WARNING: Target Missed (>170ms)")
        }
        
        // Assert reasonable performance to prevent regression
        // If we are optimized, it should certainly be under 1000ms
        XCTAssertLessThan(ms, 1000, "Performance regression: >1s per frame")
        
        // Ideally we want to assert < 200ms but CI might vary
        // XCTAssertLessThan(ms, 200)
    }
}
