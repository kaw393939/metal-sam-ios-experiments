
import XCTest
@testable import Sam3Sensor
import MLX
import Metal
import MetalPerformanceShaders

@available(macOS 15.0, *)
final class HybridBenchmarkTests: XCTestCase {
    
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    
    override func setUp() {
        super.setUp()
        device = MTLCreateSystemDefaultDevice()
        commandQueue = device.makeCommandQueue()
        continueAfterFailure = false
    }
    
    func testHybridEncoderFPS() throws {
        print("\n=== HYBRID ViT ENCODER FPS BENCHMARK ===")
        
        let encoder = try HybridViTEncoder(
            device: device,
            embedDim: 768,
            depth: 12,
            numHeads: 12,
            mlpDim: 3072,
            patchSize: 16,
            imageSize: 1024
        )
        
        print("Initializing weights randomly...")
        encoder.randomInitialize()
        
        // Create input texture
        let textureDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: 1024, height: 1024, mipmapped: false)
        textureDesc.usage = [.shaderRead, .shaderWrite]
        guard let inputTexture = device.makeTexture(descriptor: textureDesc) else {
            XCTFail("Failed to create input texture")
            return
        }
        
        // Warmup
        print("Warmup (3 iterations)...")
        for _ in 0..<3 {
            guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
            _ = try encoder.forward(image: inputTexture, commandBuffer: commandBuffer)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        
        // Measure
        print("Measuring (10 iterations)...")
        let start = CFAbsoluteTimeGetCurrent()
        let iterations = 10
        
        for _ in 0..<iterations {
            guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
            _ = try encoder.forward(image: inputTexture, commandBuffer: commandBuffer)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        
        let duration = CFAbsoluteTimeGetCurrent() - start
        let avgTime = duration / Double(iterations)
        let fps = 1.0 / avgTime
        
        print("Average Time: \(String(format: "%.4f", avgTime)) s")
        print("FPS: \(String(format: "%.2f", fps))")
        print("Total Time: \(String(format: "%.4f", duration)) s")
    }
}
