import XCTest
import Metal
@testable import Sam3Sensor

@available(macOS 15.0, *)
final class SyntheticBenchmarkTests: XCTestCase {
    var pipeline: SAM3MetalPipeline!
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    
    override func setUpWithError() throws {
        device = MTLCreateSystemDefaultDevice()
        commandQueue = device.makeCommandQueue()
        pipeline = try SAM3MetalPipeline()
        
        print("⚠️ Initializing Synthetic Weights...")
        // Init weights synthetically
        pipeline.encoder.testOnly_randomize()
        pipeline.maskDecoder.testOnly_randomize()
        pipeline.promptEncoder.testOnly_randomize()
        
        // Setup Pipeline State
        pipeline.isLoaded = true
    }
    
    func testPipelineThrougput() throws {
        // Create Dummy Inputs
        let width = 1008
        let height = 1008
        let textureDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: width, height: height, mipmapped: false)
        textureDesc.usage = [.shaderRead, .shaderWrite]
        let inputTex = device.makeTexture(descriptor: textureDesc)!
        
        let points = [SIMD2<Float>(500, 500)]
        let labels = [Int32(1)]
        
        print("🚀 Starting Throughput Test (10 Frames)...")
        
        // Warmup
        _ = try pipeline.segment(image: inputTex, points: points, labels: labels)
        
        let loops = 10
        let start = CFAbsoluteTimeGetCurrent()
        
        for i in 0..<loops {
            _ = try pipeline.segment(image: inputTex, points: points, labels: labels)
            if i % 2 == 0 { print("  Frame \(i)") }
        }
        
        let duration = CFAbsoluteTimeGetCurrent() - start
        let fps = Double(loops) / duration
        let avgMs = (duration / Double(loops)) * 1000
        
        print("✅ Result: \(String(format: "%.2f", fps)) FPS (Avg: \(String(format: "%.0f", avgMs)) ms)")
        
        XCTAssertGreaterThan(fps, 0.1, "FPS should be non-zero")
    }
    
    func testLatencyDecodeOnly() throws {
        // Test Decode Latency (Assuming Encoded Features are cached)
        let width = 1008
        let height = 1008
        let textureDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: width, height: height, mipmapped: false)
        textureDesc.usage = [.shaderRead, .shaderWrite]
        let inputTex = device.makeTexture(descriptor: textureDesc)!
        
        // 1. Encode Once
        let startEnc = CFAbsoluteTimeGetCurrent()
        let features = try pipeline.encode(image: inputTex)
        print("  Encode Time: \(String(format: "%.0f", (CFAbsoluteTimeGetCurrent() - startEnc)*1000)) ms")
        
        // 2. Loop Decode
        let loops = 20
        
        // Manually build input for MaskDecoder
        let peSparse: MTLBuffer
        let peDense: MTLBuffer
        
        let promptPoints = [PromptEncoder.PromptType.point(x: 500, y: 500, label: 1)]
        (peSparse, peDense) = try pipeline.promptEncoder.forward(points: promptPoints, boxes: [], masks: nil)
        
        let peCmd = commandQueue.makeCommandBuffer()!
        let imagePE = pipeline.geometryEncoder.computeDensePE(gridSize: 64, commandBuffer: peCmd)
        peCmd.commit()
        peCmd.waitUntilCompleted()
        
        print("🚀 Starting Decode-Only Latency Test (20 loops)...")
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<loops {
            // Call MaskDecoder directly
            let _ = try pipeline.maskDecoder.forward(
                imageEmbeddings: features,
                imagePE: imagePE,
                pointEmbeddings: peSparse,
                densePromptEmbeddings: peDense,
                highResS0: nil,
                highResS1: nil,
                commandQueue: commandQueue
            )
            // Sync?
            let cmd = commandQueue.makeCommandBuffer()!
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let duration = CFAbsoluteTimeGetCurrent() - start
        let ms = (duration / Double(loops)) * 1000
        print("✅ Decode Latency: \(String(format: "%.1f", ms)) ms")
    }
}
