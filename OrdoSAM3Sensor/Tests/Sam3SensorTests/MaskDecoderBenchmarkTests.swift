import XCTest
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
@testable import Sam3Sensor

@available(macOS 15.0, *)
final class MaskDecoderBenchmarkTests: XCTestCase {
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    var maskDecoder: MaskDecoder!
    var promptEncoder: PromptEncoder!
    var geometryEncoder: GeometryEncoder!
    
    override func setUpWithError() throws {
        device = MTLCreateSystemDefaultDevice()
        commandQueue = device.makeCommandQueue()
        
        // Init Pure Metal Components
        // PromptEncoder outputs Float32. MaskDecoder must match or we cast.
        // For simplicity, use F32 for benchmark.
        maskDecoder = MaskDecoder(device: device, embedDim: 256, numMultimaskOutputs: 3, enableHalfPrecision: false)
        promptEncoder = PromptEncoder(device: device, embedDim: 256, imageEmbeddingSize: (64, 64), inputImageSize: (1024, 1024), maskInChans: 16)
        geometryEncoder = GeometryEncoder(device: device, embedDim: 256, depth: 3, enableHalfPrecision: false)
        
        print("⚠️ Initializing MaskDecoder Synthetic Weights...")
        maskDecoder.testOnly_randomize()
        promptEncoder.testOnly_randomize()
        // GeometryEncoder doesn't need randomize
    }
    
    func testMaskDecoderLatency() throws {
        // Dummy Features [1, 256, 64, 64] (Flattened: [1, 4096, 256])
        let embedDim = 256
        let seqLen = 64 * 64
        let featBytes = 1 * seqLen * embedDim * 4 // F32
        guard let features = device.makeBuffer(length: featBytes, options: .storageModeShared) else { return }
        
        // 1. Prepare PE (Image + Point)
        // Image PE (Dense)
        print("DEBUG: Starting computeDensePE")
        let peCmd1 = commandQueue.makeCommandBuffer()!
        let imagePE = geometryEncoder.computeDensePE(gridSize: 64, commandBuffer: peCmd1)
        print("DEBUG: computeDensePE returned. Committing...")
        // peCmd1.commit()
        peCmd1.waitUntilCompleted()
        print("DEBUG: computeDensePE committed.")
        
        // Point PE (Sparse + Dense Prompt)
        print("DEBUG: Starting promptEncoder.forward")
        let promptPoints = [PromptEncoder.PromptType.point(x: 500, y: 500, label: 1)]
        let (peSparse, peDense) = try promptEncoder.forward(points: promptPoints, boxes: [], masks: nil)
        print("DEBUG: promptEncoder.forward done.")
        
        print("🚀 Starting MaskDecoder Latency Test (20 loops)...")
        
        // Warmup
        _ = try maskDecoder.forward(
            imageEmbeddings: features,
            imagePE: imagePE,
            pointEmbeddings: peSparse,
            densePromptEmbeddings: peDense,
            highResS0: nil,
            highResS1: nil,
            commandQueue: commandQueue
        )
        
        let loops = 20
        let start = CFAbsoluteTimeGetCurrent()
        
        for _ in 0..<loops {
            _ = try maskDecoder.forward(
                imageEmbeddings: features,
                imagePE: imagePE,
                pointEmbeddings: peSparse,
                densePromptEmbeddings: peDense,
                highResS0: nil,
                highResS1: nil,
                commandQueue: commandQueue
            )
            // Sync is optional for Async Pipelining check, but for "Latency" we want E2E?
            // If we don't sync, we assume queue depth handles it.
            // But to measure wall time, we should sync periodically or at end.
            // Let's sync per frame to mimic real-time constraints (latency focus).
            let cmd = commandQueue.makeCommandBuffer()!
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        
        let duration = CFAbsoluteTimeGetCurrent() - start
        let ms = (duration / Double(loops)) * 1000
        print("✅ MaskDecoder Latency: \(String(format: "%.1f", ms)) ms")
        
        XCTAssertLessThan(ms, 50.0, "Latency should be < 50ms")
    }
}
