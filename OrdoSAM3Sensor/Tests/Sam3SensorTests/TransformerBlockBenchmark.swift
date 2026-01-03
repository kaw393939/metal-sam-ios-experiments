
import XCTest
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
@testable import Sam3Sensor

@available(macOS 15.0, *)
final class TransformerBlockBenchmark: XCTestCase {
    
    var device: MTLDevice!
    var library: MTLLibrary!
    
    override func setUp() {
        super.setUp()
        device = MTLCreateSystemDefaultDevice()
        // Load library logic or mock... 
        // We reuse the logic from other tests or just use makeDefault
        if let libData = loadDefaultLib() {
             library = try? device.makeLibrary(source: libData, options: nil)
        } else {
             // Fallback
            library = device.makeDefaultLibrary()
        }
    }
    
    func loadDefaultLib() -> String? {
         // Minimal shader for benchmark
         return """
         #include <metal_stdlib>
         using namespace metal;
         kernel void add_residual_float(device float* a, device float* b, uint id [[thread_position_in_grid]]) {
             a[id] = a[id] + b[id];
         }
         kernel void add_residual_half(device half* a, device half* b, uint id [[thread_position_in_grid]]) {
             a[id] = a[id] + b[id];
         }
         kernel void gelu_float(device float* a, device float* b, uint id [[thread_position_in_grid]]) {
             float x = a[id];
             b[id] = 0.5 * x * (1.0 + tanh(0.79788456 * (x + 0.044715 * x * x * x)));
         }
         kernel void gelu_half(device half* a, device half* b, uint id [[thread_position_in_grid]]) {
             half x = a[id];
             b[id] = 0.5 * x * (1.0 + tanh(0.79788456 * (x + 0.044715 * x * x * x)));
         }
         """
    }
    
    func testBlockPerformance() throws {
        guard let device = device else { return }
        // Init Block
        // Compile simple library
        let source = loadDefaultLib()!
        let lib = try device.makeLibrary(source: source, options: nil)
        
        // 1. Create Block
        let block = try TransformerBlock(device: device, dim: 768, numHeads: 12, mlpHiddenDim: 768 * 4, library: lib, useHalfPrecision: true)
        
        // --- Init Weights ---
        func makeBuf(_ len: Int) -> MTLBuffer {
            return device.makeBuffer(length: len * 2, options: .storageModeShared)!
        }
        
        // Attention
        block.attention.loadWeights(
            qkvWeight: makeBuf(768 * 2304),
            qkvBias: makeBuf(2304),
            outputWeight: makeBuf(768 * 768),
            outputBias: makeBuf(768)
        )
        
        // MLP
        block.mlp.loadWeights(
            fc1W: makeBuf(3072 * 768),
            fc1B: makeBuf(3072),
            fc2W: makeBuf(768 * 3072),
            fc2B: makeBuf(768)
        )
        
        // LayerNorms
        print("DEBUG: Loading LN1")
        block.layerNorm1.loadWeights(gamma: makeBuf(768), beta: makeBuf(768))
        print("DEBUG: Loading LN2")
        block.layerNorm2.loadWeights(gamma: makeBuf(768), beta: makeBuf(768))
        
        print("DEBUG: Weights Loaded")

        // 2 .Setup Inputs
        let dim = 768
        let seqLen = 4096 // 64x64
        let batch = 1
        
        print("DEBUG: Creating Buffers")
        let inputLen = seqLen * dim * 2 // F16
        let inputBuffer = device.makeBuffer(length: inputLen, options: .storageModeShared)!
        
        // RoPE (Window 16x16 -> 256)
        let ropeWinSize = 16 * 16
        let ropeWinBuffer = device.makeBuffer(length: ropeWinSize * 32 * 2 * 4, options: .storageModeShared)!
        
        // RoPE (Global 64x64 -> 4096)
        let ropeGlobalSize = 4096
        let ropeGlobalBuffer = device.makeBuffer(length: ropeGlobalSize * 32 * 2 * 4, options: .storageModeShared)!
        
        let commandQueue = device.makeCommandQueue()!
        
        var recycled: [MTLBuffer] = []
        
        // --- Warmup ---
        print("DEBUG: Starting Warmup")
        for i in 0..<5 {
            print("DEBUG: Warmup \(i)")
            let cmd = commandQueue.makeCommandBuffer()!
            _ = try? block.forward(input: inputBuffer, ropeFreqs: ropeWinBuffer, seqLen: seqLen, batch: batch, windowed: true, commandBuffer: cmd, recycledBuffers: &recycled)
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        print("DEBUG: Warmup Done")
        
        // --- Measure Windowed ---
        let winStart = Date()
        let winIter = 20
        for _ in 0..<winIter {
            let cmd = commandQueue.makeCommandBuffer()!
             _ = try? block.forward(input: inputBuffer, ropeFreqs: ropeWinBuffer, seqLen: seqLen, batch: batch, windowed: true, commandBuffer: cmd, recycledBuffers: &recycled)
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let winDuration = Date().timeIntervalSince(winStart)
        print("📊 TransformerBlock (Windowed): \(String(format: "%.2f", (winDuration/Double(winIter))*1000)) ms")
        
         // --- Measure Global ---
        // Warmup Global
        for _ in 0..<2 {
            let cmd = commandQueue.makeCommandBuffer()!
            _ = try? block.forward(input: inputBuffer, ropeFreqs: ropeGlobalBuffer, seqLen: seqLen, batch: batch, windowed: false, commandBuffer: cmd, recycledBuffers: &recycled)
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        
        let globalStart = Date()
        let globalIter = 10
        for _ in 0..<globalIter {
            let cmd = commandQueue.makeCommandBuffer()!
             _ = try? block.forward(input: inputBuffer, ropeFreqs: ropeGlobalBuffer, seqLen: seqLen, batch: batch, windowed: false, commandBuffer: cmd, recycledBuffers: &recycled)
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let globalDuration = Date().timeIntervalSince(globalStart)
        print("📊 TransformerBlock (Global): \(String(format: "%.2f", (globalDuration/Double(globalIter))*1000)) ms")
        
        // --- Measure MLP Only ---
        let mlpStart = Date()
        let mlpIter = 20
        for _ in 0..<mlpIter {
            let cmd = commandQueue.makeCommandBuffer()!
            _ = try? block.mlp.forward(input: inputBuffer, seqLen: seqLen, batch: batch, commandBuffer: cmd, recycledBuffers: &recycled)
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let mlpDuration = Date().timeIntervalSince(mlpStart)
        print("📊 MLP Only: \(String(format: "%.2f", (mlpDuration/Double(mlpIter))*1000)) ms")
        
        // --- Measure Attention Only (Windowed) ---
        // Need normed input for attention? Just reuse inputBuffer
        let attnStart = Date()
        let attnIter = 20
        for _ in 0..<attnIter {
            let cmd = commandQueue.makeCommandBuffer()!
            _ = try? block.attention.forward(input: inputBuffer, ropeFreqs: ropeWinBuffer, batch: batch, seqLen: seqLen, windowed: true, commandBuffer: cmd, recycledBuffers: &recycled)
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let attnDuration = Date().timeIntervalSince(attnStart)
        print("📊 Attention Only (Windowed): \(String(format: "%.2f", (attnDuration/Double(attnIter))*1000)) ms")
        
    }
}
