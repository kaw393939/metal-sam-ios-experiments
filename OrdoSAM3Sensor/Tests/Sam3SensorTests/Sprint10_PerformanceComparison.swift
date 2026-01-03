
import XCTest
import Metal
import MetalPerformanceShaders
@testable import Sam3Sensor

@available(macOS 15.0, *)
final class Sprint10_PerformanceComparison: XCTestCase {
    
    var device: MTLDevice!
    var library: MTLLibrary!
    
    override func setUp() {
        super.setUp()
        device = MTLCreateSystemDefaultDevice()
        // Correct 4-arg kernels for TransformerBlock
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void add_residual_float(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]], constant uint& count [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
            if (gid >= count) return;
            out[gid] = a[gid] + b[gid];
        }
        kernel void add_residual_half(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]], device half* out [[buffer(2)]], constant uint& count [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
            if (gid >= count) return;
            out[gid] = a[gid] + b[gid];
        }
        """
        library = try? device.makeLibrary(source: source, options: nil)
    }
    
    func measureHelper(block: TransformerBlock, input: MTLBuffer, rope: MTLBuffer, seq: Int, batch: Int, win: Bool, count: Int, label: String) -> Double {
        guard let queue = device.makeCommandQueue() else { return 0 }
        var recycled: [MTLBuffer] = []
        
        // Warmup
        for _ in 0..<3 {
            let cmd = queue.makeCommandBuffer()!
            _ = try? block.forward(input: input, ropeFreqs: rope, seqLen: seq, batch: batch, windowed: win, commandBuffer: cmd, recycledBuffers: &recycled)
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        
        // Measure
        let start = Date()
        for _ in 0..<count {
            let cmd = queue.makeCommandBuffer()!
            _ = try? block.forward(input: input, ropeFreqs: rope, seqLen: seq, batch: batch, windowed: win, commandBuffer: cmd, recycledBuffers: &recycled)
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        return Date().timeIntervalSince(start) / Double(count) * 1000.0
    }
    
    func testCompareFusedVsUnfused() throws {
        guard let device = device, let library = library else { return }
        
        let dim = 768
        let numHeads = 12
        let mlpDim = 3072 // 768*4
        
        let block = try TransformerBlock(
            device: device,
            dim: dim,
            numHeads: numHeads,
            mlpHiddenDim: mlpDim,
            library: library,
            useHalfPrecision: true
        )
        
        // Init weights (mock)
        func makeBuf(_ len: Int) -> MTLBuffer {
            let b = device.makeBuffer(length: len * 2, options: .storageModeShared)!
             memset(b.contents(), 0, len * 2)
            return b
        }
         func makeBuf32(_ len: Int) -> MTLBuffer {
            let b = device.makeBuffer(length: len * 4, options: .storageModeShared)!
             memset(b.contents(), 0, len * 4)
            return b
        }
        
        block.attention.loadWeights(qkvWeight: makeBuf(dim*3*dim), qkvBias: makeBuf(3*dim), outputWeight: makeBuf(dim*dim), outputBias: makeBuf(dim))
        block.mlp.loadWeights(fc1W: makeBuf(mlpDim*dim), fc1B: makeBuf(mlpDim), fc2W: makeBuf(dim*mlpDim), fc2B: makeBuf(dim))
        block.layerNorm1.loadWeights(gamma: makeBuf(dim), beta: makeBuf(dim))
        block.layerNorm2.loadWeights(gamma: makeBuf(dim), beta: makeBuf(dim))
        
        // Input
        let seqLen = 4096 // 64x64
        let batch = 1
        let inputBuf = makeBuf(seqLen * dim)
        let ropeBuf = makeBuf32(seqLen * (dim/numHeads/2) * 2)
        
        print("\n⚡️ Benchmarking TransformerBlock [Seq: \(seqLen), Dim: \(dim)] F16")
        print("----------------------------------------------------------------")
        
        // 1. Unfused Windowed
        block.forceFusedBlock = false
        let unfusedWin = measureHelper(block: block, input: inputBuf, rope: ropeBuf, seq: seqLen, batch: batch, win: true, count: 10, label: "Unfused Windowed")
        
        // 2. Fused Windowed
        block.forceFusedBlock = true
        let fusedWin = measureHelper(block: block, input: inputBuf, rope: ropeBuf, seq: seqLen, batch: batch, win: true, count: 10, label: "Fused Windowed")
        
        // 3. Unfused Global
        block.forceFusedBlock = false
        let unfusedGlob = measureHelper(block: block, input: inputBuf, rope: ropeBuf, seq: seqLen, batch: batch, win: false, count: 5, label: "Unfused Global")
        
        // 4. Fused Global
        block.forceFusedBlock = true
        let fusedGlob = measureHelper(block: block, input: inputBuf, rope: ropeBuf, seq: seqLen, batch: batch, win: false, count: 5, label: "Fused Global")
        
        print("Unfused (Windowed): \(String(format: "%.2f", unfusedWin)) ms")
        print("Fused   (Windowed): \(String(format: "%.2f", fusedWin)) ms")
        let speedupWin = unfusedWin / fusedWin
        print("🚀 Speedup (Windowed): \(String(format: "%.2fx", speedupWin))")
        
        print("\nUnfused (Global):   \(String(format: "%.2f", unfusedGlob)) ms")
        print("Fused   (Global):   \(String(format: "%.2f", fusedGlob)) ms")
        let speedupGlob = unfusedGlob / fusedGlob
        print("🚀 Speedup (Global):   \(String(format: "%.2fx", speedupGlob))")
        print("----------------------------------------------------------------\n")
    }
}
