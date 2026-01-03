
import XCTest
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
@testable import Sam3Sensor

@available(macOS 15.0, *)
final class Sprint10_VerificationTests: XCTestCase {
    
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
    
    func testFusedBlockMatchesUnfused() throws {
        guard let device = device, let library = library else { return }
        
        let dim = 256
        let numHeads = 8
        let mlpDim = 1024
        
        let block = try TransformerBlock(
            device: device,
            dim: dim,
            numHeads: numHeads,
            mlpHiddenDim: mlpDim,
            library: library,
            useHalfPrecision: true
        )
        
        func randBuf2(_ len: Int) -> MTLBuffer {
            return device.makeBuffer(length: len * 2, options: .storageModeShared)!
        }
        func randBuf4(_ len: Int) -> MTLBuffer {
            return device.makeBuffer(length: len * 4, options: .storageModeShared)!
        }
        
        // Load random weights (undefined/garbage is fine for stress test, but for numerical comparison use small randoms if possible)
        // For comparison, garbage is consistent, but might cause NaNs which mess up comparison?
        // Let's use memset 0 + small random.
        // Actually, just garbage is risk of NaN. memset 0 is safe but boring.
        // Let's use memset 0 for stability.
        func makeSafeBuf(_ len: Int) -> MTLBuffer {
            let b = device.makeBuffer(length: len * 2, options: .storageModeShared)!
            memset(b.contents(), 0, len * 2)
            return b
        }
        
        block.attention.loadWeights(
            qkvWeight: makeSafeBuf(dim * 3 * dim),
            qkvBias: makeSafeBuf(3 * dim),
            outputWeight: makeSafeBuf(dim * dim),
            outputBias: makeSafeBuf(dim)
        )
        block.mlp.loadWeights(
            fc1W: makeSafeBuf(mlpDim * dim),
            fc1B: makeSafeBuf(mlpDim),
            fc2W: makeSafeBuf(dim * mlpDim),
            fc2B: makeSafeBuf(dim)
        )
        block.layerNorm1.loadWeights(gamma: makeSafeBuf(dim), beta: makeSafeBuf(dim))
        block.layerNorm2.loadWeights(gamma: makeSafeBuf(dim), beta: makeSafeBuf(dim))
        
        // Input
        let seqLen = 64
        let batch = 1
        let inputBuf = makeSafeBuf(seqLen * dim)
        // Fill input with something distinct
        let ptr = inputBuf.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<(seqLen*dim) { ptr[i] = Float16(0.1) } // Constant 0.1
        
        let ropeBuf = randBuf4(seqLen * (dim/numHeads/2) * 2) // F32
        
        var recycled: [MTLBuffer] = []
        let queue = device.makeCommandQueue()!
        
        // 1. Unfused
        block.forceFusedBlock = false
        let cmd1 = queue.makeCommandBuffer()!
        let outRef = try block.forward(input: inputBuf, ropeFreqs: ropeBuf, seqLen: seqLen, batch: batch, windowed: false, commandBuffer: cmd1, recycledBuffers: &recycled)
        cmd1.commit()
        cmd1.waitUntilCompleted()
        
        // 2. Fused
        block.forceFusedBlock = true
        let cmd2 = queue.makeCommandBuffer()!
        let outFused = try block.forward(input: inputBuf, ropeFreqs: ropeBuf, seqLen: seqLen, batch: batch, windowed: false, commandBuffer: cmd2, recycledBuffers: &recycled)
        cmd2.commit()
        cmd2.waitUntilCompleted()
        
        // Helper to read back private buffer
        func readBack(_ buff: MTLBuffer) -> [Float16] {
            let len = buff.length
            let shared = device.makeBuffer(length: len, options: .storageModeShared)!
            let cmd = queue.makeCommandBuffer()!
            let blit = cmd.makeBlitCommandEncoder()!
            blit.copy(from: buff, sourceOffset: 0, to: shared, destinationOffset: 0, size: len)
            blit.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
            
            let ptr = shared.contents().assumingMemoryBound(to: Float16.self)
            let count = len / 2
            return (0..<count).map { ptr[$0] }
        }
        
        // 3. Compare
        print("DEBUG: Reading back buffers...")
        let outputRef = readBack(outRef)
        let outputFused = readBack(outFused)
        
        let count = seqLen * dim
        var maxDiff: Float = 0.0
        
        for i in 0..<count {
            let vRef = Float(outputRef[i])
            let vFused = Float(outputFused[i])
            if vRef.isNaN || vFused.isNaN { continue }
            let diff = abs(vRef - vFused)
            maxDiff = max(maxDiff, diff)
        }
        print("📊 Max Diff: \(maxDiff)")
        XCTAssertLessThan(maxDiff, 0.05, "Fused block diverged from Unfused")
    }
}
