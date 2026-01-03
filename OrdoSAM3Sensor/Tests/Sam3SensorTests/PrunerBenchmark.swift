
import XCTest
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
@testable import Sam3Sensor

@available(macOS 15.0, *)
final class PrunerBenchmark: XCTestCase {
    
    func testPrunerPerformance() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let queue = device.makeCommandQueue()!
        
        let pruner = TokenPruner(device: device, keepK: 1024, dim: 768)
        
        // Mock Inputs
        let N = 4096
        let inputBuf = device.makeBuffer(length: N * 768 * 2, options: .storageModeShared)!
        let ropeBuf = device.makeBuffer(length: N * 64 * 4, options: .storageModeShared)! // F32
        
        var recycled: [MTLBuffer] = []
        
        // Warmup
        let cmd0 = queue.makeCommandBuffer()!
        _ = try? pruner.prune(input: inputBuf, ropeFreqs: ropeBuf, seqLen: N, batch: 1, commandBuffer: cmd0, recycledBuffers: &recycled)
        cmd0.commit()
        cmd0.waitUntilCompleted()

        print("Warmup Done")

        let start = Date()
        let iter = 50

        for _ in 0..<iter {
            let cmd = queue.makeCommandBuffer()!
            recycled.removeAll() // Simulate recycle clearing (allocator reuses)
            _ = try? pruner.prune(input: inputBuf, ropeFreqs: ropeBuf, seqLen: N, batch: 1, commandBuffer: cmd, recycledBuffers: &recycled)
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        
        let duration = Date().timeIntervalSince(start)
        let avg = (duration / Double(iter)) * 1000
        print("📊 Pruner Average: \(String(format: "%.2f", avg)) ms")
    }
}
