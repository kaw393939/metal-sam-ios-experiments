import Foundation
import Metal
import MetalPerformanceShadersGraph

public struct BenchmarkStats: Sendable {
    public let iterations: Int
    public let meanMs: Double
    public let p50Ms: Double
    public let p90Ms: Double
}

public enum BenchmarkingError: Error {
    case noMetalDevice
    case cannotCreateQueue
}

@available(macOS 15.0, iOS 18.0, *)
public enum AttentionBenchmark {
    public static func run(
        device: MTLDevice,
        queue: MTLCommandQueue,
        batch: Int,
        heads: Int,
        seqLen: Int,
        dimPerHead: Int,
        implementation: AttentionImplementation,
        warmup: Int,
        iterations: Int
    ) throws -> BenchmarkStats {
        precondition(batch > 0 && heads > 0 && seqLen > 0 && dimPerHead > 0)
        precondition(warmup >= 0 && iterations > 0)

        let count = batch * heads * seqLen * dimPerHead
        var qVals = [Float](repeating: 0, count: count)
        var kVals = [Float](repeating: 0, count: count)
        var vVals = [Float](repeating: 0, count: count)
        for i in 0..<count {
            qVals[i] = Float((i * 73) % 97) / 97.0
            kVals[i] = Float((i * 41 + 7) % 89) / 89.0
            vVals[i] = Float((i * 19 + 3) % 83) / 83.0
        }

        let qBuf = device.makeBuffer(bytes: qVals, length: count * MemoryLayout<Float>.size, options: .storageModeShared)!
        let kBuf = device.makeBuffer(bytes: kVals, length: count * MemoryLayout<Float>.size, options: .storageModeShared)!
        let vBuf = device.makeBuffer(bytes: vVals, length: count * MemoryLayout<Float>.size, options: .storageModeShared)!

        let scale = Float(1.0 / sqrt(Double(dimPerHead)))

        // Compile once for stable timings.
        let cacheKey = "Bench_Attn_\(batch)x\(heads)x\(seqLen)x\(dimPerHead)_\(String(describing: implementation))"
        let (_, placeholders, output, _) = CompiledGraphCache.shared.getOrCompile(key: cacheKey, device: device) {
            let graph = MPSGraph()
            let q = graph.placeholder(
                shape: [NSNumber(value: batch), NSNumber(value: heads), NSNumber(value: seqLen), NSNumber(value: dimPerHead)],
                dataType: .float32,
                name: "q"
            )
            let k = graph.placeholder(
                shape: [NSNumber(value: batch), NSNumber(value: heads), NSNumber(value: seqLen), NSNumber(value: dimPerHead)],
                dataType: .float32,
                name: "k"
            )
            let v = graph.placeholder(
                shape: [NSNumber(value: batch), NSNumber(value: heads), NSNumber(value: seqLen), NSNumber(value: dimPerHead)],
                dataType: .float32,
                name: "v"
            )

            let out = AttentionKernels.scaledDotProductAttentionOrReference(
                graph: graph,
                query: q,
                key: k,
                value: v,
                scale: scale,
                name: "attn",
                implementation: implementation
            )

            return (graph: graph, placeholders: ["q": q, "k": k, "v": v], output: out)
        }

        var feeds: [MPSGraphTensor: MPSGraphTensorData] = [:]
        if let qPh = placeholders["q"] {
            feeds[qPh] = MPSGraphTensorData(qBuf, shape: [NSNumber(value: batch), NSNumber(value: heads), NSNumber(value: seqLen), NSNumber(value: dimPerHead)], dataType: .float32)
        }
        if let kPh = placeholders["k"] {
            feeds[kPh] = MPSGraphTensorData(kBuf, shape: [NSNumber(value: batch), NSNumber(value: heads), NSNumber(value: seqLen), NSNumber(value: dimPerHead)], dataType: .float32)
        }
        if let vPh = placeholders["v"] {
            feeds[vPh] = MPSGraphTensorData(vBuf, shape: [NSNumber(value: batch), NSNumber(value: heads), NSNumber(value: seqLen), NSNumber(value: dimPerHead)], dataType: .float32)
        }

        // Warmup
        if warmup > 0 {
            for _ in 0..<warmup {
                _ = try CompiledGraphCache.shared.runExecutable(key: cacheKey, queue: queue, feeds: feeds, targetTensors: [output])
            }
        }

        // Measure
        var samplesNs: [UInt64] = []
        samplesNs.reserveCapacity(iterations)

        for _ in 0..<iterations {
            let t0 = DispatchTime.now().uptimeNanoseconds
            _ = try CompiledGraphCache.shared.runExecutable(key: cacheKey, queue: queue, feeds: feeds, targetTensors: [output])
            let t1 = DispatchTime.now().uptimeNanoseconds
            samplesNs.append(t1 &- t0)
        }

        samplesNs.sort()
        let meanNs = Double(samplesNs.reduce(0, +)) / Double(samplesNs.count)

        func percentile(_ p: Double) -> Double {
            let idx = Int((p * Double(samplesNs.count - 1)).rounded())
            return Double(samplesNs[min(max(idx, 0), samplesNs.count - 1)])
        }

        return BenchmarkStats(
            iterations: iterations,
            meanMs: meanNs / 1_000_000.0,
            p50Ms: percentile(0.50) / 1_000_000.0,
            p90Ms: percentile(0.90) / 1_000_000.0
        )
    }
}
