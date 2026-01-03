import XCTest
import Metal
import MetalPerformanceShadersGraph
@testable import Sam3Sensor

@available(macOS 15.0, *)
final class Sprint03_AttentionFusionFlashAttentionTDDTests: XCTestCase {
    func testSDPAMatchesReferenceAttentionWithinTolerance() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            XCTFail("Metal not supported")
            return
        }

        // Small deterministic shape.
        let b = 1
        let h = 2
        let n = 8
        let d = 4
        let count = b * h * n * d

        // Deterministic pseudo-random data.
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

        let graph = MPSGraph()
        let q = graph.placeholder(shape: [NSNumber(value: b), NSNumber(value: h), NSNumber(value: n), NSNumber(value: d)], dataType: .float32, name: "q")
        let k = graph.placeholder(shape: [NSNumber(value: b), NSNumber(value: h), NSNumber(value: n), NSNumber(value: d)], dataType: .float32, name: "k")
        let v = graph.placeholder(shape: [NSNumber(value: b), NSNumber(value: h), NSNumber(value: n), NSNumber(value: d)], dataType: .float32, name: "v")

        let scale = Float(1.0 / sqrt(Double(d)))

        let ref = AttentionKernels.scaledDotProductAttentionOrReference(
            graph: graph,
            query: q,
            key: k,
            value: v,
            scale: scale,
            name: "attn_ref",
            implementation: .referenceSoftmax
        )

        let sdpa = AttentionKernels.scaledDotProductAttentionOrReference(
            graph: graph,
            query: q,
            key: k,
            value: v,
            scale: scale,
            name: "attn_sdpa",
            implementation: .scaledDotProduct
        )

        var feeds: [MPSGraphTensor: MPSGraphTensorData] = [:]
        feeds[q] = MPSGraphTensorData(qBuf, shape: [NSNumber(value: b), NSNumber(value: h), NSNumber(value: n), NSNumber(value: d)], dataType: .float32)
        feeds[k] = MPSGraphTensorData(kBuf, shape: [NSNumber(value: b), NSNumber(value: h), NSNumber(value: n), NSNumber(value: d)], dataType: .float32)
        feeds[v] = MPSGraphTensorData(vBuf, shape: [NSNumber(value: b), NSNumber(value: h), NSNumber(value: n), NSNumber(value: d)], dataType: .float32)

        let results = graph.run(with: queue, feeds: feeds, targetTensors: [ref, sdpa], targetOperations: nil as [MPSGraphOperation]?)

        let refData = try XCTUnwrap(results[ref])
        let sdpaData = try XCTUnwrap(results[sdpa])

        let refOut = device.makeBuffer(length: count * MemoryLayout<Float>.size, options: .storageModeShared)!
        let sdpaOut = device.makeBuffer(length: count * MemoryLayout<Float>.size, options: .storageModeShared)!

        refData.mpsndarray().readBytes(refOut.contents(), strideBytes: nil)
        sdpaData.mpsndarray().readBytes(sdpaOut.contents(), strideBytes: nil)

        let rPtr = refOut.contents().bindMemory(to: Float.self, capacity: count)
        let sPtr = sdpaOut.contents().bindMemory(to: Float.self, capacity: count)

        // Tolerance: SDPA can differ slightly vs explicit softmax.
        for i in 0..<count {
            XCTAssertEqual(rPtr[i], sPtr[i], accuracy: 1e-3, "Mismatch at idx \(i)")
        }
    }
    
    func testSDPAMatchesReferenceAtSAM3Scale() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            XCTFail("Metal not supported")
            return
        }
        
        // SAM3 ViT encoder scale: B=1, H=16, N=5185, D=64
        let b = 1
        let h = 16
        let n = 5185  // 72x72 + 1 CLS token
        let d = 64
        let count = b * h * n * d
        
        // Deterministic pseudo-random data
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
        
        let scale = Float(1.0 / sqrt(Double(d)))
        
        // Use CompiledGraphCache for both implementations
        let refKey = "Sprint03_Ref_\(b)x\(h)x\(n)x\(d)"
        let (_, refPlaceholders, refOutput, _) = CompiledGraphCache.shared.getOrCompile(key: refKey, device: device) {
            let graph = MPSGraph()
            let q = graph.placeholder(shape: [NSNumber(value: b), NSNumber(value: h), NSNumber(value: n), NSNumber(value: d)], dataType: .float32, name: "q")
            let k = graph.placeholder(shape: [NSNumber(value: b), NSNumber(value: h), NSNumber(value: n), NSNumber(value: d)], dataType: .float32, name: "k")
            let v = graph.placeholder(shape: [NSNumber(value: b), NSNumber(value: h), NSNumber(value: n), NSNumber(value: d)], dataType: .float32, name: "v")
            
            let out = AttentionKernels.scaledDotProductAttentionOrReference(
                graph: graph, query: q, key: k, value: v, scale: scale, name: "attn", implementation: .referenceSoftmax
            )
            return (graph: graph, placeholders: ["q": q, "k": k, "v": v], output: out)
        }
        
        let sdpaKey = "Sprint03_SDPA_\(b)x\(h)x\(n)x\(d)"
        let (_, sdpaPlaceholders, sdpaOutput, _) = CompiledGraphCache.shared.getOrCompile(key: sdpaKey, device: device) {
            let graph = MPSGraph()
            let q = graph.placeholder(shape: [NSNumber(value: b), NSNumber(value: h), NSNumber(value: n), NSNumber(value: d)], dataType: .float32, name: "q")
            let k = graph.placeholder(shape: [NSNumber(value: b), NSNumber(value: h), NSNumber(value: n), NSNumber(value: d)], dataType: .float32, name: "k")
            let v = graph.placeholder(shape: [NSNumber(value: b), NSNumber(value: h), NSNumber(value: n), NSNumber(value: d)], dataType: .float32, name: "v")
            
            let out = AttentionKernels.scaledDotProductAttentionOrReference(
                graph: graph, query: q, key: k, value: v, scale: scale, name: "attn", implementation: .scaledDotProduct
            )
            return (graph: graph, placeholders: ["q": q, "k": k, "v": v], output: out)
        }
        
        // Prepare feeds
        var refFeeds: [MPSGraphTensor: MPSGraphTensorData] = [:]
        refFeeds[refPlaceholders["q"]!] = MPSGraphTensorData(qBuf, shape: [NSNumber(value: b), NSNumber(value: h), NSNumber(value: n), NSNumber(value: d)], dataType: .float32)
        refFeeds[refPlaceholders["k"]!] = MPSGraphTensorData(kBuf, shape: [NSNumber(value: b), NSNumber(value: h), NSNumber(value: n), NSNumber(value: d)], dataType: .float32)
        refFeeds[refPlaceholders["v"]!] = MPSGraphTensorData(vBuf, shape: [NSNumber(value: b), NSNumber(value: h), NSNumber(value: n), NSNumber(value: d)], dataType: .float32)
        
        var sdpaFeeds: [MPSGraphTensor: MPSGraphTensorData] = [:]
        sdpaFeeds[sdpaPlaceholders["q"]!] = MPSGraphTensorData(qBuf, shape: [NSNumber(value: b), NSNumber(value: h), NSNumber(value: n), NSNumber(value: d)], dataType: .float32)
        sdpaFeeds[sdpaPlaceholders["k"]!] = MPSGraphTensorData(kBuf, shape: [NSNumber(value: b), NSNumber(value: h), NSNumber(value: n), NSNumber(value: d)], dataType: .float32)
        sdpaFeeds[sdpaPlaceholders["v"]!] = MPSGraphTensorData(vBuf, shape: [NSNumber(value: b), NSNumber(value: h), NSNumber(value: n), NSNumber(value: d)], dataType: .float32)
        
        // Execute
        let refResults = try CompiledGraphCache.shared.runExecutable(key: refKey, queue: queue, feeds: refFeeds, targetTensors: [refOutput])
        let sdpaResults = try CompiledGraphCache.shared.runExecutable(key: sdpaKey, queue: queue, feeds: sdpaFeeds, targetTensors: [sdpaOutput])
        
        let refData = try XCTUnwrap(refResults[refOutput])
        let sdpaData = try XCTUnwrap(sdpaResults[sdpaOutput])
        
        // Read back results
        let refOut = device.makeBuffer(length: count * MemoryLayout<Float>.size, options: .storageModeShared)!
        let sdpaOut = device.makeBuffer(length: count * MemoryLayout<Float>.size, options: .storageModeShared)!
        
        refData.mpsndarray().readBytes(refOut.contents(), strideBytes: nil)
        sdpaData.mpsndarray().readBytes(sdpaOut.contents(), strideBytes: nil)
        
        let rPtr = refOut.contents().bindMemory(to: Float.self, capacity: count)
        let sPtr = sdpaOut.contents().bindMemory(to: Float.self, capacity: count)
        
        // Compute MSE
        var mse: Double = 0.0
        for i in 0..<count {
            let diff = Double(rPtr[i] - sPtr[i])
            mse += diff * diff
        }
        mse /= Double(count)
        
        print("Sprint03: SAM3-scale MSE = \(mse)")
        XCTAssertLessThan(mse, 1e-6, "MSE too large at SAM3 scale")
    }
    
    func testAttentionPerformanceAtViTScale() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw XCTSkip("Metal not supported")
        }
        
        // SAM3 ViT encoder scale
        let stats = try AttentionBenchmark.run(
            device: device,
            queue: queue,
            batch: 1,
            heads: 16,
            seqLen: 5185,
            dimPerHead: 64,
            implementation: .scaledDotProduct,
            warmup: 5,
            iterations: 20
        )
        
        print("Sprint03: ViT-scale SDPA: mean=\(stats.meanMs)ms, p50=\(stats.p50Ms)ms, p90=\(stats.p90Ms)ms")
        
        // Target: <500ms per attention op at ViT scale (regression gate)
        XCTAssertLessThan(stats.meanMs, 500.0, "Attention too slow at ViT scale")
    }
    
    func testSDPANumericalStability() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            XCTFail("Metal not supported")
            return
        }
        
        // Test with extreme values to verify stable softmax
        let b = 1, h = 2, n = 16, d = 4
        let count = b * h * n * d
        
        // Large values (softmax overflow risk)
        var qVals = [Float](repeating: 100.0, count: count)
        var kVals = [Float](repeating: 100.0, count: count)
        var vVals = [Float](repeating: 1.0, count: count)
        
        let qBuf = device.makeBuffer(bytes: qVals, length: count * MemoryLayout<Float>.size, options: .storageModeShared)!
        let kBuf = device.makeBuffer(bytes: kVals, length: count * MemoryLayout<Float>.size, options: .storageModeShared)!
        let vBuf = device.makeBuffer(bytes: vVals, length: count * MemoryLayout<Float>.size, options: .storageModeShared)!
        
        let graph = MPSGraph()
        let q = graph.placeholder(shape: [NSNumber(value: b), NSNumber(value: h), NSNumber(value: n), NSNumber(value: d)], dataType: .float32, name: "q")
        let k = graph.placeholder(shape: [NSNumber(value: b), NSNumber(value: h), NSNumber(value: n), NSNumber(value: d)], dataType: .float32, name: "k")
        let v = graph.placeholder(shape: [NSNumber(value: b), NSNumber(value: h), NSNumber(value: n), NSNumber(value: d)], dataType: .float32, name: "v")
        
        let scale = Float(1.0 / sqrt(Double(d)))
        let sdpa = AttentionKernels.scaledDotProductAttentionOrReference(
            graph: graph, query: q, key: k, value: v, scale: scale, name: "attn", implementation: .scaledDotProduct
        )
        
        var feeds: [MPSGraphTensor: MPSGraphTensorData] = [:]
        feeds[q] = MPSGraphTensorData(qBuf, shape: [NSNumber(value: b), NSNumber(value: h), NSNumber(value: n), NSNumber(value: d)], dataType: .float32)
        feeds[k] = MPSGraphTensorData(kBuf, shape: [NSNumber(value: b), NSNumber(value: h), NSNumber(value: n), NSNumber(value: d)], dataType: .float32)
        feeds[v] = MPSGraphTensorData(vBuf, shape: [NSNumber(value: b), NSNumber(value: h), NSNumber(value: n), NSNumber(value: d)], dataType: .float32)
        
        let results = graph.run(with: queue, feeds: feeds, targetTensors: [sdpa], targetOperations: nil as [MPSGraphOperation]?)
        let sdpaData = try XCTUnwrap(results[sdpa])
        
        let outBuf = device.makeBuffer(length: count * MemoryLayout<Float>.size, options: .storageModeShared)!
        sdpaData.mpsndarray().readBytes(outBuf.contents(), strideBytes: nil)
        
        let outPtr = outBuf.contents().bindMemory(to: Float.self, capacity: count)
        
        // Verify no NaN or Inf
        for i in 0..<count {
            XCTAssertFalse(outPtr[i].isNaN, "NaN at index \(i)")
            XCTAssertFalse(outPtr[i].isInfinite, "Inf at index \(i)")
        }
    }
}
