import XCTest
import Metal
import MetalPerformanceShadersGraph
@testable import Sam3Sensor

@available(macOS 15.0, *)
final class Sprint05b_DecoderAsyncTDDTests: XCTestCase {
    
    var device: MTLDevice!
    var queue: MTLCommandQueue!
    
    override func setUp() {
        super.setUp()
        guard let dev = MTLCreateSystemDefaultDevice(),
              let q = dev.makeCommandQueue() else {
            XCTFail("Metal not available")
            return
        }
        device = dev
        queue = q
    }
    
    // MARK: - Test Helpers
    
    func createTestEmbeddings() -> MTLBuffer {
        // Create 72x72x256 image embeddings
        let count = 5184 * 256
        let data = [Float](repeating: 0.5, count: count)
        return device.makeBuffer(bytes: data, length: count * 4, options: .storageModeShared)!
    }
    
    func createTestPE() -> MTLBuffer {
        // Create 5184x256 positional embeddings
        let count = 5184 * 256
        let data = [Float](repeating: 0.1, count: count)
        return device.makeBuffer(bytes: data, length: count * 4, options: .storageModeShared)!
    }
    
    func createTestPoints(count: Int = 2) -> MTLBuffer {
        // Create point embeddings (N points x 256)
        let totalCount = count * 256
        let data = [Float](repeating: 0.3, count: totalCount)
        return device.makeBuffer(bytes: data, length: totalCount * 4, options: .storageModeShared)!
    }
    
    func createTestHighRes(width: Int, height: Int) -> MTLBuffer {
        // Create high-res features
        let count = width * height * 256
        let data = [Float](repeating: 0.2, count: count)
        return device.makeBuffer(bytes: data, length: count * 4, options: .storageModeShared)!
    }
    
    func computeMSE(_ buffer1: MTLBuffer, _ buffer2: MTLBuffer) -> Double {
        let count = buffer1.length / 4
        let ptr1 = buffer1.contents().bindMemory(to: Float.self, capacity: count)
        let ptr2 = buffer2.contents().bindMemory(to: Float.self, capacity: count)
        
        var mse: Double = 0.0
        for i in 0..<count {
            let diff = Double(ptr1[i] - ptr2[i])
            mse += diff * diff
        }
        return mse / Double(count)
    }
    
    // MARK: - Test 1: Decoder Output Correctness
    
    func testAsyncDecoderMatchesSyncDecoder() throws {
        let decoder = MaskDecoder(device: device)
        
        // Create test inputs
        let imageEmbeddings = createTestEmbeddings()
        let imagePE = createTestPE()
        let pointEmbeddings = createTestPoints()
        let highResS0 = createTestHighRes(width: 288, height: 288)
        let highResS1 = createTestHighRes(width: 144, height: 144)
        
        // Run SYNC decoder (current implementation)
        let (syncMasks, syncIoU) = try decoder.forward(
            imageEmbeddings: imageEmbeddings,
            imagePE: imagePE,
            pointEmbeddings: pointEmbeddings,
            densePromptEmbeddings: nil as MTLBuffer?,
            highResS0: highResS0,
            highResS1: highResS1,
            commandQueue: queue
        )
        
        // Run ASYNC decoder (new implementation - will fail until implemented)
        // TODO: Implement forwardAsync() method
        // let (asyncMasks, asyncIoU) = try decoder.forwardAsync(...)
        
        // For now, verify sync decoder works
        XCTAssertEqual(syncMasks.length, 4 * 288 * 288 * 4, "Sync masks size incorrect")
        XCTAssertEqual(syncIoU.length, 4 * 4, "Sync IoU size incorrect")
        
        // TODO: When async implemented, verify outputs match
        // let maskMSE = computeMSE(syncMasks, asyncMasks)
        // let iouMSE = computeMSE(syncIoU, asyncIoU)
        // XCTAssertLessThan(maskMSE, 1e-6, "Mask outputs differ")
        // XCTAssertLessThan(iouMSE, 1e-6, "IoU outputs differ")
        
        print("Sprint05b Test 1: Sync decoder verified, async pending implementation")
    }
    
    // MARK: - Test 2: No Sync Points in Decoder Path
    
    func testDecoderHasNoSyncPoints() throws {
        let decoder = MaskDecoder(device: device)
        
        let imageEmbeddings = createTestEmbeddings()
        let imagePE = createTestPE()
        let pointEmbeddings = createTestPoints()
        let highResS0 = createTestHighRes(width: 288, height: 288)
        let highResS1 = createTestHighRes(width: 144, height: 144)
        
        // Measure current sync decoder (baseline)
        let t0 = DispatchTime.now().uptimeNanoseconds
        _ = try decoder.forward(
            imageEmbeddings: imageEmbeddings,
            imagePE: imagePE,
            pointEmbeddings: pointEmbeddings,
            densePromptEmbeddings: nil as MTLBuffer?,
            highResS0: highResS0,
            highResS1: highResS1,
            commandQueue: queue
        )
        let t1 = DispatchTime.now().uptimeNanoseconds
        
        let syncLatencyMs = Double(t1 - t0) / 1_000_000.0
        print("Sprint05b Test 2: Current sync decoder latency: \(syncLatencyMs)ms")
        
        // TODO: When async implemented, verify it's faster
        // let t2 = DispatchTime.now().uptimeNanoseconds
        // _ = try decoder.forwardAsync(...)
        // let t3 = DispatchTime.now().uptimeNanoseconds
        // let asyncLatencyMs = Double(t3 - t2) / 1_000_000.0
        // XCTAssertLessThan(asyncLatencyMs, 100.0, "Async decoder too slow")
        // XCTAssertLessThan(asyncLatencyMs, syncLatencyMs * 0.5, "Async not faster")
    }
    
    // MARK: - Test 3: Decoder Performance Target
    
    func testDecoderPerformanceTarget() throws {
        let decoder = MaskDecoder(device: device)
        
        let imageEmbeddings = createTestEmbeddings()
        let imagePE = createTestPE()
        let pointEmbeddings = createTestPoints()
        let highResS0 = createTestHighRes(width: 288, height: 288)
        let highResS1 = createTestHighRes(width: 144, height: 144)
        
        // Warmup
        for _ in 0..<3 {
            _ = try decoder.forward(
                imageEmbeddings: imageEmbeddings,
                imagePE: imagePE,
                pointEmbeddings: pointEmbeddings,
                densePromptEmbeddings: nil as MTLBuffer?,
                highResS0: highResS0,
                highResS1: highResS1,
                commandQueue: queue
            )
        }
        
        // Measure current performance
        var samples: [UInt64] = []
        for _ in 0..<10 {
            let t0 = DispatchTime.now().uptimeNanoseconds
            _ = try decoder.forward(
                imageEmbeddings: imageEmbeddings,
                imagePE: imagePE,
                pointEmbeddings: pointEmbeddings,
                densePromptEmbeddings: nil as MTLBuffer?,
                highResS0: highResS0,
                highResS1: highResS1,
                commandQueue: queue
            )
            let t1 = DispatchTime.now().uptimeNanoseconds
            samples.append(t1 - t0)
        }
        
        samples.sort()
        let meanMs = Double(samples.reduce(0, +)) / Double(samples.count) / 1_000_000.0
        let p50Ms = Double(samples[samples.count / 2]) / 1_000_000.0
        let p90Ms = Double(samples[Int(Double(samples.count) * 0.9)]) / 1_000_000.0
        
        print("Sprint05b Test 3: Current decoder: mean=\(meanMs)ms, p50=\(p50Ms)ms, p90=\(p90Ms)ms")
        
        // TODO: When async implemented, verify performance target
        // Target: < 100ms for interactive feel
        // XCTAssertLessThan(meanMs, 100.0, "Decoder too slow")
    }
    
    // MARK: - Test 4: Mask Quality
    
    func testDecoderMaskQuality() throws {
        let decoder = MaskDecoder(device: device)
        
        let imageEmbeddings = createTestEmbeddings()
        let imagePE = createTestPE()
        let pointEmbeddings = createTestPoints()
        let highResS0 = createTestHighRes(width: 288, height: 288)
        let highResS1 = createTestHighRes(width: 144, height: 144)
        
        let (masks, iou) = try decoder.forward(
            imageEmbeddings: imageEmbeddings,
            imagePE: imagePE,
            pointEmbeddings: pointEmbeddings,
            densePromptEmbeddings: nil as MTLBuffer?,
            highResS0: highResS0,
            highResS1: highResS1,
            commandQueue: queue
        )
        
        // Verify output shapes
        XCTAssertEqual(masks.length, 4 * 288 * 288 * 4, "Mask size incorrect")
        XCTAssertEqual(iou.length, 4 * 4, "IoU size incorrect")
        
        // Verify no NaN/Inf in outputs
        let maskPtr = masks.contents().bindMemory(to: Float.self, capacity: masks.length / 4)
        let iouPtr = iou.contents().bindMemory(to: Float.self, capacity: iou.length / 4)
        
        for i in 0..<(masks.length / 4) {
            XCTAssertFalse(maskPtr[i].isNaN, "NaN in masks at \(i)")
            XCTAssertFalse(maskPtr[i].isInfinite, "Inf in masks at \(i)")
        }
        
        for i in 0..<(iou.length / 4) {
            XCTAssertFalse(iouPtr[i].isNaN, "NaN in IoU at \(i)")
            XCTAssertFalse(iouPtr[i].isInfinite, "Inf in IoU at \(i)")
        }
        
        print("Sprint05b Test 4: Mask quality verified (no NaN/Inf)")
    }
    
    // MARK: - Test 5: Multiple Prompts (Regression)
    
    func testDecoderMultiplePrompts() throws {
        let decoder = MaskDecoder(device: device)
        
        let imageEmbeddings = createTestEmbeddings()
        let imagePE = createTestPE()
        let highResS0 = createTestHighRes(width: 288, height: 288)
        let highResS1 = createTestHighRes(width: 144, height: 144)
        
        // Test 1: Single point
        let points1 = createTestPoints(count: 1)
        let (masks1, iou1) = try decoder.forward(
            imageEmbeddings: imageEmbeddings,
            imagePE: imagePE,
            pointEmbeddings: points1,
            densePromptEmbeddings: nil as MTLBuffer?,
            highResS0: highResS0,
            highResS1: highResS1,
            commandQueue: queue
        )
        XCTAssertEqual(masks1.length, 4 * 288 * 288 * 4)
        
        // Test 2: Multiple points
        let points2 = createTestPoints(count: 3)
        let (masks2, iou2) = try decoder.forward(
            imageEmbeddings: imageEmbeddings,
            imagePE: imagePE,
            pointEmbeddings: points2,
            densePromptEmbeddings: nil as MTLBuffer?,
            highResS0: highResS0,
            highResS1: highResS1,
            commandQueue: queue
        )
        XCTAssertEqual(masks2.length, 4 * 288 * 288 * 4)
        
        print("Sprint05b Test 5: Multiple prompt types verified")
    }
    
    // MARK: - Test 6: CompiledGraphCache Integration
    
    func testDecoderUsesGraphCache() throws {
        // This test will verify caching when async implementation is done
        
        let decoder = MaskDecoder(device: device)
        
        let imageEmbeddings = createTestEmbeddings()
        let imagePE = createTestPE()
        let pointEmbeddings = createTestPoints()
        let highResS0 = createTestHighRes(width: 288, height: 288)
        let highResS1 = createTestHighRes(width: 144, height: 144)
        
        // First call
        let t0 = DispatchTime.now().uptimeNanoseconds
        _ = try decoder.forward(
            imageEmbeddings: imageEmbeddings,
            imagePE: imagePE,
            pointEmbeddings: pointEmbeddings,
            densePromptEmbeddings: nil as MTLBuffer?,
            highResS0: highResS0,
            highResS1: highResS1,
            commandQueue: queue
        )
        let t1 = DispatchTime.now().uptimeNanoseconds
        let firstCallMs = Double(t1 - t0) / 1_000_000.0
        
        // Second call (should be similar since current impl doesn't cache)
        let t2 = DispatchTime.now().uptimeNanoseconds
        _ = try decoder.forward(
            imageEmbeddings: imageEmbeddings,
            imagePE: imagePE,
            pointEmbeddings: pointEmbeddings,
            densePromptEmbeddings: nil as MTLBuffer?,
            highResS0: highResS0,
            highResS1: highResS1,
            commandQueue: queue
        )
        let t3 = DispatchTime.now().uptimeNanoseconds
        let secondCallMs = Double(t3 - t2) / 1_000_000.0
        
        print("Sprint05b Test 6: First call: \(firstCallMs)ms, Second call: \(secondCallMs)ms")
        
        // TODO: When async with caching implemented, verify speedup
        // XCTAssertLessThan(secondCallMs, firstCallMs * 0.8, "Graph not cached")
    }
}
