import XCTest
import Metal
import MetalPerformanceShadersGraph
@testable import Sam3Sensor

/// Sprint 01 TDD: Async execution + graph compilation.
///
/// These tests are intentionally written to fail with the current implementation,
/// because `CompiledGraphCache` disables compilation (executable is `nil`).
///
/// Sprint 01 acceptance for these tests:
/// - `CompiledGraphCache.getOrCompile*` returns a non-nil executable for shaped feeds.
/// - `CompiledGraphCache.runExecutable` runs and produces correct results.
@available(macOS 14.0, *)
final class Sprint01_AsyncExecutionAndGraphCompilationTDDTests: XCTestCase {
    private var device: MTLDevice!
    private var queue: MTLCommandQueue!

    override func setUp() {
        super.setUp()
        guard let d = MTLCreateSystemDefaultDevice(), let q = d.makeCommandQueue() else {
            XCTFail("Metal device/queue not available")
            return
        }
        device = d
        queue = q
        CompiledGraphCache.shared.clear()
    }

    override func tearDown() {
        device = nil
        queue = nil
        super.tearDown()
    }

    func testCompiledGraphCacheReturnsExecutableForShapedFeeds() throws {
        let cacheKey = "TDD_Sprint01_SimpleAdd_1x4_f32"

        let (_, placeholders, _, executable) = CompiledGraphCache.shared.getOrCompile(
            key: cacheKey,
            device: device
        ) {
            let graph = MPSGraph()
            let x = graph.placeholder(shape: [1, 4], dataType: .float32, name: "x")
            let y = graph.addition(x, x, name: "y")
            return (graph: graph, placeholders: ["x": x], output: y)
        }

        XCTAssertNotNil(placeholders["x"], "Sanity check: placeholder not wired")

        // TDD: this should become non-nil once Sprint 01 re-enables compilation.
        XCTAssertNotNil(
            executable,
            "Expected a compiled MPSGraphExecutable. Sprint 01 should enable graph compilation in CompiledGraphCache."
        )
    }

    func testRunExecutableProducesCorrectValuesAndFeedOrdering() throws {
        let cacheKey = "TDD_Sprint01_FeedOrdering_1x2_f32"

        let (graph, placeholders, output, executable) = CompiledGraphCache.shared.getOrCompile(
            key: cacheKey,
            device: device
        ) {
            let graph = MPSGraph()
            let a = graph.placeholder(shape: [1, 2], dataType: .float32, name: "a")
            let b = graph.placeholder(shape: [1, 2], dataType: .float32, name: "b")

            // If feeds are swapped, this output changes dramatically.
            // y = a + 1000*b
            let scale = graph.constant(1000.0, dataType: .float32)
            let y = graph.addition(a, graph.multiplication(b, scale, name: "bScaled"), name: "y")

            return (graph: graph, placeholders: ["a": a, "b": b], output: y)
        }

        XCTAssertNotNil(placeholders["a"])
        XCTAssertNotNil(placeholders["b"])

        // TDD: this should become non-nil once Sprint 01 re-enables compilation.
        XCTAssertNotNil(
            executable,
            "Expected a compiled MPSGraphExecutable. Sprint 01 should enable graph compilation in CompiledGraphCache."
        )

        guard let aPh = placeholders["a"], let bPh = placeholders["b"] else {
            XCTFail("Missing placeholders")
            return
        }

        let aVals: [Float] = [1.0, 2.0]
        let bVals: [Float] = [3.0, 4.0]
        let expected: [Float] = [3001.0, 4002.0]

        let aBuf = device.makeBuffer(bytes: aVals, length: aVals.count * MemoryLayout<Float>.size, options: .storageModeShared)!
        let bBuf = device.makeBuffer(bytes: bVals, length: bVals.count * MemoryLayout<Float>.size, options: .storageModeShared)!

        var feeds: [MPSGraphTensor: MPSGraphTensorData] = [:]
        feeds[aPh] = MPSGraphTensorData(aBuf, shape: [1, 2], dataType: .float32)
        feeds[bPh] = MPSGraphTensorData(bBuf, shape: [1, 2], dataType: .float32)

        // Execute via executable path.
        let results = try CompiledGraphCache.shared.runExecutable(
            key: cacheKey,
            queue: queue,
            feeds: feeds,
            targetTensors: [output]
        )

        guard let outData = results[output] else {
            XCTFail("No output tensor data")
            return
        }

        let outBuf = device.makeBuffer(length: expected.count * MemoryLayout<Float>.size, options: .storageModeShared)!
        outData.mpsndarray().readBytes(outBuf.contents(), strideBytes: nil)
        let outPtr = outBuf.contents().bindMemory(to: Float.self, capacity: expected.count)
        let actual = [Float](UnsafeBufferPointer(start: outPtr, count: expected.count))

        XCTAssertEqual(actual.count, expected.count)
        for i in 0..<expected.count {
            XCTAssertEqual(actual[i], expected[i], accuracy: 1e-5, "Mismatch at index \(i)")
        }
    }
}
