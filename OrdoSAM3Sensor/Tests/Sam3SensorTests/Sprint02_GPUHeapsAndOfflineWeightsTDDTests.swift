import XCTest
import Metal
@testable import Sam3Sensor

@available(macOS 15.0, *)
final class Sprint02_GPUHeapsAndOfflineWeightsTDDTests: XCTestCase {
    func testBufferAllocatorPrivateBuffersComeFromHeap() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("Metal not supported")
            return
        }

        CompiledGraphCache.shared.clear()

        let length = 256 * 1024
        guard let buf = BufferAllocator.shared.privateBuffer(length: length, device: device, label: "TDD_HeapPrivate") else {
            XCTFail("Failed to allocate private buffer")
            return
        }

        XCTAssertEqual(buf.storageMode, .private, "Sprint 02: private buffers should be storageModePrivate")

        // TDD: Sprint 02 requires heap-backed private allocations for predictable, low-churn memory.
        XCTAssertNotNil(
            buf.heap,
            "Sprint 02: expected BufferAllocator.privateBuffer to allocate from an MTLHeap (buf.heap != nil)."
        )
    }

    func testWeightsLoaderDoesNotAttemptRuntimePythonExtraction() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("Metal not supported")
            return
        }

        let tmp = FileManager.default.temporaryDirectory
        let npzURL = tmp.appendingPathComponent("tdd_weights").appendingPathExtension("npz")
        let jsonURL = tmp.appendingPathComponent("tdd_weights").appendingPathExtension("json")

        // Ensure only NPZ exists; JSON is intentionally missing.
        try? FileManager.default.removeItem(at: npzURL)
        try? FileManager.default.removeItem(at: jsonURL)
        FileManager.default.createFile(atPath: npzURL.path, contents: Data([0x00, 0x01, 0x02]), attributes: nil)

        let loader = WeightsLoader(device: device)

        do {
            try loader.load(from: npzURL)
            XCTFail("Expected load(from:) to fail when JSON is missing")
        } catch let err as WeightsError {
            // Sprint 02: runtime must NOT shell out to python; offline extraction must be done ahead of time.
            XCTAssertEqual(err, .offlineExtractionRequired)
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    func testPredictorHalfPrecisionRequiresOfflineFloat16Weights_NoRuntimeConversion() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("Metal not supported")
            return
        }

        let predictor = SAM3Predictor(device: device, enableHalfPrecision: true)

        // Patch-embed weight shape in `SAM3Encoder.buildGraph`:
        // [embedDim, 3, patchSize, patchSize] = [1024, 3, 14, 14]
        let elementCount = 1024 * 3 * 14 * 14
        let float32Bytes = elementCount * 4
        let float16Bytes = elementCount * 2

        // Provide Float32 payload; runtime must NOT convert it.
        let weights: [String: Data] = [
            WeightMapper.patchEmbedKeys["weight"]!: Data(count: float32Bytes)
        ]

        XCTAssertThrowsError(try predictor.loadWeights(weights)) { error in
            guard let e = error as? SAM3Predictor.PredictorWeightsError else {
                XCTFail("Unexpected error type: \(error)")
                return
            }
            XCTAssertEqual(
                e,
                .offlineFloat16Required(
                    key: WeightMapper.patchEmbedKeys["weight"]!,
                    expectedBytes: float16Bytes,
                    actualBytes: float32Bytes
                )
            )
        }
    }

    func testModelLoaderRejectsNPZAtRuntime_NoUnzip() throws {
        let tmp = FileManager.default.temporaryDirectory
        let base = "tdd_model_\(UUID().uuidString)"
        let npzURL = tmp.appendingPathComponent(base).appendingPathExtension("npz")
        let wtsURL = tmp.appendingPathComponent(base).appendingPathExtension("wts")
        try? FileManager.default.removeItem(at: npzURL)
        try? FileManager.default.removeItem(at: wtsURL)
        FileManager.default.createFile(atPath: npzURL.path, contents: Data([0x50, 0x4B, 0x03, 0x04]), attributes: nil) // ZIP header-ish

        let loader = ModelLoader()
        XCTAssertThrowsError(try loader.load(url: npzURL)) { error in
            guard let e = error as? ModelLoader.ModelLoaderError else {
                XCTFail("Unexpected error type: \(error)")
                return
            }
            XCTAssertEqual(e, .offlinePackedWeightsRequired)
        }
    }

    func testModelLoaderLoadsPackedWeightsFile() throws {
        let tmp = FileManager.default.temporaryDirectory
        let wtsURL = tmp.appendingPathComponent("tdd_model_\(UUID().uuidString)").appendingPathExtension("wts")
        try? FileManager.default.removeItem(at: wtsURL)

        // Minimal packed file: 1 entry with key and raw payload.
        let key = "detector.backbone.vision_backbone.trunk.patch_embed.proj.weight"
        let payload = Data([0x01, 0x02, 0x03, 0x04])

        var data = Data()
        data.append(contentsOf: Array("SAM3WTS1".utf8))
        data.append(UInt32(1).littleEndianData)

        data.append(UInt32(key.utf8.count).littleEndianData)
        data.append(contentsOf: Array(key.utf8))
        data.append(UInt8(1)) // dtype: float16 (opaque to loader)
        data.append(UInt8(1)) // rank
        data.append(UInt32(2).littleEndianData) // shape: [2]
        data.append(UInt32(payload.count).littleEndianData)
        data.append(payload)

        try data.write(to: wtsURL)

        let loader = ModelLoader()
        let weights = try loader.load(url: wtsURL)
        XCTAssertEqual(weights.count, 1)
        XCTAssertEqual(weights[key], payload)
    }

    func testEncoderPreuploadsWeightsToHeapBackedPrivateBuffers() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("Metal not supported")
            return
        }

        let encoder = SAM3Encoder(device: device, enableHalfPrecision: true)

        // Minimal single weight is enough to verify upload path.
        let key = WeightMapper.patchEmbedKeys["weight"]!
        let elementCount = 1024 * 3 * 14 * 14
        let f16Bytes = elementCount * 2

        encoder.loadWeights([key: Data(count: f16Bytes)])

        let buf = try XCTUnwrap(encoder._debugWeightBuffer(for: key))
        XCTAssertEqual(buf.storageMode, .private)
        XCTAssertNotNil(buf.heap, "Expected encoder weight buffer to be heap-backed")
    }
}

private extension UInt32 {
    var littleEndianData: Data {
        var v = self.littleEndian
        return Data(bytes: &v, count: MemoryLayout<UInt32>.size)
    }
}
