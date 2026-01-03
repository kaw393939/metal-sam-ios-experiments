import XCTest
import Metal
@testable import Sam3Sensor

/// Gate B: Architecture Dimension Tests
/// These tests verify that Swift implementation matches SAM3 dimensions.
final class ViTEncoderDimensionTests: XCTestCase {
    
    var device: MTLDevice!
    
    override func setUpWithError() throws {
        device = MTLCreateSystemDefaultDevice()
        guard device != nil else {
            throw XCTSkip("Metal device not available")
        }
    }
    
    // MARK: - Dimension Validation
    
    /// Test 3.1: ViTEncoder accepts embedDim=1024
    /// This test will FAIL until ViTEncoder is updated to accept configurable dimensions
    func testViTEncoderEmbedDim1024() throws {
        guard #available(macOS 15.0, *) else {
            throw XCTSkip("Requires macOS 15.0+ (ViTEncoder/MPSGraph)")
        }
        _ = try ViTEncoder(device: device, embedDim: 1024, patchSize: 14, inChannels: 3, numHeads: 16, numBlocks: 2, mlpHiddenDim: 4736, inputSize: 1008)
    }
    
    /// Test 3.2: Output buffer size matches expected shape
    func testOutputBufferSize() throws {
        // Input: 1024x1024 image
        // Patch: 14x14 → 1024/14 = 73 patches per dimension (floored to 64 for simplicity?)
        // Actually: SAM3 uses 1008px input → 1008/14 = 72 patches
        // Output: [1, 72, 72, 1024] for 1008px input
        
        let embedDim = 1024
        let patchSize = 14
        let inputSize = 1008 // SAM3 uses 1008px
        let numPatches = inputSize / patchSize // 72
        
        // Expected output bytes: 1 * 72 * 72 * 1024 * 4
        let expectedBytes = 1 * numPatches * numPatches * embedDim * 4
        
        print("Expected output: [\(1), \(numPatches), \(numPatches), \(embedDim)] = \(expectedBytes) bytes")
        
        // This test documents the expected shapes for SAM3
        XCTAssertEqual(numPatches, 72, "1008px / 14 patchSize = 72 patches")
        XCTAssertEqual(expectedBytes, 21233664, "Output buffer should be ~21.2MB")
    }
    
    /// Test 3.3: MLP hidden dimension is 4736
    func testMLPHiddenDimension() throws {
        // SAM3 uses 4736 hidden dim (4.625 × 1024), not 4 × 1024 = 4096
        let embedDim = 1024
        let mlpHiddenDim = 4736
        
        let ratio = Float(mlpHiddenDim) / Float(embedDim)
        XCTAssertEqual(ratio, 4.625, accuracy: 0.001, "MLP ratio should be 4.625")
        
        // fc1 weight: [4736, 1024]
        let fc1Bytes = mlpHiddenDim * embedDim * 4
        XCTAssertEqual(fc1Bytes, 19398656, "fc1 weight should be ~19.4MB")
    }
    
    /// Test 3.4: Number of transformer blocks
    func testTransformerBlockCount() throws {
        // SAM3-L (Large) has 31 blocks, SAM3 has varying depth
        // Check the actual weights to determine block count
        
        let weightsPath = "/Users/kwilliams/Projects/ordo/Sensors/SAM3Metal/Tests/Artifacts/sam3_weights.npz"
        guard FileManager.default.fileExists(atPath: weightsPath) else {
            throw XCTSkip("Weights file not found")
        }
        
        let loader = ModelLoader()
        let weights = try loader.load(url: URL(fileURLWithPath: weightsPath))
        
        // Count unique block indices
        var maxBlock = -1
        for key in weights.keys {
            // Pattern: backbone.vision_backbone.trunk.blocks.N.*
            if key.contains("blocks.") {
                let parts = key.components(separatedBy: ".")
                if let blockIdx = parts.firstIndex(of: "blocks"), blockIdx + 1 < parts.count,
                   let num = Int(parts[blockIdx + 1]) {
                    maxBlock = max(maxBlock, num)
                }
            }
        }
        
        let blockCount = maxBlock + 1
        print("Detected \(blockCount) transformer blocks")
        
        // SAM3 uses PE-L backbone which has 32 blocks (indices 0-31)
        XCTAssertEqual(blockCount, 32, "Expected 32 transformer blocks")
    }
}
