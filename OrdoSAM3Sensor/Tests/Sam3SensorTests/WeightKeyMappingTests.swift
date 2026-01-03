import XCTest
import Metal
@testable import Sam3Sensor

/// Gate A: Weight Key Mapping Tests
/// These tests verify that WeightMapper correctly transforms keys
final class WeightKeyMappingTests: XCTestCase {
    
    // MARK: - Key Format Tests
    
    /// Test encoder block key generation
    func testEncoderBlockKey() {
        let key = WeightMapper.encoderBlockKey(block: 0, component: "norm1.weight")
        XCTAssertEqual(key, "backbone.vision_backbone.trunk.blocks.0.norm1.weight")
        
        let key5 = WeightMapper.encoderBlockKey(block: 5, component: "attn.qkv.weight")
        XCTAssertEqual(key5, "backbone.vision_backbone.trunk.blocks.5.attn.qkv.weight")
    }
    
    /// Test patch embedding keys
    func testPatchEmbedKeys() {
        let keys = WeightMapper.patchEmbedKeys
        XCTAssertEqual(keys["weight"], "backbone.vision_backbone.trunk.patch_embed.proj.weight")
        XCTAssertEqual(keys["bias"], "backbone.vision_backbone.trunk.patch_embed.proj.bias")
    }
    
    /// Test position embedding key
    func testPosEmbedKey() {
        XCTAssertEqual(WeightMapper.posEmbedKey, "backbone.vision_backbone.trunk.pos_embed")
    }
    
    // MARK: - Architecture Constants
    
    /// Verify architecture constants match SAM3 checkpoint
    func testArchitectureConstants() {
        XCTAssertEqual(WeightMapper.embedDim, 1024)
        XCTAssertEqual(WeightMapper.mlpHiddenDim, 4736)
        XCTAssertEqual(WeightMapper.numHeads, 16)
        XCTAssertEqual(WeightMapper.numBlocks, 32)
        XCTAssertEqual(WeightMapper.patchSize, 14)
        XCTAssertEqual(WeightMapper.inputSize, 1008)
    }
    
    /// Verify MLP hidden ratio is 4.625
    func testMLPHiddenRatio() {
        let ratio = Float(WeightMapper.mlpHiddenDim) / Float(WeightMapper.embedDim)
        XCTAssertEqual(ratio, 4.625, accuracy: 0.001)
    }
    
    // MARK: - Weight Loading Tests
    
    /// Test that block keys map to actual weights in NPZ
    func testBlockKeysMapToRealWeights() throws {
        let weightsPath = "/Users/kwilliams/Projects/ordo/Sensors/SAM3Metal/Tests/Artifacts/sam3_weights.npz"
        guard FileManager.default.fileExists(atPath: weightsPath) else {
            throw XCTSkip("Weights file not found")
        }
        
        let loader = ModelLoader()
        let weights = try loader.load(url: URL(fileURLWithPath: weightsPath))
        
        // Verify block 0 keys exist
        let block0Keys = WeightMapper.encoderBlockKeys(block: 0)
        for (_, fullKey) in block0Keys {
            XCTAssertNotNil(weights[fullKey], "Key '\(fullKey)' not found in weights")
        }
        
        // Verify block 30 (last block) keys exist
        let block30Keys = WeightMapper.encoderBlockKeys(block: 30)
        for (_, fullKey) in block30Keys {
            XCTAssertNotNil(weights[fullKey], "Key '\(fullKey)' not found in weights")
        }
        
        // Verify pos_embed exists
        XCTAssertNotNil(weights[WeightMapper.posEmbedKey], "pos_embed key not found")
    }
    
    /// Verify validation function works
    func testValidateEncoderBlockWeights() throws {
        let weightsPath = "/Users/kwilliams/Projects/ordo/Sensors/SAM3Metal/Tests/Artifacts/sam3_weights.npz"
        guard FileManager.default.fileExists(atPath: weightsPath) else {
            throw XCTSkip("Weights file not found")
        }
        
        let loader = ModelLoader()
        let weights = try loader.load(url: URL(fileURLWithPath: weightsPath))
        
        // All 32 blocks should validate
        for block in 0..<32 {
            XCTAssertTrue(
                WeightMapper.validateEncoderBlockWeights(weights: weights, block: block),
                "Block \(block) failed validation"
            )
        }
    }
}
