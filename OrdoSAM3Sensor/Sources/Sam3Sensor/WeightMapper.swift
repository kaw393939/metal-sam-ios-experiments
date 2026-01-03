//
//  WeightMapper.swift
//  SAM3Metal
//
//  Maps PyTorch SAM3 weight keys to Swift component format
//

import Foundation
import Metal

/// Maps SAM3 checkpoint weight keys to Swift format
/// 
/// SAM3 keys use full paths like:
/// - `backbone.vision_backbone.trunk.blocks.0.norm1.weight`
/// 
/// Swift components expect simplified keys like:
/// - `block.0.norm1.weight`
public struct WeightMapper {
    
    // MARK: - Architecture Constants (from SAM3 checkpoint)
    
    /// Embedding dimension (verified from weights)
    public static let embedDim = 1024
    
    /// MLP hidden dimension (4.625 × embedDim)
    public static let mlpHiddenDim = 4736
    
    /// Number of attention heads
    public static let numHeads = 16
    
    /// Number of transformer blocks (verified from weights: 0-31 = 32 blocks)
    public static let numBlocks = 32
    
    /// Patch size for ViT
    public static let patchSize = 14
    
    /// Input image size
    public static let inputSize = 1008
    
    // MARK: - Key Prefixes
    
    private static let encoderPrefix = "backbone.vision_backbone.trunk"
    private static let geometryPrefix = "geometry_encoder"
    private static let segmentationPrefix = "segmentation_head"
    private static let transformerPrefix = "transformer"
    
    // MARK: - Encoder Key Mapping
    
    /// Get the full weight key for an encoder block component
    public static func encoderBlockKey(block: Int, component: String) -> String {
        return "\(encoderPrefix).blocks.\(block).\(component)"
    }
    
    /// Get keys for all weights in a transformer block
    public static func encoderBlockKeys(block: Int) -> [String: String] {
        let prefix = "\(encoderPrefix).blocks.\(block)"
        return [
            // LayerNorm 1
            "norm1.weight": "\(prefix).norm1.weight",
            "norm1.bias": "\(prefix).norm1.bias",
            
            // Attention (fused QKV)
            "attn.qkv.weight": "\(prefix).attn.qkv.weight",
            "attn.qkv.bias": "\(prefix).attn.qkv.bias",
            "attn.proj.weight": "\(prefix).attn.proj.weight",
            "attn.proj.bias": "\(prefix).attn.proj.bias",
            
            // LayerNorm 2
            "norm2.weight": "\(prefix).norm2.weight",
            "norm2.bias": "\(prefix).norm2.bias",
            
            // MLP (fc1, fc2)
            "mlp.fc1.weight": "\(prefix).mlp.fc1.weight",
            "mlp.fc1.bias": "\(prefix).mlp.fc1.bias",
            "mlp.fc2.weight": "\(prefix).mlp.fc2.weight",
            "mlp.fc2.bias": "\(prefix).mlp.fc2.bias"
        ]
    }
    
    /// Get positional embedding key
    public static var posEmbedKey: String {
        return "\(encoderPrefix).pos_embed"
    }
    
    /// Get patch embedding keys
    public static var patchEmbedKeys: [String: String] {
        return [
            "weight": "\(encoderPrefix).patch_embed.proj.weight",
            "bias": "\(encoderPrefix).patch_embed.proj.bias"
        ]
    }
    
    // MARK: - Geometry Encoder (Prompt Encoder) Keys
    
    public static func geometryEncoderKey(_ component: String) -> String {
        return "\(geometryPrefix).\(component)"
    }
    
    // MARK: - Neck Keys (Projection 1024 -> 256)
    
    /// Map normalized neck keys to SAM3 backbone convs (using Block 3 output / convs.3)
    public static var neckKeys: [String: String] {
        let prefix = "backbone.vision_backbone.convs.3"
        return [
            "conv1.weight": "\(prefix).conv_1x1.weight", // 1024 -> 256
            "conv1.bias": "\(prefix).conv_1x1.bias",
            "conv2.weight": "\(prefix).conv_3x3.weight", // 256 -> 256
            "conv2.bias": "\(prefix).conv_3x3.bias"
        ]
    }
    
    // MARK: - Segmentation Head (Mask Decoder) Keys
    
    public static func segmentationHeadKey(_ component: String) -> String {
        return "\(segmentationPrefix).\(component)"
    }
    
    // MARK: - Weight Loading Helpers
    
    /// Load a weight from dictionary and convert to MTLBuffer
    public static func loadBuffer(
        weights: [String: Data],
        key: String,
        device: MTLDevice
    ) -> MTLBuffer? {
        guard let data = weights[key] else {
            print("WARNING: Weight key '\(key)' not found")
            return nil
        }
        return ModelLoader.loadBuffer(from: data, device: device, label: key)
    }
    
    /// Validate that all required encoder block weights exist
    public static func validateEncoderBlockWeights(
        weights: [String: Data],
        block: Int
    ) -> Bool {
        let required = encoderBlockKeys(block: block)
        for (_, fullKey) in required {
            if weights[fullKey] == nil {
                print("MISSING: \(fullKey)")
                return false
            }
        }
        return true
    }
    
    /// Print weight shape information for debugging
    public static func printWeightInfo(weights: [String: Data], key: String) {
        if let data = weights[key] {
            let floats = data.count / 4
            print("Weight '\(key)': \(floats) floats (\(data.count) bytes)")
        } else {
            print("Weight '\(key)': NOT FOUND")
        }
    }
}
