//
//  MLXTransformerBlock.swift
//  Sam3Sensor - MLX Hybrid Migration
//
//  Complete transformer block with pre-norm architecture
//  Combines attention and MLP with residual connections
//

import Foundation
import Metal
import MLX
import MLXNN

/// Complete transformer block using MLX components
@available(macOS 15.0, *)
public class MLXTransformerBlock {
    private let embedDim: Int
    private let numHeads: Int
    private let mlpDim: Int
    private let device: MTLDevice
    
    // Components
    private let norm1: MLXLayerNorm
    private let attention: MLXAttention
    private let norm2: MLXLayerNorm
    private let mlp: MLXMLP
    
    public init(embedDim: Int, numHeads: Int, mlpDim: Int, device: MTLDevice) {
        self.embedDim = embedDim
        self.numHeads = numHeads
        self.mlpDim = mlpDim
        self.device = device
        
        // Initialize components
        self.norm1 = MLXLayerNorm(normalizedShape: [embedDim], device: device)
        self.attention = MLXAttention(embedDim: embedDim, numHeads: numHeads, device: device)
        self.norm2 = MLXLayerNorm(normalizedShape: [embedDim], device: device)
        self.mlp = MLXMLP(inputDim: embedDim, hiddenDim: mlpDim, outputDim: embedDim, device: device)
    }
    
    /// Load weights from MLXArrays with prefix
    public func loadWeights(prefix: String, weights: [String: MLXArray]) throws {
        // Norm1 (pre-attention)
        guard let norm1Gamma = weights["\(prefix).norm1.weight"],
              let norm1Beta = weights["\(prefix).norm1.bias"] else {
            throw SAM3Error.weightsNotLoaded("MLXTransformerBlock: missing norm1 weights for \(prefix)")
        }
        try norm1.loadWeights(gamma: norm1Gamma, beta: norm1Beta)
        
        // Attention
        guard let attnQKVW = weights["\(prefix).attn.qkv.weight"],
              let attnQKVB = weights["\(prefix).attn.qkv.bias"],
              let attnOutW = weights["\(prefix).attn.proj.weight"],
              let attnOutB = weights["\(prefix).attn.proj.bias"] else {
            throw SAM3Error.weightsNotLoaded("MLXTransformerBlock: missing attention weights for \(prefix)")
        }
        try attention.loadWeights(qkvWeight: attnQKVW, qkvBias: attnQKVB, outputWeight: attnOutW, outputBias: attnOutB)
        
        // Norm2 (pre-MLP)
        guard let norm2Gamma = weights["\(prefix).norm2.weight"],
              let norm2Beta = weights["\(prefix).norm2.bias"] else {
            throw SAM3Error.weightsNotLoaded("MLXTransformerBlock: missing norm2 weights for \(prefix)")
        }
        try norm2.loadWeights(gamma: norm2Gamma, beta: norm2Beta)
        
        // MLP
        guard let mlpFC1W = weights["\(prefix).mlp.fc1.weight"],
              let mlpFC1B = weights["\(prefix).mlp.fc1.bias"],
              let mlpFC2W = weights["\(prefix).mlp.fc2.weight"],
              let mlpFC2B = weights["\(prefix).mlp.fc2.bias"] else {
            throw SAM3Error.weightsNotLoaded("MLXTransformerBlock: missing MLP weights for \(prefix)")
        }
        try mlp.loadWeights(fc1W: mlpFC1W, fc1B: mlpFC1B, fc2W: mlpFC2W, fc2B: mlpFC2B)
    }
    
    /// Randomly initialize weights for synthetic benchmarking
    public func randomInitialize() {
        norm1.randomInitialize()
        attention.randomInitialize()
        norm2.randomInitialize()
        mlp.randomInitialize()
    }
    
    /// Forward pass with pre-norm architecture
    /// x: [B, N, D] input tensor
    /// rope: optional RoPE embeddings [N, D/numHeads]
    /// windowed: whether to use windowed attention
    public func callAsFunction(_ x: MLXArray, rope: MLXArray? = nil, windowed: Bool = false, windowSize: Int = 14, h: Int? = nil, w: Int? = nil) -> MLXArray {
        // Pre-norm attention with residual
        // h = x + attention(norm1(x))
        var hState = x
        let normed1 = norm1(hState)
        let attnOut = attention(normed1, rope: rope, windowed: windowed, windowSize: windowSize, h: h, w: w)
        hState = hState + attnOut
        
        // Pre-norm MLP with residual  
        // h = h + mlp(norm2(h))
        let normed2 = norm2(hState)
        let mlpOut = mlp(normed2)
        hState = hState + mlpOut
        
        return hState
    }
}
