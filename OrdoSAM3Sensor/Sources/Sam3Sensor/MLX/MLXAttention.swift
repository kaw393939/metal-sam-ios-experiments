//
//  MLXAttention.swift
//  Sam3Sensor - MLX Hybrid Migration
//
//  MLX-based Multi-Head Attention with RoPE support
//  Replaces MPSAttention for transformer blocks
//

import Foundation
import Metal
import MLX
import MLXNN
import MLXRandom

/// Multi-head attention using MLX with RoPE and windowed attention support
@available(macOS 15.0, *)
public class MLXAttention {
    private let embedDim: Int
    private let numHeads: Int
    private let headDim: Int
    private let device: MTLDevice
    
    // Weights
    private var qkvW: MLXArray  // Combined QKV projection
    private var qkvB: MLXArray
    private var outW: MLXArray
    private var outB: MLXArray
    
    public init(embedDim: Int, numHeads: Int, device: MTLDevice) {
        self.embedDim = embedDim
        self.numHeads = numHeads
        self.headDim = embedDim / numHeads
        self.device = device
        
        // Initialize (will be loaded from weights)
        self.qkvW = MLXArray.zeros([embedDim, embedDim * 3], type: Float16.self)
        self.qkvB = MLXArray.zeros([embedDim * 3], type: Float16.self)
        self.outW = MLXArray.zeros([embedDim, embedDim], type: Float16.self)
        self.outB = MLXArray.zeros([embedDim], type: Float16.self)
    }
    
    /// Load weights from MLXArray
    public func loadWeights(qkvWeight: MLXArray, qkvBias: MLXArray, outputWeight: MLXArray, outputBias: MLXArray) throws {
        // QKV projection: [3*embedDim, embedDim] → transpose
        self.qkvW = qkvWeight.T.asType(DType.float16)
        self.qkvB = qkvBias.asType(DType.float16)
        
        // Output projection: [embedDim, embedDim] → transpose
        self.outW = outputWeight.T.asType(DType.float16)
        self.outB = outputBias.asType(DType.float16)
    }
    
    /// Randomly initialize weights for synthetic benchmarking
    public func randomInitialize() {
        self.qkvW = MLXRandom.uniform(low: -0.1, high: 0.1, [embedDim, embedDim * 3]).asType(DType.float16)
        self.qkvB = MLXRandom.uniform(low: -0.1, high: 0.1, [embedDim * 3]).asType(DType.float16)
        self.outW = MLXRandom.uniform(low: -0.1, high: 0.1, [embedDim, embedDim]).asType(DType.float16)
        self.outB = MLXRandom.uniform(low: -0.1, high: 0.1, [embedDim]).asType(DType.float16)
    }
    
    /// Forward pass with optional RoPE and windowed attention
    public func callAsFunction(_ x: MLXArray, rope: MLXArray? = nil, windowed: Bool = false, windowSize: Int = 14, h: Int? = nil, w: Int? = nil) -> MLXArray {
        let shape = x.shape
        let B = shape[0]
        let N = shape[1]
        
        // 1. QKV Projection (Global)
        let qkv = MLX.matmul(x, qkvW) + qkvB
        let chunkSize = embedDim
        
        // Split and Reshape to [B, N, H, D]
        let q = qkv[0..., 0..., 0..<chunkSize].reshaped([B, N, numHeads, headDim])
        let k = qkv[0..., 0..., chunkSize..<(2*chunkSize)].reshaped([B, N, numHeads, headDim])
        let v = qkv[0..., 0..., (2*chunkSize)..<(3*chunkSize)].reshaped([B, N, numHeads, headDim])
        
        // 2. Apply RoPE (Global)
        var qAttn = q
        var kAttn = k
        var vAttn = v
        
        if let rope = rope {
            qAttn = applyRoPE(q, rope: rope)
            kAttn = applyRoPE(k, rope: rope) // RoPE usually on Q and K
        }
        
        // 3. Partition if windowed
        var doWindow = windowed
        if doWindow {
             if h == nil || w == nil {
                 // Warn or fallback? For correctness, if we lack h/w, strictly we can't window partition 2D.
                 // We will fallback to global attention.
                 doWindow = false
             }
        }
        
        var padH = 0
        var padW = 0
        
        if doWindow {
            let H = h!
            let W = w!
            padH = (H + windowSize - 1) / windowSize * windowSize
            padW = (W + windowSize - 1) / windowSize * windowSize
            
            // Helper to pad & partition: [B, N, NH, HD] -> [B*nW, ws*ws, NH, HD]
            func partition(_ t: MLXArray) -> MLXArray {
                // Must reshape N -> H, W first.
                // Assuming N == H*W.
                var img = t.reshaped([B, H, W, numHeads, headDim])
                
                if padH != H || padW != W {
                    var padded = MLX.zeros([B, padH, padW, numHeads, headDim], type: Float16.self)
                    padded[0..., 0..<H, 0..<W, 0..., 0...] = img
                    img = padded
                }
                
                let hB = padH / windowSize
                let wB = padW / windowSize
                
                // Reshape to [B, hB, ws, wB, ws, NH, HD]
                var p = img.reshaped([B, hB, windowSize, wB, windowSize, numHeads, headDim])
                // Transpose to [B, hB, wB, ws, ws, NH, HD]
                p = p.transposed(axes: [0, 1, 3, 2, 4, 5, 6])
                // Flatten to [B*hB*wB, ws*ws, NH, HD]
                return p.reshaped([-1, windowSize * windowSize, numHeads, headDim])
            }
            
            qAttn = partition(qAttn)
            kAttn = partition(kAttn)
            vAttn = partition(vAttn)
        }
        
        // 4. Attention (Flash)
        // MLXFast expects queries: [B, NH, L, D]
        let qT = qAttn.transposed(axes: [0, 2, 1, 3])
        let kT = kAttn.transposed(axes: [0, 2, 1, 3])
        let vT = vAttn.transposed(axes: [0, 2, 1, 3])
        
        let attnOut = MLXFast.scaledDotProductAttention(queries: qT, keys: kT, values: vT, scale: Float(1.0/sqrt(Float(headDim))), mask: .none)
        
        // 5. Reverse Partition / Reshape
        var output = attnOut.transposed(axes: [0, 2, 1, 3]) // [B_win, N_win, NH, HD]
        
        if doWindow {
            let H = h!
            let W = w!
            let hB = padH / windowSize
            let wB = padW / windowSize
            
            // [B, hB, wB, ws, ws, NH, HD] -> [B, hB, wB, ws, ws, NH, HD] if flattened incorrectly?
            // Input shape is [-1, ws*ws, NH, HD] which is [B*hB*wB, ws*ws, NH, HD]
            
            var X = output.reshaped([B, hB, wB, windowSize, windowSize, numHeads, headDim])
            
            // Transpose back: [B, hB, ws, wB, ws, NH, HD]
            X = X.transposed(axes: [0, 1, 3, 2, 4, 5, 6])
            
            // Flatten spatial: [B, padH, padW, NH, HD]
            X = X.reshaped([B, padH, padW, numHeads, headDim])
            
            // Unpad
            if padH != H || padW != W {
                X = X[0..., 0..<H, 0..<W, 0..., 0...]
            }
            output = X.reshaped([B, N, numHeads, headDim])
        }
        
        // 6. Proj Output
        let flat = output.reshaped([B, N, embedDim])
        return MLX.matmul(flat, outW) + outB
    }
    
    private func windowPartition(_ x: MLXArray, windowSize: Int, H: Int, W: Int) -> MLXArray {
         // ... implementation implicit above
         return x  
    }
    
    /// Apply rotary position embeddings
    private func applyRoPE(_ x: MLXArray, rope: MLXArray) -> MLXArray {
        let shape = x.shape
        // let B = shape[0], N = shape[1], H = shape[2]
        let D = shape[3]
        
        // Split into even/odd for rotation
        let halfD = D / 2
        let x1 = x[0..., 0..., 0..., 0..<halfD]
        let x2 = x[0..., 0..., 0..., halfD..<D]
        
        // RoPE: [N, D/2] → broadcast to [B, N, H, D/2]
        // Currently [N, halfD]. Need [1, N, 1, halfD] to match x [B, N, H, halfD]
        let cos = rope[0..., 0..<halfD].reshaped([1, -1, 1, halfD])
        let sin = rope[0..., halfD..<D].reshaped([1, -1, 1, halfD])
        
        // Rotate: x1*cos - x2*sin, x1*sin + x2*cos
        let rotated1 = x1 * cos - x2 * sin
        let rotated2 = x1 * sin + x2 * cos
        
        return concatenated([rotated1, rotated2], axis: -1)
    }
    

}
