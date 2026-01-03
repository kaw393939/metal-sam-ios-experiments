//
//  SAM3Encoder.swift
//  SAM3Metal
//
//  SAM3-compatible Vision Transformer Encoder
//  Architecture matches the actual SAM3 checkpoint dimensions
//

import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

/// SAM3-compatible Vision Transformer Encoder
/// Uses correct dimensions from SAM3 checkpoint:
/// - embedDim: 1024
/// - mlpHiddenDim: 4736
/// - numBlocks: 32
/// - numHeads: 16
/// - patchSize: 14
@available(macOS 15.0, *)
public final class SAM3Encoder {
    
    private let device: MTLDevice
    private let graph: MPSGraph

    private let config: SAM3EncoderConfig
    
    // Architecture parameters (from WeightMapper constants)
    public let embedDim: Int
    public let mlpHiddenDim: Int
    public let numBlocks: Int
    public let numHeads: Int
    public let patchSize: Int
    public let inputSize: Int
    
    // Components
    private let rope: RoPE
    private let ropeWindow: RoPE // New: For 24x24 window relative freqs
    
    // Weight storage
    private var weights: [String: Data]?
    private var weightBuffers: [String: MTLBuffer] = [:]
    
    private let enableHalfPrecision: Bool
    
    public convenience init(device: MTLDevice, enableHalfPrecision: Bool = true) {
        self.init(device: device, config: .sam3Checkpoint, enableHalfPrecision: enableHalfPrecision)
    }

    public init(device: MTLDevice, config: SAM3EncoderConfig, enableHalfPrecision: Bool = true) {
        self.device = device
        self.enableHalfPrecision = enableHalfPrecision
        self.graph = MPSGraph()

        self.config = config
        
        // Use verified SAM3 architecture constants
        self.embedDim = config.embedDim
        self.mlpHiddenDim = config.mlpHiddenDim
        self.numBlocks = config.numBlocks
        self.numHeads = config.numHeads
        self.patchSize = config.patchSize
        self.inputSize = config.inputSize

        let dimPerHead = config.dimPerHead
        let spatialTokens = config.spatialTokenCount
        let windowTokens = config.windowTokenCount
        
        // headDim = embedDim / numHeads. RoPE needs headDim.
        self.rope = RoPE(device: device, numHeads: config.numHeads, headDim: dimPerHead, maxSeqLen: spatialTokens)
        // windowSize^2 tokens for window
        self.ropeWindow = RoPE(device: device, numHeads: config.numHeads, headDim: dimPerHead, maxSeqLen: windowTokens)
    }
    
    /// Load weights from ModelWeights dictionary
    public func loadWeights(_ weights: [String: Data]) {
        self.weights = weights

        // Upload frequently-used weights to GPU once (Private/Heap-backed when possible).
        // This avoids re-materializing weights from Data on every graph execution.
        if let queue = device.makeCommandQueue(), let commandBuffer = queue.makeCommandBuffer() {
            commandBuffer.label = "SAM3Encoder.WeightUpload"

            var uploaded: [String: MTLBuffer] = [:]
            uploaded.reserveCapacity(weights.count)

            // Keep any no-copy staged Data alive until the blit completes.
            var retainedStagingData: [Data] = []
            retainedStagingData.reserveCapacity(256)

            for (key, data) in weights {
                // Skip any non-tensor blobs
                if key == "manual_gaussian_matrix" { continue }

                let shared: MTLBuffer?
                if data.count > 0 {
                    // Prefer no-copy staging to avoid an extra CPU memcpy when Data is already file-mapped.
                    shared = data.withUnsafeBytes { raw -> MTLBuffer? in
                        guard let base = raw.baseAddress else { return nil }
                        let ptr = UnsafeMutableRawPointer(mutating: base)
                        return device.makeBuffer(
                            bytesNoCopy: ptr,
                            length: data.count,
                            options: .storageModeShared,
                            deallocator: nil
                        )
                    }
                } else {
                    shared = nil
                }

                guard let sharedBuf = shared ?? device.makeBuffer(bytes: (data as NSData).bytes, length: data.count, options: .storageModeShared) else {
                    continue
                }
                if shared != nil {
                    retainedStagingData.append(data)
                }

                sharedBuf.label = "w.shared/\(key)"

                if let priv = BufferAllocator.shared.makePrivateCopy(from: sharedBuf, device: device, commandBuffer: commandBuffer, label: "w/\(key)") {
                    uploaded[key] = priv
                } else {
                    // Fallback to shared if private copy fails
                    uploaded[key] = sharedBuf
                }
            }

            self.weightBuffers = uploaded
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            // Keep alive until after GPU upload completes.
            _ = retainedStagingData
        } else {
            self.weightBuffers = [:]
        }
        
        // Validate critical weights exist
        let posEmbedKey = WeightMapper.posEmbedKey
        guard weights[posEmbedKey] != nil else {
            print("WARNING: pos_embed not found in weights")
            return
        }
        
        // Validate all blocks
        for block in 0..<numBlocks {
            if !WeightMapper.validateEncoderBlockWeights(weights: weights, block: block) {
                print("WARNING: Block \(block) weights incomplete")
            }
        }
        
        print("SAM3Encoder: Loaded weights for \(numBlocks) blocks")
    }

    // Test-only hook (via @testable import) to verify weight upload behavior.
    internal func _debugWeightBuffer(for key: String) -> MTLBuffer? {
        weightBuffers[key]
    }
    
    /// Build the encoder graph
    /// Input: [1, H, W, 3] image tensor
    /// Output: [1, H/14, W/14, embedDim] feature tensor
    public func buildGraph(
        input: MPSGraphTensor,
        graph: MPSGraph,
        computeDT: MPSDataType = .float32
    ) -> (output: MPSGraphTensor, placeholders: [String: MPSGraphTensor]) {
        var placeholders: [String: MPSGraphTensor] = [:]

        let gridSize = config.checkpointGridSize
        let spatialTokens = config.spatialTokenCount
        let windowSize = config.windowSize
        let windowTokens = config.windowTokenCount
        let headHalfDim = config.dimPerHead / 2
        let tiles = max(1, gridSize / windowSize)
        
        // Compute Data Type: Float16 for M3 performance, Float32 for legacy/fallback
        let computeDT: MPSDataType = enableHalfPrecision ? .float16 : .float32
        
        let xInput = graph.cast(input, to: computeDT, name: "input_cast")
        
        // Helper: Create Baked Constant or Placeholder
        func loadWeight(_ phName: String, shape: [NSNumber]) -> MPSGraphTensor {
            let fullKey: String
            if phName == "pos_embed" || phName == "pos_embed_in" {
                fullKey = WeightMapper.posEmbedKey
            } else if phName == "cls_token" {
                fullKey = "backbone.vision_backbone.trunk.cls_token"
            } else if phName.hasPrefix("block.") {
                let components = phName.components(separatedBy: ".")
                if components.count >= 4, let blockIdx = Int(components[1]) {
                    let suffix = components.dropFirst(2).joined(separator: ".")
                    fullKey = WeightMapper.encoderBlockKey(block: blockIdx, component: suffix)
                } else {
                    fullKey = phName
                }
            } else if phName.hasPrefix("patch_embed.") {
                let suffix = phName.replacingOccurrences(of: "patch_embed.", with: "")
                fullKey = WeightMapper.patchEmbedKeys[suffix.contains("weight") ? "weight" : "bias"] ?? phName
            } else {
                fullKey = phName
            }

            if let data = self.weights?[fullKey] {
                return graph.constant(data, shape: shape, dataType: computeDT)
            }

            let ph = graph.placeholder(shape: shape, dataType: computeDT, name: phName)
            placeholders[phName] = ph
            return ph
        }
        
        // RoPE Frequencies placeholder: [spatialTokens, headHalfDim, 2]
        let ropeFreqs = graph.placeholder(
            shape: [NSNumber(value: spatialTokens), NSNumber(value: headHalfDim), 2],
            dataType: .float32,
            name: "rope_freqs"
        )
        placeholders["rope_freqs"] = ropeFreqs
        
        // Windowed RoPE: [windowTokens, headHalfDim, 2] -> Tile to [spatialTokens, headHalfDim, 2]
        let ropeFreqsWindowCheck = graph.placeholder(
            shape: [NSNumber(value: windowTokens), NSNumber(value: headHalfDim), 2],
            dataType: .float32,
            name: "rope_freqs_window"
        )
        placeholders["rope_freqs_window"] = ropeFreqsWindowCheck
        
        // Tile Logic: [windowTokens, headHalfDim, 2] -> [W, W, headHalfDim, 2] -> Tile to [G, G, headHalfDim, 2] -> [spatialTokens, headHalfDim, 2]
        let rGrid = graph.reshape(
            ropeFreqsWindowCheck,
            shape: [NSNumber(value: windowSize), NSNumber(value: windowSize), NSNumber(value: headHalfDim), 2],
            name: "rope_win_grid"
        )

        var tiledCols = rGrid
        if tiles > 1 {
            for _ in 1..<tiles {
                tiledCols = graph.concatTensors([tiledCols, rGrid], dimension: 0, name: "rope_win_col")
            }
        }

        var tiledFull = tiledCols
        if tiles > 1 {
            for _ in 1..<tiles {
                tiledFull = graph.concatTensors([tiledFull, tiledCols], dimension: 1, name: "rope_win_full")
            }
        }

        _ = graph.reshape(
            tiledFull,
            shape: [NSNumber(value: spatialTokens), NSNumber(value: headHalfDim), 2],
            name: "rope_win_flat"
        )
        
        // 1. Patch Embedding
        // Conv2d(3, 1024, kernel_size=14, stride=14)
        // 1. Patch Embedding
        // Conv2d(3, 1024, kernel_size=14, stride=14)
        let patchWeight = loadWeight("patch_embed.proj.weight", shape: [NSNumber(value: embedDim), 3, NSNumber(value: patchSize), NSNumber(value: patchSize)])
        let patchBias = loadWeight("patch_embed.proj.bias", shape: [1, 1, 1, NSNumber(value: embedDim)])
        
        let patchDesc = MPSGraphConvolution2DOpDescriptor(
            strideInX: patchSize, strideInY: patchSize,
            dilationRateInX: 1, dilationRateInY: 1,
            groups: 1,
            paddingLeft: 0, paddingRight: 0, paddingTop: 0, paddingBottom: 0,
            paddingStyle: .explicit,
            dataLayout: .NHWC,
            weightsLayout: .OIHW
        )!
        
        var x = graph.convolution2D(xInput, weights: patchWeight, descriptor: patchDesc, name: "patch_embed")
        x = graph.addition(x, graph.cast(patchBias, to: computeDT, name: "patch_bias_cast"), name: "patch_embed_bias")
        
        // Flatten spatial dims for transformer: [1, H', W', D] -> [1, H'*W', D]
        let shape = graph.shapeOf(x, name: "x_shape")
        let batchDim = graph.sliceTensor(shape, dimension: 0, start: 0, length: 1, name: "batch")
        let hDim = graph.sliceTensor(shape, dimension: 0, start: 1, length: 1, name: "h")
        let wDim = graph.sliceTensor(shape, dimension: 0, start: 2, length: 1, name: "w")
        let spatialCount = graph.multiplication(hDim, wDim, name: "hw")
        let seqShape = graph.concatTensors([batchDim, spatialCount, graph.constant(Double(embedDim), shape: [1], dataType: .int32)], dimension: 0, name: "seq_shape")
        x = graph.reshape(x, shapeTensor: seqShape, name: "flatten")
        x = graph.cast(x, to: computeDT, name: "flatten_cast")
        
        // 2. CLS Token
        // Weights don't have cls_token key, but pos_embed is 577 (1+24x24).
        // Protocol: Add a zero-initialized CLS token if not provided.
        // Needs matching datatype
        // 2. CLS Token
        // Weights don't have cls_token key, but pos_embed is 577 (1+24x24).
        // Protocol: Add a zero-initialized CLS token if not provided.
        let clsToken = loadWeight("cls_token", shape: [1, 1, NSNumber(value: embedDim)])
        
        // Prepend CLS: [1, 1, D] + [1, HW, D] -> [1, 1+HW, D]
        x = graph.concatTensors([clsToken, x], dimension: 1, name: "prepend_cls")
        
        // 3. Add Positional Embedding
        // 3. Add Positional Embedding
        let posEmbed = loadWeight("pos_embed_in", shape: [1, NSNumber(value: config.posEmbedSeqLen), NSNumber(value: embedDim)])
        // placeholder key is mapped manually in addFeeds usually, but we need it in 'placeholders' dict.
        // loadWeight does that. But key in dict is "pos_embed_in". 
        // addFeeds handles "pos_embed" mapping.
        if let ph = placeholders["pos_embed_in"] {
            placeholders["pos_embed"] = ph
        }
        
        // Interpolate posEmbed to match sequence length (1 + gridSize*gridSize)
        let clsPos = graph.sliceTensor(posEmbed, dimension: 1, start: 0, length: 1, name: "cls_pos")
        let spatialPos = graph.sliceTensor(posEmbed, dimension: 1, start: 1, length: windowTokens, name: "spatial_pos_flat")
        
        // windowSize x windowSize -> gridSize x gridSize
        let spatialPosGrid = graph.reshape(spatialPos, shape: [1, NSNumber(value: windowSize), NSNumber(value: windowSize), NSNumber(value: embedDim)], name: "spatial_pos_grid")
        let upsampledPosGrid = graph.resize(spatialPosGrid, 
                          size: [NSNumber(value: gridSize), NSNumber(value: gridSize)], 
                                          mode: .bilinear, 
                                          centerResult: true, 
                                          alignCorners: false, 
                                          layout: .NHWC,
                                          name: "upsample_pos")
        let upsampledPosFlat = graph.reshape(upsampledPosGrid, shape: [1, NSNumber(value: spatialTokens), NSNumber(value: embedDim)], name: "upsample_pos_flat")
        
        // Concat: [1, 1, D] + [1, spatialTokens, D] -> [1, 1+spatialTokens, D]
        let finalPosEmbed = graph.concatTensors([clsPos, upsampledPosFlat], dimension: 1, name: "final_pos_embed")
        let finalPosEmbedCast = graph.cast(finalPosEmbed, to: computeDT, name: "final_pos_embed_cast")
        
        x = graph.addition(x, finalPosEmbedCast, name: "add_pos_embed")
        
        // 3. Transformer Blocks
        // Global Attention Layers (0-indexed): roughly every 1/4 of blocks.
        // For 32 blocks -> {7,15,23,31}
        let globalStride = max(1, numBlocks / 4)
        let globalLayers = Set(stride(from: globalStride - 1, through: max(0, numBlocks - 1), by: globalStride))
        
        for blockIdx in 0..<numBlocks {
            let isGlobal = globalLayers.contains(blockIdx)
            // If Global, windowed = false. If not, windowed = true.
            let windowed = !isGlobal 
            
            // Choose RoPE: Global or Window-Relative (Tiled)
            // Since we are forcing Global, use ropeFreqs
            let currentRoPE = ropeFreqs
            
            let (blockOutput, blockPlaceholders) = buildTransformerBlock(
                input: x,
                graph: graph,
                blockIndex: blockIdx,
                ropeFreqs: graph.cast(currentRoPE, to: computeDT, name: "block.\(blockIdx).rope_cast"),
                windowed: windowed,
                computeDT: computeDT
            )
            x = blockOutput
            
            // Merge placeholders
            for (key, ph) in blockPlaceholders {
                placeholders[key] = ph
            }
        }

        
        // Final output: Slice off CLS token and reshape back to gridSize x gridSize
        let spatialOut = graph.sliceTensor(x, dimension: 1, start: 1, length: spatialTokens, name: "slice_spatial_out")
        let finalOut = graph.reshape(spatialOut, shape: [1, NSNumber(value: gridSize), NSNumber(value: gridSize), NSNumber(value: embedDim)], name: "encoder_final_reshape")
        
        return (finalOut, placeholders)
    }
    
    /// Build a single transformer block
    private func buildTransformerBlock(
        input: MPSGraphTensor,
        graph: MPSGraph,
        blockIndex: Int,
        ropeFreqs: MPSGraphTensor,
        windowed: Bool,
        computeDT: MPSDataType
    ) -> (output: MPSGraphTensor, placeholders: [String: MPSGraphTensor]) {
        var placeholders: [String: MPSGraphTensor] = [:]
        let prefix = "block.\(blockIndex)"
        
        // LayerNorm 1
        let (ln1Out, ln1Ph) = buildLayerNorm(input: input, graph: graph, name: "\(prefix).norm1", computeDT: computeDT)
        placeholders.merge(ln1Ph) { $1 }
        
        // Attention (with fused QKV and RoPE)
        let (attnOut, attnPh) = buildFusedQKVAttention(
            input: ln1Out, 
            graph: graph, 
            name: "\(prefix).attn",
            ropeFreqs: ropeFreqs,
            windowed: windowed,
            computeDT: computeDT
        )
        placeholders.merge(attnPh) { $1 }
        
        // Residual 1
        let res1 = graph.addition(input, attnOut, name: "\(prefix).res1")
        
        // LayerNorm 2
        let (ln2Out, ln2Ph) = buildLayerNorm(input: res1, graph: graph, name: "\(prefix).norm2", computeDT: computeDT)
        placeholders.merge(ln2Ph) { $1 }
        
        // MLP
        let (mlpOut, mlpPh) = buildMLP(input: ln2Out, graph: graph, name: "\(prefix).mlp", computeDT: computeDT)
        placeholders.merge(mlpPh) { $1 }
        
        // Residual 2
        let output = graph.addition(res1, mlpOut, name: "\(prefix).res2")
        
        return (output, placeholders)
    }
    
    /// Build LayerNorm
    private func buildLayerNorm(
        input: MPSGraphTensor,
        graph: MPSGraph,
        name: String,
        computeDT: MPSDataType
    ) -> (output: MPSGraphTensor, placeholders: [String: MPSGraphTensor]) {
        var localPlaceholders: [String: MPSGraphTensor] = [:]
        
        func loadLocal(_ phName: String, shape: [NSNumber]) -> MPSGraphTensor {
            let fullKey: String
            if phName.hasPrefix("block.") {
                let components = phName.components(separatedBy: ".")
                if components.count >= 4, let blockIdx = Int(components[1]) {
                    let suffix = components.dropFirst(2).joined(separator: ".")
                    fullKey = WeightMapper.encoderBlockKey(block: blockIdx, component: suffix)
                } else {
                    fullKey = phName
                }
            } else {
                fullKey = phName
            }

            if let data = self.weights?[fullKey] {
                return graph.constant(data, shape: shape, dataType: computeDT)
            }

            let ph = graph.placeholder(shape: shape, dataType: computeDT, name: phName)
            localPlaceholders[phName] = ph
            return ph
        }
        
        let gamma = loadLocal("\(name).weight", shape: [1, 1, NSNumber(value: embedDim)])
        let beta = loadLocal("\(name).bias", shape: [1, 1, NSNumber(value: embedDim)])
        
        // Compute in F16 if possible
        let mean = graph.mean(of: input, axes: [-1], name: "\(name).mean")
        let centered = graph.subtraction(input, mean, name: "\(name).center")
        let variance = graph.mean(of: graph.square(with: centered, name: "\(name).sq"), axes: [-1], name: "\(name).var")
        let epsilon = graph.constant(1e-5, dataType: computeDT)
        let std = graph.squareRoot(with: graph.addition(variance, epsilon, name: "\(name).eps"), name: "\(name).std")
        let normalized = graph.division(centered, std, name: "\(name).norm")
        let scaled = graph.multiplication(normalized, gamma, name: "\(name).scale")
        let output = graph.addition(scaled, beta, name: "\(name).shift")
        
        return (output, localPlaceholders)
    }
    
    /// Window Partition: [B, H, W, C] -> [B*numWindows, windowSize*windowSize, C]
    private func windowPartition(
        input: MPSGraphTensor,
        graph: MPSGraph,
        windowSize: Int = 24,
        name: String
    ) -> MPSGraphTensor {
        let gridSize = config.checkpointGridSize
        let partitionsCount = max(1, gridSize / windowSize)
        
        // Assume input is [B, gridSize*gridSize, C] (flattened grid)
        // 1. Reshape to [B, gridSize, gridSize, C]
        let shape = graph.shapeOf(input, name: "\(name)/shape")
        let b = graph.sliceTensor(shape, dimension: 0, start: 0, length: 1, name: "\(name)/b")
        let c = graph.sliceTensor(shape, dimension: 0, start: 2, length: 1, name: "\(name)/c")
        
        let gridShape = graph.concatTensors([
            b, 
            graph.constant(Double(gridSize), shape: [1], dataType: .int32), 
            graph.constant(Double(gridSize), shape: [1], dataType: .int32), 
            c
        ], dimension: 0, name: "\(name)/gridShape")
        
        let x = graph.reshape(input, shapeTensor: gridShape, name: "\(name)/grid")
        
        // 2. Reshape to [B, P, windowSize, P, windowSize, C] where P = gridSize / windowSize
        let partitions = graph.constant(Double(partitionsCount), shape: [1], dataType: .int32)
        let wSize = graph.constant(Double(windowSize), shape: [1], dataType: .int32)
        
        let partitionShape = graph.concatTensors([b, partitions, wSize, partitions, wSize, c], dimension: 0, name: "\(name)/partShape")
        let xPart = graph.reshape(x, shapeTensor: partitionShape, name: "\(name)/partitioned")
        
        // 3. Transpose to [B, P, P, windowSize, windowSize, C]
        let xTrans = graph.transposeTensor(xPart, dimension: 2, withDimension: 3, name: "\(name)/trans")
        
        // 4. Reshape to [B*(P*P), windowSize*windowSize, C]
        let pSq = partitionsCount * partitionsCount
        let bTimesP2 = graph.multiplication(b, graph.constant(Double(pSq), shape: [1], dataType: .int32), name: "\(name)/bP2")
        let wSq = graph.constant(Double(windowSize * windowSize), shape: [1], dataType: .int32)
        
        let finalShape = graph.concatTensors([bTimesP2, wSq, c], dimension: 0, name: "\(name)/finalShape")
        return graph.reshape(xTrans, shapeTensor: finalShape, name: "\(name)/out")
    }
    
    /// Window Reverse: [B*numWindows, windowSize*windowSize, C] -> [B, H, W, C]
    private func windowReverse(
        windows: MPSGraphTensor,
        graph: MPSGraph,
        windowSize: Int = 24,
        originalBatchSize: MPSGraphTensor,
        name: String
    ) -> MPSGraphTensor {
        let gridSize = config.checkpointGridSize
        let partitionsCount = max(1, gridSize / windowSize)

        // windows: [B*(P*P), windowSize*windowSize, C]
        let shape = graph.shapeOf(windows, name: "\(name)/shape")
        let c = graph.sliceTensor(shape, dimension: 0, start: 2, length: 1, name: "\(name)/c")
        
        // 1. Reshape to [B, P, P, windowSize, windowSize, C]
        let partitions = graph.constant(Double(partitionsCount), shape: [1], dataType: .int32)
        let wSize = graph.constant(Double(windowSize), shape: [1], dataType: .int32)
        
        let viewShape = graph.concatTensors([originalBatchSize, partitions, partitions, wSize, wSize, c], dimension: 0, name: "\(name)/viewShape")
        let xView = graph.reshape(windows, shapeTensor: viewShape, name: "\(name)/view")
        
        // 2. Transpose to [B, P, windowSize, P, windowSize, C]
        let xTrans = graph.transposeTensor(xView, dimension: 2, withDimension: 3, name: "\(name)/trans")
        
        // 3. Reshape to [B, gridSize, gridSize, C]
        let h = graph.constant(Double(gridSize), shape: [1], dataType: .int32)
        let w = graph.constant(Double(gridSize), shape: [1], dataType: .int32)
        
        let gridShape = graph.concatTensors([originalBatchSize, h, w, c], dimension: 0, name: "\(name)/gridShape")
        let xGrid = graph.reshape(xTrans, shapeTensor: gridShape, name: "\(name)/grid")
        
        // 4. Flatten to [B, gridSize*gridSize, C]
        let hw = graph.constant(Double(gridSize * gridSize), shape: [1], dataType: .int32)
        let flatShape = graph.concatTensors([originalBatchSize, hw, c], dimension: 0, name: "\(name)/flatShape")
        
        return graph.reshape(xGrid, shapeTensor: flatShape, name: "\(name)/out")
    }

    /// Build Attention with Fused QKV weights
    private func buildFusedQKVAttention(
        input: MPSGraphTensor,
        graph: MPSGraph,
        name: String,
        ropeFreqs: MPSGraphTensor,
        windowed: Bool,
        computeDT: MPSDataType
    ) -> (output: MPSGraphTensor, placeholders: [String: MPSGraphTensor]) {
        let dimPerHead = embedDim / numHeads
        var localPlaceholders: [String: MPSGraphTensor] = [:]
        
        func loadLocal(_ phName: String, shape: [NSNumber]) -> MPSGraphTensor {
            let fullKey: String
            if phName.hasPrefix("block.") {
                let components = phName.components(separatedBy: ".")
                if components.count >= 4, let blockIdx = Int(components[1]) {
                    let suffix = components.dropFirst(2).joined(separator: ".")
                    fullKey = WeightMapper.encoderBlockKey(block: blockIdx, component: suffix)
                } else {
                    fullKey = phName
                }
            } else {
                fullKey = phName
            }

            if let data = self.weights?[fullKey] {
                return graph.constant(data, shape: shape, dataType: computeDT)
            }

            let ph = graph.placeholder(shape: shape, dataType: computeDT, name: phName)
            localPlaceholders[phName] = ph
            return ph
        }
        
        // Fused QKV weight: [3*embedDim, embedDim]
        let qkvWeight = loadLocal("\(name).qkv.weight", shape: [NSNumber(value: 3 * embedDim), NSNumber(value: embedDim)])
        let qkvBias = loadLocal("\(name).qkv.bias", shape: [1, 1, NSNumber(value: 3 * embedDim)])
        
        // Output projection
        let projWeight = loadLocal("\(name).proj.weight", shape: [NSNumber(value: embedDim), NSNumber(value: embedDim)])
        let projBias = loadLocal("\(name).proj.bias", shape: [1, 1, NSNumber(value: embedDim)])
        
        // Compute QKV: [B, N, 3*D] (N=5185)
        let qkvWeightT = graph.transposeTensor(qkvWeight, dimension: 0, withDimension: 1, name: "\(name).qkv.wT")
        var qkv = graph.matrixMultiplication(primary: input, secondary: qkvWeightT, name: "\(name).qkv.mm")
        qkv = graph.addition(qkv, qkvBias, name: "\(name).qkv.add")
        
        // Split into Q, K, V
        let q = graph.sliceTensor(qkv, dimension: 2, start: 0, length: embedDim, name: "\(name).q")
        let k = graph.sliceTensor(qkv, dimension: 2, start: embedDim, length: embedDim, name: "\(name).k")
        let v = graph.sliceTensor(qkv, dimension: 2, start: 2 * embedDim, length: embedDim, name: "\(name).v")
        
        // Apply RoPE to Q and K (Skipping CLS token)
        // Note: RoPE is applied Globally BEFORE windowing
        func applyRoPE(_ x: MPSGraphTensor, name: String) -> MPSGraphTensor {
            let spatialTokens = config.spatialTokenCount
            let dimPerHead = embedDim / numHeads
            let headHalfDim = dimPerHead / 2

            let cls = graph.sliceTensor(x, dimension: 1, start: 0, length: 1, name: "\(name)/cls")
            let spatial = graph.sliceTensor(x, dimension: 1, start: 1, length: spatialTokens, name: "\(name)/spatial")
            
            // Reshape spatial to [1, spatialTokens, numHeads, dimPerHead]
            let xR = graph.reshape(spatial, shape: [1, NSNumber(value: spatialTokens), NSNumber(value: numHeads), NSNumber(value: dimPerHead)], name: "\(name)/reshape")
            // Reshape to [1, spatialTokens, numHeads, headHalfDim, 2] to separate real/imag pairs
            let xP = graph.reshape(xR, shape: [1, NSNumber(value: spatialTokens), NSNumber(value: numHeads), NSNumber(value: headHalfDim), 2], name: "\(name)/pairs")
            
            // x0/x1: [1, spatialTokens, numHeads, headHalfDim]
            let x0 = graph.sliceTensor(xP, dimension: 4, start: 0, length: 1, name: "\(name)/x0")
            let x1 = graph.sliceTensor(xP, dimension: 4, start: 1, length: 1, name: "\(name)/x1")
            
            // x_rotated = [-x1, x0]
            let negX1 = graph.multiplication(x1, graph.constant(-1.0, dataType: computeDT), name: "\(name)/negX1")
            let xRotated = graph.concatTensors([negX1, x0], dimension: 4, name: "\(name)/rotated")
            
            // Freqs is [spatialTokens, headHalfDim, 2]. Broadcast to [1, spatialTokens, numHeads, headHalfDim, 2]
            let cosFreq = graph.sliceTensor(ropeFreqs, dimension: 2, start: 0, length: 1, name: "\(name)/cos")
            let sinFreq = graph.sliceTensor(ropeFreqs, dimension: 2, start: 1, length: 1, name: "\(name)/sin")
            
            let cosE = graph.reshape(cosFreq, shape: [1, NSNumber(value: spatialTokens), 1, NSNumber(value: headHalfDim), 1], name: "\(name)/cosE")
            let sinE = graph.reshape(sinFreq, shape: [1, NSNumber(value: spatialTokens), 1, NSNumber(value: headHalfDim), 1], name: "\(name)/sinE")
            
            let outSpatial = graph.addition(
                graph.multiplication(xP, cosE, name: "\(name)/mul_cos"),
                graph.multiplication(xRotated, sinE, name: "\(name)/mul_sin"),
                name: "\(name)/result_pairs"
            )
            
            let flattenedSpatial = graph.reshape(outSpatial, shape: [1, NSNumber(value: spatialTokens), NSNumber(value: embedDim)], name: "\(name)/flatten")
            
            // Re-concat CLS
            return graph.concatTensors([cls, flattenedSpatial], dimension: 1, name: "\(name)/final")
        }
        
        let qRoPE = applyRoPE(q, name: "\(name).q.rope")
        let kRoPE = applyRoPE(k, name: "\(name).k.rope")
        
        // Prepare for Attention
        var qEffective = qRoPE
        var kEffective = kRoPE
        var vEffective = v
        
        let inputShape = graph.shapeOf(input, name: "\(name).in_shape")
        let originalBatchSize = graph.sliceTensor(inputShape, dimension: 0, start: 0, length: 1, name: "\(name).dimB")
        
        if windowed {
            let spatialTokens = config.spatialTokenCount
            let windowSize = config.windowSize
            // Split CLS and Spatial
            let spatialQ = graph.sliceTensor(qEffective, dimension: 1, start: 1, length: spatialTokens, name: "\(name)/spatialQ")
            let spatialK = graph.sliceTensor(kEffective, dimension: 1, start: 1, length: spatialTokens, name: "\(name)/spatialK")
            let spatialV = graph.sliceTensor(vEffective, dimension: 1, start: 1, length: spatialTokens, name: "\(name)/spatialV")
            
            // Window Partition: [B, spatialTokens, D] -> [B*(P*P), windowTokens, D]
            qEffective = windowPartition(input: spatialQ, graph: graph, windowSize: windowSize, name: "\(name)/winQ")
            kEffective = windowPartition(input: spatialK, graph: graph, windowSize: windowSize, name: "\(name)/winK")
            vEffective = windowPartition(input: spatialV, graph: graph, windowSize: windowSize, name: "\(name)/winV")
        }
        
        // Reshape for multi-head: [B', N', H, D/H] -> [B', H, N', D/H]
        // Construct dynamic shape [B', N', H, D/H]
        let effectiveShape = graph.shapeOf(qEffective, name: "\(name).eff_shape")
        let batchDim = graph.sliceTensor(effectiveShape, dimension: 0, start: 0, length: 1, name: "\(name).eff_dimB")
        let seqDim = graph.sliceTensor(effectiveShape, dimension: 0, start: 1, length: 1, name: "\(name).eff_dimN")
        let headDim = graph.constant(Double(numHeads), shape: [1], dataType: .int32)
        let dimPerHeadConst = graph.constant(Double(dimPerHead), shape: [1], dataType: .int32)
        
        // Shape: [B, N, H, D/H]
        let mhShape = graph.concatTensors([batchDim, seqDim, headDim, dimPerHeadConst], dimension: 0, name: "\(name).mh_shape")
        
        let qReshaped = graph.reshape(qEffective, shapeTensor: mhShape, name: "\(name).q.reshape")
        let qMH = graph.transposeTensor(qReshaped, dimension: 1, withDimension: 2, name: "\(name).q.mh")
        
        let kReshaped = graph.reshape(kEffective, shapeTensor: mhShape, name: "\(name).k.reshape")
        let kMH = graph.transposeTensor(kReshaped, dimension: 1, withDimension: 2, name: "\(name).k.mh")
        
        let vReshaped = graph.reshape(vEffective, shapeTensor: mhShape, name: "\(name).v.reshape")
        let vMH = graph.transposeTensor(vReshaped, dimension: 1, withDimension: 2, name: "\(name).v.mh")
        
        // Attention: prefer fused SDPA (FlashAttention-style) when available.
        let scale = Float(1.0 / sqrt(Double(dimPerHead)))
        var attnOut = AttentionKernels.scaledDotProductAttentionOrReference(
            graph: graph,
            query: qMH,
            key: kMH,
            value: vMH,
            scale: scale,
            name: "\(name).sdpa",
            implementation: .auto
        )
        
        // Reshape back: [B, H, N, D/H] -> [B, N, D]
        attnOut = graph.transposeTensor(attnOut, dimension: 1, withDimension: 2, name: "\(name).out.mh")
        attnOut = graph.reshape(attnOut, shapeTensor: effectiveShape, name: "\(name).out.reshape")
        
        // If windowed, reverse partitioning
        if windowed {
            // Recover spatial: [B*9, 576, D] -> [B, 5184, D]
            let spatialOut = windowReverse(windows: attnOut, graph: graph, windowSize: config.windowSize, originalBatchSize: originalBatchSize, name: "\(name)/winRev")
            
            // For CLS, we output zeros (Identity residual) because CLS was not part of attention
            // [B, 1, D] zeros
            let clsZeros = graph.constant(0.0, shape: [1, 1, NSNumber(value: embedDim)], dataType: computeDT)
            // Broadcast zeros to batch? B is usually 1, so [1,1,D] is fine.
            
            attnOut = graph.concatTensors([clsZeros, spatialOut], dimension: 1, name: "\(name)/final_concat")
        }
        
        // Output projection
        let projWeightT = graph.transposeTensor(projWeight, dimension: 0, withDimension: 1, name: "\(name).proj.wT")
        var output = graph.matrixMultiplication(primary: attnOut, secondary: projWeightT, name: "\(name).proj.mm")
        output = graph.addition(output, projBias, name: "\(name).proj.add")
        
        return (output, localPlaceholders)
    }
    
    /// Build MLP (2-layer with GELU)
    private func buildMLP(
        input: MPSGraphTensor,
        graph: MPSGraph,
        name: String,
        computeDT: MPSDataType
    ) -> (output: MPSGraphTensor, placeholders: [String: MPSGraphTensor]) {
        var localPlaceholders: [String: MPSGraphTensor] = [:]
        
        func loadLocal(_ phName: String, shape: [NSNumber]) -> MPSGraphTensor {
            let fullKey: String
            if phName.hasPrefix("block.") {
                let components = phName.components(separatedBy: ".")
                if components.count >= 4, let blockIdx = Int(components[1]) {
                    let suffix = components.dropFirst(2).joined(separator: ".")
                    fullKey = WeightMapper.encoderBlockKey(block: blockIdx, component: suffix)
                } else {
                    fullKey = phName
                }
            } else {
                fullKey = phName
            }

            if let data = self.weights?[fullKey] {
                return graph.constant(data, shape: shape, dataType: computeDT)
            }

            let ph = graph.placeholder(shape: shape, dataType: computeDT, name: phName)
            localPlaceholders[phName] = ph
            return ph
        }
    
        // fc1: [embedDim, mlpHiddenDim]
        let fc1Weight = loadLocal("\(name).fc1.weight", shape: [NSNumber(value: mlpHiddenDim), NSNumber(value: embedDim)])
        let fc1Bias = loadLocal("\(name).fc1.bias", shape: [1, 1, NSNumber(value: mlpHiddenDim)])
        
        // fc2: [mlpHiddenDim, embedDim]
        let fc2Weight = loadLocal("\(name).fc2.weight", shape: [NSNumber(value: embedDim), NSNumber(value: mlpHiddenDim)])
        let fc2Bias = loadLocal("\(name).fc2.bias", shape: [1, 1, NSNumber(value: embedDim)])
        
        // fc1
        let fc1WeightT = graph.transposeTensor(fc1Weight, dimension: 0, withDimension: 1, name: "\(name).fc1.wT")
        var x = graph.matrixMultiplication(primary: input, secondary: fc1WeightT, name: "\(name).fc1.mm")
        x = graph.addition(x, fc1Bias, name: "\(name).fc1.add")
        
        // GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        let pointFive = graph.constant(0.5, dataType: computeDT)
        let one = graph.constant(1.0, dataType: computeDT)
        let sqrtTwo = graph.constant(1.41421356, dataType: computeDT)
        let div = graph.division(x, sqrtTwo, name: "\(name).gelu_div")
        let erf = graph.erf(with: div, name: "\(name).gelu_erf")
        let onePlusErf = graph.addition(one, erf, name: "\(name).gelu_add")
        let halfX = graph.multiplication(pointFive, x, name: "\(name).gelu_half")
        let geluOut = graph.multiplication(halfX, onePlusErf, name: "\(name).gelu")
        
        // fc2
        let fc2WeightT = graph.transposeTensor(fc2Weight, dimension: 0, withDimension: 1, name: "\(name).fc2.wT")
        var output = graph.matrixMultiplication(primary: geluOut, secondary: fc2WeightT, name: "\(name).fc2.mm")
        output = graph.addition(output, fc2Bias, name: "\(name).fc2.add")
        
        return (output, localPlaceholders)
    }

    
    /// Add feeds for the encoder placeholders
    public func addFeeds(
        placeholders: [String: MPSGraphTensor],
        feeds: inout [MPSGraphTensor: MPSGraphTensorData]
    ) {
        guard let weights = self.weights else {
            print("SAM3Encoder: No weights loaded")
            return
        }
        
        for (key, ph) in placeholders {
            var fullKey = key
            
            // Map placeholder name to full weight key
            if key == "pos_embed" || key == "pos_embed_in" {
                fullKey = WeightMapper.posEmbedKey
            } else if key == "cls_token" {
                fullKey = "backbone.vision_backbone.trunk.cls_token"
            } else if key.hasPrefix("block.") {
                let components = key.components(separatedBy: ".")
                if components.count >= 4, let blockIdx = Int(components[1]) {
                    let suffix = components.dropFirst(2).joined(separator: ".")
                    fullKey = WeightMapper.encoderBlockKey(block: blockIdx, component: suffix)
                }
            } else if key == "rope_freqs" {
                if let freqBuf = rope.getFrequencies() {
                    feeds[ph] = MPSGraphTensorData(
                        freqBuf,
                        shape: [NSNumber(value: config.spatialTokenCount), NSNumber(value: config.dimPerHead / 2), 2],
                        dataType: .float32
                    )
                }
                continue
            } else if key == "rope_freqs_window" {
                if let freqBuf = ropeWindow.getFrequencies() {
                    feeds[ph] = MPSGraphTensorData(
                        freqBuf,
                        shape: [NSNumber(value: config.windowTokenCount), NSNumber(value: config.dimPerHead / 2), 2],
                        dataType: .float32
                    )
                }
                continue
            } else if key.hasPrefix("patch_embed.") {
                let suffix = key.replacingOccurrences(of: "patch_embed.", with: "")
                if let k = WeightMapper.patchEmbedKeys[suffix.contains("weight") ? "weight" : "bias"] {
                    fullKey = k
                }
            } else {
                // Not an encoder key (e.g. neck/, mask_decoder/), skip it.
                continue
            }
            
            // Standard loading
            if let b = weightBuffers[fullKey] {
                feeds[ph] = MPSGraphTensorData(
                    b,
                    shape: ph.shape!,
                    dataType: enableHalfPrecision ? .float16 : .float32
                )
                continue
            }

            guard let data = weights[fullKey] else {
                // Bias/CLS fallback: Create zeros if missing
                if key.hasSuffix("bias") || key == "cls_token" {
                    let shape = ph.shape!.map { $0.intValue }
                    let count = shape.reduce(1, *)
                    
                    if self.enableHalfPrecision {
                        let zeros = [Float16](repeating: 0, count: count)
                        let data = zeros.withUnsafeBytes { raw in
                            Data(bytes: raw.baseAddress!, count: raw.count)
                        }
                        feeds[ph] = MPSGraphTensorData(
                            device: MPSGraphDevice(mtlDevice: device),
                            data: data,
                            shape: ph.shape!,
                            dataType: .float16
                        )
                    } else {
                        let zeros = [Float](repeating: 0, count: count)
                        let data = zeros.withUnsafeBytes { raw in
                            Data(bytes: raw.baseAddress!, count: raw.count)
                        }
                        feeds[ph] = MPSGraphTensorData(
                            device: MPSGraphDevice(mtlDevice: device),
                            data: data,
                            shape: ph.shape!,
                            dataType: .float32
                        )
                    }
                    continue
                }
                print("MISSING WEIGHT: \(fullKey)")
                continue
            }

            // Fallback (slower): feed directly from Data.
            feeds[ph] = MPSGraphTensorData(
                device: MPSGraphDevice(mtlDevice: device),
                data: data,
                shape: ph.shape!,
                dataType: enableHalfPrecision ? .float16 : .float32
            )
        }
    }
}
