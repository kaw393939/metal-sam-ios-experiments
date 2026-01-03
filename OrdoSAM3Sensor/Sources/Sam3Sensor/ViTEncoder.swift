//
//  ViTEncoder.swift
//  SAM3Metal
//
//  Vision Transformer Encoder - 50x speedup target (8.5s → 170ms)
//

import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

/// Neck Layer (FPN-like)
/// Reduces encoder dimension (checkpoint: 1024) -> 256 using Conv1x1 and Conv3x3
@available(macOS 15.0, *)
public final class NeckLayer {
    private let device: MTLDevice
    private let inDim: Int
    private let outDim: Int
    private let graph: MPSGraph
    
    // Weights
    private var conv1W: MTLBuffer?
    private var conv1B: MTLBuffer?
    private var conv2W: MTLBuffer?
    private var conv2B: MTLBuffer?
    
    // Float16 Support
    private let useHalfPrecision: Bool
    private var ioDataType: MPSDataType { useHalfPrecision ? .float16 : .float32 }
    private var bytesPerElement: Int { useHalfPrecision ? 2 : 4 }
    
    public init(device: MTLDevice, inDim: Int = WeightMapper.embedDim, outDim: Int = 256, useHalfPrecision: Bool = true) {
        self.device = device
        self.inDim = inDim
        self.outDim = outDim
        self.useHalfPrecision = useHalfPrecision
        self.graph = MPSGraph()
        
        // Weights must be loaded via loadWeights() - no random fallback
        self.conv1W = nil
        self.conv1B = nil
        self.conv2W = nil
        self.conv2B = nil
    }
    
    public func loadWeights(conv1W: MTLBuffer, conv1B: MTLBuffer, conv2W: MTLBuffer, conv2B: MTLBuffer) {
        self.conv1W = conv1W
        self.conv1B = conv1B
        self.conv2W = conv2W
        self.conv2B = conv2B
    }

    public func forward(
        input: MTLBuffer,
        batch: Int,
        height: Int,
        width: Int,
        commandBuffer: MTLCommandBuffer,
        recycledBuffers: inout [MTLBuffer]
    ) throws -> MTLBuffer {
         let queue = commandBuffer.commandQueue
         let dt = ioDataType
         
         // Key: Neck_{Batch}_{H}_{W}_{Prec}
         let cacheKey = "Neck_\(batch)_\(height)_\(width)_\(dt == .float16 ? "F16" : "F32")"
         
         let (graph, placeholders, finalTensor, executable) = CompiledGraphCache.shared.getOrCompile(key: cacheKey, device: device) {
             let graph = MPSGraph()
             
             // Input: [N, H*W, C_in]
             // Note: Input shape is [B, H*W, D]. But Conv2D implies spatial.
             // We need to reshape input to [B, H, W, D] or just treat H*W as sequence?
             // Graph Convolution2D expects [N, H, W, C].
             // The inputs are FLAT [N, Seq, C].
             // We must reshape in graph!
             
             let inputShape: [NSNumber] = [NSNumber(value: batch), NSNumber(value: height), NSNumber(value: width), NSNumber(value: inDim)]
             let inputT = graph.placeholder(shape: inputShape, dataType: dt, name: "input")
             
             // Conv1 (1x1): Indim -> OutDim
             let c1W = graph.placeholder(shape: [NSNumber(value: outDim), NSNumber(value: inDim), 1, 1], dataType: dt, name: "c1W")
             let c1Desc = MPSGraphConvolution2DOpDescriptor(strideInX: 1, strideInY: 1, dilationRateInX: 1, dilationRateInY: 1, groups: 1, paddingLeft: 0, paddingRight: 0, paddingTop: 0, paddingBottom: 0, paddingStyle: .explicit, dataLayout: .NHWC, weightsLayout: .OIHW)!
             
             let c1 = graph.convolution2D(inputT, weights: c1W, descriptor: c1Desc, name: "conv1")
             let c1B = graph.placeholder(shape: [1, 1, 1, NSNumber(value: outDim)], dataType: dt, name: "c1B")
             let c1Val = graph.addition(c1, c1B, name: "c1Add")
             
             // Conv2 (3x3): OutDim -> OutDim
             let c2W = graph.placeholder(shape: [NSNumber(value: outDim), NSNumber(value: outDim), 3, 3], dataType: dt, name: "c2W")
             // Padding 1 for 3x3 to maintain size
             let c2Desc = MPSGraphConvolution2DOpDescriptor(strideInX: 1, strideInY: 1, dilationRateInX: 1, dilationRateInY: 1, groups: 1, paddingLeft: 1, paddingRight: 1, paddingTop: 1, paddingBottom: 1, paddingStyle: .explicit, dataLayout: .NHWC, weightsLayout: .OIHW)!
             
             let c2 = graph.convolution2D(c1Val, weights: c2W, descriptor: c2Desc, name: "conv2")
             let c2B = graph.placeholder(shape: [1, 1, 1, NSNumber(value: outDim)], dataType: dt, name: "c2B")
             let final = graph.addition(c2, c2B, name: "c2Add")
             
             let phs: [String: MPSGraphTensor] = [
                 "input": inputT,
                 "c1W": c1W, "c1B": c1B,
                 "c2W": c2W, "c2B": c2B
             ]
             
             return (graph, phs, final)
         }
         
         // Prepare feeds
         var feeds: [MPSGraphTensor: MPSGraphTensorData] = [:]
         
         // Data
         let inputShape: [NSNumber] = [NSNumber(value: batch), NSNumber(value: height), NSNumber(value: width), NSNumber(value: inDim)]
         feeds[placeholders["input"]!] = MPSGraphTensorData(input, shape: inputShape, dataType: dt)
         
         // Weights
         if let w = conv1W { feeds[placeholders["c1W"]!] = MPSGraphTensorData(w, shape: [NSNumber(value: outDim), NSNumber(value: inDim), 1, 1], dataType: dt) }
         if let b = conv1B { feeds[placeholders["c1B"]!] = MPSGraphTensorData(b, shape: [1, 1, 1, NSNumber(value: outDim)], dataType: dt) }
         if let w = conv2W { feeds[placeholders["c2W"]!] = MPSGraphTensorData(w, shape: [NSNumber(value: outDim), NSNumber(value: outDim), 3, 3], dataType: dt) }
         if let b = conv2B { feeds[placeholders["c2B"]!] = MPSGraphTensorData(b, shape: [1, 1, 1, NSNumber(value: outDim)], dataType: dt) }
         
         // Execute
         // Sprint 01: Fail fast if compilation failed
         guard executable != nil else {
            throw SAM3Error.graphCompilationFailed("NeckLayer: \(cacheKey)")
         }
         let results = try CompiledGraphCache.shared.runExecutable(key: cacheKey, queue: queue, feeds: feeds, targetTensors: [finalTensor])
         
         guard let resultData = results[finalTensor] else {
            throw SAM3Error.executionFailed("NeckLayer: No result")
         }
         
         let outputBytes = batch * height * width * outDim * bytesPerElement
         guard let output = BufferAllocator.shared.privateBuffer(length: outputBytes, device: device, label: "NeckOut") else {
            throw SAM3Error.bufferAllocationFailed("NeckLayer output buffer")
         }
         recycledBuffers.append(output)
         
         resultData.mpsndarray().exportData(with: commandBuffer, to: output, destinationDataType: dt, offset: 0, rowStrides: nil as UnsafeMutablePointer<Int>?)
         
         return output
    }
    public func randomInitialize() {
        // Allocate dummy weights for benchmark
        func alloc(_ shape: [Int]) -> MTLBuffer {
            let count = shape.reduce(1, *)
            return device.makeBuffer(length: count * bytesPerElement, options: .storageModeShared)!
        }
        
        // Shapes derived from graph placeholders
        self.conv1W = alloc([outDim, inDim, 1, 1])
        self.conv1B = alloc([1, 1, 1, outDim])
        self.conv2W = alloc([outDim, outDim, 3, 3])
        self.conv2B = alloc([1, 1, 1, outDim])
    }
}

/// Converts image [3, H, W] → patches [N, D] where N = (H/patchSize)*(W/patchSize)
@available(macOS 15.0, *)
/// Converts image [3, H, W] → patches [N, D] where N = (H/patchSize)*(W/patchSize)
@available(macOS 15.0, *)
public final class PatchEmbedding {
    private let device: MTLDevice
    private let patchSize: Int
    private let inChannels: Int
    private let embedDim: Int
    
    // MPS convolution (16x16 kernel, stride 16)
    private var convolution: MPSCNNConvolution?
    private let useHalfPrecision: Bool
    
    public init(device: MTLDevice, embedDim: Int = WeightMapper.embedDim, patchSize: Int = WeightMapper.patchSize, inChannels: Int = 3, useHalfPrecision: Bool = true) {
        self.device = device
        self.embedDim = embedDim
        self.patchSize = patchSize
        self.inChannels = inChannels
        self.useHalfPrecision = useHalfPrecision
        
        // No random/garbage initialization - wait for loadWeights
        self.convolution = nil
    }
    
    public func loadWeights(weights: MTLBuffer, bias: MTLBuffer) {
        let desc = MPSCNNConvolutionDescriptor(
            kernelWidth: patchSize,
            kernelHeight: patchSize,
            inputFeatureChannels: inChannels,
            outputFeatureChannels: embedDim
        )
        desc.strideInPixelsX = patchSize
        desc.strideInPixelsY = patchSize
        
        // Update with real weights using DataSource
        let dataSource = PatchEmbedDataSource(
            weights: weights,
            bias: bias,
            descriptor: desc,
            dataType: useHalfPrecision ? .float16 : .float32
        )
        
        self.convolution = MPSCNNConvolution(device: device, weights: dataSource)
        // Force reload? init new one is safer to drop old refs.
    }
    
    public func randomInitialize() {
        let wCount = embedDim * inChannels * patchSize * patchSize
        let bCount = embedDim
        let bytesPer = useHalfPrecision ? 2 : 4
        
        let wBuf = device.makeBuffer(length: wCount * bytesPer, options: .storageModeShared)!
        let bBuf = device.makeBuffer(length: bCount * bytesPer, options: .storageModeShared)!
        
        // Zero init is fine for benchmark
        memset(wBuf.contents(), 0, wBuf.length)
        memset(bBuf.contents(), 0, bBuf.length)
        
        loadWeights(weights: wBuf, bias: bBuf)
    }
    
    public func forward(
        input: MTLTexture,
        commandBuffer: MTLCommandBuffer
    ) throws -> MTLTexture {
        guard let conv = convolution else {
            throw SAM3Error.weightsNotLoaded("PatchEmbedding weights")
        }
        
        // Create output texture: [H/14, W/14, embedDim]
        let outH = input.height / patchSize
        let outW = input.width / patchSize
        
        // Sprint 12: Enforce Invariant Format (.rgba16Float)
        // regardless of input format.
        let outputDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float,
            width: outW,
            height: outH,
            mipmapped: false
        )
        outputDesc.usage = [.shaderRead, .shaderWrite]
        // Sprint 12: Enforce .private storage for GPU locality
        outputDesc.storageMode = .private
        
        // MPS needs array texture for multiple channels
        outputDesc.textureType = .type2DArray
        // Calculate slices: each slice holds 4 channels in RGBA
        outputDesc.arrayLength = (embedDim + 3) / 4
        
        guard let output = device.makeTexture(descriptor: outputDesc) else {
            throw SAM3Error.bufferAllocationFailed("PatchEmbedding output texture")
        }
        
        // Create MPSImages
        // Note: MPSImage featureChannels must match exactly
        let sourceImage = MPSImage(texture: input, featureChannels: self.inChannels)
        let destImage = MPSImage(texture: output, featureChannels: embedDim)
        
        conv.encode(
            commandBuffer: commandBuffer,
            sourceImage: sourceImage,
            destinationImage: destImage
        )
        
        return output
    }
}

/// Data source for patch embedding weights
class PatchEmbedDataSource: NSObject, MPSCNNConvolutionDataSource, NSCopying {
    let weightsBuffer: MTLBuffer
    let biasBuffer: MTLBuffer
    let convDescriptor: MPSCNNConvolutionDescriptor
    let _dataType: MPSDataType
    
    init(weights: MTLBuffer, bias: MTLBuffer, descriptor: MPSCNNConvolutionDescriptor, dataType: MPSDataType) {
        self.weightsBuffer = weights
        self.biasBuffer = bias
        self.convDescriptor = descriptor
        self._dataType = dataType
    }
    
    func copy(with zone: NSZone? = nil) -> Any {
        return PatchEmbedDataSource(weights: weightsBuffer, bias: biasBuffer, descriptor: convDescriptor, dataType: _dataType)
    }
    
    func dataType() -> MPSDataType { _dataType }
    func descriptor() -> MPSCNNConvolutionDescriptor { convDescriptor }
    func weights() -> UnsafeMutableRawPointer { weightsBuffer.contents() }
    func biasTerms() -> UnsafeMutablePointer<Float>? {
        // Bias is always Float32 for MPSCNNConvolution
        return biasBuffer.contents().assumingMemoryBound(to: Float.self)
    }
    func load() -> Bool { true }
    func purge() {}
    func label() -> String? { "PatchEmbed" }
}

/// Transformer Block
/// Self-attention + MLP with residual connections
@available(macOS 15.0, *)
public final class TransformerBlock {
    private let device: MTLDevice
    private let dim: Int
    private let numHeads: Int
    private let mlpHiddenDim: Int
    
    // Components
    let layerNorm1: LayerNorm
    let attention: MPSAttentionLayer
    let layerNorm2: LayerNorm
    let mlp: MLP
    
    // RoPE kernel access
    // private let ropeKernel: MTLComputePipelineState // Removed: RoPE is graph-fused now
    private let addResidualKernel: MTLComputePipelineState

    // Opt-in fused block execution (single MPSGraph per block)
    public var forceFusedBlock: Bool? = nil
    private var useFusedBlock: Bool {
        if let forced = forceFusedBlock { return forced }
        return ProcessInfo.processInfo.environment["SAM3_FUSED_BLOCK"] == "1"
    }
    
    public init(
        device: MTLDevice,
        dim: Int = WeightMapper.embedDim,
        numHeads: Int = WeightMapper.numHeads,
        mlpHiddenDim: Int? = WeightMapper.mlpHiddenDim,
        library: MTLLibrary,
        useHalfPrecision: Bool = true  // Float16 for maximum M3+ performance
    ) throws {
        self.device = device
        self.dim = dim
        self.numHeads = numHeads
        self.mlpHiddenDim = mlpHiddenDim ?? (dim * 4)
        
        // All components use Float16 by default on Apple Silicon
        // storageModeShared enables zero-copy unified memory access
        self.layerNorm1 = LayerNorm(device: device, dim: dim, useHalfPrecision: useHalfPrecision)
        self.attention = MPSAttentionLayer(
            device: device,
            numHeads: numHeads,
            dimPerHead: dim / numHeads,
            useHalfPrecision: useHalfPrecision
        )
        self.layerNorm2 = LayerNorm(device: device, dim: dim, useHalfPrecision: useHalfPrecision)
        self.mlp = MLP(
            device: device,
            inputDim: dim,
            hiddenDim: self.mlpHiddenDim,
            useHalfPrecision: useHalfPrecision
        )
        
        // Load RoPE kernel - REMOVED (Graph Fused)
        // let ropeFunction = library.makeFunction(name: "apply_rope_2d_batch")!
        // self.ropeKernel = try device.makeComputePipelineState(function: ropeFunction)
        
        // Load Residual Add kernel (unified, bounds-safe)
        let addName = useHalfPrecision ? "add_residual_half" : "add_residual_float"
        guard let addFunction = library.makeFunction(name: addName) else {
            throw SAM3Error.noMetalDevice
        }
        self.addResidualKernel = try device.makeComputePipelineState(function: addFunction)
        
        // Optimization 1: Neck Layer (now Float16 aware)
        // self.neck = NeckLayer(device: device, inDim: dim, outDim: 256, useHalfPrecision: useHalfPrecision)
    }
    
    /// Forward pass with RoPE - Optimized: all ops on single command buffer
    public func forward(
        input: MTLBuffer,
        ropeFreqs: MTLBuffer,
        seqLen: Int,
        batch: Int,
        windowed: Bool = false, // Added windowed support
        commandBuffer: MTLCommandBuffer,
        recycledBuffers: inout [MTLBuffer]
    ) throws -> MTLBuffer { // Added throws
        if useFusedBlock {
            return try forwardFused( // Added try
                input: input,
                ropeFreqs: ropeFreqs,
                seqLen: seqLen,
                batch: batch,
                windowed: windowed && (seqLen == 4096), // Sprint 13 Rule: If pruned (seqLen < 4096), MUST use Global Attention.
                commandBuffer: commandBuffer,
                recycledBuffers: &recycledBuffers
            )
        }
        // 1. LayerNorm
        let normed1 = try layerNorm1.forward( // Added try
            input: input,
            seqLen: seqLen, // Added seqLen usage
            batch: batch,
            commandBuffer: commandBuffer,
            recycledBuffers: &recycledBuffers
        )
        
        // 2. Self-Attention (With Graph-Fused RoPE)
        let attnOut = try attention.forward( // Added try
            input: normed1, // FIX: Use normed input
            ropeFreqs: ropeFreqs,
            batch: batch,
            seqLen: seqLen,
            windowed: windowed && (seqLen == 4096), // Sprint 13 Rule: Force Global if pruned
            commandBuffer: commandBuffer,
            recycledBuffers: &recycledBuffers
        )
        
        // 3. Residual
        let residual1 = addResidual(input, attnOut, commandBuffer)
        
        // 4. LayerNorm
        let normed2 = try layerNorm2.forward( // Added try
            input: residual1,
            seqLen: seqLen,
            batch: batch,
            commandBuffer: commandBuffer,
            recycledBuffers: &recycledBuffers
        )
        
        // 5. MLP
        let mlpOut = try mlp.forward( // Added try
            input: normed2,
            seqLen: seqLen,
            batch: batch,
            commandBuffer: commandBuffer,
            recycledBuffers: &recycledBuffers
        )
        
        // 6. Residual
        return addResidual(residual1, mlpOut, commandBuffer)
    }

    private func forwardFused(
        input: MTLBuffer,
        ropeFreqs: MTLBuffer,
        seqLen: Int,
        batch: Int,
        windowed: Bool,
        commandBuffer: MTLCommandBuffer,
        recycledBuffers: inout [MTLBuffer]
    ) throws -> MTLBuffer { // Added throws
        let totalDim = dim
        let dimPerHead = dim / numHeads
        let dt: MPSDataType = attention.useHalfPrecisionFlag ? .float16 : .float32
        let bytesPerElement = attention.useHalfPrecisionFlag ? 2 : 4

        // Match existing attention RoPE effective sequence length logic
        let side = Int(sqrt(Double(seqLen)))
        let winSize = (windowed && side % 16 == 0) ? 16 : 24
        let sEff = windowed ? winSize * winSize : seqLen

        let mode = windowed ? "W" : "G"
        let prec = (dt == .float16) ? "F16" : "F32"
        let cacheKey = "TB_FUSED_\(batch)_\(seqLen)_\(sEff)_\(mode)_\(prec)"
        let queue = commandBuffer.commandQueue

        let (graph, placeholders, outputTensor, executable) = CompiledGraphCache.shared.getOrCompile(key: cacheKey, device: device) {
            let graph = MPSGraph()

            // Input: [B, S, D]
            let x = graph.placeholder(
                shape: [NSNumber(value: batch), NSNumber(value: seqLen), NSNumber(value: totalDim)],
                dataType: dt,
                name: "x"
            )

            // --- LN1 ---
            let gamma1 = graph.placeholder(shape: [1, 1, NSNumber(value: totalDim)], dataType: dt, name: "gamma1")
            let beta1 = graph.placeholder(shape: [1, 1, NSNumber(value: totalDim)], dataType: dt, name: "beta1")
            let mean1 = graph.mean(of: x, axes: [2], name: "ln1_mean")
            let var1 = graph.variance(of: x, mean: mean1, axes: [2], name: "ln1_var")
            let eps = graph.constant(1e-6, dataType: dt)
            let centered1 = graph.subtraction(x, mean1, name: "ln1_center")
            let denom1 = graph.squareRoot(with: graph.addition(var1, eps, name: "ln1_var_eps"), name: "ln1_denom")
            let norm1 = graph.division(centered1, denom1, name: "ln1_norm")
            let ln1 = graph.addition(graph.multiplication(norm1, gamma1, name: "ln1_scale"), beta1, name: "ln1")

            // --- Attention (QKV + RoPE + SDPA + proj) ---
            let qkvW = graph.placeholder(shape: [NSNumber(value: totalDim), NSNumber(value: 3 * totalDim)], dataType: dt, name: "qkvW")
            let qkvB = graph.placeholder(shape: [1, 1, NSNumber(value: 3 * totalDim)], dataType: dt, name: "qkvB")

            let qkv = graph.matrixMultiplication(primary: ln1, secondary: qkvW, name: "qkvMM")
            let qkvBiased = graph.addition(qkv, qkvB, name: "qkvBias")

            // [B, S, 3, NH, DPH]
            let qkvReshaped = graph.reshape(
                qkvBiased,
                shape: [NSNumber(value: batch), NSNumber(value: seqLen), 3, NSNumber(value: numHeads), NSNumber(value: dimPerHead)],
                name: "qkvReshape"
            )

            let q = graph.sliceTensor(qkvReshaped, dimension: 2, start: 0, length: 1, name: "sliceQ")
            let k = graph.sliceTensor(qkvReshaped, dimension: 2, start: 1, length: 1, name: "sliceK")
            let v = graph.sliceTensor(qkvReshaped, dimension: 2, start: 2, length: 1, name: "sliceV")

            let q4 = graph.reshape(q, shape: [NSNumber(value: batch), NSNumber(value: seqLen), NSNumber(value: numHeads), NSNumber(value: dimPerHead)], name: "q4")
            let k4 = graph.reshape(k, shape: [NSNumber(value: batch), NSNumber(value: seqLen), NSNumber(value: numHeads), NSNumber(value: dimPerHead)], name: "k4")
            let v4 = graph.reshape(v, shape: [NSNumber(value: batch), NSNumber(value: seqLen), NSNumber(value: numHeads), NSNumber(value: dimPerHead)], name: "v4")

            // RoPE freqs: [sEff, DPH/2, 2] in float32
            let ropeT = graph.placeholder(shape: [NSNumber(value: sEff), NSNumber(value: dimPerHead / 2), 2], dataType: .float32, name: "ropeFreqs")

            func applyRoPE(_ x: MPSGraphTensor, name: String) -> MPSGraphTensor {
                let xP = graph.reshape(
                    x,
                    shape: [NSNumber(value: batch), NSNumber(value: seqLen), NSNumber(value: numHeads), NSNumber(value: dimPerHead / 2), 2],
                    name: "\(name)/pairs"
                )
                let xR = graph.sliceTensor(xP, dimension: 4, start: 0, length: 1, name: "\(name)/real")
                let xI = graph.sliceTensor(xP, dimension: 4, start: 1, length: 1, name: "\(name)/imag")

                let cos = graph.sliceTensor(ropeT, dimension: 2, start: 0, length: 1, name: "\(name)/cos")
                let sin = graph.sliceTensor(ropeT, dimension: 2, start: 1, length: 1, name: "\(name)/sin")

                var cosB = graph.broadcast(cos, shape: [NSNumber(value: batch), NSNumber(value: seqLen), NSNumber(value: numHeads), NSNumber(value: dimPerHead / 2), 1], name: "\(name)/cosB")
                var sinB = graph.broadcast(sin, shape: [NSNumber(value: batch), NSNumber(value: seqLen), NSNumber(value: numHeads), NSNumber(value: dimPerHead / 2), 1], name: "\(name)/sinB")

                cosB = graph.cast(cosB, to: dt, name: "\(name)/cosB_cast")
                sinB = graph.cast(sinB, to: dt, name: "\(name)/sinB_cast")

                let outR = graph.subtraction(
                    graph.multiplication(xR, cosB, name: "\(name)/re_re"),
                    graph.multiplication(xI, sinB, name: "\(name)/im_im"),
                    name: "\(name)/outR"
                )
                let outI = graph.addition(
                    graph.multiplication(xR, sinB, name: "\(name)/re_im"),
                    graph.multiplication(xI, cosB, name: "\(name)/im_re"),
                    name: "\(name)/outI"
                )

                let concat = graph.concatTensors([outR, outI], dimension: 4, name: "\(name)/concat")
                return graph.reshape(
                    concat,
                    shape: [NSNumber(value: batch), NSNumber(value: seqLen), NSNumber(value: numHeads), NSNumber(value: dimPerHead)],
                    name: "\(name)/final"
                )
            }

            let qRope = applyRoPE(q4, name: "q")
            let kRope = applyRoPE(k4, name: "k")

            // [B, NH, S, DPH]
            let qT = graph.transposeTensor(qRope, dimension: 1, withDimension: 2, name: "qT")
            let kT = graph.transposeTensor(kRope, dimension: 1, withDimension: 2, name: "kT")
            let vT = graph.transposeTensor(v4, dimension: 1, withDimension: 2, name: "vT")

            let scale = Float(1.0 / sqrt(Double(dimPerHead)))
            let attn = graph.scaledDotProductAttention(query: qT, key: kT, value: vT, mask: nil, scale: scale, name: "sdpa")

            let attnBack = graph.transposeTensor(attn, dimension: 1, withDimension: 2, name: "attnBack")
            let attnFlat = graph.reshape(attnBack, shape: [NSNumber(value: batch), NSNumber(value: seqLen), NSNumber(value: totalDim)], name: "attnFlat")

            let outW = graph.placeholder(shape: [NSNumber(value: totalDim), NSNumber(value: totalDim)], dataType: dt, name: "outW")
            let outB = graph.placeholder(shape: [1, 1, NSNumber(value: totalDim)], dataType: dt, name: "outB")
            let proj = graph.matrixMultiplication(primary: attnFlat, secondary: outW, name: "outMM")
            let attnOut = graph.addition(proj, outB, name: "outBias")

            let residual1 = graph.addition(x, attnOut, name: "residual1")

            // --- LN2 ---
            let gamma2 = graph.placeholder(shape: [1, 1, NSNumber(value: totalDim)], dataType: dt, name: "gamma2")
            let beta2 = graph.placeholder(shape: [1, 1, NSNumber(value: totalDim)], dataType: dt, name: "beta2")
            let mean2 = graph.mean(of: residual1, axes: [2], name: "ln2_mean")
            let var2 = graph.variance(of: residual1, mean: mean2, axes: [2], name: "ln2_var")
            let centered2 = graph.subtraction(residual1, mean2, name: "ln2_center")
            let denom2 = graph.squareRoot(with: graph.addition(var2, eps, name: "ln2_var_eps"), name: "ln2_denom")
            let norm2 = graph.division(centered2, denom2, name: "ln2_norm")
            let ln2 = graph.addition(graph.multiplication(norm2, gamma2, name: "ln2_scale"), beta2, name: "ln2")

            // --- MLP ---
            let hiddenDim = mlpHiddenDim
            let fc1W = graph.placeholder(shape: [NSNumber(value: hiddenDim), NSNumber(value: totalDim)], dataType: dt, name: "fc1W")
            let fc1B = graph.placeholder(shape: [1, 1, NSNumber(value: hiddenDim)], dataType: dt, name: "fc1B")
            let fc2W = graph.placeholder(shape: [NSNumber(value: totalDim), NSNumber(value: hiddenDim)], dataType: dt, name: "fc2W")
            let fc2B = graph.placeholder(shape: [1, 1, NSNumber(value: totalDim)], dataType: dt, name: "fc2B")

            let fc1W_T = graph.transposeTensor(fc1W, dimension: 0, withDimension: 1, name: "fc1W_T")
            let fc1 = graph.matrixMultiplication(primary: ln2, secondary: fc1W_T, name: "fc1")
            let fc1Biased = graph.addition(fc1, fc1B, name: "fc1_biased")

            let pointFive = graph.constant(0.5, dataType: dt)
            let one = graph.constant(1.0, dataType: dt)
            let sqrtTwo = graph.constant(1.41421356, dataType: dt)
            let div = graph.division(fc1Biased, sqrtTwo, name: "gelu_div")
            let erf = graph.erf(with: div, name: "gelu_erf")
            let onePlusErf = graph.addition(one, erf, name: "gelu_add")
            let halfX = graph.multiplication(pointFive, fc1Biased, name: "gelu_half")
            let geluOut = graph.multiplication(halfX, onePlusErf, name: "gelu_out")

            let fc2W_T = graph.transposeTensor(fc2W, dimension: 0, withDimension: 1, name: "fc2W_T")
            let fc2 = graph.matrixMultiplication(primary: geluOut, secondary: fc2W_T, name: "fc2")
            let mlpOut = graph.addition(fc2, fc2B, name: "mlp_out")

            let y = graph.addition(residual1, mlpOut, name: "residual2")

            let phs: [String: MPSGraphTensor] = [
                "x": x,
                "gamma1": gamma1,
                "beta1": beta1,
                "qkvW": qkvW,
                "qkvB": qkvB,
                "outW": outW,
                "outB": outB,
                "ropeFreqs": ropeT,
                "gamma2": gamma2,
                "beta2": beta2,
                "fc1W": fc1W,
                "fc1B": fc1B,
                "fc2W": fc2W,
                "fc2B": fc2B
            ]

            return (graph, phs, y)
        }

        // Feeds
        var feeds: [MPSGraphTensor: MPSGraphTensorData] = [:]
        if let p = placeholders["x"] {
            feeds[p] = MPSGraphTensorData(input, shape: [NSNumber(value: batch), NSNumber(value: seqLen), NSNumber(value: totalDim)], dataType: dt)
        }

        // LayerNorm weights (gamma/beta are 1D; feed as [1,1,D])
        if let p = placeholders["gamma1"] {
            feeds[p] = MPSGraphTensorData(layerNorm1.gammaOrOnes(device: device), shape: [1, 1, NSNumber(value: totalDim)], dataType: dt)
        }
        if let p = placeholders["beta1"] {
            feeds[p] = MPSGraphTensorData(layerNorm1.betaOrZeros(device: device), shape: [1, 1, NSNumber(value: totalDim)], dataType: dt)
        }
        if let p = placeholders["gamma2"] {
            feeds[p] = MPSGraphTensorData(layerNorm2.gammaOrOnes(device: device), shape: [1, 1, NSNumber(value: totalDim)], dataType: dt)
        }
        if let p = placeholders["beta2"] {
            feeds[p] = MPSGraphTensorData(layerNorm2.betaOrZeros(device: device), shape: [1, 1, NSNumber(value: totalDim)], dataType: dt)
        }

        // Attention weights
        if let p = placeholders["qkvW"] {
            let w = attention.qkvWeightsBuffer
            feeds[p] = MPSGraphTensorData(w, shape: [NSNumber(value: totalDim), NSNumber(value: 3 * totalDim)], dataType: dt)
        }
        if let p = placeholders["qkvB"] {
            feeds[p] = MPSGraphTensorData(attention.qkvBiasBufferOrZeros(device: device), shape: [1, 1, NSNumber(value: 3 * totalDim)], dataType: dt)
        }
        if let p = placeholders["outW"] {
            let w = attention.outputWeightsBuffer
            feeds[p] = MPSGraphTensorData(w, shape: [NSNumber(value: totalDim), NSNumber(value: totalDim)], dataType: dt)
        }
        if let p = placeholders["outB"] {
            feeds[p] = MPSGraphTensorData(attention.outputBiasBufferOrZeros(device: device), shape: [1, 1, NSNumber(value: totalDim)], dataType: dt)
        }

        // RoPE
        if let p = placeholders["ropeFreqs"] {
            feeds[p] = MPSGraphTensorData(ropeFreqs, shape: [NSNumber(value: sEff), NSNumber(value: dimPerHead / 2), 2], dataType: .float32)
        }

        // MLP weights
        let hiddenDim = mlpHiddenDim
        if let p = placeholders["fc1W"] {
            feeds[p] = MPSGraphTensorData(mlp.fc1WOrZeros(device: device, bytesPerElement: bytesPerElement), shape: [NSNumber(value: hiddenDim), NSNumber(value: totalDim)], dataType: dt)
        }
        if let p = placeholders["fc1B"] {
            feeds[p] = MPSGraphTensorData(mlp.fc1BOrZeros(device: device, bytesPerElement: bytesPerElement), shape: [1, 1, NSNumber(value: hiddenDim)], dataType: dt)
        }
        if let p = placeholders["fc2W"] {
            feeds[p] = MPSGraphTensorData(mlp.fc2WOrZeros(device: device, bytesPerElement: bytesPerElement), shape: [NSNumber(value: totalDim), NSNumber(value: hiddenDim)], dataType: dt)
        }
        if let p = placeholders["fc2B"] {
            feeds[p] = MPSGraphTensorData(mlp.fc2BOrZeros(device: device, bytesPerElement: bytesPerElement), shape: [1, 1, NSNumber(value: totalDim)], dataType: dt)
        }

        // Execute
        // Sprint 01: Fail fast if compilation failed
        guard executable != nil else {
            throw SAM3Error.graphCompilationFailed("TransformerBlock fused: \(cacheKey)")
        }
        let results = try CompiledGraphCache.shared.runExecutable(key: cacheKey, queue: queue, feeds: feeds, targetTensors: [outputTensor])

        guard let resultData = results[outputTensor] else {
            throw SAM3Error.executionFailed("TransformerBlock fused: No result")
        }

        let outputLength = batch * seqLen * totalDim * bytesPerElement
        guard let output = BufferAllocator.shared.privateBuffer(length: outputLength, device: device, label: "TBOut") else {
            throw SAM3Error.bufferAllocationFailed("TransformerBlock fused output buffer")
        }
        recycledBuffers.append(output)
        resultData.mpsndarray().exportData(with: commandBuffer, to: output, destinationDataType: dt, offset: 0, rowStrides: nil as UnsafeMutablePointer<Int>?)
        return output
    }
    
    public func loadWeights(_ weights: [String: Data], prefix: String) {
        // LayerNorm 1
        if let gamma = weights.buffer(for: "\(prefix).norm1.weight", device: device),
            let beta = weights.buffer(for: "\(prefix).norm1.bias", device: device) {
            layerNorm1.loadWeights(gamma: gamma, beta: beta)
        }
        
        // Attention
        if let qkvW = weights.buffer(for: "\(prefix).attn.qkv.weight", device: device),
            let qkvB = weights.buffer(for: "\(prefix).attn.qkv.bias", device: device),
            let outW = weights.buffer(for: "\(prefix).attn.proj.weight", device: device),
            let outB = weights.buffer(for: "\(prefix).attn.proj.bias", device: device) {
            attention.loadWeights(qkvWeight: qkvW, qkvBias: qkvB, outputWeight: outW, outputBias: outB)
        }
        
        // LayerNorm 2
        if let gamma = weights.buffer(for: "\(prefix).norm2.weight", device: device),
            let beta = weights.buffer(for: "\(prefix).norm2.bias", device: device) {
            layerNorm2.loadWeights(gamma: gamma, beta: beta)
        }
        
        // MLP
        if let fc1W = weights.buffer(for: "\(prefix).mlp.lin1.weight", device: device),
            let fc1B = weights.buffer(for: "\(prefix).mlp.lin1.bias", device: device),
            let fc2W = weights.buffer(for: "\(prefix).mlp.lin2.weight", device: device),
            let fc2B = weights.buffer(for: "\(prefix).mlp.lin2.bias", device: device) {
            mlp.loadWeights(fc1W: fc1W, fc1B: fc1B, fc2W: fc2W, fc2B: fc2B)
        }
    }
    
    private func addResidual(
        _ a: MTLBuffer,
        _ b: MTLBuffer,
        _ commandBuffer: MTLCommandBuffer
    ) -> MTLBuffer {
        let bytesPerElement = attention.useHalfPrecisionFlag ? 2 : 4
        let count = a.length / bytesPerElement
        // Sprint 02: Use BufferAllocator for GPU-private memory
        let out = BufferAllocator.shared.privateBuffer(length: a.length, device: device, label: "addResidual/out")!
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(addResidualKernel)
        encoder.setBuffer(a, offset: 0, index: 0)
        encoder.setBuffer(b, offset: 0, index: 1)
        encoder.setBuffer(out, offset: 0, index: 2)
        var countU = UInt32(count)
        encoder.setBytes(&countU, length: MemoryLayout<UInt32>.stride, index: 3)
        
        let threadsPerGroup = MTLSize(width: 256, height: 1, depth: 1)
        let numThreadgroups = MTLSize(width: (count + 255) / 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        
        return out
    }
}

/// Layer Normalization - Supports Float16 I/O with Float32 internals for stability
@available(macOS 15.0, *)
public final class LayerNorm {
    private let device: MTLDevice
    private let dim: Int
    private var gamma: MTLBuffer?
    private var beta: MTLBuffer?
    
    // Precision control - LayerNorm uses F32 internally for stability
    private let useHalfPrecision: Bool
    private var ioDataType: MPSDataType { useHalfPrecision ? .float16 : .float32 }
    private var bytesPerElement: Int { useHalfPrecision ? 2 : 4 }
    
    public init(device: MTLDevice, dim: Int, useHalfPrecision: Bool = false) {
        self.device = device
        self.dim = dim
        self.useHalfPrecision = useHalfPrecision
    }
    
    public func loadWeights(gamma: MTLBuffer, beta: MTLBuffer) {
        self.gamma = gamma
        self.beta = beta
    }

    fileprivate func gammaOrOnes(device: MTLDevice) -> MTLBuffer {
        if let g = gamma { return g }
        // Sprint 02: Keep as shared - needs CPU write access for initialization
        let ones = device.makeBuffer(length: dim * bytesPerElement, options: .storageModeShared)!
        if useHalfPrecision {
            let ptr = ones.contents().assumingMemoryBound(to: Float16.self)
            for i in 0..<dim { ptr[i] = 1.0 }
        } else {
            let ptr = ones.contents().assumingMemoryBound(to: Float.self)
            for i in 0..<dim { ptr[i] = 1.0 }
        }
        return ones
    }

    fileprivate func betaOrZeros(device: MTLDevice) -> MTLBuffer {
        if let b = beta { return b }
        // Sprint 02: Keep as shared - needs CPU write access for initialization
        let zeros = device.makeBuffer(length: dim * bytesPerElement, options: .storageModeShared)!
        memset(zeros.contents(), 0, dim * bytesPerElement)
        return zeros
    }
    
    public func forward(
        input: MTLBuffer,
        seqLen: Int,
        batch: Int,
        commandBuffer: MTLCommandBuffer,
        recycledBuffers: inout [MTLBuffer]
    ) throws -> MTLBuffer { // Added throws
        let queue = commandBuffer.commandQueue
        let dt = ioDataType  // Float16 or Float32
        
        let totalSeq = seqLen * batch
        
        // Key: LN_{Seq}_{Dim}_{Prec}
        let cacheKey = "LN_\(totalSeq)_\(dim)_\(dt == .float16 ? "F16" : "F32")"
        
        let (graph, placeholders, outputTensor, executable) = CompiledGraphCache.shared.getOrCompile(key: cacheKey, device: device) {
            let graph = MPSGraph()
            
            let shape: [NSNumber] = [NSNumber(value: totalSeq), NSNumber(value: dim)]
            
            // Placeholders
            let inputTensor = graph.placeholder(shape: shape, dataType: dt, name: "input")
            let gammaPlaceholder = graph.placeholder(shape: [NSNumber(value: 1), NSNumber(value: dim)], dataType: dt, name: "gamma")
            let betaPlaceholder = graph.placeholder(shape: [NSNumber(value: 1), NSNumber(value: dim)], dataType: dt, name: "beta")
            
            // LayerNorm: compute mean and variance (internally uses Float32 for stability)
            let axes: [NSNumber] = [NSNumber(value: 1)]
            let meanTensor = graph.mean(of: inputTensor, axes: axes, name: "mean")
            let varianceTensor = graph.variance(of: inputTensor, mean: meanTensor, axes: axes, name: "variance")
            
            let normalized = graph.normalize(inputTensor,
                                             mean: meanTensor,
                                             variance: varianceTensor,
                                             gamma: gammaPlaceholder,
                                             beta: betaPlaceholder,
                                             epsilon: 1e-6,
                                             name: "output")
            
            return (graph, ["input": inputTensor, "gamma": gammaPlaceholder, "beta": betaPlaceholder], normalized)
        }
        
        let inputData = MPSGraphTensorData(input, shape: [NSNumber(value: totalSeq), NSNumber(value: dim)], dataType: dt)
        
        // Weight data with correct precision
        let gammaData: MPSGraphTensorData
        if let g = gamma {
            gammaData = MPSGraphTensorData(g, shape: [1, NSNumber(value: dim)], dataType: dt)
        } else {
            let ones = device.makeBuffer(length: dim * bytesPerElement, options: .storageModeShared)!
            if useHalfPrecision {
                let ptr = ones.contents().assumingMemoryBound(to: Float16.self)
                for i in 0..<dim { ptr[i] = 1.0 }
            } else {
                let ptr = ones.contents().assumingMemoryBound(to: Float.self)
                for i in 0..<dim { ptr[i] = 1.0 }
            }
            gammaData = MPSGraphTensorData(ones, shape: [1, NSNumber(value: dim)], dataType: dt)
        }
        
        let betaData: MPSGraphTensorData
        if let b = beta {
            betaData = MPSGraphTensorData(b, shape: [1, NSNumber(value: dim)], dataType: dt)
        } else {
            let zeros = device.makeBuffer(length: dim * bytesPerElement, options: .storageModeShared)!
            memset(zeros.contents(), 0, dim * bytesPerElement)
            betaData = MPSGraphTensorData(zeros, shape: [1, NSNumber(value: dim)], dataType: dt)
        }
        
        // Explicitly typed feeds to avoid compiler confusion with [String: Tensor]
        var feeds: [MPSGraphTensor : MPSGraphTensorData] = [:]
        if let p = placeholders["input"] { feeds[p] = inputData }
        if let p = placeholders["gamma"] { feeds[p] = gammaData }
        if let p = placeholders["beta"] { feeds[p] = betaData }
        
        // Execute cached graph
        // Sprint 01: Fail fast if compilation failed
        guard executable != nil else {
            throw SAM3Error.graphCompilationFailed("LayerNorm: \(cacheKey)")
        }
        let results = try CompiledGraphCache.shared.runExecutable(key: cacheKey, queue: queue, feeds: feeds, targetTensors: [outputTensor])
        
        guard let resultData = results[outputTensor] else {
            throw SAM3Error.executionFailed("LayerNorm: No result")
        }
        
        guard let output = BufferAllocator.shared.privateBuffer(length: input.length, device: device, label: "LNOut") else {
            throw SAM3Error.bufferAllocationFailed("LayerNorm output buffer")
        }
        recycledBuffers.append(output)
        resultData.mpsndarray().exportData(with: commandBuffer, to: output, destinationDataType: dt, offset: 0, rowStrides: nil as UnsafeMutablePointer<Int>?)
        
        return output
    }
}

/// MLP (Feed-Forward Network) - Supports Float16 I/O for 2x memory bandwidth
@available(macOS 15.0, *)
public final class MLP {
    private let device: MTLDevice
    private let inputDim: Int
    private let hiddenDim: Int
    
    private var fc1W: MTLBuffer?
    private var fc1B: MTLBuffer?
    private var fc2W: MTLBuffer?
    private var fc2B: MTLBuffer?
    
    // Precision control
    private let useHalfPrecision: Bool
    private var ioDataType: MPSDataType { useHalfPrecision ? .float16 : .float32 }
    private var bytesPerElement: Int { useHalfPrecision ? 2 : 4 }
    
    public init(device: MTLDevice, inputDim: Int, hiddenDim: Int, useHalfPrecision: Bool = false) {
        self.device = device
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.useHalfPrecision = useHalfPrecision
    }
    
    public func loadWeights(fc1W: MTLBuffer, fc1B: MTLBuffer, fc2W: MTLBuffer, fc2B: MTLBuffer) {
        self.fc1W = fc1W
        self.fc1B = fc1B
        self.fc2W = fc2W
        self.fc2B = fc2B
    }

    fileprivate func fc1WOrZeros(device: MTLDevice, bytesPerElement: Int) -> MTLBuffer {
        if let b = fc1W { return b }
        let count = hiddenDim * inputDim
        let buf = device.makeBuffer(length: count * bytesPerElement, options: .storageModeShared)!
        memset(buf.contents(), 0, count * bytesPerElement)
        return buf
    }

    fileprivate func fc1BOrZeros(device: MTLDevice, bytesPerElement: Int) -> MTLBuffer {
        if let b = fc1B { return b }
        let count = hiddenDim
        let buf = device.makeBuffer(length: count * bytesPerElement, options: .storageModeShared)!
        memset(buf.contents(), 0, count * bytesPerElement)
        return buf
    }

    fileprivate func fc2WOrZeros(device: MTLDevice, bytesPerElement: Int) -> MTLBuffer {
        if let b = fc2W { return b }
        let count = inputDim * hiddenDim
        let buf = device.makeBuffer(length: count * bytesPerElement, options: .storageModeShared)!
        memset(buf.contents(), 0, count * bytesPerElement)
        return buf
    }

    fileprivate func fc2BOrZeros(device: MTLDevice, bytesPerElement: Int) -> MTLBuffer {
        if let b = fc2B { return b }
        let count = inputDim
        let buf = device.makeBuffer(length: count * bytesPerElement, options: .storageModeShared)!
        memset(buf.contents(), 0, count * bytesPerElement)
        return buf
    }
    
    public func forward(
        input: MTLBuffer,
        seqLen: Int,
        batch: Int,
        commandBuffer: MTLCommandBuffer,
        recycledBuffers: inout [MTLBuffer]
    ) throws -> MTLBuffer { // Added throws
        let queue = commandBuffer.commandQueue
        let dt = ioDataType  // Float16 or Float32
        
        let totalSeq = seqLen * batch
        
        // Key: MLP_{Seq}_{In}_{Hidden}_{Prec}
        let cacheKey = "MLP_\(totalSeq)_\(inputDim)_\(hiddenDim)_\(dt == .float16 ? "F16" : "F32")"
        
        let (graph, placeholders, outputTensor, executable) = CompiledGraphCache.shared.getOrCompile(key: cacheKey, device: device) {
            let graph = MPSGraph()
            let inputShape: [NSNumber] = [NSNumber(value: totalSeq), NSNumber(value: inputDim)]
            
            // Placeholders
            let inputTensor = graph.placeholder(shape: inputShape, dataType: dt, name: "input")
            let fc1WTensor = graph.placeholder(shape: [NSNumber(value: hiddenDim), NSNumber(value: inputDim)], dataType: dt, name: "fc1W")
            let fc1BTensor = graph.placeholder(shape: [1, NSNumber(value: hiddenDim)], dataType: dt, name: "fc1B")
            let fc2WTensor = graph.placeholder(shape: [NSNumber(value: inputDim), NSNumber(value: hiddenDim)], dataType: dt, name: "fc2W")
            let fc2BTensor = graph.placeholder(shape: [1, NSNumber(value: inputDim)], dataType: dt, name: "fc2B")
            
            // FC1
            let fc1W_T = graph.transposeTensor(fc1WTensor, dimension: 0, withDimension: 1, name: "fc1W_T")
            let fc1Out = graph.matrixMultiplication(primary: inputTensor, secondary: fc1W_T, name: "fc1")
            let fc1Biased = graph.addition(fc1Out, fc1BTensor, name: "fc1_biased")
            
            // GELU with correct precision
            let pointFive = graph.constant(0.5, dataType: dt)
            let one = graph.constant(1.0, dataType: dt)
            let sqrtTwo = graph.constant(1.41421356, dataType: dt)
            let div = graph.division(fc1Biased, sqrtTwo, name: "gelu_div")
            let erf = graph.erf(with: div, name: "gelu_erf")
            let onePlusErf = graph.addition(one, erf, name: "gelu_add")
            let halfX = graph.multiplication(pointFive, fc1Biased, name: "gelu_half")
            let geluOut = graph.multiplication(halfX, onePlusErf, name: "gelu_out")
            
            // FC2
            let fc2W_T = graph.transposeTensor(fc2WTensor, dimension: 0, withDimension: 1, name: "fc2W_T")
            let fc2Out = graph.matrixMultiplication(primary: geluOut, secondary: fc2W_T, name: "fc2")
            let finalOutput = graph.addition(fc2Out, fc2BTensor, name: "mlp_output")
            
            let placeholders: [String: MPSGraphTensor] = [
                "input": inputTensor,
                "fc1W": fc1WTensor, "fc1B": fc1BTensor,
                "fc2W": fc2WTensor, "fc2B": fc2BTensor
            ]
            
            return (graph, placeholders, finalOutput)
        }
        
        let inputData = MPSGraphTensorData(input, shape: [NSNumber(value: totalSeq), NSNumber(value: inputDim)], dataType: dt)
        
        func makeWeightData(_ buffer: MTLBuffer?, shape: [NSNumber]) -> MPSGraphTensorData {
            if let b = buffer {
                return MPSGraphTensorData(b, shape: shape, dataType: dt)
            } else {
                let count = shape.map { $0.intValue }.reduce(1, *)
                let b = device.makeBuffer(length: count * bytesPerElement, options: .storageModeShared)!
                memset(b.contents(), 0, count * bytesPerElement)
                return MPSGraphTensorData(b, shape: shape, dataType: dt)
            }
        }
        
        let fc1WData = makeWeightData(fc1W, shape: [NSNumber(value: hiddenDim), NSNumber(value: inputDim)])
        let fc1BData = makeWeightData(fc1B, shape: [1, NSNumber(value: hiddenDim)])
        let fc2WData = makeWeightData(fc2W, shape: [NSNumber(value: inputDim), NSNumber(value: hiddenDim)])
        let fc2BData = makeWeightData(fc2B, shape: [1, NSNumber(value: inputDim)])
        
        var feeds: [MPSGraphTensor : MPSGraphTensorData] = [:]
        if let p = placeholders["input"] { feeds[p] = inputData }
        if let p = placeholders["fc1W"] { feeds[p] = fc1WData }
        if let p = placeholders["fc1B"] { feeds[p] = fc1BData }
        if let p = placeholders["fc2W"] { feeds[p] = fc2WData }
        if let p = placeholders["fc2B"] { feeds[p] = fc2BData }
        
        // Sprint 01: Fail fast if compilation failed
        guard executable != nil else {
            throw SAM3Error.graphCompilationFailed("MLP: \(cacheKey)")
        }
        let results = try CompiledGraphCache.shared.runExecutable(key: cacheKey, queue: queue, feeds: feeds, targetTensors: [outputTensor])
        
        guard let resultData = results[outputTensor] else {
            throw SAM3Error.executionFailed("MLP: No result")
        }
        
        guard let output = BufferAllocator.shared.privateBuffer(length: input.length, device: device, label: "MLPOut") else {
            throw SAM3Error.bufferAllocationFailed("MLP output buffer")
        }
        recycledBuffers.append(output)
        resultData.mpsndarray().exportData(with: commandBuffer, to: output, destinationDataType: dt, offset: 0, rowStrides: nil as UnsafeMutablePointer<Int>?)
        
        return output
    }
}
/// Complete ViT Encoder

/// Complete ViT Encoder
@available(macOS 15.0, *)
public final class ViTEncoder {
    let device: MTLDevice
    let patchEmbed: PatchEmbedding
    let blocks: [TransformerBlock]
    let neck: NeckLayer
    private var posEmbed: MTLBuffer?
    // Cached RoPE frequencies by grid/window size
    private var ropeFreqsByGridSize: [Int: MTLBuffer] = [:]
    private var ropeFreqsByWindowSize: [Int: MTLBuffer] = [:]
    private var tokenPruner: TokenPruner? // Optim 6: Pruning
    
    // Kernel for texture->buffer conversion
    private let flattenKernel: MTLComputePipelineState
    private let addResidualKernel: MTLComputePipelineState
    private let useHalfPrecision: Bool
    private let config: SAM3EncoderConfig
    
    // Optimization: Cached position encodings
    private var cachedRoPEFreqs: MTLBuffer?
    private var cachedInputSize: (Int, Int)? 
    
    public convenience init(
        device: MTLDevice,
        embedDim: Int,
        patchSize: Int,
        inChannels: Int = 3,
        numHeads: Int = WeightMapper.numHeads,
        numBlocks: Int = WeightMapper.numBlocks,
        mlpHiddenDim: Int = WeightMapper.mlpHiddenDim,
        inputSize: Int = WeightMapper.inputSize,
        useHalfPrecision: Bool = true
    ) throws {
        let cfg = SAM3EncoderConfig(
            embedDim: embedDim,
            numHeads: numHeads,
            numBlocks: numBlocks,
            patchSize: patchSize,
            inputSize: inputSize,
            mlpHiddenDim: mlpHiddenDim,
            inChannels: inChannels
        )
        try self.init(device: device, config: cfg, numBlocks: numBlocks, useHalfPrecision: useHalfPrecision)
    }

    public convenience init(device: MTLDevice, numBlocks: Int? = nil, useHalfPrecision: Bool = true) throws {
        try self.init(device: device, config: .sam3Checkpoint, numBlocks: numBlocks, useHalfPrecision: useHalfPrecision)
    }

    public init(device: MTLDevice, config: SAM3EncoderConfig = .sam3Checkpoint, numBlocks: Int? = nil, useHalfPrecision: Bool = true) throws {
        self.device = device
        self.useHalfPrecision = useHalfPrecision
        self.config = config
        
        // Robust library loading
        var lib: MTLLibrary?
        
        if let bundle = Bundle.moduleIfAvailable {
            print("DEBUG: Trying to load library from Bundle.module")
            // Try standard bundle loading first
            do {
                lib = try device.makeDefaultLibrary(bundle: bundle)
            } catch {
                // Fallback to searching for default.metallib
                if let path = bundle.path(forResource: "default", ofType: "metallib") {
                     lib = try? device.makeLibrary(filepath: path)
                } else {
                    // RUNTIME COMPILATION FALLBACK
                    // If metallib is missing, try compiling .metal sources from bundle
                    let fileManager = FileManager.default
                    if let contents = try? fileManager.contentsOfDirectory(atPath: bundle.bundlePath) {
                        let metalFiles = contents.filter { $0.hasSuffix(".metal") }
                        if !metalFiles.isEmpty {
                             var source = ""
                             for file in metalFiles {
                                 let path = bundle.bundlePath + "/" + file
                                 if let content = try? String(contentsOfFile: path, encoding: .utf8) {
                                     source += "\n// File: \(file)\n"
                                     source += content
                                 }
                             }
                             
                             do {
                                 lib = try device.makeLibrary(source: source, options: nil)
                             } catch {
                                 print("Runtime compilation failed: \(error)")
                             }
                        }
                    }
                }
            }
        }
        if lib == nil {
            lib = device.makeDefaultLibrary()
        }
        
        guard let library = lib else {
            throw SAM3Error.noMetalDevice
        }
        
        self.patchEmbed = PatchEmbedding(device: device, embedDim: config.embedDim, patchSize: config.patchSize, inChannels: config.inChannels, useHalfPrecision: useHalfPrecision)
        
        let blockCount = numBlocks ?? config.numBlocks
        self.blocks = try (0..<blockCount).map { _ in
            try TransformerBlock(
                device: device,
                dim: config.embedDim,
                numHeads: config.numHeads,
                mlpHiddenDim: config.mlpHiddenDim,
                library: library,
                useHalfPrecision: useHalfPrecision
            )
        }
        
        // Neck Layer
        self.neck = NeckLayer(device: device, inDim: config.embedDim, outDim: 256, useHalfPrecision: useHalfPrecision)
        
          // Load Utility Kernels
          let flattenName = useHalfPrecision ? "texture_to_buffer_flat_half" : "texture_to_buffer_flat"
          let addName = useHalfPrecision ? "add_residual_half" : "add_residual_float"
          guard let flattenFunc = library.makeFunction(name: flattenName),
              let addFunc = library.makeFunction(name: addName) else {
            throw SAM3Error.noMetalDevice
        }
        self.flattenKernel = try device.makeComputePipelineState(function: flattenFunc)
        self.addResidualKernel = try device.makeComputePipelineState(function: addFunc)
        
        // Init Pruner (keepK fixed for now; dim must match embedDim)
        self.tokenPruner = TokenPruner(device: device, keepK: 1024, dim: config.embedDim, ropeDim: config.dimPerHead)
    }

    private func ropeFreqsForGridSize(_ gridSize: Int) -> MTLBuffer {
        if let buf = ropeFreqsByGridSize[gridSize] { return buf }
        let maxSeq = gridSize * gridSize
        let rope = RoPE(device: device, numHeads: config.numHeads, headDim: config.dimPerHead, maxSeqLen: maxSeq)
        let buf = rope.getFrequencies() ?? device.makeBuffer(length: maxSeq * config.dimPerHead * MemoryLayout<Float>.stride, options: .storageModeShared)!
        ropeFreqsByGridSize[gridSize] = buf
        return buf
    }

    private func ropeFreqsForWindowSize(_ windowSize: Int) -> MTLBuffer {
        if let buf = ropeFreqsByWindowSize[windowSize] { return buf }
        let maxSeq = windowSize * windowSize
        let rope = RoPE(device: device, numHeads: config.numHeads, headDim: config.dimPerHead, maxSeqLen: maxSeq)
        let buf = rope.getFrequencies() ?? device.makeBuffer(length: maxSeq * config.dimPerHead * MemoryLayout<Float>.stride, options: .storageModeShared)!
        ropeFreqsByWindowSize[windowSize] = buf
        return buf
    }
    
    public func loadWeights(_ weights: [String: Data]) {
        // 1. Patch Embed
        if let w = weights.buffer(for: "patch_embed.proj.weight", device: device),
           let b = weights.buffer(for: "patch_embed.proj.bias", device: device) {
             patchEmbed.loadWeights(weights: w, bias: b)
        }
        
        // 2. Pos Embed
        if let p = weights.buffer(for: "pos_embed", device: device) {
             self.posEmbed = p
        }
        
        // 3. Blocks
        for (i, block) in blocks.enumerated() {
             block.loadWeights(weights, prefix: "blocks.\(i)")
        }
        
        // 4. Neck
        if let c1w = weights.buffer(for: "neck.0.weight", device: device),
           let c1b = weights.buffer(for: "neck.0.bias", device: device),
           let c2w = weights.buffer(for: "neck.2.weight", device: device),
           let c2b = weights.buffer(for: "neck.2.bias", device: device) {
            neck.loadWeights(conv1W: c1w, conv1B: c1b, conv2W: c2w, conv2B: c2b)
        }
    }
    
    /// Forward pass of the ViT Image Encoder
    /// - Parameters:
    ///   - image: Input texture (e.g. from ARKit or Camera)
    ///   - commandBuffer: Metal command buffer
    /// - Returns: Image embeddings [1, 256, 16, 16] flattened
    public func forward(image: MTLTexture, commandBuffer: MTLCommandBuffer) throws -> MTLBuffer { // Added throws
        let clock = ContinuousClock()
        let totalStart = clock.now

        // 1. Patch embedding [H, W] -> Texture [H/patchSize, W/patchSize, D]
        let patchTexture = try patchEmbed.forward(
            input: image,
            commandBuffer: commandBuffer
        )
        
        // 2. Convert to buffer for Transformer blocks
        let seqLen = patchTexture.width * patchTexture.height
        let dim = config.embedDim

        let bytesPerElement = useHalfPrecision ? 2 : 4

        // RoPE must match the current token grid.
        // Current attention assumes square grids.
        let gridSize = patchTexture.width
        let windowSize = (gridSize % 16 == 0) ? 16 : 24
        let ropeFreqs = ropeFreqsForGridSize(gridSize)
        let ropeWindowFreqs = ropeFreqsForWindowSize(windowSize)
        
        // Track buffers for recycling
        var recycledBuffers: [MTLBuffer] = []
        
        // Create embedding buffer
        // Note: Embeddings are input to the transformer.
        // We use BufferAllocator.shared.sharedBuffer (or private?)
        // FlattenKernel runs on GPU, so private is fine?
        // Wait, current impl used .storageModeShared for embeddings "zero copy" idea?
        // But for consistency with Allocator, use sharedBuffer if we want CPU access (debug) or privateBuffer.
        // Let's use privateBuffer for speed.
        
           let embedBytes = seqLen * dim * bytesPerElement
        guard let embeddings = BufferAllocator.shared.privateBuffer(length: embedBytes, device: device, label: "Embeddings") else {
            throw SAM3Error.bufferAllocationFailed("PatchEmbedding embeddings buffer")
        }
        recycledBuffers.append(embeddings)
        
        // Run Flatten Kernel
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(flattenKernel)
            encoder.setTexture(patchTexture, index: 0)
            encoder.setBuffer(embeddings, offset: 0, index: 0)
            var dimParam = UInt32(dim)
            encoder.setBytes(&dimParam, length: 4, index: 1)
            
            let threadGroupSize = MTLSize(width: 8, height: 8, depth: 1)
            let threadGroups = MTLSize(
                width: (patchTexture.width + 7) / 8,
                height: (patchTexture.height + 7) / 8,
                depth: 1
            )
            
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            encoder.endEncoding()
        }
        
        // Add CLS token & Pos Embed (Simplified for test: just add pos embed)
        if let pos = posEmbed {
            let enc = commandBuffer.makeComputeCommandEncoder()!
            enc.setComputePipelineState(addResidualKernel)
            enc.setBuffer(embeddings, offset: 0, index: 0)
            enc.setBuffer(pos, offset: 0, index: 1)
            enc.setBuffer(embeddings, offset: 0, index: 2) // allow in-place by aliasing out=a
            var countU = UInt32(seqLen * dim)
            enc.setBytes(&countU, length: MemoryLayout<UInt32>.stride, index: 3)
            // Grid size... (Simplified)
            // let w = enc.setThreadgroupMemoryLength(length: 1, index: 0) // Dummy - removed
            enc.dispatchThreads(MTLSizeMake(seqLen * dim, 1, 1), threadsPerThreadgroup: MTLSizeMake(256, 1, 1))
            enc.endEncoding()
        }
        
        var features = embeddings
        let batch = 1 // Basic ViT assumes batch 1 per image for now
        
        var currentSeqLen = seqLen
        var currentRoPEFreqsWindow = ropeWindowFreqs
        var currentRoPEFreqsGlobal = ropeFreqs
        var isPruned = false
        var pruneIndices: MTLBuffer? = nil
        var patchAndPosMs: Double = 0
        var blocksPrePruneMs: Double = 0
        var pruneMs: Double = 0
        var blocksPostPruneMs: Double = 0
        var restoreMs: Double = 0
        var neckMs: Double = 0

        func ms(_ duration: Duration) -> Double {
            Double(duration.components.seconds) * 1000.0 + Double(duration.components.attoseconds) / 1.0e15
        }

        // Everything up to the transformer blocks (patch embed + flatten + pos embed)
        let blocksStart = clock.now
        patchAndPosMs = ms(totalStart.duration(to: blocksStart))

        #if DEBUG
        Sam3Log.debug("Starting ViT Forward Loop. Blocks: \(blocks.count), seqLen: \(seqLen)")
        #endif

        let globalStride = max(1, blocks.count / 4)
        let globalBlockIndices = Set(stride(from: globalStride - 1, through: max(0, blocks.count - 1), by: globalStride))

        for (i, block) in blocks.enumerated() {
            #if DEBUG
            Sam3Log.debug("Block \(i) starting...")
            #endif
            
            // Optim 6: Prune after block 2 (before block 3)
            if i == 3 && !isPruned {
                #if DEBUG
                Sam3Log.debug("Pruning...")
                #endif
                let pruneStart = clock.now
                if let pruner = tokenPruner,
                   let (prunedFeats, prunedRoPE, indices) = try pruner.prune(
                    input: features,
                    ropeFreqs: currentRoPEFreqsGlobal, // Gather from Global RoPE
                    seqLen: currentSeqLen,
                    batch: batch,
                    commandBuffer: commandBuffer,
                    recycledBuffers: &recycledBuffers
                   ) {
                    features = prunedFeats
                    // Update State
                    currentSeqLen = pruner.keepK
                    currentRoPEFreqsGlobal = prunedRoPE
                    currentRoPEFreqsWindow = prunedRoPE // Not used but keeps consistency
                    isPruned = true
                    pruneIndices = indices
                    #if DEBUG
                    Sam3Log.debug("Pruning complete. New seqLen: \(currentSeqLen)")
                    #endif
                }
                pruneMs += ms(pruneStart.duration(to: clock.now))
            }
        
            // Determine if Global or Windowed.
            // Checkpoint patterns: 32 blocks -> {7,15,23,31}; 24 blocks -> {5,11,17,23}.
            let isGlobalIndex = globalBlockIndices.contains(i)
            // If Pruned, Force Global (Windowing breaks on unstructured data)
            let isGlobal = isGlobalIndex || isPruned
            let windowed = !isGlobal
            
            // Select correct RoPE
            let currentRoPE = windowed ? currentRoPEFreqsWindow : currentRoPEFreqsGlobal
            
            features = try block.forward(
                input: features,
                ropeFreqs: currentRoPE,
                seqLen: currentSeqLen,
                batch: batch,
                windowed: windowed,
                commandBuffer: commandBuffer,
                recycledBuffers: &recycledBuffers
            )

            if i == 2 {
                blocksPrePruneMs = ms(blocksStart.duration(to: clock.now))
            }
        }

        // Remaining blocks after pruning (or full pass if pruning didn't run)
        let blocksTotalMs = ms(blocksStart.duration(to: clock.now))
        blocksPostPruneMs = blocksTotalMs - blocksPrePruneMs - pruneMs
        
        // Restore Spatial if pruned, before Neck
        var finalFeatures = features
        if isPruned, let indices = pruneIndices, let pruner = tokenPruner {
               #if DEBUG
               Sam3Log.debug("Restoring spatial...")
               #endif
               let restoreStart = clock.now
             finalFeatures = try pruner.restoreSpatial(
                pruned: features,
                indices: indices,
                originalSeqLen: seqLen, // 4096
                batch: batch,
                commandBuffer: commandBuffer, // FIX: Matches new signature
                recycledBuffers: &recycledBuffers
             )
               #if DEBUG
               Sam3Log.debug("Restoration complete.")
               #endif
               restoreMs += ms(restoreStart.duration(to: clock.now))
        }
        
        // Neck
           let neckStart = clock.now
        let neckFeatures = try neck.forward( // Added try
            input: finalFeatures,
            batch: batch,
            height: image.height / config.patchSize,
            width: image.width / config.patchSize,
            commandBuffer: commandBuffer,
            recycledBuffers: &recycledBuffers
        )
        neckMs += ms(neckStart.duration(to: clock.now))
        
        let totalMs = ms(totalStart.duration(to: clock.now))
        Sam3Log.debug("[\(Date())] DEBUG: ViT Forward Loop Complete. Total time: \(Int(totalMs))ms")

        Sam3Log.stageTiming(
            String(
                format:
                    "SAM3_STAGE_TIMING ms: total=%.2f patch+pos=%.2f blocks_preprune=%.2f prune=%.2f blocks_postprune=%.2f restore=%.2f neck=%.2f",
                totalMs,
                patchAndPosMs,
                blocksPrePruneMs,
                pruneMs,
                blocksPostPruneMs,
                restoreMs,
                neckMs
            )
        )
        
        // Register cleanup for intermediate buffers
        // We must NOT recycle the `neckFeatures` (return value) yet.
        if let last = recycledBuffers.last, last === neckFeatures {
            recycledBuffers.removeLast()
        }
        
        // Capture buffer list for closure
        let buffersToRecycle = recycledBuffers
        commandBuffer.addCompletedHandler { _ in
            for buffer in buffersToRecycle {
                BufferAllocator.shared.recycle(buffer)
            }
        }
        
        return neckFeatures
    }
    
    private static func computeRoPEFrequencies(device: MTLDevice, size: Int = 64) -> MTLBuffer {
        let cfg = SAM3EncoderConfig.sam3Checkpoint
        let maxSeq = size * size
        let rope = RoPE(device: device, numHeads: cfg.numHeads, headDim: cfg.dimPerHead, maxSeqLen: maxSeq)

        if let buf = rope.getFrequencies() {
            return buf
        }

        // Fallback placeholder if RoPE init fails.
        return device.makeBuffer(length: maxSeq * cfg.dimPerHead * MemoryLayout<Float>.stride, options: .storageModeShared)!
    }
}

// Add helper specifically for testing Bundle access
extension Bundle {
    static var moduleIfAvailable: Bundle? {
        #if SWIFT_PACKAGE
        return Bundle.module
        #else
        return nil
        #endif
    }
}
