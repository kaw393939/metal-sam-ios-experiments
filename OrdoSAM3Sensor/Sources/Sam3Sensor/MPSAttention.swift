//
//  MPSAttention.swift
//  Sam3Sensor
//
//  FlashAttention (Optimization 3) using MPSGraph.scaledDotProductAttention (macOS 15+)
//

import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

@available(macOS 15.0, *)
public final class MPSAttentionLayer {
    private let device: MTLDevice
    private let numHeads: Int
    private let dimPerHead: Int
    private let useHalfPrecision: Bool
    
    // Weights (Initialized with dummies, updated via loadWeights)
    private var qkvWeights: MTLBuffer
    private var qkvBias: MTLBuffer?
    private var outputWeights: MTLBuffer
    private var outputBias: MTLBuffer?

    // Internal accessors for fused-block execution
    internal var qkvWeightsBuffer: MTLBuffer { qkvWeights }
    internal var outputWeightsBuffer: MTLBuffer { outputWeights }
    internal var useHalfPrecisionFlag: Bool { useHalfPrecision }

    internal func qkvBiasBufferOrZeros(device: MTLDevice) -> MTLBuffer {
        if let b = qkvBias { return b }
        let totalDim = numHeads * dimPerHead
        let bytesPerElement = useHalfPrecision ? 2 : 4
        let count = 3 * totalDim
        let buf = device.makeBuffer(length: count * bytesPerElement, options: .storageModeShared)!
        memset(buf.contents(), 0, count * bytesPerElement)
        return buf
    }

    internal func outputBiasBufferOrZeros(device: MTLDevice) -> MTLBuffer {
        if let b = outputBias { return b }
        let totalDim = numHeads * dimPerHead
        let bytesPerElement = useHalfPrecision ? 2 : 4
        let count = totalDim
        let buf = device.makeBuffer(length: count * bytesPerElement, options: .storageModeShared)!
        memset(buf.contents(), 0, count * bytesPerElement)
        return buf
    }
    
    public init(
        device: MTLDevice,
        numHeads: Int,
        dimPerHead: Int,
        useHalfPrecision: Bool = true
    ) {
        self.device = device
        self.numHeads = numHeads
        self.dimPerHead = dimPerHead
        self.useHalfPrecision = useHalfPrecision

        // Properly sized dummy buffers for macOS 26 strict validation
        let totalDim = numHeads * dimPerHead
        let bytesPerElement = useHalfPrecision ? 2 : 4
        // qkvWeights: [totalDim, 3 * totalDim]
        let qkvSize = totalDim * 3 * totalDim * bytesPerElement
        // outputWeights: [totalDim, totalDim]
        let outSize = totalDim * totalDim * bytesPerElement

        self.qkvWeights = device.makeBuffer(length: qkvSize, options: .storageModeShared)!
        self.outputWeights = device.makeBuffer(length: outSize, options: .storageModeShared)!
    }
    
    public func loadWeights(
        qkvWeight: MTLBuffer,
        qkvBias: MTLBuffer?,
        outputWeight: MTLBuffer,
        outputBias: MTLBuffer?
    ) {
        self.qkvWeights = qkvWeight
        self.qkvBias = qkvBias
        self.outputWeights = outputWeight
        self.outputBias = outputBias
    }
    
    public func forward(
        input: MTLBuffer, // [B, S, D]
        ropeFreqs: MTLBuffer, // [S, DPH/2, 2]
        batch: Int,
        seqLen: Int,
        windowed: Bool,
        commandBuffer: MTLCommandBuffer,
        recycledBuffers: inout [MTLBuffer]
    ) throws -> MTLBuffer {
        let totalDim = numHeads * dimPerHead
        let dt: MPSDataType = useHalfPrecision ? .float16 : .float32

        // Calculate RoPE effective sequence length - must match exactly between graph and feed
        // Sprint 12 Warning: "Windowed" RoPE logic was reducing sEff to 256 while Input was 4096.
        // This causes graph.broadcast to fail (256 -> 4096). 
        // We enforce sEff = seqLen until proper Window handling (reshaping input) is implemented.
        let sEff = seqLen

        // Cache Key: Attn_{B}_{S}_{sEff}_{W}_{Prec}
        let cacheKey = "Attn_\(batch)_\(seqLen)_\(sEff)_\(windowed ? "W" : "G")_\(useHalfPrecision ? "F16" : "F32")"

        let (graph, placeholders, outputTensor, executable) = CompiledGraphCache.shared.getOrCompile(key: cacheKey, device: device) {
            let graph = MPSGraph()

            // 1. Inputs: [Batch, Seq, Dim]
            let inputT = graph.placeholder(shape: [NSNumber(value: batch), NSNumber(value: seqLen), NSNumber(value: totalDim)], dataType: dt, name: "input")

            // 2. QKV Projection
            let qkvW = graph.placeholder(shape: [NSNumber(value: totalDim), NSNumber(value: 3 * totalDim)], dataType: dt, name: "qkvW")
            let qkvB = graph.placeholder(shape: [1, NSNumber(value: 3 * totalDim)], dataType: dt, name: "qkvB")

            let qkv = graph.matrixMultiplication(primary: inputT, secondary: qkvW, name: "qkvMM")
            let qkvBiased = graph.addition(qkv, qkvB, name: "qkvBias")

            // Split Q, K, V
            // Reshape to [B, S, 3, NH, DPH]
            let qkvReshaped = graph.reshape(qkvBiased, shape: [NSNumber(value: batch), NSNumber(value: seqLen), 3, NSNumber(value: numHeads), NSNumber(value: dimPerHead)], name: "qkvReshape")

            // Slice into Q, K, V
            let q = graph.sliceTensor(qkvReshaped, dimension: 2, start: 0, length: 1, name: "sliceQ")
            let k = graph.sliceTensor(qkvReshaped, dimension: 2, start: 1, length: 1, name: "sliceK")
            let v = graph.sliceTensor(qkvReshaped, dimension: 2, start: 2, length: 1, name: "sliceV")

            // Final shapes: [B, S, 1, NH, DPH] -> [B, S, NH, DPH]
            let q4 = graph.reshape(q, shape: [NSNumber(value: batch), NSNumber(value: seqLen), NSNumber(value: numHeads), NSNumber(value: dimPerHead)], name: "q4")
            let k4 = graph.reshape(k, shape: [NSNumber(value: batch), NSNumber(value: seqLen), NSNumber(value: numHeads), NSNumber(value: dimPerHead)], name: "k4")
            let v4 = graph.reshape(v, shape: [NSNumber(value: batch), NSNumber(value: seqLen), NSNumber(value: numHeads), NSNumber(value: dimPerHead)], name: "v4")

            // --- RoPE (Optimization 5) ---
            // ropeFreqs: [S_eff, DPH/2, 2] - sEff is captured from outer scope and matches feed data
            let ropeT = graph.placeholder(shape: [NSNumber(value: sEff), NSNumber(value: dimPerHead / 2), 2], dataType: .float32, name: "ropeFreqs")
            
            func applyRoPE(_ x: MPSGraphTensor, name: String) -> MPSGraphTensor {
                // x: [B, S, NH, DPH]
                // Reshape to [B, S, NH, DPH/2, 2]
                let xP = graph.reshape(x, shape: [NSNumber(value: batch), NSNumber(value: seqLen), NSNumber(value: numHeads), NSNumber(value: dimPerHead/2), 2], name: "\(name)/pairs")
                let xR = graph.sliceTensor(xP, dimension: 4, start: 0, length: 1, name: "\(name)/real")
                let xI = graph.sliceTensor(xP, dimension: 4, start: 1, length: 1, name: "\(name)/imag")

                let cos = graph.sliceTensor(ropeT, dimension: 2, start: 0, length: 1, name: "\(name)/cos")
                let sin = graph.sliceTensor(ropeT, dimension: 2, start: 1, length: 1, name: "\(name)/sin")

                // BroadCast RoPE to [B, S, NH, DPH/2, 1]
                var cosB = graph.broadcast(cos, shape: [NSNumber(value: batch), NSNumber(value: seqLen), NSNumber(value: numHeads), NSNumber(value: dimPerHead/2), 1], name: "\(name)/cosB")
                var sinB = graph.broadcast(sin, shape: [NSNumber(value: batch), NSNumber(value: seqLen), NSNumber(value: numHeads), NSNumber(value: dimPerHead/2), 1], name: "\(name)/sinB")

                // Cast RoPE to match x's datatype (Float16) for macOS 26 strict type checking
                cosB = graph.cast(cosB, to: dt, name: "\(name)/cosB_cast")
                sinB = graph.cast(sinB, to: dt, name: "\(name)/sinB_cast")

                let outR = graph.subtraction(graph.multiplication(xR, cosB, name: "re_re"), graph.multiplication(xI, sinB, name: "im_im"), name: "\(name)/outR")
                let outI = graph.addition(graph.multiplication(xR, sinB, name: "re_im"), graph.multiplication(xI, cosB, name: "im_re"), name: "\(name)/outI")

                let concat = graph.concatTensors([outR, outI], dimension: 4, name: "\(name)/concat")
                return graph.reshape(concat, shape: [NSNumber(value: batch), NSNumber(value: seqLen), NSNumber(value: numHeads), NSNumber(value: dimPerHead)], name: "\(name)/final")
            }
            
            let qRope = applyRoPE(q4, name: "q")
            let kRope = applyRoPE(k4, name: "k")
            
            // Transpose to [B, NH, S, DPH]
            let qT = graph.transposeTensor(qRope, dimension: 1, withDimension: 2, name: "qT")
            let kT = graph.transposeTensor(kRope, dimension: 1, withDimension: 2, name: "kT")
            let vT = graph.transposeTensor(v4, dimension: 1, withDimension: 2, name: "vT")
            
            // 3. FlashAttention (Optimization 3)
            let scale = Float(1.0 / sqrt(Double(dimPerHead)))
            let attn = graph.scaledDotProductAttention(query: qT, key: kT, value: vT, mask: nil, scale: scale, name: "sdpa")
            
            // attn: [B, NH, S, DPH] -> [B, S, NH, DPH]
            let attnBack = graph.transposeTensor(attn, dimension: 1, withDimension: 2, name: "attnBack")
            let attnFlat = graph.reshape(attnBack, shape: [NSNumber(value: batch), NSNumber(value: seqLen), NSNumber(value: totalDim)], name: "attnFlat")
            
            // 4. Output Projection
            let outW = graph.placeholder(shape: [NSNumber(value: totalDim), NSNumber(value: totalDim)], dataType: dt, name: "outW")
            let outB = graph.placeholder(shape: [1, NSNumber(value: totalDim)], dataType: dt, name: "outB")
            
            let proj = graph.matrixMultiplication(primary: attnFlat, secondary: outW, name: "outMM")
            let final = graph.addition(proj, outB, name: "outBias")
            
            let phs: [String: MPSGraphTensor] = [
                "input": inputT,
                "qkvW": qkvW, "qkvB": qkvB,
                "outW": outW, "outB": outB,
                "ropeFreqs": ropeT
            ]
            
            return (graph, phs, final)
        }
        
        let queue = commandBuffer.commandQueue

        // --- Prepare Feeds ---
        let inputData = MPSGraphTensorData(input, shape: [NSNumber(value: batch), NSNumber(value: seqLen), NSNumber(value: totalDim)], dataType: dt)
        
        func makeWrapper(_ buf: MTLBuffer?, shape: [NSNumber]) -> MPSGraphTensorData {
            guard let b = buf else {
                let zeros = [Float16](repeating: 0, count: shape.reduce(1){ $0 * $1.intValue })
                let data = zeros.withUnsafeBufferPointer { Data(buffer: $0) }
                return MPSGraphTensorData(device: MPSGraphDevice(mtlDevice: device), data: data, shape: shape, dataType: dt)
            }
            return MPSGraphTensorData(b, shape: shape, dataType: dt)
        }
        
        let qkvWData = makeWrapper(qkvWeights, shape: [NSNumber(value: totalDim), NSNumber(value: 3 * totalDim)])
        let qkvBData = makeWrapper(qkvBias, shape: [NSNumber(value: 1), NSNumber(value: 3 * totalDim)])
        let outWData = makeWrapper(outputWeights, shape: [NSNumber(value: totalDim), NSNumber(value: totalDim)])
        let outBData = makeWrapper(outputBias, shape: [NSNumber(value: 1), NSNumber(value: totalDim)])

        // sEff is already calculated above - reuse it for ropeData
        let ropeData = MPSGraphTensorData(ropeFreqs, shape: [NSNumber(value: sEff), NSNumber(value: dimPerHead / 2), 2], dataType: .float32)
        
        var feeds: [MPSGraphTensor: MPSGraphTensorData] = [:]
        if let p = placeholders["input"] { feeds[p] = inputData }
        if let p = placeholders["qkvW"] { feeds[p] = qkvWData }
        if let p = placeholders["qkvB"] { feeds[p] = qkvBData }
        if let p = placeholders["outW"] { feeds[p] = outWData }
        if let p = placeholders["outB"] { feeds[p] = outBData }
        if let p = placeholders["ropeFreqs"] { feeds[p] = ropeData }
        
        // Sprint 01: Fail fast if compilation failed
        guard executable != nil else {
            throw SAM3Error.graphCompilationFailed("MPSAttention: \(cacheKey)")
        }
        let results = try CompiledGraphCache.shared.runExecutable(key: cacheKey, queue: queue, feeds: feeds, targetTensors: [outputTensor])
        
        guard let resultData = results[outputTensor] else {
            throw SAM3Error.executionFailed("MPSAttention: No result")
        }
        
        let outputLength = batch * seqLen * totalDim * (useHalfPrecision ? 2 : 4)
        guard let output = BufferAllocator.shared.privateBuffer(length: outputLength, device: device, label: "AttnOut") else {
            throw SAM3Error.bufferAllocationFailed("MPSAttention output buffer")
        }
        recycledBuffers.append(output)
        
        resultData.mpsndarray().exportData(with: commandBuffer, to: output, destinationDataType: dt, offset: 0, rowStrides: nil as UnsafeMutablePointer<Int>?)
        return output
    }
}

/// Cross-attention variant (for Decoder)
@available(macOS 15.0, *)
public final class MPSCrossAttentionLayer {
    private let selfAttention: MPSAttentionLayer
    private let device: MTLDevice
    
    public init(device: MTLDevice, numHeads: Int, dimPerHead: Int, useHalfPrecision: Bool = true) {
        self.device = device
        self.selfAttention = MPSAttentionLayer(
            device: device,
            numHeads: numHeads,
            dimPerHead: dimPerHead,
            useHalfPrecision: useHalfPrecision
        )
    }
}
