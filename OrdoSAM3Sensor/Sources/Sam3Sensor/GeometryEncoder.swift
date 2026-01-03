
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

public class GeometryEncoder {
    let device: MTLDevice
    let embedDim: Int
    let enableHalfPrecision: Bool
    private var ioDataType: MPSDataType { enableHalfPrecision ? .float16 : .float32 }
    
    // Projections
    var pointsDirectProj: MTLBuffer?
    var pointsDirectBias: MTLBuffer?
    var labelEmbed: MTLBuffer?
    var clsEmbed: MTLBuffer?
    
    // Gaussian Matrix: [2, 128]
    var gaussianMatrix: MTLBuffer?
    
    var pointsPosEncProj: MTLBuffer?
    var pointsPosEncBias: MTLBuffer?
    
    // Final Proj/Norm
    var finalProjW: MTLBuffer?
    var finalProjB: MTLBuffer?
    var imagePreNorm: TwoWayLayerNorm?
    var encodeNorm: TwoWayLayerNorm?
    var finalNorm: TwoWayLayerNorm?
    
    var blocks: [GeometryBlock] = []
    
    public init(device: MTLDevice, embedDim: Int = 256, depth: Int = 3, enableHalfPrecision: Bool = true) {
        self.device = device
        self.embedDim = embedDim
        self.enableHalfPrecision = enableHalfPrecision
        
        for _ in 0..<depth {
            blocks.append(GeometryBlock(device: device, embedDim: embedDim, enableHalfPrecision: enableHalfPrecision))
        }
        
        imagePreNorm = TwoWayLayerNorm(device: device, normalizedShape: [NSNumber(value: embedDim)], enableHalfPrecision: enableHalfPrecision)
        encodeNorm = TwoWayLayerNorm(device: device, normalizedShape: [NSNumber(value: embedDim)], enableHalfPrecision: enableHalfPrecision)
        finalNorm = TwoWayLayerNorm(device: device, normalizedShape: [NSNumber(value: embedDim)], enableHalfPrecision: enableHalfPrecision)
    }
    
    public func loadWeights(weights: [String: Data]) {
        let prefix = "geometry_encoder"
        
        if let data = weights["\(prefix).gaussian_matrix"] {
             print("GeometryEncoder: Loaded WTS Gaussian size: \(data.count)")
             gaussianMatrix = data.withUnsafeBytes { ptr in
                device.makeBuffer(bytes: ptr.baseAddress!, length: data.count, options: .storageModeShared)
            }
        } else if let data = weights["manual_gaussian_matrix"] {
            print("GeometryEncoder: Loaded Manual Gaussian size: \(data.count)")
            gaussianMatrix = data.withUnsafeBytes { ptr in
                device.makeBuffer(bytes: ptr.baseAddress!, length: data.count, options: .storageModeShared)
            }
        }
    
        // Label/CLS Embeds
        labelEmbed = weights.buffer(for: "\(prefix).label_embed.weight", device: device)
        clsEmbed = weights.buffer(for: "\(prefix).cls_embed.weight", device: device)
        
        // Projections
        pointsDirectProj = weights.buffer(for: "\(prefix).points_direct_project.weight", device: device)
        pointsDirectBias = weights.buffer(for: "\(prefix).points_direct_project.bias", device: device)
        pointsPosEncProj = weights.buffer(for: "\(prefix).points_pos_enc_project.weight", device: device)
        pointsPosEncBias = weights.buffer(for: "\(prefix).points_pos_enc_project.bias", device: device)
        
        finalProjW = weights.buffer(for: "\(prefix).final_proj.weight", device: device)
        finalProjB = weights.buffer(for: "\(prefix).final_proj.bias", device: device)
        
        // Norms
        if let g = weights.buffer(for: "\(prefix).img_pre_norm.weight", device: device),
           let b = weights.buffer(for: "\(prefix).img_pre_norm.bias", device: device) {
            imagePreNorm?.loadWeights(gamma: g, beta: b)
        }
        if let g = weights.buffer(for: "\(prefix).norm.weight", device: device),
           let b = weights.buffer(for: "\(prefix).norm.bias", device: device) {
            finalNorm?.loadWeights(gamma: g, beta: b)
        }
        if let g = weights.buffer(for: "\(prefix).encode_norm.weight", device: device),
           let b = weights.buffer(for: "\(prefix).encode_norm.bias", device: device) {
            encodeNorm?.loadWeights(gamma: g, beta: b)
        }
        
        // Blocks
        for (i, block) in blocks.enumerated() {
            block.loadWeights(weights: weights, prefix: "\(prefix).encode.\(i)")
        }
        
        print("GeometryEncoder: ✅ Loaded weights (DirectProj: \(pointsDirectProj != nil), LabelEmbed: \(labelEmbed != nil), CLS: \(clsEmbed != nil))")
    }
    
    public func forward(
        points: MTLBuffer, 
        pointCount: Int,
        imageEmbeddings: MTLBuffer,
        labelEmbeddings: MTLBuffer? = nil,
        imageResolution: (Int, Int) = (64, 64),
        commandBuffer: MTLCommandBuffer
    ) throws -> MTLBuffer {
        
        let graph = MPSGraph()
        
        // Debug Buffer Size
        let expectedBytes = 1 * 4096 * embedDim * (enableHalfPrecision ? 2 : 4)
        print("GeometryEncoder.forward: points=\(points.length) imageEmbeddings=\(imageEmbeddings.length) expected=\(expectedBytes) enableHalfPrecision=\(enableHalfPrecision)")
        
        // 1. Inputs
        let pointsT = graph.placeholder(shape: [1, NSNumber(value: pointCount), 2], dataType: .float32, name: "points_in")
        let gaussianT = graph.placeholder(shape: [2, 128], dataType: .float32, name: "gaussian_matrix")
        
        let wPe = graph.placeholder(shape: [NSNumber(value: embedDim), NSNumber(value: embedDim)], dataType: ioDataType, name: "w_pe")
        let bPe = graph.placeholder(shape: [1, NSNumber(value: embedDim)], dataType: ioDataType, name: "b_pe")
        
        let wDirect = graph.placeholder(shape: [NSNumber(value: embedDim), 2], dataType: ioDataType, name: "w_direct")
        let bDirect = graph.placeholder(shape: [1, NSNumber(value: embedDim)], dataType: ioDataType, name: "b_direct")
        
        let labelT = graph.placeholder(shape: [1, NSNumber(value: pointCount), NSNumber(value: embedDim)], dataType: ioDataType, name: "label_in")
        let clsT = graph.placeholder(shape: [1, 1, NSNumber(value: embedDim)], dataType: ioDataType, name: "cls_token")
        let imgT_in = graph.placeholder(shape: [1, 4096, NSNumber(value: embedDim)], dataType: ioDataType, name: "image_in")
        
        let wFinal = graph.placeholder(shape: [NSNumber(value: embedDim), NSNumber(value: embedDim)], dataType: ioDataType, name: "w_final")
        let bFinal = graph.placeholder(shape: [1, NSNumber(value: embedDim)], dataType: ioDataType, name: "b_final")
        
        // Build Graph
        // Compute Gaussian PE
        let coordsMM = graph.matrixMultiplication(primary: pointsT, secondary: gaussianT, name: "gauss_mm")
        let scaleC = graph.constant(2.0 * Double.pi, dataType: .float32)
        let scaledC = graph.multiplication(coordsMM, scaleC, name: "gauss_scale")
        
        let sinPart = graph.sin(with: scaledC, name: "gauss_sin")
        let cosPart = graph.cos(with: scaledC, name: "gauss_cos")
        let peRaw = graph.concatTensors([sinPart, cosPart], dimension: 2, name: "gauss_cat")
        
        // Direct Projection
        let wDirectT = graph.transposeTensor(wDirect, dimension: 0, withDimension: 1, name: "w_direct_t")
        let pointsCast = graph.cast(pointsT, to: ioDataType, name: "points_cast")
        let directEmb = graph.addition(graph.matrixMultiplication(primary: pointsCast, secondary: wDirectT, name: "direct_mm"), bDirect, name: "direct_add")
        
        // PE Projection
        let wPeT = graph.transposeTensor(wPe, dimension: 0, withDimension: 1, name: "w_pe_t")
        let peRawCast = graph.cast(peRaw, to: ioDataType, name: "pe_raw_cast")
        let peEmb = graph.addition(graph.matrixMultiplication(primary: peRawCast, secondary: wPeT, name: "pe_mm"), bPe, name: "pe_add")
        
        // Sum Points
        var tokens = graph.addition(graph.addition(directEmb, peEmb, name: "token_sum_1"), labelT, name: "token_sum")
        
        // Prepend CLS
        tokens = graph.concatTensors([clsT, tokens], dimension: 1, name: "prepend_cls")
        
        // Image Pre-Norm
        var phs: [String: MPSGraphTensor] = [:]
        
        var imgT: MPSGraphTensor = imgT_in
        if let ipn = imagePreNorm {
             let (res, ipnPhs) = ipn.buildGraph(input: imgT_in, graph: graph, name: "img_pre_norm")
             imgT = res
             phs.merge(ipnPhs) { $1 }
        }
        
        // Blocks
        var x = tokens
        for (i, block) in blocks.enumerated() {
            let (res, blockPhs) = block.buildGraph(input: x, image: imgT, graph: graph, namePrefix: "geo_b\(i)")
            x = res
            phs.merge(blockPhs) { $1 }
        }
        
        // Encode Norm
        if let en = encodeNorm {
             let (res, enPhs) = en.buildGraph(input: x, graph: graph, name: "encode_norm")
             x = res
             phs.merge(enPhs) { $1 }
        }
        
        // Final Proj/Norm
        let wFinalT = graph.transposeTensor(wFinal, dimension: 0, withDimension: 1, name: "w_final_t")
        var finalOut = graph.addition(graph.matrixMultiplication(primary: x, secondary: wFinalT, name: "final_mm"), bFinal, name: "final_add")
        
        if let fn = finalNorm {
             let (res, fnPhs) = fn.buildGraph(input: finalOut, graph: graph, name: "geo_norm")
             finalOut = res
             phs.merge(fnPhs) { $1 }
        }
        
        // Feeds
        var feeds: [MPSGraphTensor: MPSGraphTensorData] = [
            pointsT: MPSGraphTensorData(points, shape: [1, NSNumber(value: pointCount), 2], dataType: .float32),
            imgT_in: MPSGraphTensorData(imageEmbeddings, shape: [1, 4096, NSNumber(value: embedDim)], dataType: ioDataType)
        ]
        
        if let g = gaussianMatrix {
            feeds[gaussianT] = MPSGraphTensorData(g, shape: [2, 128], dataType: .float32)
        }
        
        if let l = labelEmbeddings {
            feeds[labelT] = MPSGraphTensorData(l, shape: [1, NSNumber(value: pointCount), NSNumber(value: embedDim)], dataType: ioDataType)
        } else {
             let zeros = [Float](repeating: 0, count: pointCount * embedDim)
             let zBuf = device.makeBuffer(bytes: zeros, length: zeros.count*4, options: .storageModeShared)!
             feeds[labelT] = MPSGraphTensorData(zBuf, shape: [1, NSNumber(value: pointCount), NSNumber(value: embedDim)], dataType: .float32) // Fallback F32
        }
        
        if let c = clsEmbed {
            feeds[clsT] = MPSGraphTensorData(c, shape: [1, 1, NSNumber(value: embedDim)], dataType: ioDataType)
        } else {
            let zeros = [Float](repeating: 0, count: embedDim)
            let zBuf = device.makeBuffer(bytes: zeros, length: 4 * embedDim, options: .storageModeShared)!
            feeds[clsT] = MPSGraphTensorData(zBuf, shape: [1, 1, NSNumber(value: embedDim)], dataType: .float32)
        }
        
        feeds[wDirect] = MPSGraphTensorData(pointsDirectProj!, shape: [NSNumber(value: embedDim), 2], dataType: ioDataType)
        feeds[bDirect] = MPSGraphTensorData(pointsDirectBias!, shape: [1, NSNumber(value: embedDim)], dataType: ioDataType)
        feeds[wPe] = MPSGraphTensorData(pointsPosEncProj!, shape: [NSNumber(value: embedDim), NSNumber(value: embedDim)], dataType: ioDataType)
        feeds[bPe] = MPSGraphTensorData(pointsPosEncBias!, shape: [1, NSNumber(value: embedDim)], dataType: ioDataType)
        feeds[wFinal] = MPSGraphTensorData(finalProjW!, shape: [NSNumber(value: embedDim), NSNumber(value: embedDim)], dataType: ioDataType)
        feeds[bFinal] = MPSGraphTensorData(finalProjB!, shape: [1, NSNumber(value: embedDim)], dataType: ioDataType)
        
        imagePreNorm?.addFeeds(placeholders: phs, to: &feeds, name: "img_pre_norm")
        encodeNorm?.addFeeds(placeholders: phs, to: &feeds, name: "encode_norm")
        finalNorm?.addFeeds(placeholders: phs, to: &feeds, name: "geo_norm")
        
        for (i, block) in blocks.enumerated() {
            block.addFeeds(placeholders: phs, to: &feeds, prefix: "geo_b\(i)")
        }
        
        let mpsCmd = MPSCommandBuffer(commandBuffer: commandBuffer)
        let results = graph.encode(to: mpsCmd, feeds: feeds, targetTensors: [finalOut], targetOperations: nil, executionDescriptor: nil)
        
        let outTokenCount = pointCount + 1
        let outBuffer = device.makeBuffer(length: outTokenCount * embedDim * 4, options: .storageModeShared)!
        results[finalOut]?.mpsndarray().exportData(with: mpsCmd, to: outBuffer, destinationDataType: .float32, offset: 0, rowStrides: nil as UnsafeMutablePointer<Int>?)
        
        mpsCmd.commit()
        mpsCmd.waitUntilCompleted()
        return outBuffer
    }
    
    public func forwardAndBridge(
        points: [CGPoint],
        labels: [Int],
        imageEmbeddings: MTLBuffer,
        commandBuffer: MTLCommandBuffer
    ) throws -> (sparse: MTLBuffer, dense: MTLTexture?) {
        let pointCount = points.count
        var coordsData: [Float] = []
        let inputImageSize: (Float, Float) = (1008, 1008)
        
        for p in points {
             let nx = Float(p.x) / inputImageSize.0
             let ny = Float(p.y) / inputImageSize.1
             coordsData.append(nx)
             coordsData.append(ny)
        }
        
        let pointsBuffer = device.makeBuffer(bytes: coordsData, length: coordsData.count * 4, options: .storageModeShared)!
        
        var labelBuf: MTLBuffer?
        if let le = labelEmbed {
            let expectedHalfBytes = 2 * embedDim * 2
            let expectedFloatBytes = 2 * embedDim * 4

            if enableHalfPrecision {
                var gatheredLabels: [Float16] = []
                gatheredLabels.reserveCapacity(labels.count * embedDim)
                
                // Read from weights (Handle F16 or F32 weights)
                if le.length == expectedHalfBytes {
                    let lePtr = le.contents().bindMemory(to: Float16.self, capacity: le.length / 2)
                    for label in labels {
                        let safeLabel = max(0, min(1, label))
                        let startIdx = safeLabel * embedDim
                        for i in 0..<embedDim {
                            gatheredLabels.append(lePtr[startIdx + i])
                        }
                    }
                } else {
                    // Convert F32 weights to F16 buffer
                    let lePtr = le.contents().bindMemory(to: Float.self, capacity: le.length / 4)
                    for label in labels {
                        let safeLabel = max(0, min(1, label))
                        let startIdx = safeLabel * embedDim
                        for i in 0..<embedDim {
                            gatheredLabels.append(Float16(lePtr[startIdx + i]))
                        }
                    }
                }
                
                if !gatheredLabels.isEmpty {
                    labelBuf = gatheredLabels.withUnsafeBytes { ptr in
                        device.makeBuffer(bytes: ptr.baseAddress!, length: gatheredLabels.count * 2, options: .storageModeShared)
                    }
                }
            } else {
                var gatheredLabels: [Float] = []
                gatheredLabels.reserveCapacity(labels.count * embedDim)

                if le.length == expectedHalfBytes {
                    let lePtr = le.contents().bindMemory(to: Float16.self, capacity: le.length / 2)
                    for label in labels {
                        let safeLabel = max(0, min(1, label))
                        let startIdx = safeLabel * embedDim
                        for i in 0..<embedDim {
                            gatheredLabels.append(Float(lePtr[startIdx + i]))
                        }
                    }
                } else {
                    if le.length != expectedFloatBytes {
                        print("WARNING: GeometryEncoder labelEmbed unexpected byte size=\(le.length); expected \(expectedHalfBytes) (F16) or \(expectedFloatBytes) (F32). Interpreting as Float32.")
                    }
                    let lePtr = le.contents().bindMemory(to: Float.self, capacity: le.length / 4)
                    for label in labels {
                        let safeLabel = max(0, min(1, label))
                        let startIdx = safeLabel * embedDim
                        for i in 0..<embedDim {
                            gatheredLabels.append(lePtr[startIdx + i])
                        }
                    }
                }

                if !gatheredLabels.isEmpty {
                    labelBuf = device.makeBuffer(bytes: gatheredLabels, length: gatheredLabels.count * 4, options: .storageModeShared)
                }
            }
        }
        
        let sparseEmbeds = try forward(
            points: pointsBuffer, 
            pointCount: pointCount, 
            imageEmbeddings: imageEmbeddings, 
            labelEmbeddings: labelBuf, 
            commandBuffer: commandBuffer
        )
        
        return (sparseEmbeds, nil)
    }
    
    public func computeDensePE(gridSize: Int = 64, commandBuffer: MTLCommandBuffer) -> MTLBuffer {
        let count = gridSize * gridSize
        var coords: [Float] = []
        for h in 0..<gridSize {
            for w in 0..<gridSize {
                coords.append((Float(w) + 0.5) / Float(gridSize))
                coords.append((Float(h) + 0.5) / Float(gridSize))
            }
        }
        
        let coordBuf = device.makeBuffer(bytes: coords, length: coords.count * 4, options: .storageModeShared)!
        let graph = MPSGraph()
        let coordsT = graph.placeholder(shape: [NSNumber(value: count), 2], dataType: .float32, name: "coords")
        let gaussT = graph.placeholder(shape: [2, 128], dataType: .float32, name: "gauss")
        
        let proj = graph.matrixMultiplication(primary: coordsT, secondary: gaussT, name: "proj")
        let scale = graph.constant(2.0 * Double.pi, dataType: .float32)
        let scaled = graph.multiplication(proj, scale, name: "scale")
        
        let pe = graph.concatTensors([graph.sin(with: scaled, name: "sin"), graph.cos(with: scaled, name: "cos")], dimension: 1, name: "pe")
        let batchPE = graph.expandDims(pe, axis: 0, name: "batch_pe")
        
        var feeds: [MPSGraphTensor: MPSGraphTensorData] = [
            coordsT: MPSGraphTensorData(coordBuf, shape: [NSNumber(value: count), 2], dataType: .float32)
        ]
        if let g = gaussianMatrix {
            feeds[gaussT] = MPSGraphTensorData(g, shape: [2, 128], dataType: .float32)
        } else {
             let z = [Float](repeating: 0, count: 256)
             feeds[gaussT] = MPSGraphTensorData(device.makeBuffer(bytes: z, length: 1024)!, shape: [2, 128], dataType: .float32)
        }
        
        let mpsCmd = MPSCommandBuffer(commandBuffer: commandBuffer)
        let results = graph.encode(to: mpsCmd, feeds: feeds, targetTensors: [batchPE], targetOperations: nil, executionDescriptor: nil)
        
        let bytesPerElem = enableHalfPrecision ? 2 : 4
        let outBuf = device.makeBuffer(length: count * embedDim * bytesPerElem + 16384, options: .storageModeShared)!
        results[batchPE]?.mpsndarray().exportData(with: mpsCmd, to: outBuf, destinationDataType: ioDataType, offset: 0, rowStrides: nil as UnsafeMutablePointer<Int>?)
        // mpsCmd.commit() -> Caller commits
        // mpsCmd.waitUntilCompleted() -> Caller waits
        return outBuf
    }
}

class GeometryBlock {
    let device: MTLDevice
    let embedDim: Int
    let selfAttn: AttentionLayer
    let norm1: TwoWayLayerNorm
    let crossAttnImage: AttentionLayer
    let norm2: TwoWayLayerNorm
    let mlp: MLPLayer
    let norm3: TwoWayLayerNorm
    
    
    init(device: MTLDevice, embedDim: Int, enableHalfPrecision: Bool = true) {
        self.device = device
        self.embedDim = embedDim
        self.selfAttn = AttentionLayer(device: device, embedDim: embedDim, numHeads: 8, enableHalfPrecision: enableHalfPrecision)
        self.norm1 = TwoWayLayerNorm(device: device, normalizedShape: [NSNumber(value: embedDim)], enableHalfPrecision: enableHalfPrecision)
        self.crossAttnImage = AttentionLayer(device: device, embedDim: embedDim, numHeads: 8, enableHalfPrecision: enableHalfPrecision)
        self.norm2 = TwoWayLayerNorm(device: device, normalizedShape: [NSNumber(value: embedDim)], enableHalfPrecision: enableHalfPrecision)
        self.mlp = MLPLayer(device: device, inputDim: embedDim, hiddenDim: 2048, outputDim: embedDim, enableHalfPrecision: enableHalfPrecision)
        self.norm3 = TwoWayLayerNorm(device: device, normalizedShape: [NSNumber(value: embedDim)], enableHalfPrecision: enableHalfPrecision)
    }
    
    func loadWeights(weights: [String: Data], prefix: String) {
        if let fw = weights["\(prefix).self_attn.in_proj_weight"], let fb = weights["\(prefix).self_attn.in_proj_bias"] {
            splitAndLoad(fw, fb, into: selfAttn)
        }
        if let w = weights.buffer(for: "\(prefix).self_attn.out_proj.weight", device: device) { selfAttn.out_proj = w }
        if let b = weights.buffer(for: "\(prefix).self_attn.out_proj.bias", device: device) { selfAttn.out_bias = b }
        loadNorm(norm1, key: "norm1", weights: weights, prefix: prefix)
        
        if let fw = weights["\(prefix).cross_attn_image.in_proj_weight"], let fb = weights["\(prefix).cross_attn_image.in_proj_bias"] {
            splitAndLoad(fw, fb, into: crossAttnImage)
        }
        if let w = weights.buffer(for: "\(prefix).cross_attn_image.out_proj.weight", device: device) { crossAttnImage.out_proj = w }
        if let b = weights.buffer(for: "\(prefix).cross_attn_image.out_proj.bias", device: device) { crossAttnImage.out_bias = b }
        loadNorm(norm2, key: "norm2", weights: weights, prefix: prefix)
        
        if let w = weights.buffer(for: "\(prefix).linear1.weight", device: device) { mlp.w1 = w }
        if let b = weights.buffer(for: "\(prefix).linear1.bias", device: device) { mlp.b1 = b }
        if let w = weights.buffer(for: "\(prefix).linear2.weight", device: device) { mlp.w2 = w }
        if let b = weights.buffer(for: "\(prefix).linear2.bias", device: device) { mlp.b2 = b }
        loadNorm(norm3, key: "norm3", weights: weights, prefix: prefix)
    }
    
    func buildGraph(input: MPSGraphTensor, image: MPSGraphTensor, graph: MPSGraph, namePrefix: String) -> (MPSGraphTensor, [String: MPSGraphTensor]) {
        var phs: [String: MPSGraphTensor] = [:]
        
        let (n1, n1Phs) = norm1.buildGraph(input: input, graph: graph, name: "\(namePrefix)/n1")
        phs.merge(n1Phs) { $1 }
        
        let (sa, saPhs) = selfAttn.buildGraph(query: n1, key: n1, value: n1, graph: graph, name: "\(namePrefix)/sa")
        phs.merge(saPhs) { $1 }
        
        var x = graph.addition(input, sa, name: "\(namePrefix)/add1")
        
        let (n2, n2Phs) = norm2.buildGraph(input: x, graph: graph, name: "\(namePrefix)/n2")
        phs.merge(n2Phs) { $1 }
        
        let (ca, caPhs) = crossAttnImage.buildGraph(query: n2, key: image, value: image, graph: graph, name: "\(namePrefix)/ca")
        phs.merge(caPhs) { $1 }
        
        x = graph.addition(x, ca, name: "\(namePrefix)/add2")
        
        let (n3, n3Phs) = norm3.buildGraph(input: x, graph: graph, name: "\(namePrefix)/n3")
        phs.merge(n3Phs) { $1 }
        
        let (m, mPhs) = mlp.buildGraph(input: n3, graph: graph, name: "\(namePrefix)/mlp")
        phs.merge(mPhs) { $1 }
        
        let out = graph.addition(x, m, name: "\(namePrefix)/add3")
        return (out, phs)
    }
    
    func addFeeds(placeholders: [String: MPSGraphTensor], to feeds: inout [MPSGraphTensor: MPSGraphTensorData], prefix: String) {
        selfAttn.addFeeds(placeholders: placeholders, to: &feeds, name: "\(prefix)/sa")
        norm1.addFeeds(placeholders: placeholders, to: &feeds, name: "\(prefix)/n1")
        crossAttnImage.addFeeds(placeholders: placeholders, to: &feeds, name: "\(prefix)/ca")
        norm2.addFeeds(placeholders: placeholders, to: &feeds, name: "\(prefix)/n2")
        mlp.addFeeds(placeholders: placeholders, to: &feeds, name: "\(prefix)/mlp")
        norm3.addFeeds(placeholders: placeholders, to: &feeds, name: "\(prefix)/n3")
    }
    
    private func splitAndLoad(_ fw: Data, _ fb: Data, into layer: AttentionLayer) {
        let dim = embedDim
        
        // Detect if weights are float16 or float32 based on size
        let expectedFloat32Bytes = 3 * dim * dim * 4
        let expectedFloat16Bytes = 3 * dim * dim * 2
        let bytesPerElement: Int
        
        if fw.count == expectedFloat16Bytes {
            bytesPerElement = 2  // float16
        } else if fw.count == expectedFloat32Bytes {
            bytesPerElement = 4  // float32
        } else {
            print("WARNING: GeometryBlock splitAndLoad unexpected fw size=\(fw.count); expected \(expectedFloat16Bytes) (F16) or \(expectedFloat32Bytes) (F32). Assuming F32.")
            bytesPerElement = 4
        }
        
        let dimBytes = dim * dim * bytesPerElement
        layer.q_proj = ModelLoader.loadBuffer(from: fw.subdata(in: 0..<dimBytes), device: device)
        layer.k_proj = ModelLoader.loadBuffer(from: fw.subdata(in: dimBytes..<(2*dimBytes)), device: device)
        layer.v_proj = ModelLoader.loadBuffer(from: fw.subdata(in: (2*dimBytes)..<(3*dimBytes)), device: device)
        
        let dimBytesB = dim * bytesPerElement
        layer.q_bias = ModelLoader.loadBuffer(from: fb.subdata(in: 0..<dimBytesB), device: device)
        layer.k_bias = ModelLoader.loadBuffer(from: fb.subdata(in: dimBytesB..<(2*dimBytesB)), device: device)
        layer.v_bias = ModelLoader.loadBuffer(from: fb.subdata(in: (2*dimBytesB)..<(3*dimBytesB)), device: device)
    }
    
    private func loadNorm(_ n: TwoWayLayerNorm, key: String, weights: [String: Data], prefix: String) {
        if let g = weights.buffer(for: "\(prefix).\(key).weight", device: device),
           let b = weights.buffer(for: "\(prefix).\(key).bias", device: device) {
            n.loadWeights(gamma: g, beta: b)
        }
    }
}
