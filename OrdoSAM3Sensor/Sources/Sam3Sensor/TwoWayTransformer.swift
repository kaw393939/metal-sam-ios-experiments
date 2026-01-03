
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

public class TwoWayTransformer {
    let device: MTLDevice
    let depth: Int
    let embedDim: Int
    let numHeads: Int
    let mlpDim: Int
    let enableHalfPrecision: Bool

    private var ioDataType: MPSDataType { enableHalfPrecision ? .float16 : .float32 }
    private var bytesPerElement: Int { enableHalfPrecision ? 2 : 4 }
    
    var blocks: [TwoWayTransformerBlock] = []
    var finalNorm: TwoWayLayerNorm?
    let finalAttnTokenToImage: AttentionLayer
    
    public init(device: MTLDevice, depth: Int = 2, embedDim: Int = 256, numHeads: Int = 8, mlpDim: Int = 2048, enableHalfPrecision: Bool = true) {
        self.device = device
        self.depth = depth
        self.embedDim = embedDim
        self.numHeads = numHeads
        self.mlpDim = mlpDim
        self.enableHalfPrecision = enableHalfPrecision
        
        // Final Attn must be initialized before blocks use self
        self.finalAttnTokenToImage = AttentionLayer(device: device, embedDim: embedDim, numHeads: numHeads, internalDim: 128, enableHalfPrecision: enableHalfPrecision)
        
        for i in 0..<depth {
            blocks.append(TwoWayTransformerBlock(device: device, embedDim: embedDim, numHeads: numHeads, mlpDim: mlpDim, enableHalfPrecision: enableHalfPrecision, skipFirstLayerPE: i == 0))
        }
        finalNorm = TwoWayLayerNorm(device: device, normalizedShape: [NSNumber(value: embedDim)], enableHalfPrecision: enableHalfPrecision)
    }
    
    public func buildGraph(
        graph: MPSGraph,
        imageEmbeddings: MPSGraphTensor,
        imagePE: MPSGraphTensor,
        pointEmbeddings: MPSGraphTensor
    ) -> (pointEmbeddings: MPSGraphTensor, imageEmbeddings: MPSGraphTensor, placeholders: [String: MPSGraphTensor]) {
        var placeholders: [String: MPSGraphTensor] = [:]
        
        var currentImage = imageEmbeddings
        var currentPoint = pointEmbeddings
        let currentImagePE = imagePE
        let pointPE = pointEmbeddings 
        
        // Blocks
        for (i, block) in blocks.enumerated() {
            let (newPoint, newImage, blockPhs) = block.buildGraph(
                pointInput: currentPoint, 
                imageInput: currentImage, 
                pointPE: pointPE, 
                imagePE: currentImagePE,
                graph: graph, 
                namePrefix: "twt/b\(i)"
            )
            currentPoint = newPoint
            currentImage = newImage
            placeholders.merge(blockPhs) { $1 }
        }
        
        // Final Attn (Token -> Image)
        let faQ = graph.addition(currentPoint, pointPE, name: "final/q_pe")
        let faK = graph.addition(currentImage, currentImagePE, name: "final/k_pe")
        let faV = currentImage
        let (fa, faPhs) = finalAttnTokenToImage.buildGraph(query: faQ, key: faK, value: faV, graph: graph, name: "twt/final_attn")
        currentPoint = graph.addition(currentPoint, fa, name: "twt/final_add")
        placeholders.merge(faPhs) { $1 }
        
        // Final Norm
        if let fn = finalNorm {
            let (finalP, fnPhs) = fn.buildGraph(input: currentPoint, graph: graph, name: "twt/norm")
            placeholders.merge(fnPhs) { $1 }
            return (finalP, currentImage, placeholders)
        }
        
        return (currentPoint, currentImage, placeholders)
    }
    
    public func addFeeds(placeholders: [String: MPSGraphTensor], to feeds: inout [MPSGraphTensor: MPSGraphTensorData]) {
        for (i, block) in blocks.enumerated() {
            block.addFeeds(placeholders: placeholders, to: &feeds, namePrefix: "twt/b\(i)")
        }
        finalAttnTokenToImage.addFeeds(placeholders: placeholders, to: &feeds, name: "twt/final_attn")
        finalNorm?.addFeeds(placeholders: placeholders, to: &feeds, name: "twt/norm")
    }

    public func loadWeights(weights: [String: Data], prefix: String) {
        for (i, block) in blocks.enumerated() {
             block.loadWeights(weights: weights, prefix: "\(prefix).layers.\(i)")
        }
        print("TwoWayTransformer: ✅ Loaded \(blocks.count) blocks")
        
        // Final Attn: "final_attn_token_to_image" (Unfused in SAM3)
        if loadUnfusedAttn(finalAttnTokenToImage, key: "final_attn_token_to_image", weights: weights, prefix: prefix) {
            print("DEBUG: Loaded final_attn_token_to_image weights (unfused)")
        } else if let fusedW = weights["\(prefix).final_attn_token_to_image.in_proj_weight"],
                  let fusedB = weights["\(prefix).final_attn_token_to_image.in_proj_bias"] {
             splitAndLoad(fusedW, fusedB, into: finalAttnTokenToImage)
             print("DEBUG: Loaded final_attn_token_to_image weights (fused)")
        } else {
            print("WARNING: Failed to load final_attn_token_to_image weights")
        }
        
        if let w = weights.buffer(for: "\(prefix).final_attn_token_to_image.out_proj.weight", device: device) { 
            finalAttnTokenToImage.out_proj = w 
            print("DEBUG: Loaded final_attn out_proj.weight")
        }
        if let b = weights.buffer(for: "\(prefix).final_attn_token_to_image.out_proj.bias", device: device) { 
            finalAttnTokenToImage.out_bias = b 
            print("DEBUG: Loaded final_attn out_proj.bias")
        }
    }
    
    // Helper to load unfused QKV
    private func loadUnfusedAttn(_ layer: AttentionLayer, key: String, weights: [String: Data], prefix: String) -> Bool {
        var loaded = false
        let qKey = "\(prefix).\(key).q_proj.weight"
        print("DEBUG: loadUnfusedAttn trying key: \(qKey)")
        if let q = weights.buffer(for: qKey, device: device) { 
            layer.q_proj = q; loaded = true
            print("DEBUG: Found q_proj")
        } else {
            print("DEBUG: NOT FOUND: \(qKey)")
        }
        if let k = weights.buffer(for: "\(prefix).\(key).k_proj.weight", device: device) { layer.k_proj = k; loaded = true }
        if let v = weights.buffer(for: "\(prefix).\(key).v_proj.weight", device: device) { layer.v_proj = v; loaded = true }
        
        if let qb = weights.buffer(for: "\(prefix).\(key).q_proj.bias", device: device) { layer.q_bias = qb }
        if let kb = weights.buffer(for: "\(prefix).\(key).k_proj.bias", device: device) { layer.k_bias = kb }
        if let vb = weights.buffer(for: "\(prefix).\(key).v_proj.bias", device: device) { layer.v_bias = vb }
        return loaded
    }
    
    // CPU Splitter for Fused Weights
    private func splitAndLoad(_ fusedW: Data, _ fusedB: Data, into layer: AttentionLayer) {
        let dim = layer.embedDim
        let bpe = bytesPerElement
        guard fusedW.count % bpe == 0, fusedW.count >= 3 * dim * dim * bpe else { return }
        
        let inDim = fusedW.count / bpe / (3 * dim)
        let qRange = 0..<(dim * inDim * bpe)
        let kRange = (dim * inDim * bpe)..<(2 * dim * inDim * bpe)
        let vRange = (2 * dim * inDim * bpe)..<(3 * dim * inDim * bpe)
        
        layer.q_proj = ModelLoader.loadBuffer(from: fusedW.subdata(in: qRange), device: device)
        layer.k_proj = ModelLoader.loadBuffer(from: fusedW.subdata(in: kRange), device: device)
        layer.v_proj = ModelLoader.loadBuffer(from: fusedW.subdata(in: vRange), device: device)
        
        guard fusedB.count >= 3 * dim * bpe else { return }
        layer.q_bias = ModelLoader.loadBuffer(from: fusedB.subdata(in: 0..<(dim*bpe)), device: device)
        layer.k_bias = ModelLoader.loadBuffer(from: fusedB.subdata(in: (dim*bpe)..<(2*dim*bpe)), device: device)
        layer.v_bias = ModelLoader.loadBuffer(from: fusedB.subdata(in: (2*dim*bpe)..<(3*dim*bpe)), device: device)
    }



    public func forward(
        imageEmbeddings: MTLBuffer, 
        imagePE: MTLBuffer,
        pointEmbeddings: MTLBuffer
    ) throws -> (pointEmbeddings: MTLBuffer, imageEmbeddings: MTLBuffer) {
        
        // Use MPSGraph
        let graph = MPSGraph()
        
        // Define Inputs
        // Image: [1, 4096, 256] ? Or [Batch, Seq, Dim]
        // Points: [1, N, 256]
        
        // We need to know shapes. length / bytesPerElement / embedDim => Count
        let imageCount = imageEmbeddings.length / bytesPerElement / embedDim
        let pointCount = pointEmbeddings.length / bytesPerElement / embedDim
        
        let imageSeqLen = imageCount
        let pointSeqLen = pointCount
        
        let imageTensor = graph.placeholder(shape: [1, NSNumber(value: imageSeqLen), NSNumber(value: embedDim)], dataType: ioDataType, name: "image_emb")
        let pointTensor = graph.placeholder(shape: [1, NSNumber(value: pointSeqLen), NSNumber(value: embedDim)], dataType: ioDataType, name: "point_emb")
        let imagePETensor = graph.placeholder(shape: [1, NSNumber(value: imageSeqLen), NSNumber(value: embedDim)], dataType: ioDataType, name: "image_pe")
        
        let (finalPoint, finalImage, phs) = buildGraph(graph: graph, imageEmbeddings: imageTensor, imagePE: imagePETensor, pointEmbeddings: pointTensor)
        
        // Execute
        guard let queue = device.makeCommandQueue(),
              let cmd = queue.makeCommandBuffer() else {
            throw NSError(domain: "TwoWayTransformer", code: 1, userInfo: nil)
        }
        let mpsCmd = MPSCommandBuffer(commandBuffer: cmd)
        
        // Feeds
        // Create wrappers
        let imageFeed = MPSGraphTensorData(imageEmbeddings, shape: [1, NSNumber(value: imageSeqLen), NSNumber(value: embedDim)], dataType: ioDataType)
        let pointFeed = MPSGraphTensorData(pointEmbeddings, shape: [1, NSNumber(value: pointSeqLen), NSNumber(value: embedDim)], dataType: ioDataType)
        let imagePEFeed = MPSGraphTensorData(imagePE, shape: [1, NSNumber(value: imageSeqLen), NSNumber(value: embedDim)], dataType: ioDataType)
        
        // Block weights feeds?
        // We need to collect all weights from blocks and feed them
        var feeds: [MPSGraphTensor: MPSGraphTensorData] = [
            imageTensor: imageFeed,
            pointTensor: pointFeed,
            imagePETensor: imagePEFeed
        ]
        
        addFeeds(placeholders: phs, to: &feeds)
        
        // Targets
        let results = graph.encode(
            to: mpsCmd,
            feeds: feeds,
            targetTensors: [finalPoint, finalImage], // target both
            targetOperations: nil,
            executionDescriptor: nil
        )
        
        guard let pRes = results[finalPoint], let iRes = results[finalImage] else {
             throw NSError(domain: "TwoWayTransformer", code: 2, userInfo: nil)
        }
        
        // Output buffers
        let outPBuf = device.makeBuffer(length: pointEmbeddings.length, options: .storageModeShared)!
        let outIBuf = device.makeBuffer(length: imageEmbeddings.length, options: .storageModeShared)!
        
        pRes.mpsndarray().exportData(with: mpsCmd, to: outPBuf, destinationDataType: ioDataType, offset: 0, rowStrides: nil as UnsafeMutablePointer<Int>?)
        iRes.mpsndarray().exportData(with: mpsCmd, to: outIBuf, destinationDataType: ioDataType, offset: 0, rowStrides: nil as UnsafeMutablePointer<Int>?)
        
        if cmd.status == .notEnqueued {
            cmd.commit()
        }
        cmd.waitUntilCompleted()
        
        return (outPBuf, outIBuf)
    }
    
    public func randomInitialize() {
        for block in blocks {
            block.randomInitialize()
        }
        finalAttnTokenToImage.randomInitialize()
        finalNorm?.randomInitialize()
    }
}

// MARK: - Internal Components

public class TwoWayTransformerBlock {
    let selfAttn: AttentionLayer
    let crossAttnTokenToImage: AttentionLayer
    let mlp: MLPLayer
    let crossAttnImageToToken: AttentionLayer
    
    let norm1: TwoWayLayerNorm
    let norm2: TwoWayLayerNorm
    let norm3: TwoWayLayerNorm

    let norm4: TwoWayLayerNorm
    
    let device: MTLDevice
    let enableHalfPrecision: Bool
    let skipFirstLayerPE: Bool
    
    init(device: MTLDevice, embedDim: Int, numHeads: Int, mlpDim: Int, enableHalfPrecision: Bool, skipFirstLayerPE: Bool = false) {
        self.device = device
        self.enableHalfPrecision = enableHalfPrecision
        self.skipFirstLayerPE = skipFirstLayerPE
        self.selfAttn = AttentionLayer(device: device, embedDim: embedDim, numHeads: numHeads, internalDim: embedDim, enableHalfPrecision: enableHalfPrecision)
        
        // SAM3 Cross Attention uses 128 dim projection (256 -> 128 -> 256)
        // With 8 heads, headDim = 128 / 8 = 16
        let crossDim = 128
        self.crossAttnTokenToImage = AttentionLayer(device: device, embedDim: embedDim, numHeads: numHeads, internalDim: crossDim, enableHalfPrecision: enableHalfPrecision)
        self.mlp = MLPLayer(device: device, inputDim: embedDim, hiddenDim: mlpDim, outputDim: embedDim, enableHalfPrecision: enableHalfPrecision)
        self.crossAttnImageToToken = AttentionLayer(device: device, embedDim: embedDim, numHeads: numHeads, internalDim: crossDim, enableHalfPrecision: enableHalfPrecision)
        
        self.norm1 = TwoWayLayerNorm(device: device, normalizedShape: [NSNumber(value: embedDim)], enableHalfPrecision: enableHalfPrecision)
        self.norm2 = TwoWayLayerNorm(device: device, normalizedShape: [NSNumber(value: embedDim)], enableHalfPrecision: enableHalfPrecision)
        self.norm3 = TwoWayLayerNorm(device: device, normalizedShape: [NSNumber(value: embedDim)], enableHalfPrecision: enableHalfPrecision)
        self.norm4 = TwoWayLayerNorm(device: device, normalizedShape: [NSNumber(value: embedDim)], enableHalfPrecision: enableHalfPrecision)
    }
    
    public func randomInitialize() {
        selfAttn.randomInitialize()
        crossAttnTokenToImage.randomInitialize()
        mlp.randomInitialize()
        crossAttnImageToToken.randomInitialize()
        
        norm1.randomInitialize()
        norm2.randomInitialize()
        norm3.randomInitialize()
        norm4.randomInitialize()
    }
    
    func buildGraph(pointInput: MPSGraphTensor, imageInput: MPSGraphTensor, pointPE: MPSGraphTensor, imagePE: MPSGraphTensor, graph: MPSGraph, namePrefix: String) -> (MPSGraphTensor, MPSGraphTensor, [String: MPSGraphTensor]) {
        // PyTorch Logic: Post-Norm
        // query_pe is assumed to be pointInput (initially) or should be passed separately?
        // In MaskDecoder, pointEmbeddings IS passed as `point_embedding` AND `queries`.
        // BUT `queries` evolves through layers. `point_embedding` (PE) stays constant?
        // Wait. PyTorch `forward` in Transformer:
        // loops: queries, keys = layer(queries, keys, query_pe=point_embedding, key_pe=image_pe)
        // So `query_pe` is CONSTANT (original point embeddings).
        // `queries` changes.
        
        // I need to accept `queryPE` (original) separately from `pointInput` (current)?
        // My current signature: `pointInput` (current), `imageInput` (current), `imagePE` (constant).
        // I MISS `pointPE` (constant)!
        // `pointInput` in first layer IS `pointPE` (if initialized that way).
        // But for subsequent layers, `pointInput` is modified.
        // So I must pass `pointPE` (constant) to the block!
        
        // Refactor required: transformer.buildGraph needs `pointPE`?
        // In MaskDecoder: `tokens` is passed as `pointEmbeddings` to `buildGraph`.
        // `tokens` is `cat([iou, mask, sparse])`.
        // This `tokens` acts as both initial content AND positional encoding.
        // So `pointPE` should be `tokens` (before loop).
        // `currentPoint` updates.
        
        // I must update `buildGraph` signature to take `pointPE`.
        // But for this chunk I can just use `pointInput` if I assume it's layer 0? No.
        // 1. Self Attn (Sparse)
        // PyTorch: if skip_first_layer_pe, use raw queries (no PE, no residual)
        //          else: q = queries + query_pe, attn_out, queries = queries + attn_out
        var phs: [String: MPSGraphTensor] = [:]
        
        // 1. Self Attn (Sparse)
        // PyTorch: if skip_first_layer_pe, use raw queries (no PE, no residual)
        //          else: q = queries + query_pe, attn_out, queries = queries + attn_out
        var q = pointInput
        
        if skipFirstLayerPE {
            // Block 0: Use raw queries, NO residual connection
            let (res, saPhs) = selfAttn.buildGraph(query: q, key: q, value: q, graph: graph, name: "\(namePrefix)/sa")
            q = res
            phs.merge(saPhs) { $1 }
        } else {
            // Other blocks: Add PE to Q/K, keep residual
            let saQ = graph.addition(q, pointPE, name: "\(namePrefix)/sa_q_pe")
            let saK = saQ // Same
            let saV = q // Content only
            
            let (sa, saPhs) = selfAttn.buildGraph(query: saQ, key: saK, value: saV, graph: graph, name: "\(namePrefix)/sa")
            phs.merge(saPhs) { $1 }
            q = graph.addition(q, sa, name: "\(namePrefix)/add1")
        }
        let (n1, n1Phs) = norm1.buildGraph(input: q, graph: graph, name: "\(namePrefix)/norm1")
        q = n1
        phs.merge(n1Phs) { $1 }
        
        // 2. Cross Attn (Token -> Image)
        // q = q + pointPE, k = img + imgPE, v = img
        let ca1Q = graph.addition(q, pointPE, name: "\(namePrefix)/ca1_q_pe")
        let ca1K = graph.addition(imageInput, imagePE, name: "\(namePrefix)/ca1_k_pe")
        let ca1V = imageInput
        
        let (ca1, ca1Phs) = crossAttnTokenToImage.buildGraph(query: ca1Q, key: ca1K, value: ca1V, graph: graph, name: "\(namePrefix)/ca1")
        phs.merge(ca1Phs) { $1 }
        q = graph.addition(q, ca1, name: "\(namePrefix)/add2")
        let (n2, n2Phs) = norm2.buildGraph(input: q, graph: graph, name: "\(namePrefix)/norm2")
        q = n2
        phs.merge(n2Phs) { $1 }
        
        // 3. MLP
        let (m, mPhs) = mlp.buildGraph(input: q, graph: graph, name: "\(namePrefix)/mlp")
        phs.merge(mPhs) { $1 }
        q = graph.addition(q, m, name: "\(namePrefix)/add3")
        let (n3, n3Phs) = norm3.buildGraph(input: q, graph: graph, name: "\(namePrefix)/norm3")
        q = n3
        phs.merge(n3Phs) { $1 }
        
        // 4. Cross Attn (Image -> Token)
        var img = imageInput
        let ca2Q = graph.addition(img, imagePE, name: "\(namePrefix)/ca2_q_pe")
        let ca2K = graph.addition(q, pointPE, name: "\(namePrefix)/ca2_k_pe")
        let ca2V = q 
        
        let (ca2, ca2Phs) = crossAttnImageToToken.buildGraph(query: ca2Q, key: ca2K, value: ca2V, graph: graph, name: "\(namePrefix)/ca2")
        phs.merge(ca2Phs) { $1 }
        img = graph.addition(img, ca2, name: "\(namePrefix)/add4")
        let (n4, n4Phs) = norm4.buildGraph(input: img, graph: graph, name: "\(namePrefix)/norm4")
        img = n4
        phs.merge(n4Phs) { $1 }
        
        return (q, img, phs)
    }
    
    func addFeeds(placeholders: [String: MPSGraphTensor], to feeds: inout [MPSGraphTensor: MPSGraphTensorData], namePrefix: String) {
        selfAttn.addFeeds(placeholders: placeholders, to: &feeds, name: "\(namePrefix)/sa")
        crossAttnTokenToImage.addFeeds(placeholders: placeholders, to: &feeds, name: "\(namePrefix)/ca1")
        mlp.addFeeds(placeholders: placeholders, to: &feeds, name: "\(namePrefix)/mlp")
        crossAttnImageToToken.addFeeds(placeholders: placeholders, to: &feeds, name: "\(namePrefix)/ca2")
        norm1.addFeeds(placeholders: placeholders, to: &feeds, name: "\(namePrefix)/norm1")
        norm2.addFeeds(placeholders: placeholders, to: &feeds, name: "\(namePrefix)/norm2")
        norm3.addFeeds(placeholders: placeholders, to: &feeds, name: "\(namePrefix)/norm3")
        norm4.addFeeds(placeholders: placeholders, to: &feeds, name: "\(namePrefix)/norm4")
    }
    
    public func loadWeights(weights: [String: Data], prefix: String) {
        // let dim = 1024 // Unused
        
        // Helper to load or split fused weights
        func loadAttn(layer: AttentionLayer, keyPrefix: String) {
            var loadedCount = 0
            // Try logic for separate if legacy (or if we manually split later)
             if let q = weights.buffer(for: "\(prefix).\(keyPrefix).q_proj.weight", device: device) { layer.q_proj = q; loadedCount += 1 }
             if let k = weights.buffer(for: "\(prefix).\(keyPrefix).k_proj.weight", device: device) { layer.k_proj = k; loadedCount += 1 }
             if let v = weights.buffer(for: "\(prefix).\(keyPrefix).v_proj.weight", device: device) { layer.v_proj = v; loadedCount += 1 }
             
             // Output
             if let o = weights.buffer(for: "\(prefix).\(keyPrefix).out_proj.weight", device: device) { layer.out_proj = o; loadedCount += 1 }
             if let ob = weights.buffer(for: "\(prefix).\(keyPrefix).out_proj.bias", device: device) { layer.out_bias = ob }
             
             if loadedCount > 0 {
                 print("DEBUG: Loaded \(keyPrefix) weights: \(loadedCount)/4")
             }
        }
        
        // Refined Loading Logic that handles FUSED weights by looking for them and if found, 
        // assigning them to QKV via crude pointer offset if possible, or just identifying them.
        
        // 1. Self Attn: "self_attn"
        // Try separate QKV first (SAM style)
        if loadUnfusedAttn(selfAttn, key: "self_attn", weights: weights, prefix: prefix) {
             // Loaded unfused
        } else if let fusedW = weights[weightsKey(prefix, "self_attn.in_proj_weight")],
           let fusedB = weights[weightsKey(prefix, "self_attn.in_proj_bias")] {
            splitAndLoad(fusedW, fusedB, into: selfAttn)
        }
        if let w = weights.buffer(for: "\(prefix).self_attn.out_proj.weight", device: device) { selfAttn.out_proj = w }
        if let b = weights.buffer(for: "\(prefix).self_attn.out_proj.bias", device: device) { selfAttn.out_bias = b }
        
        // 2. Norm1: "norm1"
        loadNorm(norm1, key: "norm1", weights: weights, prefix: prefix)
        
        // 3. Cross Attn (Token->Image): "cross_attn_image" (NPZ key) for DETR, "cross_attn_token_to_image" for SAM
        // Try SAM key first
        if loadUnfusedAttn(crossAttnTokenToImage, key: "cross_attn_token_to_image", weights: weights, prefix: prefix) {
            // Loaded SAM style
        } else if let fusedW = weights[weightsKey(prefix, "cross_attn_image.in_proj_weight")],
           let fusedB = weights[weightsKey(prefix, "cross_attn_image.in_proj_bias")] {
            splitAndLoad(fusedW, fusedB, into: crossAttnTokenToImage)
        }
        
        if let w = weights.buffer(for: "\(prefix).cross_attn_token_to_image.out_proj.weight", device: device) { crossAttnTokenToImage.out_proj = w }
        if let b = weights.buffer(for: "\(prefix).cross_attn_token_to_image.out_proj.bias", device: device) { crossAttnTokenToImage.out_bias = b }
        
        // Try fallback legacy naming
        if let w = weights.buffer(for: "\(prefix).cross_attn_image.out_proj.weight", device: device) { crossAttnTokenToImage.out_proj = w }
        if let b = weights.buffer(for: "\(prefix).cross_attn_image.out_proj.bias", device: device) { crossAttnTokenToImage.out_bias = b }
        
        // 4. Norm2: "norm2"
        loadNorm(norm2, key: "norm2", weights: weights, prefix: prefix)
        
        // 5. MLP: "mlp.lin1", "mlp.lin2"
        var mlpCount = 0
        if let w = weights.buffer(for: "\(prefix).mlp.lin1.weight", device: device) { mlp.w1 = w; mlpCount += 1 }
        if let b = weights.buffer(for: "\(prefix).mlp.lin1.bias", device: device) { mlp.b1 = b; mlpCount += 1 }
        if let w = weights.buffer(for: "\(prefix).mlp.lin2.weight", device: device) { mlp.w2 = w; mlpCount += 1 }
        if let b = weights.buffer(for: "\(prefix).mlp.lin2.bias", device: device) { mlp.b2 = b; mlpCount += 1 }
        if mlpCount > 0 {
            print("DEBUG: Loaded MLP weights: \(mlpCount)/4")
        }
        
        // 6. Norm3: "norm3"
        loadNorm(norm3, key: "norm3", weights: weights, prefix: prefix)
        
        // 7. Cross Attn (Image->Token): "cross_attn_image_to_token" for SAM
        if loadUnfusedAttn(crossAttnImageToToken, key: "cross_attn_image_to_token", weights: weights, prefix: prefix) {
           // Loaded SAM style
        }
        if let w = weights.buffer(for: "\(prefix).cross_attn_image_to_token.out_proj.weight", device: device) { crossAttnImageToToken.out_proj = w }
        if let b = weights.buffer(for: "\(prefix).cross_attn_image_to_token.out_proj.bias", device: device) { crossAttnImageToToken.out_bias = b }

        
        // 8. Norm4: "norm4"
        loadNorm(norm4, key: "norm4", weights: weights, prefix: prefix)
        
        print("TwoWayTransformerBlock(\(prefix)): ✅ Loaded weights")
    }
    
    private func weightsKey(_ prefix: String, _ suffix: String) -> String {
        return "\(prefix).\(suffix)"
    }
    
    // Helper to load unfused QKV
    private func loadUnfusedAttn(_ layer: AttentionLayer, key: String, weights: [String: Data], prefix: String) -> Bool {
        var loaded = false
        if let q = weights.buffer(for: "\(prefix).\(key).q_proj.weight", device: device) { layer.q_proj = q; loaded = true }
        if let k = weights.buffer(for: "\(prefix).\(key).k_proj.weight", device: device) { layer.k_proj = k; loaded = true }
        if let v = weights.buffer(for: "\(prefix).\(key).v_proj.weight", device: device) { layer.v_proj = v; loaded = true }
        
        if let qb = weights.buffer(for: "\(prefix).\(key).q_proj.bias", device: device) { layer.q_bias = qb }
        if let kb = weights.buffer(for: "\(prefix).\(key).k_proj.bias", device: device) { layer.k_bias = kb }
        if let vb = weights.buffer(for: "\(prefix).\(key).v_proj.bias", device: device) { layer.v_bias = vb }
        return loaded
    }
    
    // CPU Splitter for Fused Weights
    private func splitAndLoad(_ fusedW: Data, _ fusedB: Data, into layer: AttentionLayer) {
        let dim = layer.embedDim
        let bpe = enableHalfPrecision ? 2 : 4
        guard fusedW.count % bpe == 0 else {
            print("CRTICAL ERROR: Fused weights not \(bpe)-byte aligned count: \(fusedW.count)")
            return
        }
        
        let elementCount = fusedW.count / bpe
        let inDim = elementCount / (3 * dim)
        
        if inDim != dim {
            print("WARNING: Fused weights inDim (\(inDim)) != dim (\(dim)). Count: \(fusedW.count)")
            // If inDim is 0, we crash below.
            if inDim == 0 { return }
        }
        
           // Expected size: 3 * dim * dim * bytesPerElement
           let expectedSize = 3 * dim * dim * bpe
        if fusedW.count < expectedSize {
               print("CRITICAL ERROR: Fused weights size \(fusedW.count) too small for 3*\(dim)*\(dim)*\(bpe) (\(expectedSize))")
             return
        }
        
           let qRange = 0..<(dim * inDim * bpe)
           let kRange = (dim * inDim * bpe)..<(2 * dim * inDim * bpe)
           let vRange = (2 * dim * inDim * bpe)..<(3 * dim * inDim * bpe)
        
        // print("DEBUG: Splitting fused weights size \(fusedW.count)")
        
        layer.q_proj = ModelLoader.loadBuffer(from: fusedW.subdata(in: qRange), device: device)
        layer.k_proj = ModelLoader.loadBuffer(from: fusedW.subdata(in: kRange), device: device)
        layer.v_proj = ModelLoader.loadBuffer(from: fusedW.subdata(in: vRange), device: device)
        
        // Bias Check
           let expectedBiasSize = 3 * dim * bpe
        if fusedB.count < expectedBiasSize {
               print("CRITICAL ERROR: Fused bias size \(fusedB.count) too small for 3*\(dim)*\(bpe) (\(expectedBiasSize))")
             return
        }
        
           layer.q_bias = ModelLoader.loadBuffer(from: fusedB.subdata(in: 0..<(dim*bpe)), device: device)
           layer.k_bias = ModelLoader.loadBuffer(from: fusedB.subdata(in: (dim*bpe)..<(2*dim*bpe)), device: device)
           layer.v_bias = ModelLoader.loadBuffer(from: fusedB.subdata(in: (2*dim*bpe)..<(3*dim*bpe)), device: device)
    }
    
    private func loadNorm(_ norm: TwoWayLayerNorm, key: String, weights: [String: Data], prefix: String) {
        if let g = weights.buffer(for: "\(prefix).\(key).weight", device: device),
           let b = weights.buffer(for: "\(prefix).\(key).bias", device: device) {
             norm.loadWeights(gamma: g, beta: b)
        }
    }
}

public class AttentionLayer {
    // Basic Multi-head Attention
    let embedDim: Int
    let internalDim: Int
    let numHeads: Int
    let headDim: Int
    
    // Weights (Random/Placeholder)
    var q_proj: MTLBuffer?
    var k_proj: MTLBuffer?
    var v_proj: MTLBuffer?
    var out_proj: MTLBuffer?
    
    var q_bias: MTLBuffer?
    var k_bias: MTLBuffer?
    var v_bias: MTLBuffer?
    var out_bias: MTLBuffer?
    
    // Stateless placeholders
    // var placeholders ... REMOVED
    
    let device: MTLDevice
    let enableHalfPrecision: Bool

    private var ioDataType: MPSDataType { enableHalfPrecision ? .float16 : .float32 }
    private var bytesPerElement: Int { enableHalfPrecision ? 2 : 4 }
    
    init(device: MTLDevice, embedDim: Int, numHeads: Int, internalDim: Int? = nil, enableHalfPrecision: Bool = true) {
        self.device = device
        self.enableHalfPrecision = enableHalfPrecision
        self.embedDim = embedDim
        self.internalDim = internalDim ?? embedDim
        self.numHeads = numHeads
        self.headDim = self.internalDim / numHeads
        
        // Weights must be loaded via loadWeights() - no random fallback
        q_proj = nil
        k_proj = nil
        v_proj = nil
        out_proj = nil
        q_bias = nil
        k_bias = nil
        v_bias = nil
        out_bias = nil
    }
    
    public func loadWeights(qkvWeight: MTLBuffer, qkvBias: MTLBuffer, outputWeight: MTLBuffer, outputBias: MTLBuffer) {
        // Legacy - not used for separate proj
    }
    
    public func randomInitialize() {
        let bpe = bytesPerElement
        
        // Q, K, V Proj: [Internal, Embed]
        let qkvSize = internalDim * embedDim * bpe
        let qkvBiasSize = internalDim * bpe
        
        // Out Proj: [Embed, Internal]
        let outSize = embedDim * internalDim * bpe
        let outBiasSize = embedDim * bpe
        
        func alloc(_ len: Int) -> MTLBuffer {
            let b = device.makeBuffer(length: len, options: .storageModeShared)!
            memset(b.contents(), 0, len)
            return b
        }
        
        q_proj = alloc(qkvSize)
        k_proj = alloc(qkvSize)
        v_proj = alloc(qkvSize)
        out_proj = alloc(outSize)
        
        q_bias = alloc(qkvBiasSize)
        k_bias = alloc(qkvBiasSize)
        v_bias = alloc(qkvBiasSize)
        out_bias = alloc(outBiasSize)
    }
    
    public func loadWeights(q: MTLBuffer, k: MTLBuffer, v: MTLBuffer, o: MTLBuffer, qb: MTLBuffer, kb: MTLBuffer, vb: MTLBuffer, ob: MTLBuffer) {
        self.q_proj = q
        self.k_proj = k
        self.v_proj = v
        self.out_proj = o
        self.q_bias = qb
        self.k_bias = kb
        self.v_bias = vb
        self.out_bias = ob
    }
    
    func buildGraph(query: MPSGraphTensor, key: MPSGraphTensor, value: MPSGraphTensor, graph: MPSGraph, name: String) -> (MPSGraphTensor, [String: MPSGraphTensor]) {
        // Placeholders for weights - MATCH PYTORCH SHAPE [Out, In]
        // W: [Internal, Embed]
        let w_q = graph.placeholder(shape: [NSNumber(value: internalDim), NSNumber(value: embedDim)], dataType: ioDataType, name: "\(name)/wq")
        let w_k = graph.placeholder(shape: [NSNumber(value: internalDim), NSNumber(value: embedDim)], dataType: ioDataType, name: "\(name)/wk")
        let w_v = graph.placeholder(shape: [NSNumber(value: internalDim), NSNumber(value: embedDim)], dataType: ioDataType, name: "\(name)/wv")
        
        // Output: [Embed, Internal] (PyTorch shape)
        let w_o = graph.placeholder(shape: [NSNumber(value: embedDim), NSNumber(value: internalDim)], dataType: ioDataType, name: "\(name)/wo")
        
        let b_q = graph.placeholder(shape: [1, NSNumber(value: internalDim)], dataType: ioDataType, name: "\(name)/bq")
        let b_k = graph.placeholder(shape: [1, NSNumber(value: internalDim)], dataType: ioDataType, name: "\(name)/bk")
        let b_v = graph.placeholder(shape: [1, NSNumber(value: internalDim)], dataType: ioDataType, name: "\(name)/bv")
        let b_o = graph.placeholder(shape: [1, NSNumber(value: embedDim)], dataType: ioDataType, name: "\(name)/bo")
        
        var phs: [String: MPSGraphTensor] = [
            "\(name)/w_q": w_q, "\(name)/w_k": w_k, "\(name)/w_v": w_v, "\(name)/w_o": w_o,
            "\(name)/b_q": b_q, "\(name)/b_k": b_k, "\(name)/b_v": b_v, "\(name)/b_o": b_o
        ]
        
        // Transpose weights for MatMul: [Out, In] -> [In, Out]
        let w_q_t = graph.transposeTensor(w_q, dimension: 0, withDimension: 1, name: "\(name)/wq_t")
        let w_k_t = graph.transposeTensor(w_k, dimension: 0, withDimension: 1, name: "\(name)/wk_t")
        let w_v_t = graph.transposeTensor(w_v, dimension: 0, withDimension: 1, name: "\(name)/wv_t")
        // w_o is [Embed, Internal], transpose to [Internal, Embed] for (Internal @ Internal->Embed)
        // Wait. Context is [B, S, Internal]. OutProj is [Internal -> Embed].
        // PyTorch W_o is [Embed, Internal].
        // So we need [Internal, Embed].
        // So transpose W_o.
        let w_o_t = graph.transposeTensor(w_o, dimension: 0, withDimension: 1, name: "\(name)/wo_t")
        
        // Projections
        let q = graph.addition(graph.matrixMultiplication(primary: query, secondary: w_q_t, name: "\(name)/q_mm"), b_q, name: "\(name)/q_add")
        let k = graph.addition(graph.matrixMultiplication(primary: key, secondary: w_k_t, name: "\(name)/k_mm"), b_k, name: "\(name)/k_add")
        let v = graph.addition(graph.matrixMultiplication(primary: value, secondary: w_v_t, name: "\(name)/v_mm"), b_v, name: "\(name)/v_add")
        
        // Reshape & Transpose -> [B, N, S, H]
        let targetShape = [NSNumber(value: 1), NSNumber(value: -1), NSNumber(value: numHeads), NSNumber(value: headDim)]
        
        let q_r = graph.reshape(q, shape: targetShape, name: "\(name)/q_res")
        let k_r = graph.reshape(k, shape: targetShape, name: "\(name)/k_res")
        let v_r = graph.reshape(v, shape: targetShape, name: "\(name)/v_res")
        
        // Transpose to [B, N, S, H]
        let q_t = graph.transposeTensor(q_r, dimension: 1, withDimension: 2, name: "\(name)/q_t")
        let k_t = graph.transposeTensor(k_r, dimension: 1, withDimension: 2, name: "\(name)/k_t")
        let v_t = graph.transposeTensor(v_r, dimension: 1, withDimension: 2, name: "\(name)/v_t")
        
        // Attention: Scores = Q @ K.T / sqrt(d)
        let k_tt = graph.transposeTensor(k_t, dimension: 2, withDimension: 3, name: "\(name)/k_tt")
        var scores = graph.matrixMultiplication(primary: q_t, secondary: k_tt, name: "\(name)/scores")
        let scale = graph.constant(1.0 / sqrt(Double(headDim)), dataType: ioDataType)
        scores = graph.multiplication(scores, scale, name: "\(name)/scaled")
        
        let attn = graph.softMax(with: scores, axis: -1, name: "\(name)/softmax")
        
        // Context = Attn @ V
        let ctx = graph.matrixMultiplication(primary: attn, secondary: v_t, name: "\(name)/ctx")
        
        // Transpose back [B, S, N, H]
        let ctx_t = graph.transposeTensor(ctx, dimension: 1, withDimension: 2, name: "\(name)/ctx_t")
        
        // Flatten [B, S, InternalDim]
        let ctx_f = graph.reshape(ctx_t, shape: [NSNumber(value: 1), NSNumber(value: -1), NSNumber(value: internalDim)], name: "\(name)/flatten")
        
        // Out Projection
        let out = graph.addition(graph.matrixMultiplication(primary: ctx_f, secondary: w_o_t, name: "\(name)/out_mm"), b_o, name: "\(name)/out_add")
        
        return (out, phs)
    }
    
    func addFeeds(placeholders: [String: MPSGraphTensor], to feeds: inout [MPSGraphTensor: MPSGraphTensorData], name: String) {
        // Feeds use PyTorch shape [Out, In]
        if let ph = placeholders["\(name)/w_q"], let b = q_proj { feeds[ph] = MPSGraphTensorData(b, shape: [NSNumber(value: internalDim), NSNumber(value: embedDim)], dataType: ioDataType) }
        if let ph = placeholders["\(name)/w_k"], let b = k_proj { feeds[ph] = MPSGraphTensorData(b, shape: [NSNumber(value: internalDim), NSNumber(value: embedDim)], dataType: ioDataType) }
        if let ph = placeholders["\(name)/w_v"], let b = v_proj { feeds[ph] = MPSGraphTensorData(b, shape: [NSNumber(value: internalDim), NSNumber(value: embedDim)], dataType: ioDataType) }
        
        // Out Proj [Embed, Internal]
        if let ph = placeholders["\(name)/w_o"], let b = out_proj { feeds[ph] = MPSGraphTensorData(b, shape: [NSNumber(value: embedDim), NSNumber(value: internalDim)], dataType: ioDataType) }
        
        if let ph = placeholders["\(name)/b_q"], let b = q_bias { feeds[ph] = MPSGraphTensorData(b, shape: [1, NSNumber(value: internalDim)], dataType: ioDataType) }
        if let ph = placeholders["\(name)/b_k"], let b = k_bias { feeds[ph] = MPSGraphTensorData(b, shape: [1, NSNumber(value: internalDim)], dataType: ioDataType) }
        if let ph = placeholders["\(name)/b_v"], let b = v_bias { feeds[ph] = MPSGraphTensorData(b, shape: [1, NSNumber(value: internalDim)], dataType: ioDataType) }
        if let ph = placeholders["\(name)/b_o"], let b = out_bias { feeds[ph] = MPSGraphTensorData(b, shape: [1, NSNumber(value: embedDim)], dataType: ioDataType) }
    }
}

public class MLPLayer {
    let inputDim: Int
    let hiddenDim: Int
    let outputDim: Int
    
    var w1: MTLBuffer?
    var b1: MTLBuffer?
    var w2: MTLBuffer?
    var b2: MTLBuffer?
    
    // Stateless placeholders
    // var placeholders ... REMOVED
    
    let device: MTLDevice
    let enableHalfPrecision: Bool

    private var ioDataType: MPSDataType { enableHalfPrecision ? .float16 : .float32 }
    private var bytesPerElement: Int { enableHalfPrecision ? 2 : 4 }
    
    public init(device: MTLDevice, inputDim: Int, hiddenDim: Int, outputDim: Int, enableHalfPrecision: Bool = true) {
        self.device = device
        self.enableHalfPrecision = enableHalfPrecision
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.outputDim = outputDim
        
        // Weights must be loaded via loadWeights() - no random fallback
        w1 = nil
        b1 = nil
        w2 = nil
        b2 = nil
    }
    
    public func loadWeights(fc1W: MTLBuffer, fc1B: MTLBuffer, fc2W: MTLBuffer, fc2B: MTLBuffer) {
        self.w1 = fc1W
        self.b1 = fc1B
        self.w2 = fc2W
        self.b2 = fc2B
    }
    
    public func randomInitialize() {
        let bpe = bytesPerElement
        
        // W1: [Hidden, Input]
        let w1Size = hiddenDim * inputDim * bpe
        let b1Size = hiddenDim * bpe
        
        // W2: [Output, Hidden]
        let w2Size = outputDim * hiddenDim * bpe
        let b2Size = outputDim * bpe
        
        func alloc(_ len: Int) -> MTLBuffer {
            let b = device.makeBuffer(length: len, options: .storageModeShared)!
            memset(b.contents(), 0, len)
            return b
        }
        
        w1 = alloc(w1Size)
        b1 = alloc(b1Size)
        w2 = alloc(w2Size)
        b2 = alloc(b2Size)
    }
    
    public func buildGraph(input: MPSGraphTensor, graph: MPSGraph, name: String) -> (MPSGraphTensor, [String: MPSGraphTensor]) {
        // PyTorch Shapes [Out, In]
        let t_w1 = graph.placeholder(shape: [NSNumber(value: hiddenDim), NSNumber(value: inputDim)], dataType: ioDataType, name: "\(name)/w1")
        let t_b1 = graph.placeholder(shape: [1, NSNumber(value: hiddenDim)], dataType: ioDataType, name: "\(name)/b1")
        let t_w2 = graph.placeholder(shape: [NSNumber(value: outputDim), NSNumber(value: hiddenDim)], dataType: ioDataType, name: "\(name)/w2")
        let t_b2 = graph.placeholder(shape: [1, NSNumber(value: outputDim)], dataType: ioDataType, name: "\(name)/b2")
        
        var phs: [String: MPSGraphTensor] = [
            "\(name)/w1": t_w1, "\(name)/b1": t_b1,
            "\(name)/w2": t_w2, "\(name)/b2": t_b2
        ]
        
        // Transpose [Out, In] -> [In, Out]
        let w1_t = graph.transposeTensor(t_w1, dimension: 0, withDimension: 1, name: "\(name)/w1_t")
        let w2_t = graph.transposeTensor(t_w2, dimension: 0, withDimension: 1, name: "\(name)/w2_t")
        
        // Lin 1
        let x1 = graph.addition(graph.matrixMultiplication(primary: input, secondary: w1_t, name: "\(name)/mm1"), t_b1, name: "\(name)/add1")
        // ReLU
        let x2 = graph.reLU(with: x1, name: "\(name)/relu")
        // Lin 2
        let x3 = graph.addition(graph.matrixMultiplication(primary: x2, secondary: w2_t, name: "\(name)/mm2"), t_b2, name: "\(name)/add2")
        return (x3, phs)
    }
    
    public func addFeeds(placeholders: [String: MPSGraphTensor], to feeds: inout [MPSGraphTensor: MPSGraphTensorData], name: String) {
        // Feeds [Out, In]
        if let ph = placeholders["\(name)/w1"], let b = w1 { feeds[ph] = MPSGraphTensorData(b, shape: [NSNumber(value: hiddenDim), NSNumber(value: inputDim)], dataType: ioDataType) }
        if let ph = placeholders["\(name)/b1"], let b = b1 { feeds[ph] = MPSGraphTensorData(b, shape: [1, NSNumber(value: hiddenDim)], dataType: ioDataType) }
        if let ph = placeholders["\(name)/w2"], let b = w2 { feeds[ph] = MPSGraphTensorData(b, shape: [NSNumber(value: outputDim), NSNumber(value: hiddenDim)], dataType: ioDataType) }
        if let ph = placeholders["\(name)/b2"], let b = b2 { feeds[ph] = MPSGraphTensorData(b, shape: [1, NSNumber(value: outputDim)], dataType: ioDataType) }
    }
}

public class TwoWayLayerNorm {
    let normalizedShape: [NSNumber]
    var gamma: MTLBuffer?
    var beta: MTLBuffer?
    let enableHalfPrecision: Bool

    private var ioDataType: MPSDataType { enableHalfPrecision ? .float16 : .float32 }
    private var bytesPerElement: Int { enableHalfPrecision ? 2 : 4 }
    
    // Stateless placeholders
    // var placeholders ... REMOVED
    
    let device: MTLDevice
    
    public init(device: MTLDevice, normalizedShape: [NSNumber], enableHalfPrecision: Bool = true) {
        self.device = device
        self.normalizedShape = normalizedShape
        self.enableHalfPrecision = enableHalfPrecision
        let len = normalizedShape.map{$0.intValue}.reduce(1,*)
        
        // Weights must be loaded via loadWeights() - no random fallback
        gamma = nil
        beta = nil
    }
    
    public func loadWeights(gamma: MTLBuffer, beta: MTLBuffer) {
        self.gamma = gamma
        self.beta = beta
    }
    
    public func randomInitialize() {
        let len = normalizedShape.map{$0.intValue}.reduce(1,*)
        let bpe = bytesPerElement
        let size = len * bpe
        
        func alloc(_ len: Int) -> MTLBuffer {
            let b = device.makeBuffer(length: len, options: .storageModeShared)!
            memset(b.contents(), 0, len)
             // Gamma needs 1.0 ideally, but 0 is fine for bench
            return b
        }
        
        gamma = alloc(size)
        beta = alloc(size)
    }
    
    public func buildGraph(input: MPSGraphTensor, graph: MPSGraph, name: String) -> (MPSGraphTensor, [String: MPSGraphTensor]) {
        let t_gamma = graph.placeholder(shape: normalizedShape, dataType: ioDataType, name: "\(name)/gamma")
        let t_beta = graph.placeholder(shape: normalizedShape, dataType: ioDataType, name: "\(name)/beta")
        
        let phs: [String: MPSGraphTensor] = ["\(name)/gamma": t_gamma, "\(name)/beta": t_beta]
        
        // Mean/Var - reduce over last dim
        let axes: [NSNumber] = [NSNumber(value: -1)]
        let mean = graph.mean(of: input, axes: axes, name: "\(name)/mean")
        let variance = graph.variance(of: input, axes: axes, name: "\(name)/var")
        
        // Normalize: (x - mean) / sqrt(var + eps) * gamma + beta
        let sub = graph.subtraction(input, mean, name: "\(name)/sub")
        let epsilon = graph.constant(1e-6, dataType: ioDataType)
        let stdDev = graph.squareRoot(with: graph.addition(variance, epsilon, name: "\(name)/addEps"), name: "\(name)/std")
        let div = graph.division(sub, stdDev, name: "\(name)/div")
        
        let mul = graph.multiplication(div, t_gamma, name: "\(name)/mul")
        let res = graph.addition(mul, t_beta, name: "\(name)/add")
        return (res, phs)
    }
    
    public func addFeeds(placeholders: [String: MPSGraphTensor], to feeds: inout [MPSGraphTensor: MPSGraphTensorData], name: String) {
        if let ph = placeholders["\(name)/gamma"], let b = gamma { feeds[ph] = MPSGraphTensorData(b, shape: normalizedShape, dataType: ioDataType) }
        if let ph = placeholders["\(name)/beta"], let b = beta { feeds[ph] = MPSGraphTensorData(b, shape: normalizedShape, dataType: ioDataType) }
    }
}
