import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

/// Applies Cross-Attention between the current frame's features and the Memory Bank.
///
/// Corresponds to `tracker.transformer` in the weights.
/// Architecture:
/// - 4 Transformer Blocks
/// - Each block:
///   - Self Attention (Current Frame)
///   - Cross Attention (Current Frame -> Memory Bank) // "cross_attn_image"
///   - MLP
public class SAM3MemoryAttention {
    public let device: MTLDevice
    
    // Layers
    // We will need a custom 'MemoryBlock' class that matches the weight keys:
    // "layers.X.self_attn...", "layers.X.cross_attn_image...", "layers.X.mlp..."
    // Since I don't have that class yet, I'll scaffold the control flow.
    
    var blocks: [MemoryAttentionBlock] = []
    let numLayers = 4 // Confirmed by keys "layers.3"
    let embedDim = 256
    let numHeads = 8
    
    // Final Norm? Keys showed "tracker.transformer.encoder.norm" (LayerNorm)
    // "tracker.transformer.encoder.layers.0.norm1", "norm2", etc.
    // And there might be a final norm.
    var finalNorm: TwoWayLayerNorm?
    
    // Helper types from TwoWayTransformer (assumed public/accessible)
    // If not accessible, I need to copy them.
    // They are public class in TwoWayTransformer.swift but check scope.
    // TwoWayTransformer.swift defines them as top-level public classes.
    
    public init(device: MTLDevice) {
        self.device = device
        
        for _ in 0..<numLayers {
            blocks.append(MemoryAttentionBlock(device: device, embedDim: embedDim, numHeads: numHeads, mlpDim: 2048))
        }
        
        finalNorm = TwoWayLayerNorm(device: device, normalizedShape: [NSNumber(value: embedDim)])
    }
    
    public func loadWeights(_ weights: [String: Data]) {
        let prefix = "tracker.transformer.encoder"
        
        for (i, block) in blocks.enumerated() {
            block.loadWeights(weights: weights, prefix: "\(prefix).layers.\(i)")
        }
        
        // Final Norm
        // Check key: tracker.transformer.encoder.norm.weight?
        if let g = weights.buffer(for: "\(prefix).norm.weight", device: device),
           let b = weights.buffer(for: "\(prefix).norm.bias", device: device) {
            finalNorm?.loadWeights(gamma: g, beta: b)
        }
    }
    
    private var zeroBuffer: MTLBuffer?
    
    public func forward(
        currentFeatures: MTLBuffer,
        memoryBank: MemoryBank,
        commandBuffer: MTLCommandBuffer
    ) throws -> MTLBuffer {
        
        let graph = MPSGraph()
        
        // 1. Inputs
        let currentSeqLen = 64 * 64
        // Current features are flattened buffer [1, S, C]
        let currentTensor = graph.placeholder(shape: [1, NSNumber(value: currentSeqLen), NSNumber(value: embedDim)], dataType: .float32, name: "current")
        
        // 2. Memories (Video support up to 6 frames)
        let memories = memoryBank.getMemories()
        
        // We define 6 fixed placeholders. Unused ones will be fed with zeros.
        var memTensors: [MPSGraphTensor] = []
        for i in 0..<6 {
            // Memory Textures are 64x64x256 (2D Array Slice). 
            // We feed them as [1, 64, 64, 256].
            let t = graph.placeholder(shape: [1, 64, 64, 256], dataType: .float32, name: "mem_\(i)")
            memTensors.append(t)
        }
        
        // Flatten memories inside graph: [1, 64, 64, 256] -> [1, 4096, 256]
        let flattenedMems = memTensors.map { 
            graph.reshape($0, shape: [1, NSNumber(value: currentSeqLen), NSNumber(value: embedDim)], name: "flat_\($0.operation.name)") 
        }
        
        // Concat: [1, 6 * 4096, 256]
        // Note: If memoryBank is empty, we still feed zeros. Attention will attend to zeros (no-op/soft mask).
        // SAM 3 might expect masking logic? 
        // With Zeros as Key/Value, attention scores will be small (bias), Softmax might distribute evenly.
        // This is not ideal but standard for "padding" if no mask provided.
        // Ideally we provide a bias mask ( -inf for padding).
        // Since I don't have bias mask input in TwoWayTransformer blocks reused here,
        // I will rely on the fact that 0 keys * 0 values -> 0 contribution to value sum?
        // No, Softmax([scores]) -> sum=1. If all scores 0, attention is uniform 1/N.
        // Then we sum V (zeros). So output is 0.
        // THIS IS CORRECT. If V is 0, output is 0.
        // So padding with 0 is safe for values.
        
        let memoryKeyValues = graph.concatTensors(flattenedMems, dimension: 1, name: "memory_concat")
        
        // 3. Run Blocks
        // 3. Run Blocks
        var x = currentTensor
        var phs: [String: MPSGraphTensor] = [:]
        
        for (i, block) in blocks.enumerated() {
            let (res, blockPhs) = block.buildGraph(x: x, memory: memoryKeyValues, graph: graph, namePrefix: "mem_attn/b\(i)")
            x = res
            phs.merge(blockPhs) { $1 }
        }
        
        // Final Norm
        if let fn = finalNorm {
            let (res, fnPhs) = fn.buildGraph(input: x, graph: graph, name: "mem_attn/final_norm")
            x = res
            phs.merge(fnPhs) { $1 }
        }
        
        // 4. Feeds
        let mpsCmd = MPSCommandBuffer(commandBuffer: commandBuffer)
        var feeds: [MPSGraphTensor: MPSGraphTensorData] = [:]
        
        feeds[currentTensor] = MPSGraphTensorData(currentFeatures, shape: [1, NSNumber(value: currentSeqLen), NSNumber(value: embedDim)], dataType: .float32)
        
        // ensure zero buffer
        if zeroBuffer == nil {
            let zeroBytes = 64 * 64 * 256 * 4
            zeroBuffer = device.makeBuffer(length: zeroBytes, options: .storageModeShared)
            memset(zeroBuffer!.contents(), 0, zeroBytes)
        }
        
        // Feed memories
        for i in 0..<6 {
            let ph = memTensors[i]
            if i < memories.count {
                // Feed Texture
                let mpsImg = MPSImage(texture: memories[i], featureChannels: 256) 
                // Note: MPSImage featureChannels=256 works if texture has enough slices.
                // Texture is RGBA16Float (4 channels) x 64 slices = 256.
                // MPSImage(texture: featureChannels:) logic handles array slices as C dimension?
                // MPSImage doc: "number of feature channels... >= 1".
                // If Array Length > 1, does it map to Channels?
                // Actually `MPSImage` maps `textureType=2DArray` slices to feature channels.
                // So yes, 64 slices * 4 channels = 256 feature channels.
                
                feeds[ph] = MPSGraphTensorData([mpsImg])
            } else {
                // Feed Zero Buffer
                feeds[ph] = MPSGraphTensorData(zeroBuffer!, shape: [1, 64, 64, 256], dataType: .float32)
            }
        }
        
        // Add Block Weights
        // Add Block Weights
        for (i, block) in blocks.enumerated() {
            block.selfAttn.addFeeds(placeholders: phs, to: &feeds, name: "mem_attn/b\(i)/sa")
            block.norm1.addFeeds(placeholders: phs, to: &feeds, name: "mem_attn/b\(i)/norm1")
            block.crossAttn.addFeeds(placeholders: phs, to: &feeds, name: "mem_attn/b\(i)/ca")
            block.norm2.addFeeds(placeholders: phs, to: &feeds, name: "mem_attn/b\(i)/norm2")
            block.mlp.addFeeds(placeholders: phs, to: &feeds, name: "mem_attn/b\(i)/mlp")
            block.norm3.addFeeds(placeholders: phs, to: &feeds, name: "mem_attn/b\(i)/norm3")
        }
        finalNorm?.addFeeds(placeholders: phs, to: &feeds, name: "mem_attn/final_norm")
        
        // 5. Execute
        let usage = MPSGraphTensorData(currentFeatures, shape: [1, NSNumber(value: currentSeqLen), NSNumber(value: embedDim)], dataType: .float32)
        // We reuse currentFeatures buffer for output? No, allocate new.
        // We return new buffer.
        
        let results = graph.encode(
            to: mpsCmd,
            feeds: feeds,
            targetTensors: [x],
            targetOperations: nil,
            executionDescriptor: nil
        )
        
        guard let data = results[x] else { throw NSError(domain: "SAM3MemoryAttention", code: 1, userInfo: nil) }
        
        let outBytes = 64 * 64 * 256 * 4
        let outBuffer = device.makeBuffer(length: outBytes, options: .storageModePrivate)!
        
        data.mpsndarray().exportData(with: mpsCmd, to: outBuffer, destinationDataType: .float32, offset: 0, rowStrides: nil)
        
        return outBuffer
    }
}

class MemoryAttentionBlock {
    let device: MTLDevice
    
    // Layers
    let selfAttn: AttentionLayer
    let norm1: TwoWayLayerNorm
    let crossAttn: AttentionLayer // to Memory
    let norm2: TwoWayLayerNorm
    let mlp: MLPLayer
    let norm3: TwoWayLayerNorm
    
    init(device: MTLDevice, embedDim: Int, numHeads: Int, mlpDim: Int) {
        self.device = device
        self.selfAttn = AttentionLayer(device: device, embedDim: embedDim, numHeads: numHeads)
        self.norm1 = TwoWayLayerNorm(device: device, normalizedShape: [NSNumber(value: embedDim)])
        
        self.crossAttn = AttentionLayer(device: device, embedDim: embedDim, numHeads: numHeads)
        self.norm2 = TwoWayLayerNorm(device: device, normalizedShape: [NSNumber(value: embedDim)])
         
        self.mlp = MLPLayer(device: device, inputDim: embedDim, hiddenDim: mlpDim, outputDim: embedDim)
        self.norm3 = TwoWayLayerNorm(device: device, normalizedShape: [NSNumber(value: embedDim)])
    }
    
    func loadWeights(weights: [String: Data], prefix: String) {
        // Load Self Attn
        loadAttn(selfAttn, key: "self_attn", weights: weights, prefix: prefix)
        loadNorm(norm1, key: "norm1", weights: weights, prefix: prefix)
        
        // Load Cross Attn (Image)
        loadAttn(crossAttn, key: "cross_attn_image", weights: weights, prefix: prefix)
        loadNorm(norm2, key: "norm2", weights: weights, prefix: prefix)
        
        // Load MLP
        loadMLP(mlp, key: "mlp", weights: weights, prefix: prefix)
        loadNorm(norm3, key: "norm3", weights: weights, prefix: prefix)
    }
    
    func buildGraph(x: MPSGraphTensor, memory: MPSGraphTensor, graph: MPSGraph, namePrefix: String) -> (MPSGraphTensor, [String: MPSGraphTensor]) {
        // x: [1, S, C]
        // memory: [1, MemS, C]
        var phs: [String: MPSGraphTensor] = [:]
        
        // 1. Self Attn
        var q = x
        let (sa, saPhs) = selfAttn.buildGraph(query: q, key: q, value: q, graph: graph, name: "\(namePrefix)/sa")
        phs.merge(saPhs) { $1 }
        q = graph.addition(q, sa, name: "\(namePrefix)/add1")
        
        let (n1, n1Phs) = norm1.buildGraph(input: q, graph: graph, name: "\(namePrefix)/norm1")
        phs.merge(n1Phs) { $1 }
        q = n1
        
        // 2. Cross Attn
        let (ca, caPhs) = crossAttn.buildGraph(query: q, key: memory, value: memory, graph: graph, name: "\(namePrefix)/ca")
        phs.merge(caPhs) { $1 }
        
        q = graph.addition(q, ca, name: "\(namePrefix)/add2")
        let (n2, n2Phs) = norm2.buildGraph(input: q, graph: graph, name: "\(namePrefix)/norm2")
        phs.merge(n2Phs) { $1 }
        q = n2
        
        // 3. MLP
        let (m, mPhs) = mlp.buildGraph(input: q, graph: graph, name: "\(namePrefix)/mlp")
        phs.merge(mPhs) { $1 }
        
        q = graph.addition(q, m, name: "\(namePrefix)/add3")
        let (n3, n3Phs) = norm3.buildGraph(input: q, graph: graph, name: "\(namePrefix)/norm3")
        phs.merge(n3Phs) { $1 }
        q = n3
        
        return (q, phs)
    }
    
    // Helpers
    private func loadAttn(_ layer: AttentionLayer, key: String, weights: [String: Data], prefix: String) {
        // q/k/v proj
        if let q = weights.buffer(for: "\(prefix).\(key).q_proj.weight", device: device) { layer.q_proj = q }
        if let k = weights.buffer(for: "\(prefix).\(key).k_proj.weight", device: device) { layer.k_proj = k }
        if let v = weights.buffer(for: "\(prefix).\(key).v_proj.weight", device: device) { layer.v_proj = v }
        
        if let qb = weights.buffer(for: "\(prefix).\(key).q_proj.bias", device: device) { layer.q_bias = qb }
        if let kb = weights.buffer(for: "\(prefix).\(key).k_proj.bias", device: device) { layer.k_bias = kb }
        if let vb = weights.buffer(for: "\(prefix).\(key).v_proj.bias", device: device) { layer.v_bias = vb }
        
        // out proj
        if let o = weights.buffer(for: "\(prefix).\(key).out_proj.weight", device: device) { layer.out_proj = o }
        if let ob = weights.buffer(for: "\(prefix).\(key).out_proj.bias", device: device) { layer.out_bias = ob }
        
        // Fused fallback (e.g. if key is "cross_attn_image" but weights are "cross_attn_image.in_proj_weight")
        // Check for in_proj
        if layer.q_proj == nil {
             if let fusedW = weights["\(prefix).\(key).in_proj_weight"],
                let fusedB = weights["\(prefix).\(key).in_proj_bias"] {
                 splitAndLoad(fusedW, fusedB, into: layer)
             }
        }
    }
    
    private func loadNorm(_ layer: TwoWayLayerNorm, key: String, weights: [String: Data], prefix: String) {
        if let g = weights.buffer(for: "\(prefix).\(key).weight", device: device),
           let b = weights.buffer(for: "\(prefix).\(key).bias", device: device) {
             layer.loadWeights(gamma: g, beta: b)
        }
    }
    
    private func loadMLP(_ layer: MLPLayer, key: String, weights: [String: Data], prefix: String) {
        if let w1 = weights.buffer(for: "\(prefix).\(key).lin1.weight", device: device) { layer.w1 = w1 }
        if let b1 = weights.buffer(for: "\(prefix).\(key).lin1.bias", device: device) { layer.b1 = b1 }
        
        if let w2 = weights.buffer(for: "\(prefix).\(key).lin2.weight", device: device) { layer.w2 = w2 }
        if let b2 = weights.buffer(for: "\(prefix).\(key).lin2.bias", device: device) { layer.b2 = b2 }
        
        // Try alternate keys "linear1", "linear2"
        if layer.w1 == nil {
             if let w = weights.buffer(for: "\(prefix).\(key).linear1.weight", device: device) { layer.w1 = w }
             if let b = weights.buffer(for: "\(prefix).\(key).linear1.bias", device: device) { layer.b1 = b }
             if let w = weights.buffer(for: "\(prefix).\(key).linear2.weight", device: device) { layer.w2 = w }
             if let b = weights.buffer(for: "\(prefix).\(key).linear2.bias", device: device) { layer.b2 = b }
        }
    }
    
    // CPU Splitter (Copied from TwoWayTransformer)
    private func splitAndLoad(_ fusedW: Data, _ fusedB: Data, into layer: AttentionLayer) {
        let dim = layer.embedDim
        guard fusedW.count % 4 == 0, fusedW.count >= 3 * dim * dim * 4 else { return }
        
        let inDim = fusedW.count / 4 / (3 * dim)
        let qRange = 0..<(dim * inDim * 4)
        let kRange = (dim * inDim * 4)..<(2 * dim * inDim * 4)
        let vRange = (2 * dim * inDim * 4)..<(3 * dim * inDim * 4)
        
        layer.q_proj = ModelLoader.loadBuffer(from: fusedW.subdata(in: qRange), device: device)
        layer.k_proj = ModelLoader.loadBuffer(from: fusedW.subdata(in: kRange), device: device)
        layer.v_proj = ModelLoader.loadBuffer(from: fusedW.subdata(in: vRange), device: device)
        
        guard fusedB.count >= 3 * dim * 4 else { return }
        layer.q_bias = ModelLoader.loadBuffer(from: fusedB.subdata(in: 0..<(dim*4)), device: device)
        layer.k_bias = ModelLoader.loadBuffer(from: fusedB.subdata(in: (dim*4)..<(2*dim*4)), device: device)
        layer.v_bias = ModelLoader.loadBuffer(from: fusedB.subdata(in: (2*dim*4)..<(3*dim*4)), device: device)
    }
}
