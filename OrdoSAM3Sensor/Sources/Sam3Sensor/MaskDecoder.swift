import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

public class MaskDecoder {
    let device: MTLDevice
    let transformer: TwoWayTransformer
    let embedDim: Int
    let numMultimaskOutputs: Int
    let enableHalfPrecision: Bool

    private var ioDataType: MPSDataType { enableHalfPrecision ? .float16 : .float32 }
    private var bytesPerElement: Int { enableHalfPrecision ? 2 : 4 }
    
    // Learnable Tokens
    var iouToken: MTLBuffer?
    var maskTokens: MTLBuffer?
    
    // Upscaling (ConvTranspose2d + LN + GELU)
    var up1Weights: MTLBuffer?
    var up1Bias: MTLBuffer?
    var up1LN: TwoWayLayerNorm
    
    var up2Weights: MTLBuffer?
    var up2Bias: MTLBuffer?
    // up2 has no LN
    
    // Hypernetworks (3-layer MLPs)
    var outputHypernetworksMLPs: [ThreeLayerMLP] = []
    
    // IoU Head (3-layer MLP)
    var iouPredictionHead: ThreeLayerMLP
    
    // High Res Projection Weights
    var convS0Weights: MTLBuffer?
    var convS0Bias: MTLBuffer?
    
    var convS1Weights: MTLBuffer?
    var convS1Bias: MTLBuffer?
    
    public init(device: MTLDevice, embedDim: Int = 256, numMultimaskOutputs: Int = 3, enableHalfPrecision: Bool = true) {
        self.device = device
        self.embedDim = embedDim
        self.numMultimaskOutputs = numMultimaskOutputs
        self.enableHalfPrecision = enableHalfPrecision
        
        self.transformer = TwoWayTransformer(device: device, embedDim: embedDim, enableHalfPrecision: enableHalfPrecision)
        
        // Tokens and weights must be loaded via loadWeights() - no random fallback
        self.iouToken = nil
        self.maskTokens = nil
        self.up1Weights = nil
        self.up1Bias = nil
        self.up1LN = TwoWayLayerNorm(device: device, normalizedShape: [NSNumber(value: 64)], enableHalfPrecision: enableHalfPrecision)
        self.up2Weights = nil
        self.up2Bias = nil
        
        // Init MLPs (3-layers for SAM Output)
        for _ in 0..<4 {
            outputHypernetworksMLPs.append(ThreeLayerMLP(device: device, inputDim: embedDim, hiddenDim: 256, outputDim: 32, enableHalfPrecision: enableHalfPrecision))
        }
        
        // IoU Head: 256 -> 256 -> 256 -> 4 (one score per mask)
        iouPredictionHead = ThreeLayerMLP(device: device, inputDim: embedDim, hiddenDim: 256, outputDim: 4, enableHalfPrecision: enableHalfPrecision)
    }
    
    public func loadWeights(_ weights: [String: Data]) {
        // Tokens (remapped: tracker.sam_mask_decoder.* -> sam_mask_decoder.*)
        if let iou = weights.buffer(for: "sam_mask_decoder.iou_token.weight", device: device) { 
            iouToken = iou 
            print("DEBUG: Loaded iou_token.weight size=\(iou.length)")
        } else {
            print("WARNING: iou_token.weight NOT FOUND")
        }
        if let mask = weights.buffer(for: "sam_mask_decoder.mask_tokens.weight", device: device) { 
            maskTokens = mask 
            print("DEBUG: Loaded mask_tokens.weight size=\(mask.length)")
        } else {
            print("WARNING: mask_tokens.weight NOT FOUND")
        }
        
        // Output Upscaling (remapped: tracker.sam_mask_decoder.* -> sam_mask_decoder.*)
        if let w = weights.buffer(for: "sam_mask_decoder.output_upscaling.0.weight", device: device) { up1Weights = w }
        if let b = weights.buffer(for: "sam_mask_decoder.output_upscaling.0.bias", device: device) { up1Bias = b }
        if let g = weights.buffer(for: "sam_mask_decoder.output_upscaling.1.weight", device: device),
           let bv = weights.buffer(for: "sam_mask_decoder.output_upscaling.1.bias", device: device) {
             up1LN.loadWeights(gamma: g, beta: bv)
        }
        
        if let w = weights.buffer(for: "sam_mask_decoder.output_upscaling.3.weight", device: device) { up2Weights = w }
        if let b = weights.buffer(for: "sam_mask_decoder.output_upscaling.3.bias", device: device) { up2Bias = b }
        
        // Hypernetworks
        for (i, mlp) in outputHypernetworksMLPs.enumerated() {
             let prefix = "sam_mask_decoder.output_hypernetworks_mlps.\(i)"
             mlp.loadWeights(weights: weights, prefix: prefix)
        }
        
        // IoU Head
        iouPredictionHead.loadWeights(weights: weights, prefix: "sam_mask_decoder.iou_prediction_head")

        // Transformer (remapped: tracker.sam_mask_decoder.transformer -> sam_mask_decoder.transformer)
        transformer.loadWeights(weights: weights, prefix: "sam_mask_decoder.transformer")
        
        // Final Norm (Handled by transformer.loadWeights? No, implementation plan kept finalNorm in transformer but said MaskDecoder calls it.
        // Wait, TwoWayTransformer.loadWeights added in previous step handles blocks and Final Attn, but does it handle Final Norm?
        // Checking TwoWayTransformer.loadWeights implementation... I didn't see explicit Final Norm loading in the snippet I added.
        // I added "Final Attn" loading.
        // Let's rely on MaskDecoder to load Final Norm for now OR update TwoWayTransformer to do it?
        // The previous step added loadWeights to TWT which handles "finalAttnTokenToImage".
        // It does NOT handle "norm_final_attn".
        // Final Norm (remapped: tracker.sam_mask_decoder.transformer.norm_final_attn -> sam_mask_decoder.transformer.norm_final_attn)
        if let g = weights.buffer(for: "sam_mask_decoder.transformer.norm_final_attn.weight", device: device), 
           let b = weights.buffer(for: "sam_mask_decoder.transformer.norm_final_attn.bias", device: device) { 
             transformer.finalNorm?.loadWeights(gamma: g, beta: b)
        }
        
        // High Res Projections
        if let w = weights.buffer(for: "sam_mask_decoder.conv_s0.weight", device: device) { convS0Weights = w }
        if let b = weights.buffer(for: "sam_mask_decoder.conv_s0.bias", device: device) { convS0Bias = b }
        
        if let w = weights.buffer(for: "sam_mask_decoder.conv_s1.weight", device: device) { convS1Weights = w }
        if let b = weights.buffer(for: "sam_mask_decoder.conv_s1.bias", device: device) { convS1Bias = b }
    }
    
    public func testOnly_randomize() {
        print("⚠️ Randomizing MaskDecoder for Benchmark")
        
        // 1. Tokens (IoU, Mask)
        func alloc(_ count: Int) -> MTLBuffer {
             let len = count * (enableHalfPrecision ? 2 : 4)
             let b = device.makeBuffer(length: len, options: .storageModeShared)!
             memset(b.contents(), 0, len)
             return b
        }
        
        iouToken = alloc(embedDim)
        maskTokens = alloc(4 * embedDim)
        
        // 2. Transformer
        transformer.randomInitialize()
        
        // 3. Heads
        iouPredictionHead.randomInitialize()
        for mlp in outputHypernetworksMLPs {
            mlp.randomInitialize()
        }
        
        // 4. Upscaling
        convS0Weights = alloc(32 * 256 * 1 * 1); convS0Bias = alloc(32)
        convS1Weights = alloc(64 * 256 * 1 * 1); convS1Bias = alloc(64)
        
        // Up1: [256, 64, 2, 2] 
        up1Weights = alloc(256 * 64 * 2 * 2); up1Bias = alloc(64)
        up1LN.randomInitialize()
        
        // Up2: [64, 32, 2, 2]
        up2Weights = alloc(64 * 32 * 2 * 2); up2Bias = alloc(32)
    }
    
    // Sprint 05b: Refactored async methods - modular approach
    
    // Helper: Build token placeholders and concatenate
    private func buildTokenPlaceholders(
        graph: MPSGraph,
        pointCount: Int
    ) -> (iouToken: MPSGraphTensor, maskTokens: MPSGraphTensor, points: MPSGraphTensor, tokens: MPSGraphTensor) {
        let iouT = graph.placeholder(shape: [1, 1, NSNumber(value: embedDim)], dataType: ioDataType, name: "iou_token")
        let maskT = graph.placeholder(shape: [1, 4, NSNumber(value: embedDim)], dataType: ioDataType, name: "mask_tokens")
        let outputTokens = graph.concatTensors([iouT, maskT], dimension: 1, name: "output_tokens")
        let pT = graph.placeholder(shape: [1, NSNumber(value: pointCount), NSNumber(value: embedDim)], dataType: ioDataType, name: "points")
        let tokens = graph.concatTensors([outputTokens, pT], dimension: 1, name: "tokens")
        return (iouT, maskT, pT, tokens)
    }
    
    // Helper: Build upscaling layers
    private func buildUpscalingLayers(
        graph: MPSGraph,
        input: MPSGraphTensor,
        s0: MPSGraphTensor?,
        s1: MPSGraphTensor?,
        hasS0: Bool,
        hasS1: Bool
    ) -> (output: MPSGraphTensor, placeholders: [String: MPSGraphTensor]) {
        var phs: [String: MPSGraphTensor] = [:]
        
        let imgReshaped = graph.reshape(input, shape: [1, 64, 64, NSNumber(value: embedDim)], name: "img_reshaped")
        
        // Layer 1: 256->64
        let up1W = graph.placeholder(shape: [256, 64, 2, 2], dataType: ioDataType, name: "up1/w")
        let desc1 = MPSGraphConvolution2DOpDescriptor(strideInX: 2, strideInY: 2, dilationRateInX: 1, dilationRateInY: 1, groups: 1, paddingLeft: 0, paddingRight: 0, paddingTop: 0, paddingBottom: 0, paddingStyle: .TF_SAME, dataLayout: .NHWC, weightsLayout: .OIHW)!
        let up1 = graph.convolutionTranspose2D(imgReshaped, weights: up1W, outputShape: [1, 128, 128, 64], descriptor: desc1, name: "up1/conv")
        let up1b = graph.placeholder(shape: [1, 1, 1, 64], dataType: ioDataType, name: "up1/b")
        let up1BiasAdded = graph.addition(up1, up1b, name: "up1/bias_add")
        
        // S1 Integration
        var up1Fused = up1BiasAdded
        if hasS1 {
            let s1W = graph.placeholder(shape: [64, 256, 1, 1], dataType: ioDataType, name: "s1/proj/w")
            let s1B = graph.placeholder(shape: [1, 1, 1, 64], dataType: ioDataType, name: "s1/proj/b")
            let desc1x1 = MPSGraphConvolution2DOpDescriptor(strideInX: 1, strideInY: 1, dilationRateInX: 1, dilationRateInY: 1, groups: 1, paddingLeft: 0, paddingRight: 0, paddingTop: 0, paddingBottom: 0, paddingStyle: .explicit, dataLayout: .NHWC, weightsLayout: .OIHW)!
            let s1Proj = graph.convolution2D(s1!, weights: s1W, descriptor: desc1x1, name: "s1/proj")
            let s1ProjBias = graph.addition(s1Proj, s1B, name: "s1/proj/add")
            up1Fused = graph.addition(up1BiasAdded, s1ProjBias, name: "up1/add_s1")
            phs["s1/proj/w"] = s1W
            phs["s1/proj/b"] = s1B
        }
        
        let (up1n, up1nPhs) = up1LN.buildGraph(input: up1Fused, graph: graph, name: "up1/ln")
        phs.merge(up1nPhs) { $1 }
        let up1a = applyGELU(graph: graph, input: up1n, name: "up1/gelu")
        
        // Layer 2: 64->32
        let up2W = graph.placeholder(shape: [64, 32, 2, 2], dataType: ioDataType, name: "up2/w")
        let desc2 = MPSGraphConvolution2DOpDescriptor(strideInX: 2, strideInY: 2, dilationRateInX: 1, dilationRateInY: 1, groups: 1, paddingLeft: 0, paddingRight: 0, paddingTop: 0, paddingBottom: 0, paddingStyle: .TF_SAME, dataLayout: .NHWC, weightsLayout: .OIHW)!
        let up2 = graph.convolutionTranspose2D(up1a, weights: up2W, outputShape: [1, 256, 256, 32], descriptor: desc2, name: "up2/conv")
        let up2b = graph.placeholder(shape: [1, 1, 1, 32], dataType: ioDataType, name: "up2/b")
        let up2BiasAdded = graph.addition(up2, up2b, name: "up2/bias_add")
        
        // S0 Integration
        var up2Fused = up2BiasAdded
        if hasS0 {
            let s0W = graph.placeholder(shape: [32, 256, 1, 1], dataType: ioDataType, name: "s0/proj/w")
            let s0B = graph.placeholder(shape: [1, 1, 1, 32], dataType: ioDataType, name: "s0/proj/b")
            let desc1x1 = MPSGraphConvolution2DOpDescriptor(strideInX: 1, strideInY: 1, dilationRateInX: 1, dilationRateInY: 1, groups: 1, paddingLeft: 0, paddingRight: 0, paddingTop: 0, paddingBottom: 0, paddingStyle: .explicit, dataLayout: .NHWC, weightsLayout: .OIHW)!
            let s0Proj = graph.convolution2D(s0!, weights: s0W, descriptor: desc1x1, name: "s0/proj")
            let s0ProjBias = graph.addition(s0Proj, s0B, name: "s0/proj/add")
            up2Fused = graph.addition(up2BiasAdded, s0ProjBias, name: "up2/add_s0")
            phs["s0/proj/w"] = s0W
            phs["s0/proj/b"] = s0B
        }
        
        let up2a = applyGELU(graph: graph, input: up2Fused, name: "up2/gelu")
        
        phs["up1/w"] = up1W
        phs["up1/b"] = up1b
        phs["up2/w"] = up2W
        phs["up2/b"] = up2b
        
        return (up2a, phs)
    }
    
    // Helper: Apply GELU activation
    private func applyGELU(graph: MPSGraph, input: MPSGraphTensor, name: String) -> MPSGraphTensor {
        let pointFive = graph.constant(0.5, dataType: ioDataType)
        let one = graph.constant(1.0, dataType: ioDataType)
        let sqrtTwo = graph.constant(1.41421356, dataType: ioDataType)
        let div = graph.division(input, sqrtTwo, name: "\(name)_div")
        let erf = graph.erf(with: div, name: "\(name)_erf")
        let onePlusErf = graph.addition(one, erf, name: "\(name)_add")
        let halfX = graph.multiplication(pointFive, input, name: "\(name)_half")
        return graph.multiplication(halfX, onePlusErf, name: name)
    }
    
    // Main buildGraph method - now clean and modular
    public func buildGraph(
        graph: MPSGraph,
        pointCount: Int,
        hasDensePrompt: Bool,
        hasS0: Bool,
        hasS1: Bool
    ) -> (placeholders: [String: MPSGraphTensor], outputs: (masks: MPSGraphTensor, iouPred: MPSGraphTensor)) {
        
        // 1. Build tokens
        let (iouT, maskT, pT, tokens) = buildTokenPlaceholders(graph: graph, pointCount: pointCount)
        
        // 2. Image embeddings
        // [1, 4096, 256] flat inputs for M3
        let flatImgCount = 4096 * embedDim
        let imgTFlat = graph.placeholder(shape: [NSNumber(value: flatImgCount)], dataType: ioDataType, name: "image_flat")
        let imgPeTFlat = graph.placeholder(shape: [NSNumber(value: flatImgCount)], dataType: ioDataType, name: "image_pe_flat")
        
        let imgT = graph.reshape(imgTFlat, shape: [NSNumber(value: 1), NSNumber(value: 4096), NSNumber(value: embedDim)], name: "image_reshape")
        let imgPeT = graph.reshape(imgPeTFlat, shape: [NSNumber(value: 1), NSNumber(value: 4096), NSNumber(value: embedDim)], name: "image_pe_reshape")
        
        var src = imgT
        var denseT: MPSGraphTensor? = nil
        if hasDensePrompt {
            denseT = graph.placeholder(shape: [1, -1, NSNumber(value: embedDim)], dataType: ioDataType, name: "dense_prompt")
            src = graph.addition(imgT, denseT!, name: "image_plus_dense")
        }
        
        // 3. High-res features
        // Only layout placeholders if integration is enabled
        var s0T: MPSGraphTensor? = nil
        if hasS0 { s0T = graph.placeholder(shape: [1, 256, 256, 256], dataType: ioDataType, name: "s0_raw") }
        
        var s1T: MPSGraphTensor? = nil
        if hasS1 { s1T = graph.placeholder(shape: [1, 128, 128, 256], dataType: ioDataType, name: "s1_raw") }
        
        // 4. Transformer
        let (finalPoint, currentImage, transPhs) = transformer.buildGraph(graph: graph, imageEmbeddings: src, imagePE: imgPeT, pointEmbeddings: tokens)
        
        // 5. Upscaling
        let (upscaled, upscalePhs) = buildUpscalingLayers(graph: graph, input: currentImage, s0: s0T, s1: s1T, hasS0: hasS0, hasS1: hasS1)
        
        // 6. Hypernetworks
        let outTokens = graph.sliceTensor(finalPoint, dimension: 1, start: 0, length: 5, name: "slice_tokens")
        var mlpPhs: [String: MPSGraphTensor] = [:]
        
        let iouTokenSlice = graph.sliceTensor(outTokens, dimension: 1, start: 0, length: 1, name: "slice_iou_token")
        let iouFlat = graph.reshape(iouTokenSlice, shape: [1, NSNumber(value: embedDim)], name: "iou_flat")
        let (iouScore, iouPhs) = iouPredictionHead.buildGraph(input: iouFlat, graph: graph, name: "iou_head")
        mlpPhs.merge(iouPhs) { $1 }
        
        var hyperWeightsList: [MPSGraphTensor] = []
        for i in 0..<4 {
            let mt = graph.sliceTensor(outTokens, dimension: 1, start: 1 + i, length: 1, name: "slice_mask_\(i)")
            let mtFlat = graph.reshape(mt, shape: [1, NSNumber(value: embedDim)], name: "mask_flat_\(i)")
            let (w, phs) = outputHypernetworksMLPs[i].buildGraph(input: mtFlat, graph: graph, name: "hyper_mlp_\(i)")
            hyperWeightsList.append(w)
            mlpPhs.merge(phs) { $1 }
        }
        let hyperWeights = graph.concatTensors(hyperWeightsList, dimension: 0, name: "hyper_weights")
        
        // 7. Masks
        let imgFlat = graph.reshape(upscaled, shape: [NSNumber(value: 256*256), 32], name: "img_flat")
        let hwT = graph.transposeTensor(hyperWeights, dimension: 0, withDimension: 1, name: "hw_T")
        let logits = graph.matrixMultiplication(primary: imgFlat, secondary: hwT, name: "mask_logits")
        let logitsT = graph.transposeTensor(logits, dimension: 0, withDimension: 1, name: "logits_T")
        let masks = graph.reshape(logitsT, shape: [1, 4, 256, 256], name: "masks_out")
        
        // Collect placeholders
        var placeholders: [String: MPSGraphTensor] = [
            "iou_token": iouT, "mask_tokens": maskT, "points": pT,
            "image_flat": imgTFlat, "image_pe_flat": imgPeTFlat
        ]
        if let s0 = s0T { placeholders["s0_raw"] = s0 }
        if let s1 = s1T { placeholders["s1_raw"] = s1 }
        if let dt = denseT { placeholders["dense_prompt"] = dt }
        if let dt = denseT { placeholders["dense_prompt"] = dt }
        placeholders.merge(transPhs) { $1 }
        placeholders.merge(upscalePhs) { $1 }
        placeholders.merge(mlpPhs) { $1 }
        
        return (placeholders: placeholders, outputs: (masks: masks, iouPred: iouScore))
    }

    public func forward(
        imageEmbeddings: MTLBuffer, // [1, 4096, 256]
        imagePE: MTLBuffer,         // [1, 4096, 256] (Dense PE)
        pointEmbeddings: MTLBuffer,  // [1, N, 256] (Direct + PE + Labels)
        densePromptEmbeddings: MTLBuffer? = nil, // [1, 1, 256] (No Mask) or [1, 4096, 256] (Mask)
        highResS0: MTLBuffer? = nil, // [1, 32, 256, 256] (Projected? No, Raw? 256->32)
        highResS1: MTLBuffer? = nil,  // [1, 64, 128, 128] (Projected? No, Raw? 256->64)
        commandQueue: MTLCommandQueue // B7: Reuse queue
    ) throws -> (masks: MTLBuffer, iouPred: MTLBuffer) {
        
        // 1. Determine Cache Key
        let pointCount = pointEmbeddings.length / bytesPerElement / embedDim
        let hasDense = densePromptEmbeddings != nil
        let s0Available = (highResS0 != nil && convS0Weights != nil)
        let s1Available = (highResS1 != nil && convS1Weights != nil)
        let precision = enableHalfPrecision ? "F16" : "F32"
        let cacheKey = "MaskDec_P\(pointCount)_\(hasDense ? "Dense" : "Sparse")_S0\(s0Available ? 1:0)_S1\(s1Available ? 1:0)_\(precision)"
        
        let queue = commandQueue
        
        let (graph, placeholders, outputs, executable) = CompiledGraphCache.shared.getOrCompileMulti(key: cacheKey, device: device) {
             let graph = MPSGraph()
             let (phs, outs) = self.buildGraph(
                 graph: graph,
                 pointCount: pointCount,
                 hasDensePrompt: hasDense,
                 hasS0: s0Available,
                 hasS1: s1Available
             )
             return (graph, phs, [outs.masks, outs.iouPred])
        }
        
        // 2. Prepare Feeds
        var feeds: [MPSGraphTensor: MPSGraphTensorData] = [:]
        
        // Tokens
        if let t = placeholders["iou_token"] { feeds[t] = MPSGraphTensorData(iouToken!, shape: [1, 1, NSNumber(value: embedDim)], dataType: ioDataType) }
        if let t = placeholders["mask_tokens"] { feeds[t] = MPSGraphTensorData(maskTokens!, shape: [1, 4, NSNumber(value: embedDim)], dataType: ioDataType) }
        if let t = placeholders["points"] { feeds[t] = MPSGraphTensorData(pointEmbeddings, shape: [1, NSNumber(value: pointCount), NSNumber(value: embedDim)], dataType: ioDataType) }
        
        // Image (Flat)
        let flatImgCount = 4096 * embedDim
        if let t = placeholders["image_flat"] { feeds[t] = MPSGraphTensorData(imageEmbeddings, shape: [NSNumber(value: flatImgCount)], dataType: ioDataType) }
        if let t = placeholders["image_pe_flat"] { feeds[t] = MPSGraphTensorData(imagePE, shape: [NSNumber(value: flatImgCount)], dataType: ioDataType) }

        // Dense
        if let dBuf = densePromptEmbeddings, let t = placeholders["dense_prompt"] {
            let dCount = dBuf.length / bytesPerElement / embedDim
            feeds[t] = MPSGraphTensorData(dBuf, shape: [1, NSNumber(value: dCount), NSNumber(value: embedDim)], dataType: ioDataType)
        }
        
        // HighRes
        if s0Available, let s0 = highResS0, let t = placeholders["s0_raw"] {
            feeds[t] = MPSGraphTensorData(s0, shape: [1, 256, 256, 256], dataType: ioDataType)
        }
        if s1Available, let s1 = highResS1, let t = placeholders["s1_raw"] {
            feeds[t] = MPSGraphTensorData(s1, shape: [1, 128, 128, 256], dataType: ioDataType)
        }
        
        // Weights (Transformer)
        transformer.addFeeds(placeholders: placeholders, to: &feeds)
        // Wrappers from Component properties
        if let b = up1Weights, let t = placeholders["up1/w"] { feeds[t] = MPSGraphTensorData(b, shape: [256, 64, 2, 2], dataType: ioDataType) }
        if let b = up1Bias, let t = placeholders["up1/b"] { feeds[t] = MPSGraphTensorData(b, shape: [1, 1, 1, 64], dataType: ioDataType) }
        up1LN.addFeeds(placeholders: placeholders, to: &feeds, name: "up1/ln")
        
        if let b = up2Weights, let t = placeholders["up2/w"] { feeds[t] = MPSGraphTensorData(b, shape: [64, 32, 2, 2], dataType: ioDataType) }
        if let b = up2Bias, let t = placeholders["up2/b"] { feeds[t] = MPSGraphTensorData(b, shape: [1, 1, 1, 32], dataType: ioDataType) }
        
        if s0Available, let w = convS0Weights, let b = convS0Bias, let tw = placeholders["s0/proj/w"], let tb = placeholders["s0/proj/b"] {
            feeds[tw] = MPSGraphTensorData(w, shape: [32, 256, 1, 1], dataType: ioDataType)
            feeds[tb] = MPSGraphTensorData(b, shape: [1, 1, 1, 32], dataType: ioDataType)
        }
        if s1Available, let w = convS1Weights, let b = convS1Bias, let tw = placeholders["s1/proj/w"], let tb = placeholders["s1/proj/b"] {
            feeds[tw] = MPSGraphTensorData(w, shape: [64, 256, 1, 1], dataType: ioDataType)
            feeds[tb] = MPSGraphTensorData(b, shape: [1, 1, 1, 64], dataType: ioDataType)
        }
        
        for (i, mlp) in outputHypernetworksMLPs.enumerated() { mlp.addFeeds(placeholders: placeholders, feeds: &feeds, name: "hyper_mlp_\(i)") }
        iouPredictionHead.addFeeds(placeholders: placeholders, feeds: &feeds, name: "iou_head")

        // 3. Run Executable
        // NOTE: We rely on runExecutable's deterministic ordering logic.
        let results = try CompiledGraphCache.shared.runExecutable(
            key: cacheKey,
            queue: queue,
            feeds: feeds,
            targetTensors: [outputs[0], outputs[1]]
        )
        
        guard let maskResult = results[outputs[0]], let iouResult = results[outputs[1]] else {
            throw NSError(domain: "MaskDecoder", code: 2, userInfo: nil)
        }
        
        // Export (Sync for now)
        let outMasks = device.makeBuffer(length: 4 * 256 * 256 * 4, options: .storageModeShared)!
        let outIoU = device.makeBuffer(length: 4 * 4, options: .storageModeShared)!
        
        guard let exportCmd = queue.makeCommandBuffer() else { throw NSError(domain: "MaskDecoder", code: 3, userInfo: nil) }
        let exportMpsCmd = MPSCommandBuffer(commandBuffer: exportCmd)
        
        maskResult.mpsndarray().exportData(with: exportMpsCmd, to: outMasks, destinationDataType: MPSDataType.float32, offset: 0, rowStrides: nil as UnsafeMutablePointer<Int>?)
        iouResult.mpsndarray().exportData(with: exportMpsCmd, to: outIoU, destinationDataType: MPSDataType.float32, offset: 0, rowStrides: nil as UnsafeMutablePointer<Int>?)
        
        exportCmd.commit()
        // B5: Removed waitUntilCompleted. Caller must sync if reading CPU.
        // exportCmd.waitUntilCompleted()
        
        return (outMasks, outIoU)
    }
}

/// Helper: 3-Layer MLP for SAM Output
class ThreeLayerMLP {
    let device: MTLDevice
    let inputDim: Int
    let hiddenDim: Int
    let outputDim: Int
    let enableHalfPrecision: Bool

    private var ioDataType: MPSDataType { enableHalfPrecision ? .float16 : .float32 }
    private var bytesPerElement: Int { enableHalfPrecision ? 2 : 4 }
    
    // Weights
    var fc1W: MTLBuffer?
    var fc1B: MTLBuffer?
    var fc2W: MTLBuffer?
    var fc2B: MTLBuffer?
    var fc3W: MTLBuffer?
    var fc3B: MTLBuffer?
    
    init(device: MTLDevice, inputDim: Int, hiddenDim: Int, outputDim: Int, enableHalfPrecision: Bool = true) {
        self.device = device
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.outputDim = outputDim
        self.enableHalfPrecision = enableHalfPrecision
        
        // No random initialization - wait for loadWeights
        fc1W = nil; fc1B = nil
        fc2W = nil; fc2B = nil
        fc3W = nil; fc3B = nil
    }
    
    func loadWeights(weights: [String: Data], prefix: String) {
        func load(_ suffix: String) -> MTLBuffer? {
            return weights.buffer(for: "\(prefix).\(suffix)", device: device)
        }
        
        // SAM3 Checkpoint Format: layers.0, layers.1, layers.2
        if let w = load("layers.0.weight") { fc1W = w }
        if let b = load("layers.0.bias") { fc1B = b }
        
        if let w = load("layers.1.weight") { fc2W = w }
        if let b = load("layers.1.bias") { fc2B = b }
        
        if let w = load("layers.2.weight") { fc3W = w }
        if let b = load("layers.2.bias") { fc3B = b }
        
        var count = 0
        if fc1W != nil { count += 1 }
        if fc2W != nil { count += 1 }
        if fc3W != nil { count += 1 }
        if count > 0 {
            print("DEBUG: ThreeLayerMLP(\(prefix)) loaded \(count)/3 weights")
        }
        
        // Fallback for legacy keys (just in case)
        if fc1W == nil {
             if let w = load("0.weight") { fc1W = w }
             if let b = load("0.bias") { fc1B = b }
             if let w = load("2.weight") { fc2W = w }
             if let b = load("2.bias") { fc2B = b }
             if let w = load("4.weight") { fc3W = w }
             if let b = load("4.bias") { fc3B = b }
        }
    }
    
    public func randomInitialize() {
        let bpe = bytesPerElement
        
        let w1S = hiddenDim * inputDim * bpe
        let b1S = hiddenDim * bpe
        
        let w2S = hiddenDim * hiddenDim * bpe
        let b2S = hiddenDim * bpe
        
        let w3S = outputDim * hiddenDim * bpe
        let b3S = outputDim * bpe
        
        func alloc(_ len: Int) -> MTLBuffer {
            let b = device.makeBuffer(length: len, options: .storageModeShared)!
            memset(b.contents(), 0, len)
            return b
        }
        
        fc1W = alloc(w1S); fc1B = alloc(b1S)
        fc2W = alloc(w2S); fc2B = alloc(b2S)
        fc3W = alloc(w3S); fc3B = alloc(b3S)
    }
    

    
    func buildGraph(input: MPSGraphTensor, graph: MPSGraph, name: String) -> (MPSGraphTensor, [String: MPSGraphTensor]) {
        // Placeholders
        let w1 = graph.placeholder(shape: [NSNumber(value: hiddenDim), NSNumber(value: inputDim)], dataType: ioDataType, name: "\(name)/w1")
        let b1 = graph.placeholder(shape: [1, NSNumber(value: hiddenDim)], dataType: ioDataType, name: "\(name)/b1")
        
        let w2 = graph.placeholder(shape: [NSNumber(value: hiddenDim), NSNumber(value: hiddenDim)], dataType: ioDataType, name: "\(name)/w2")
        let b2 = graph.placeholder(shape: [1, NSNumber(value: hiddenDim)], dataType: ioDataType, name: "\(name)/b2")
        
        let w3 = graph.placeholder(shape: [NSNumber(value: outputDim), NSNumber(value: hiddenDim)], dataType: ioDataType, name: "\(name)/w3")
        let b3 = graph.placeholder(shape: [1, NSNumber(value: outputDim)], dataType: ioDataType, name: "\(name)/b3")
        
        var phs: [String: MPSGraphTensor] = [
            "\(name)/w1": w1, "\(name)/b1": b1,
            "\(name)/w2": w2, "\(name)/b2": b2,
            "\(name)/w3": w3, "\(name)/b3": b3
        ]
        
        // GELU Helper
        func gelu(_ x: MPSGraphTensor) -> MPSGraphTensor {
             let half = graph.constant(0.5, dataType: ioDataType)
             let one = graph.constant(1.0, dataType: ioDataType)
             let sqrt2pi = graph.constant(0.7978845608, dataType: ioDataType)
             let coeff = graph.constant(0.044715, dataType: ioDataType)
             
             let x3 = graph.multiplication(graph.multiplication(x, x, name: nil), x, name: nil)
             let inner = graph.addition(x, graph.multiplication(coeff, x3, name: nil), name: nil)
             let arg = graph.multiplication(sqrt2pi, inner, name: nil)
             let tanh = graph.tanh(with: arg, name: nil)
             let res = graph.multiplication(graph.multiplication(half, x, name: nil), graph.addition(one, tanh, name: nil), name: nil)
             return res
        }
        
        // fc1
        let w1T = graph.transposeTensor(w1, dimension: 0, withDimension: 1, name: nil)
        var x = graph.matrixMultiplication(primary: input, secondary: w1T, name: nil)
        x = graph.addition(x, b1, name: nil)
        x = graph.reLU(with: x, name: "\(name)/relu1")
        
        // fc2
        let w2T = graph.transposeTensor(w2, dimension: 0, withDimension: 1, name: nil)
        x = graph.matrixMultiplication(primary: x, secondary: w2T, name: nil)
        x = graph.addition(x, b2, name: nil)
        x = graph.reLU(with: x, name: "\(name)/relu2")
        
        // fc3
        let w3T = graph.transposeTensor(w3, dimension: 0, withDimension: 1, name: nil)
        x = graph.matrixMultiplication(primary: x, secondary: w3T, name: nil)
        x = graph.addition(x, b3, name: nil)
        
        return (x, phs)
    }
    
    func addFeeds(placeholders: [String: MPSGraphTensor], feeds: inout [MPSGraphTensor: MPSGraphTensorData], name: String) {
        func add(_ suffix: String, _ buffer: MTLBuffer?, shape: [NSNumber]) {
            if let b = buffer, let ph = placeholders["\(name)/\(suffix)"] {
                feeds[ph] = MPSGraphTensorData(b, shape: shape, dataType: ioDataType)
            }
        }
        
        add("w1", fc1W, shape: [NSNumber(value: hiddenDim), NSNumber(value: inputDim)])
        add("b1", fc1B, shape: [1, NSNumber(value: hiddenDim)])
        
        add("w2", fc2W, shape: [NSNumber(value: hiddenDim), NSNumber(value: hiddenDim)])
        add("b2", fc2B, shape: [1, NSNumber(value: hiddenDim)])
        
        add("w3", fc3W, shape: [NSNumber(value: outputDim), NSNumber(value: hiddenDim)])
        add("b3", fc3B, shape: [1, NSNumber(value: outputDim)])
    }
}



// --- FILE: Sources/Sam3Sensor/SAM3MemoryAttention.swift ---

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
