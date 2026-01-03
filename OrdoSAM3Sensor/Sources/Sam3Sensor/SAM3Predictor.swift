import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

public struct SAM3Result {
    public let masks: MTLTexture
    public let iouScores: [Float]
}

@available(macOS 15.0, *)
public class SAM3Predictor {
    public enum PredictorWeightsError: Error, Equatable {
        case offlineFloat16Required(key: String, expectedBytes: Int, actualBytes: Int)
    }

    public let device: MTLDevice
    internal let commandQueue: MTLCommandQueue
    
    internal let imageEncoder: SAM3Encoder
    internal let neck: SAM3Neck
    internal let promptEncoder: PromptEncoder
    internal let geometryEncoder: GeometryEncoder // New
    internal let maskDecoder: MaskDecoder
    
    internal var imageEmbeddings: MTLBuffer?
    internal var highResS0: MTLBuffer? 
    internal var highResS1: MTLBuffer? 
    
    // Sprint 01: Removed manual graph caching (cachedGraph, cachedInputTensor, etc.)
    // Now using CompiledGraphCache for all graph compilation and execution
    
    private let enableHalfPrecision: Bool
    
    public init(device: MTLDevice, enableHalfPrecision: Bool = true) {
        self.device = device
        self.enableHalfPrecision = enableHalfPrecision
        
        guard let cq = device.makeCommandQueue() else { fatalError("No Queue") }
        self.commandQueue = cq
        
        self.imageEncoder = SAM3Encoder(device: device, enableHalfPrecision: enableHalfPrecision)
        self.neck = SAM3Neck(device: device, enableHalfPrecision: enableHalfPrecision)
        
        self.promptEncoder = PromptEncoder(
            device: device,
            embedDim: 256,
            imageEmbeddingSize: (64, 64),
            inputImageSize: (1008, 1008)
        )
        self.geometryEncoder = GeometryEncoder(device: device, embedDim: 256, enableHalfPrecision: enableHalfPrecision)
        self.maskDecoder = MaskDecoder(device: device, embedDim: 256, enableHalfPrecision: enableHalfPrecision)
    }
    
    public func loadWeights(from url: URL) throws {
        let loader = ModelLoader()
        var weights = try loader.load(url: url)
        
        let weightsDir = url.deletingLastPathComponent()
        let gaussianURL = weightsDir.appendingPathComponent("gaussian_matrix.bin")
        if FileManager.default.fileExists(atPath: gaussianURL.path) {
            do {
                let data = try Data(contentsOf: gaussianURL)
                weights["manual_gaussian_matrix"] = data
                print("SAM3Predictor: ✅ Loaded Gaussian Gen Matrix")
            } catch {
                print("SAM3Predictor: ❌ Error reading Gaussian: \(error)")
            }
        }
        
        try loadWeights(weights)
    }
    
    public func loadWeights(_ rawWeights: [String: Data]) throws {
        var weights = rawWeights
        
        var keysToRemove: [String] = []
        var newEntries: [String: Data] = [:]
        
        for (key, data) in weights {
            // Mapping Logic
            if key.hasPrefix("tracker.sam_mask_decoder.") {
                let newKey = key.replacingOccurrences(of: "tracker.sam_mask_decoder.", with: "sam_mask_decoder.")
                newEntries[newKey] = data
                keysToRemove.append(key)
            } else if key.hasPrefix("detector.geometry_encoder.") {
                let newKey = key.replacingOccurrences(of: "detector.geometry_encoder.", with: "geometry_encoder.")
                newEntries[newKey] = data
                keysToRemove.append(key)
            } else if key.hasPrefix("tracker.sam_prompt_encoder.") {
                let newKey = key.replacingOccurrences(of: "tracker.sam_prompt_encoder.", with: "")
                newEntries[newKey] = data
                keysToRemove.append(key)
            } else if key.hasPrefix("tracker.transformer.encoder.") {
                keysToRemove.append(key)
            } else if key.hasPrefix("tracker.transformer.") {
                 keysToRemove.append(key)
            }
            else if key.hasPrefix("inst_interactive_predictor.model.sam_mask_decoder.") {
                let newKey = key.replacingOccurrences(of: "inst_interactive_predictor.model.sam_mask_decoder.", with: "sam_mask_decoder.")
                newEntries[newKey] = data
                keysToRemove.append(key)
            } else if key.hasPrefix("inst_interactive_predictor.model.sam_prompt_encoder.") {
                let newKey = key.replacingOccurrences(of: "inst_interactive_predictor.model.sam_prompt_encoder.", with: "")
                newEntries[newKey] = data
                keysToRemove.append(key)
            }
        }
        
        for k in keysToRemove { weights.removeValue(forKey: k) }
        for (k, v) in newEntries { weights[k] = v }



        // Sprint 02 performance rule: runtime must not convert Float32→Float16.
        // If half precision is enabled, weights must be preconverted offline.
        if enableHalfPrecision {
            try validateOfflineFloat16Required(weights)
        }
        
        print("SAM3Predictor: Distributing weights...")
        self.imageEncoder.loadWeights(weights)
        self.neck.loadWeights(weights)
        self.promptEncoder.loadWeights(weights)
        self.geometryEncoder.loadWeights(weights: weights)
        self.maskDecoder.loadWeights(weights)
    }

    private func validateOfflineFloat16Required(_ weights: [String: Data]) throws {
        // Validate encoder/backbone weights using known SAM3 shapes.
        // This stays O(numKeys) and avoids building an MPSGraph at load time.
        let embedDim = WeightMapper.embedDim
        let mlpHiddenDim = WeightMapper.mlpHiddenDim
        let patchSize = WeightMapper.patchSize

        var expectedBytesByKey: [String: Int] = [:]
        expectedBytesByKey.reserveCapacity(1_000)

        func setExpectedBytes(_ key: String, elementCount: Int) {
            expectedBytesByKey[key] = elementCount * 2
        }

        // Patch embed
        if let wKey = WeightMapper.patchEmbedKeys["weight"] {
            setExpectedBytes(wKey, elementCount: embedDim * 3 * patchSize * patchSize)
        }
        if let bKey = WeightMapper.patchEmbedKeys["bias"] {
            setExpectedBytes(bKey, elementCount: embedDim)
        }

        // Positional embedding + CLS token
        setExpectedBytes(WeightMapper.posEmbedKey, elementCount: 1 * 577 * embedDim)
        setExpectedBytes("detector.backbone.vision_backbone.trunk.cls_token", elementCount: 1 * 1 * embedDim)

        // Transformer blocks
        for block in 0..<WeightMapper.numBlocks {
            setExpectedBytes(WeightMapper.encoderBlockKey(block: block, component: "norm1.weight"), elementCount: embedDim)
            setExpectedBytes(WeightMapper.encoderBlockKey(block: block, component: "norm1.bias"), elementCount: embedDim)

            setExpectedBytes(WeightMapper.encoderBlockKey(block: block, component: "attn.qkv.weight"), elementCount: (3 * embedDim) * embedDim)
            setExpectedBytes(WeightMapper.encoderBlockKey(block: block, component: "attn.qkv.bias"), elementCount: 3 * embedDim)
            setExpectedBytes(WeightMapper.encoderBlockKey(block: block, component: "attn.proj.weight"), elementCount: embedDim * embedDim)
            setExpectedBytes(WeightMapper.encoderBlockKey(block: block, component: "attn.proj.bias"), elementCount: embedDim)

            setExpectedBytes(WeightMapper.encoderBlockKey(block: block, component: "norm2.weight"), elementCount: embedDim)
            setExpectedBytes(WeightMapper.encoderBlockKey(block: block, component: "norm2.bias"), elementCount: embedDim)

            setExpectedBytes(WeightMapper.encoderBlockKey(block: block, component: "mlp.fc1.weight"), elementCount: mlpHiddenDim * embedDim)
            setExpectedBytes(WeightMapper.encoderBlockKey(block: block, component: "mlp.fc1.bias"), elementCount: mlpHiddenDim)
            setExpectedBytes(WeightMapper.encoderBlockKey(block: block, component: "mlp.fc2.weight"), elementCount: embedDim * mlpHiddenDim)
            setExpectedBytes(WeightMapper.encoderBlockKey(block: block, component: "mlp.fc2.bias"), elementCount: embedDim)
        }

        // Validate neck weights (fixed shapes / fixed keys).
        let neckPrefix = "backbone.vision_backbone.convs"
        let neckSpecs: [(key: String, shape: [Int])] = [
            ("\(neckPrefix).0.dconv_2x2_0.weight", [1024, 512, 2, 2]),
            ("\(neckPrefix).0.dconv_2x2_0.bias", [512]),
            ("\(neckPrefix).0.dconv_2x2_1.weight", [512, 256, 2, 2]),
            ("\(neckPrefix).0.dconv_2x2_1.bias", [256]),
            ("\(neckPrefix).0.conv_1x1.weight", [256, 256, 1, 1]),
            ("\(neckPrefix).0.conv_1x1.bias", [256]),
            ("\(neckPrefix).0.conv_3x3.weight", [256, 256, 3, 3]),
            ("\(neckPrefix).0.conv_3x3.bias", [256]),

            ("\(neckPrefix).1.dconv_2x2.weight", [1024, 512, 2, 2]),
            ("\(neckPrefix).1.dconv_2x2.bias", [512]),
            ("\(neckPrefix).1.conv_1x1.weight", [256, 512, 1, 1]),
            ("\(neckPrefix).1.conv_1x1.bias", [256]),
            ("\(neckPrefix).1.conv_3x3.weight", [256, 256, 3, 3]),
            ("\(neckPrefix).1.conv_3x3.bias", [256]),

            ("\(neckPrefix).2.conv_1x1.weight", [256, 1024, 1, 1]),
            ("\(neckPrefix).2.conv_1x1.bias", [256]),
            ("\(neckPrefix).2.conv_3x3.weight", [256, 256, 3, 3]),
            ("\(neckPrefix).2.conv_3x3.bias", [256])
        ]

        for spec in neckSpecs {
            guard let data = weights[spec.key] else { continue }
            let count = spec.shape.reduce(1, *)
            let expectedBytes = count * 2
            if data.count != expectedBytes {
                throw PredictorWeightsError.offlineFloat16Required(
                    key: spec.key,
                    expectedBytes: expectedBytes,
                    actualBytes: data.count
                )
            }
        }

        // Validate mask decoder weights (Float16 only when half precision is enabled).
        let mdEmbedDim = 256

        func setExpectedBytesMD(_ key: String, elementCount: Int) {
            expectedBytesByKey[key] = elementCount * 2
        }

        // Tokens
        setExpectedBytesMD("sam_mask_decoder.iou_token.weight", elementCount: mdEmbedDim)
        setExpectedBytesMD("sam_mask_decoder.mask_tokens.weight", elementCount: 4 * mdEmbedDim)

        // Output upscaling
        setExpectedBytesMD("sam_mask_decoder.output_upscaling.0.weight", elementCount: 256 * 64 * 2 * 2)
        setExpectedBytesMD("sam_mask_decoder.output_upscaling.0.bias", elementCount: 64)
        setExpectedBytesMD("sam_mask_decoder.output_upscaling.1.weight", elementCount: 64)
        setExpectedBytesMD("sam_mask_decoder.output_upscaling.1.bias", elementCount: 64)
        setExpectedBytesMD("sam_mask_decoder.output_upscaling.3.weight", elementCount: 64 * 32 * 2 * 2)
        setExpectedBytesMD("sam_mask_decoder.output_upscaling.3.bias", elementCount: 32)

        // High-res projections
        setExpectedBytesMD("sam_mask_decoder.conv_s0.weight", elementCount: 32 * mdEmbedDim)
        setExpectedBytesMD("sam_mask_decoder.conv_s0.bias", elementCount: 32)
        setExpectedBytesMD("sam_mask_decoder.conv_s1.weight", elementCount: 64 * mdEmbedDim)
        setExpectedBytesMD("sam_mask_decoder.conv_s1.bias", elementCount: 64)

        // IoU prediction head (3-layer MLP)
        setExpectedBytesMD("sam_mask_decoder.iou_prediction_head.layers.0.weight", elementCount: mdEmbedDim * mdEmbedDim)
        setExpectedBytesMD("sam_mask_decoder.iou_prediction_head.layers.0.bias", elementCount: mdEmbedDim)
        setExpectedBytesMD("sam_mask_decoder.iou_prediction_head.layers.1.weight", elementCount: mdEmbedDim * mdEmbedDim)
        setExpectedBytesMD("sam_mask_decoder.iou_prediction_head.layers.1.bias", elementCount: mdEmbedDim)
        setExpectedBytesMD("sam_mask_decoder.iou_prediction_head.layers.2.weight", elementCount: 4 * mdEmbedDim)
        setExpectedBytesMD("sam_mask_decoder.iou_prediction_head.layers.2.bias", elementCount: 4)

        // Output hypernetworks (4x 3-layer MLPs)
        for i in 0..<4 {
            let prefix = "sam_mask_decoder.output_hypernetworks_mlps.\(i).layers"
            setExpectedBytesMD("\(prefix).0.weight", elementCount: mdEmbedDim * mdEmbedDim)
            setExpectedBytesMD("\(prefix).0.bias", elementCount: mdEmbedDim)
            setExpectedBytesMD("\(prefix).1.weight", elementCount: mdEmbedDim * mdEmbedDim)
            setExpectedBytesMD("\(prefix).1.bias", elementCount: mdEmbedDim)
            setExpectedBytesMD("\(prefix).2.weight", elementCount: 32 * mdEmbedDim)
            setExpectedBytesMD("\(prefix).2.bias", elementCount: 32)
        }

        // TwoWayTransformer inside mask decoder
        let twtPrefix = "sam_mask_decoder.transformer"
        let twtDepth = 2
        let twtMlpDim = 2048
        let crossDim = 128

        func setAttnExpected(_ base: String, internalDim: Int) {
            // Unfused (SAM-style) q/k/v/out
            setExpectedBytesMD("\(base).q_proj.weight", elementCount: internalDim * mdEmbedDim)
            setExpectedBytesMD("\(base).k_proj.weight", elementCount: internalDim * mdEmbedDim)
            setExpectedBytesMD("\(base).v_proj.weight", elementCount: internalDim * mdEmbedDim)
            setExpectedBytesMD("\(base).out_proj.weight", elementCount: mdEmbedDim * internalDim)

            setExpectedBytesMD("\(base).q_proj.bias", elementCount: internalDim)
            setExpectedBytesMD("\(base).k_proj.bias", elementCount: internalDim)
            setExpectedBytesMD("\(base).v_proj.bias", elementCount: internalDim)
            setExpectedBytesMD("\(base).out_proj.bias", elementCount: mdEmbedDim)

            // Fused (PyTorch-style) in_proj_{weight,bias}
            setExpectedBytesMD("\(base).in_proj_weight", elementCount: (3 * internalDim) * mdEmbedDim)
            setExpectedBytesMD("\(base).in_proj_bias", elementCount: 3 * internalDim)
        }

        func setNormExpected(_ base: String) {
            setExpectedBytesMD("\(base).weight", elementCount: mdEmbedDim)
            setExpectedBytesMD("\(base).bias", elementCount: mdEmbedDim)
        }

        for layer in 0..<twtDepth {
            let layerPrefix = "\(twtPrefix).layers.\(layer)"

            setAttnExpected("\(layerPrefix).self_attn", internalDim: mdEmbedDim)
            setNormExpected("\(layerPrefix).norm1")

            setAttnExpected("\(layerPrefix).cross_attn_token_to_image", internalDim: crossDim)
            setNormExpected("\(layerPrefix).norm2")

            setExpectedBytesMD("\(layerPrefix).mlp.lin1.weight", elementCount: twtMlpDim * mdEmbedDim)
            setExpectedBytesMD("\(layerPrefix).mlp.lin1.bias", elementCount: twtMlpDim)
            setExpectedBytesMD("\(layerPrefix).mlp.lin2.weight", elementCount: mdEmbedDim * twtMlpDim)
            setExpectedBytesMD("\(layerPrefix).mlp.lin2.bias", elementCount: mdEmbedDim)
            setNormExpected("\(layerPrefix).norm3")

            setAttnExpected("\(layerPrefix).cross_attn_image_to_token", internalDim: crossDim)
            setNormExpected("\(layerPrefix).norm4")
        }

        setAttnExpected("\(twtPrefix).final_attn_token_to_image", internalDim: crossDim)
        setNormExpected("\(twtPrefix).norm_final_attn")

        // Validate all known Float16-only weights that are present in this weight pack.
        for (key, data) in weights {
            guard let expectedBytes = expectedBytesByKey[key] else { continue }
            if data.count != expectedBytes {
                throw PredictorWeightsError.offlineFloat16Required(
                    key: key,
                    expectedBytes: expectedBytes,
                    actualBytes: data.count
                )
            }
        }
    }
    
    public func setImage(_ texture: MTLTexture) throws {
        if texture.width != 1008 || texture.height != 1008 {
             throw NSError(domain: "SAM3Predictor", code: 4, userInfo: [NSLocalizedDescriptionKey: "Texture must be 1008x1008"])
        }

        // Sprint 01: Use CompiledGraphCache to eliminate 6.2s blocking stall
        let cacheKey = "SAM3Predictor_Encoder_1008x1008_\(enableHalfPrecision ? "F16" : "F32")"
        print("SAM3Predictor.setImage: Getting graph cacheKey=\(cacheKey)")
        // 1. Get or compile graph executable (once)
        let (graph, placeholders, targetTensors, executable) = CompiledGraphCache.shared.getOrCompileMulti(
            key: cacheKey,
            device: device
        ) {
            let graph = MPSGraph()
            let inputRaw = graph.placeholder(shape: [1, 1008, 1008, 4], dataType: .float32, name: "input_raw")
            
            let inputRGB = graph.sliceTensor(inputRaw, dimension: 3, start: 0, length: 3, name: "slice_rgb")
            let meanFloats: [Float] = [0.485, 0.456, 0.406]
            let stdFloats: [Float] = [0.229, 0.224, 0.225]
            let computeDT: MPSDataType = self.enableHalfPrecision ? .float16 : .float32
            let meanData = Data(bytes: meanFloats, count: meanFloats.count * 4)
            let stdData = Data(bytes: stdFloats, count: stdFloats.count * 4)
            let mean = graph.constant(meanData, shape: [1, 1, 1, 3], dataType: .float32)
            let std = graph.constant(stdData, shape: [1, 1, 1, 3], dataType: .float32)
            let normalized = graph.division(graph.subtraction(inputRGB, mean, name: "sub_mean"), std, name: "div_std")
            
            let (encOut, encPH) = imageEncoder.buildGraph(input: normalized, graph: graph, computeDT: computeDT)
            let (s0, s1, s2, neckPH) = neck.buildGraph(input: encOut, graph: graph, computeDT: computeDT)
            
            // Merge placeholders
            var allPlaceholders: [String: MPSGraphTensor] = ["input_raw": inputRaw]
            for (k, v) in encPH { allPlaceholders[k] = v }
            for (k, v) in neckPH { allPlaceholders[k] = v }
            
            // Output order: [s2, s0, s1] (image embeddings, high-res S0, high-res S1)
            return (graph: graph, placeholders: allPlaceholders, outputs: [s2, s0, s1])
        }
        
        // Sprint 01: Fail fast if compilation failed
        guard executable != nil else {
            throw NSError(domain: "SAM3Predictor", code: 11, userInfo: [NSLocalizedDescriptionKey: "Graph compilation failed for key: \(cacheKey)"])
        }
        
        // 2. Prepare feeds
        var feeds: [MPSGraphTensor: MPSGraphTensorData] = [:]
        let mpsImage = MPSImage(texture: texture, featureChannels: 4)
        if let inputRaw = placeholders["input_raw"] {
            feeds[inputRaw] = MPSGraphTensorData([mpsImage])
        }
        
        // Sprint 01: addFeeds requires weights to be loaded
        // If weights aren't loaded (e.g., in tests), we still need to provide feeds for all placeholders
        // The encoder and neck will handle missing weights by creating zeros
        imageEncoder.addFeeds(placeholders: placeholders, feeds: &feeds)
        neck.addFeeds(placeholders: placeholders, feeds: &feeds)
        
        // Ensure all required placeholders have feeds (for tests without weights)
        // If any placeholder is still missing, create zero-filled data
        for (key, ph) in placeholders {
            if feeds[ph] == nil && key != "input_raw" {
                // Create zero-filled data for missing placeholders
                if let shape = ph.shape {
                    let count = Int(shape.map { $0.intValue }.reduce(1, *))
                    let dt = ph.dataType  // Use placeholder's actual data type
                    let bytesPerElement = (dt == .float16) ? 2 : 4
                    let zeros = [UInt8](repeating: 0, count: count * bytesPerElement)
                    let data = Data(zeros)
                    feeds[ph] = MPSGraphTensorData(
                        device: MPSGraphDevice(mtlDevice: device),
                        data: data,
                        shape: shape,
                        dataType: dt
                    )
                }
            }
        }
        
        // 3. Execute via compiled executable (async, no blocking!)
        // Sprint 01: This replaces the 6.2s blocking graph.run() call
        print("SAM3Predictor.setImage: Executing graph...")
        let results = try CompiledGraphCache.shared.runExecutable(
            key: cacheKey,
            queue: commandQueue,
            feeds: feeds,
            targetTensors: targetTensors
        )
        print("SAM3Predictor.setImage: Graph execution complete.")
        
        let s2 = targetTensors[0]  // Image embeddings
        let s0 = targetTensors[1]  // High-res S0
        let s1 = targetTensors[2]  // High-res S1
        
        func export(_ tensor: MPSGraphTensor, count: Int) throws -> MTLBuffer {
            guard let data = results[tensor] else { throw NSError(domain: "SAM3Predictor", code: 5, userInfo: nil) }
            let bytes = count * (enableHalfPrecision ? 2 : 4)
            print("SAM3Predictor.export: count=\(count) bytes=\(bytes)")
            // M3 Optimization: Use .storageModeShared for visibility during debugging
            // Adding padding to rule out alignment assertions
            guard let buf = device.makeBuffer(length: bytes + 16384, options: .storageModeShared) else { throw NSError(domain: "SAM3Predictor", code: 6, userInfo: nil) }
            
            // Export requires its own command buffer
            guard let cb = commandQueue.makeCommandBuffer() else { throw NSError(domain: "SAM3Predictor", code: 7, userInfo: nil) }
            let mpsCb = MPSCommandBuffer(commandBuffer: cb)
            
            data.mpsndarray().exportData(with: mpsCb, to: buf, destinationDataType: enableHalfPrecision ? MPSDataType.float16 : MPSDataType.float32, offset: 0, rowStrides: nil as UnsafeMutablePointer<Int>?)
            
            cb.commit()
            // Sprint 01: No waitUntilCompleted - let export run async
            
            return buf
        }
        
        self.imageEmbeddings = try export(s2, count: 64 * 64 * 256)
        self.highResS0 = try export(s0, count: 256 * 256 * 256)
        self.highResS1 = try export(s1, count: 128 * 128 * 256)
 
            

    }

    
    public func predict(
        points: [CGPoint],
        labels: [Int],
        box: CGRect? = nil,
        multimaskOutput: Bool = true
    ) throws -> SAM3Result {
        guard let imageEmbeddings = imageEmbeddings else {
            throw NSError(domain: "SAM3Predictor", code: 1, userInfo: [NSLocalizedDescriptionKey: "Image not set"])
        }
        
        // 1. Get Sparse and Dense Embeddings
        let sparse: MTLBuffer
        let dense: MTLBuffer
        
        if useGeometryEncoder {
            let (geoSparse, _) = try geometryEncoder.forwardAndBridge(points: points, labels: labels, imageEmbeddings: imageEmbeddings, commandBuffer: commandQueue.makeCommandBuffer()!)
            sparse = geoSparse
            
            // Dense: Use no_mask_embed from PromptEncoder (or zero if missing)
            if let noMask = promptEncoder.noMaskEmbed {
                dense = noMask
            } else {
                // Fallback (Zero)
                print("WARNING: noMaskEmbed missing, using zeros")
                dense = device.makeBuffer(length: 256 * 4, options: .storageModeShared)!
            }
        } else {
            // Convert points
            let typePoints = points.enumerated().map { (i, p) in 
                PromptEncoder.PromptType.point(x: Float(p.x), y: Float(p.y), label: labels.indices.contains(i) ? labels[i] : 1)
            }
            
            let (peSparse, peDense) = try promptEncoder.forward(points: typePoints, boxes: [], masks: nil)
            sparse = peSparse
            dense = peDense
        }
            
        let height = 256
        let width = 256
        
        // Generate Dense PE (64x64) for image features
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let imagePE = geometryEncoder.computeDensePE(gridSize: 64, commandBuffer: commandBuffer)
        
        print("SAM3Predictor.predict: imagePE=\(imagePE.length) enableHalfPrecision=\(enableHalfPrecision)")
        
        let (masksBuffer, iouBuffer) = try maskDecoder.forward(
            imageEmbeddings: imageEmbeddings, 
            imagePE: imagePE, 
            pointEmbeddings: sparse,
            densePromptEmbeddings: dense,
            highResS0: highResS0,
            highResS1: highResS1,
            commandQueue: self.commandQueue
        )
        
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: width, height: height, mipmapped: false)
        desc.textureType = .type2DArray
        desc.arrayLength = 4
        desc.usage = [.shaderRead, .shaderWrite]
        
        guard let outTex = device.makeTexture(descriptor: desc) else {
             throw NSError(domain: "SAM3Predictor", code: 3, userInfo: [NSLocalizedDescriptionKey: "Texture Alloc Failed"])
        }
        
        let bytesPerRow = width * 4
        let bytesPerImage = height * bytesPerRow
        
        // B5: Perform transfer on GPU using Blit (Avoid CPU stall until final sync)
        guard let blitCmd = self.commandQueue.makeCommandBuffer(),
              let blitEnc = blitCmd.makeBlitCommandEncoder() else {
             throw NSError(domain: "SAM3Predictor", code: 4, userInfo: nil)
        }
        
        let sourceSize = MTLSize(width: width, height: height, depth: 1)
        
        for i in 0..<4 {
             let offset = i * bytesPerImage
             blitEnc.copy(from: masksBuffer, sourceOffset: offset, sourceBytesPerRow: bytesPerRow, sourceBytesPerImage: bytesPerImage, sourceSize: sourceSize, to: outTex, destinationSlice: i, destinationLevel: 0, destinationOrigin: MTLOriginMake(0, 0, 0))
        }
        blitEnc.endEncoding()
        blitCmd.commit()
        
        // Final Sync: Wait for Blit (which waits for MaskDec due to serial queue)
        // This ensures outTex is ready AND iouBuffer is ready for CPU read
        blitCmd.waitUntilCompleted()
        
        var scores: [Float] = []
        let iouCount = 4
        let ptr = iouBuffer.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<iouCount {
             let logit = ptr[i]
             let score = Float(1.0 / (1.0 + exp(-Double(logit))))
             scores.append(score)
        }
        
        return SAM3Result(masks: outTex, iouScores: scores)
    } 
    
    private var useGeometryEncoder: Bool { return geometryEncoder.pointsDirectProj != nil }
}
