//
//  HybridViTEncoder.swift
//  Sam3Sensor - MLX Hybrid Migration
//
//  Full Implementation with MPS Pre/Post Processing and MLX Transformers
//

import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
import MLX
import MLXNN
import MLXRandom

/// Hybrid ViT Encoder
/// Uses MPS for Convolutions (PatchEmbed, Neck)
/// Uses MLX for Transformer Blocks
@available(macOS 15.0, *)
public class HybridViTEncoder {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    
    // Components
    private let patchEmbed: PatchEmbedding
    private let neck: NeckLayer
    private var transformerBlocks: [MLXTransformerBlock] = []
    
    // Kernels & State
    private let flattenKernel: MTLComputePipelineState
    private let addResidualKernel: MTLComputePipelineState
    private var posEmbed: MTLBuffer?
    
    // Caching for Zero-Copy
    private var cachedFeaturesMLX: MLXArray?
    private var cachedFeaturesBuffer: MTLBuffer?
    
    // RoPE (Real, not random)
    private var rope: RoPE?
    private var cachedRoPE: MLXArray?
    
    // Configuration
    private let numBlocks: Int
    private let embedDim: Int
    private let numHeads: Int
    private let mlpDim: Int
    private let patchSize: Int
    private let imageSize: Int
    
    // Benchmark Mode
    public var isBenchmarkMode = false
    
    public init(
        device: MTLDevice,
        embedDim: Int = 1024,
        depth: Int = 24,
        numHeads: Int = 16,
        mlpDim: Int = 4096,
        patchSize: Int = 14,
        imageSize: Int = 1024
    ) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw SAM3Error.cannotCreateQueue
        }
        self.commandQueue = queue
        
        self.embedDim = embedDim
        self.numBlocks = depth
        self.numHeads = numHeads
        self.mlpDim = mlpDim
        self.patchSize = patchSize
        self.imageSize = imageSize
        
        // Initialize RoPE (Max Seq Len based on *Padded* image size)
        // Window partitioning requires padding input to multiple of windowSize (14)
        let maxGrid = (imageSize + patchSize - 1) / patchSize // e.g. 74
        let window = 14
        let paddedGrid = (maxGrid + window - 1) / window * window // e.g. 84
        let maxSeq = paddedGrid * paddedGrid // 84*84 = 7056
        
        self.rope = RoPE(device: device, numHeads: numHeads, headDim: embedDim / numHeads, maxSeqLen: maxSeq)
        
        if let ropeBuf = self.rope?.getFrequencies() {
             // Wrap as MLXArray [MaxSeq, HeadDim] (Float32)
             self.cachedRoPE = ropeBuf.toMLXArray(shape: [maxSeq, embedDim / numHeads], dtype: .float32)
        }
        
        // Load Library & Kernels (reusing existing kernels from bundle)
        let bundle = Bundle.module
        
        var library: MTLLibrary?
        
        // 1. Try default.metallib
        if let path = bundle.path(forResource: "default", ofType: "metallib") {
             library = try? device.makeLibrary(URL: URL(fileURLWithPath: path))
        }
        
        // 2. Try makeDefaultLibrary
        if library == nil {
            library = try? device.makeDefaultLibrary(bundle: bundle)
        }
        
        // 3. Runtime Source Compilation Fallback
        if library == nil {
            let metalFiles = [
                "Common", "Dequantize", "MemoryEncoder", "RoIAlign", "RoPE", "ViTUtils"
            ]
            var source = ""
            for name in metalFiles {
                if let path = bundle.path(forResource: name, ofType: "metal") {
                    let content = try String(contentsOfFile: path, encoding: .utf8)
                    source += "\n// File: \(name).metal\n"
                    source += content
                }
            }
            if !source.isEmpty {
                 library = try device.makeLibrary(source: source, options: nil)
            }
        }
        
        guard let lib = library else {
            throw SAM3Error.executionFailed("Failed to load Metal library")
        }
        
        // Load kernels
        let flattenFunc = lib.makeFunction(name: "texture_to_buffer_flat_half") ?? lib.makeFunction(name: "texture_to_buffer_flat")
        let addFunc = lib.makeFunction(name: "add_residual_half") ?? lib.makeFunction(name: "add_residual_float")
        
        guard let ff = flattenFunc, let af = addFunc else {
            throw SAM3Error.executionFailed("Kernels not found: flatten/add_residual")
        }
        self.flattenKernel = try device.makeComputePipelineState(function: ff)
        self.addResidualKernel = try device.makeComputePipelineState(function: af)
        
        // Initialize MPS Components
        self.patchEmbed = PatchEmbedding(
            device: device,
            embedDim: embedDim,
            patchSize: patchSize
        )
        
        self.neck = NeckLayer(
            device: device,
            inDim: embedDim,
            outDim: 256
        )
        
        // Initialize MLX Blocks
        for _ in 0..<depth {
            let block = MLXTransformerBlock(
                embedDim: embedDim,
                numHeads: numHeads,
                mlpDim: mlpDim,
                device: device
            )
            transformerBlocks.append(block)
        }
    }
    
    /// Load weights from safetensors file
    public func loadWeights(url: URL) throws {
        print("DEBUG: Loading weights from \(url.path)")
        let weights = try MLX.loadArrays(url: url)
        
        // 1. Separate MPS and MLX weights
        var mpsWeights: [String: MTLBuffer] = [:]
        var mlxWeights: [String: MLXArray] = [:]
        
        for (key, array) in weights {
            if key.contains("patch_embed") || key.contains("vision_backbone.convs") {
                 // Convert to MTLBuffer for MPS
                 MLX.eval(array)
                 if let buffer = array.asMTLBuffer(device: device) {
                     mpsWeights[key] = buffer
                 } else {
                     print("WARNING: Failed to convert \(key) to MTLBuffer")
                 }
            } else {
                // Keep as MLXArray for Transformers
                mlxWeights[key] = array
            }
        }
        
        // 2. Load MPS Components
        if let w = mpsWeights["backbone.patch_embed.proj.weight"],
           let b = mpsWeights["backbone.patch_embed.proj.bias"] {
            patchEmbed.loadWeights(weights: w, bias: b)
        } else {
             print("WARNING: Missing PatchEmbedding weights")
        }
        
        let p = "backbone.vision_backbone.convs"
        if let s4c1w = mpsWeights["\(p).0.conv_1x1.weight"],
           let s4c1b = mpsWeights["\(p).0.conv_1x1.bias"],
           let s4c2w = mpsWeights["\(p).0.conv_3x3.weight"],
           let s4c2b = mpsWeights["\(p).0.conv_3x3.bias"] {
            neck.loadWeights(conv1W: s4c1w, conv1B: s4c1b, conv2W: s4c2w, conv2B: s4c2b)
        } else {
             print("WARNING: Missing NeckLayer weights")
        }
        
        // 3. Load MLX Transformers
        for (i, block) in transformerBlocks.enumerated() {
            let prefix = "backbone.blocks.\(i)"
            try block.loadWeights(prefix: prefix, weights: mlxWeights)
        }
        
        print("DEBUG: HybridViTEncoder weights loaded successfully")
    }
    
    /// Initialize with random weights for benchmarking (No file needed)
    public func testOnly_randomize() {
        print("⚠️ Randomizing HybridViTEncoder for Benchmark")
        self.isBenchmarkMode = true
        
        // 1. MPS Components
        patchEmbed.randomInitialize()
        neck.randomInitialize()
        
        // 2. RoPE (Padded)
        let maxGrid = (imageSize + patchSize - 1) / patchSize // e.g. 74
        let window = 14
        let paddedGrid = (maxGrid + window - 1) / window * window // e.g. 84
        let maxSeq = paddedGrid * paddedGrid // 84*84 = 7056
        
        self.rope = RoPE(device: device, numHeads: numHeads, headDim: embedDim/numHeads, maxSeqLen: maxSeq)
        if let ropeBuf = self.rope?.getFrequencies() {
             self.cachedRoPE = ropeBuf.toMLXArray(shape: [maxSeq, embedDim / numHeads], dtype: .float32)
        }
        
        // 3. Pos Embed
        // Note: Pos Embed logic in addResidualKernel assumes specific size.
        // We use maxSeq size for buffer safety.
        let posCount = 1 * maxSeq * embedDim
        let posBuf = device.makeBuffer(length: posCount * 2, options: .storageModeShared)!
        let ptr = posBuf.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<posCount { ptr[i] = Float16(0.01) }
        self.posEmbed = posBuf
        
        // 4. Transformers
        for block in transformerBlocks { block.randomInitialize() }
        
        // 5. Pre-allocate Output Buffer (Metal Only)
        let outBytes = 1 * maxSeq * embedDim * 2 // F16
        cachedFeaturesBuffer = device.makeBuffer(length: outBytes, options: .storageModeShared)!
    }
    
    /// Randomly initialize weights (Alias for existing API compat)
    public func randomInitialize() {
        testOnly_randomize()
    }
    
    public func forward(image: MTLTexture, commandBuffer: MTLCommandBuffer) throws -> MTLBuffer {
        // Removed bypass: We want to actually run the pipeline!
        // if isBenchmarkMode, let cached = cachedFeaturesBuffer { return cached }

        // fputs("DEBUG: HybridViTEncoder check point 1\n", stderr)
        // 1. Patch Embedding (MPS) -> Texture
        let patchTexture = try patchEmbed.forward(input: image, commandBuffer: commandBuffer)
        
        // 2. Flatten & Add Pos Embed (Compute) -> Buffer
        let seqLen = patchTexture.width * patchTexture.height 
        
        // Cache MLX Array to avoid re-allocation
        if cachedFeaturesMLX == nil {
            let arr = MLX.zeros([1, seqLen, embedDim], dtype: .float16)
            MLX.eval(arr)
            cachedFeaturesMLX = arr
            cachedFeaturesBuffer = arr.asMTLBuffer(device: device, noCopy: true)
        }
        
        guard let featuresMLX = cachedFeaturesMLX,
              let embeddings = cachedFeaturesBuffer else {
             throw SAM3Error.bufferAllocationFailed("MLX Cached Buffer")
        }
        
        var recycledBuffers: [MTLBuffer] = [embeddings]
        
        // Run Flatten
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(flattenKernel)
            encoder.setTexture(patchTexture, index: 0)
            encoder.setBuffer(embeddings, offset: 0, index: 0)
            var dimParam = UInt32(embedDim)
            encoder.setBytes(&dimParam, length: 4, index: 1)
            
            let w = patchTexture.width
            let h = patchTexture.height
            let threadGroups = MTLSize(width: (w + 7)/8, height: (h + 7)/8, depth: 1)
            let threadsPerGroup = MTLSize(width: 8, height: 8, depth: 1)
            
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
            encoder.endEncoding()
        }
        
        // Add Pos Embed
        if let pos = posEmbed {
            if let enc = commandBuffer.makeComputeCommandEncoder() {
                enc.setComputePipelineState(addResidualKernel)
                enc.setBuffer(embeddings, offset: 0, index: 0)
                enc.setBuffer(pos, offset: 0, index: 1)
                enc.setBuffer(embeddings, offset: 0, index: 2) // In-place
                
                var count = UInt32(seqLen * embedDim)
                enc.setBytes(&count, length: 4, index: 3)
                
                let numThreads = seqLen * embedDim
                let threadGroups = MTLSize(width: (numThreads + 1023)/1024, height: 1, depth: 1)
                let threadsPerGroup = MTLSize(width: 1024, height: 1, depth: 1)
                
                enc.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
                enc.endEncoding()
            }
        }
        
        // Sync MPS -> MLX
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // 3. Transformers (MLX)
        // featuresMLX is already populated by MPS!
        let transformersOut = try encodeMLX(
            featuresMLX,
            seqLen: seqLen,
            h: patchTexture.height,
            w: patchTexture.width
        )
        
        // 4. Neck (MPS)
        // Need new command buffer for post-MLX work?
        // Reuse original queue?
        let neckCmdBuffer = commandQueue.makeCommandBuffer()!
        
        let finalOut = try neck.forward(
            input: transformersOut,
            batch: 1,
            height: patchTexture.height,
            width: patchTexture.width,
            commandBuffer: neckCmdBuffer,
            recycledBuffers: &recycledBuffers
        )
        
        neckCmdBuffer.commit()
        neckCmdBuffer.waitUntilCompleted()
        
        return finalOut
    }
    
    // Cache valid graph functions by resolution (e.g. "73x73") -> Compiled Closure
    private var compiledStackCache: [String: ([MLXArray]) -> [MLXArray]] = [:]
    
    internal func encodeMLX(_ featuresMLX: MLXArray, seqLen: Int, h: Int, w: Int) throws -> MTLBuffer {
        // print("DEBUG: encodeMLX start. Device: \(MLX.GPU)")
        
        let resolutionKey = "\(h)x\(w)"
        
        // Use cached RoPE (Sliced to current seqLen)
        guard let fullRoPE = cachedRoPE else {
            throw SAM3Error.weightsNotLoaded("RoPE frequencies not initialized")
        }
        
        // Slice: rope[0..<seqLen]
        let currentRoPE = fullRoPE[0..<seqLen].asType(.float16)
        
        // Compile if needed
        if compiledStackCache[resolutionKey] == nil {
            print("INFO: Compiling Transformer Stack for resolution \(resolutionKey)")
            
            // Function to compile: Takes [x, rope]
            // We bake 'h' and 'w' as constants into the closure (recompiles for new res)
            func stack(args: [MLXArray]) -> [MLXArray] {
                var x = args[0]
                let rope = args.count > 1 ? args[1] : nil
                
                for (i, block) in self.transformerBlocks.enumerated() {
                    let isGlobal = ((i + 1) % 4 == 0)
                    let windowed = !isGlobal
                    // block handles padding internally if needed based on h/w
                    x = block(x, rope: rope, windowed: windowed, windowSize: 14, h: h, w: w)
                }
                return [x]
            }
            
            // Compile
            compiledStackCache[resolutionKey] = MLX.compile(stack)
        }
        
        guard let runner = compiledStackCache[resolutionKey] else {
            throw SAM3Error.executionFailed("Failed to retrieve compiled function")
        }
        
        // Execute Compiled Stack
        let inputs: [MLXArray] = [featuresMLX, currentRoPE]
        let outputs = runner(inputs) 
        let output = outputs[0]
        
        // Output to Metal
        guard let outputBuffer = output.toMTLBuffer(device: device) else {
            throw SAM3Error.bufferAllocationFailed("MLX Output")
        }
        return outputBuffer
    }
}
