import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

/// The centralized manager for SAM 3 Video Object Segmentation.
///
/// Coordinates:
/// 1. `SAM3Predictor` (Image-level Detector)
/// 2. `MemoryBank` (Temporal Context)
/// 3. `MemoryEncoder` (Fusion Logic)
/// 4. `MemoryAttention` (Cross Attn)
@available(macOS 15.0, *)
public class SAM3Tracker {
    public let device: MTLDevice
    public let predictor: SAM3Predictor
    public let memoryBank: MemoryBank
    public let memoryEncoder: SAM3MemoryEncoder
    public let memoryAttention: SAM3MemoryAttention
    
    public init(device: MTLDevice, predictor: SAM3Predictor) {
        self.device = device
        self.predictor = predictor
        self.memoryBank = MemoryBank(device: device)
        self.memoryEncoder = SAM3MemoryEncoder(device: device, promptEncoder: predictor.promptEncoder)
        self.memoryAttention = SAM3MemoryAttention(device: device)
    }
    
    /// Loads weights for both the Predictor (Detector) and Tracker components.
    public func loadWeights(from url: URL) throws {
        print("SAM3Tracker: Loading weights from \(url.lastPathComponent)...")
        let loader = ModelLoader()
        var weights = try loader.load(url: url)
        
        // Load Gaussian if present (legacy support)
        let weightsDir = url.deletingLastPathComponent()
        let gaussianURL = weightsDir.appendingPathComponent("gaussian_matrix.bin")
        if FileManager.default.fileExists(atPath: gaussianURL.path) {
            do {
                let data = try Data(contentsOf: gaussianURL)
                weights["manual_gaussian_matrix"] = data
            } catch {
                print("SAM3Tracker: ⚠️ Error reading Gaussian matrix: \(error)")
            }
        }
        
        // 1. Load Predictor Weights (Detector, Prompts, etc.)
        // This will strip "tracker.*" keys from its internal copy, efficiently.
        try predictor.loadWeights(weights)
        
        // 2. Load Memory Attention Weights (Tracker Transformer)
        // We pass the full dict; it filters for "tracker.transformer.encoder"
        memoryAttention.loadWeights(weights)
        
        print("SAM3Tracker: All weights loaded.")
    }
    
    /// Clears the memory bank. Call this when starting a new video or finding a new object.
    public func reset() {
        memoryBank.reset()
    }
    
    /// Processes a single frame in the video sequence.
    ///
    /// - Parameters:
    ///   - texture: The input video frame (RGB).
    ///   - points: Optional user prompts (clicks) to initialize or correct the track.
    /// - Returns: The predicted segmentation mask for the frame.
    public func track(texture: MTLTexture, points: [CGPoint] = [], labels: [Int] = []) throws -> SAM3Result {
        // 1. Run the Image Encoder + Neck (Generates Raw Features)
        // This sets `predictor.imageEmbeddings` (Raw)
        try predictor.setImage(texture)
        
        // Capture Raw Features for Memory Encoding later
        guard let rawFeatures = predictor.imageEmbeddings else {
            throw NSError(domain: "SAM3Tracker", code: 1, userInfo: [NSLocalizedDescriptionKey: "Image encoding failed"])
        }
        
        // 2. Memory Attention (Condition current features on MemoryBank)
        // If MemoryBank is empty, this returns rawFeatures (Top-down logic).
        // If used, it returns Contextualized Features.
        
        guard let commandBuffer = predictor.commandQueue.makeCommandBuffer() else {
             throw NSError(domain: "SAM3Tracker", code: 2, userInfo: [NSLocalizedDescriptionKey: "Command Buffer Alloc Failed"])
        }
        
        let ctxFeatures = try memoryAttention.forward(
            currentFeatures: rawFeatures,
            memoryBank: memoryBank,
            commandBuffer: commandBuffer
        )
        
        // Commit Memory Attention (It uses MPSGraph, likely encoded to buffer)
        // We should probably wait or let dependency tracking handle it?
        // Since we pass buffers, Metal handles dependencies if on same device/queue?
        // `MemoryAttention` uses `MPSCommandBuffer(commandBuffer)`.
        // We haven't committed `commandBuffer` yet.
        // `predictor.predict` typically uses `commandQueue` internally to make NEW buffers.
        // Ideally we chain them.
        // `predictor` methods are blocking `graph.run` usually?
        // `predict` (MaskDecoder) uses `graph.encode`? No, `run`.
        // `SAM3Predictor.predict` calls `maskDecoder.forward`.
        // If `MemoryAttention` encodes to `commandBuffer`, we must commit it before `predict` if `predict` uses a different queue/buffer synchronously?
        // `SAM3Predictor.predict` creates its own command buffer.
        // So we must commit `commandBuffer` (Attn) and wait, OR depend on it.
        // For safety/simplicity in V1: Commit and Wait (or Commit).
        commandBuffer.commit()
        // commandBuffer.waitUntilCompleted() // Optional, but safe.
        
        // 3. Inject Contextualized Features into Predictor
        // We temporarily swap the embeddings.
        predictor.imageEmbeddings = ctxFeatures
        
        // 4. Run Mask Decoder
        let result = try predictor.predict(points: points, labels: labels)
        
        // Restore Raw Features (Clean state)
        predictor.imageEmbeddings = rawFeatures
        
        // 5. Memory Encoder (Fuse Raw Features + Result Mask -> New Memory)
        // Only if we have a mask?
        // SAM usually adds memory if mask is valid.
        // If no mask found (scores low?), maybe skip?
        // For now, always add (or let upper layer decide? No, 'track' implies auto-update).
        // We add the memory.
        
        // Note: `result.masks` is MTLTexture (4 slices). We usually take the best one?
        // SAM 3 might store multimask or best.
        // Usually we pick the one with highest score for memory.
        // `SAM3Predictor.predict` returns 4 masks.
        // Logic: Find max score index.
        // We need CPU score access. `result.iouScores` is available.
        
        if let maxScore = result.iouScores.max(), let maxIdx = result.iouScores.firstIndex(of: maxScore) {
            // Threshold? If score is very low, maybe don't add memory?
            // "Track" logic usually updates.
            // We need a specific slice from the texture.
            // `MemoryEncoder` expects `mask: MTLTexture` (Single slice? Or Array?).
            // `MemoryEncoder.encodeMemory` uses `promptEncoder.processDenseMask` which expects [1, H, W, 1].
            // `result.masks` is 2D Array. We need to extract the slice.
            // We can create a view or copy.
            
            // Slice Extraction Helper
            // Using `newTextureView` if compatible, or blit.
            // 2D Array -> 2D View is possible if same pixel format.
            // slices: (maxIdx)..<(maxIdx+1)
            let sliceDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: result.masks.pixelFormat, width: result.masks.width, height: result.masks.height, mipmapped: false)
            sliceDesc.usage = [.shaderRead, .shaderWrite]
            
            // Create view?
            // MTLTexture.makeTextureView(pixelFormat: textureType: levels: slices:)
            if let maskSlice = result.masks.makeTextureView(pixelFormat: result.masks.pixelFormat, textureType: .type2D, levels: 0..<1, slices: maxIdx..<(maxIdx+1)) {
                
                guard let encCmdBuffer = predictor.commandQueue.makeCommandBuffer() else {
                     throw NSError(domain: "SAM3Tracker", code: 2, userInfo: nil)
                }
                
                let encodedMem = try memoryEncoder.encodeMemory(
                    imageEmbeddings: rawFeatures,
                    mask: maskSlice,
                    commandBuffer: encCmdBuffer
                )
                
                encCmdBuffer.commit() // Commit encoding
                
                // Add to Bank
                // Note: MemoryBank stores Textures. `encodedMem` is Texture.
                // Sync?
                // memoryBank.add takes texture. It copies or stores?
                // It stores.
                memoryBank.add(memoryFeature: encodedMem)
            }
        }
        
        return result
    }
}
