import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

/// Encodes a frame's mask and image features into a memory representation.
///
/// Logic:
/// 1. Downscale the mask using `PromptEncoder.mask_downscaling` (reused).
/// 2. Fuse with unconditioned image embeddings (Element-wise Addition).
/// 3. Output is stored in `MemoryBank`.
public class SAM3MemoryEncoder {
    public let device: MTLDevice
    public let promptEncoder: PromptEncoder
    
    private var fusePipeline: MTLComputePipelineState!
    
    public init(device: MTLDevice, promptEncoder: PromptEncoder) {
        self.device = device
        self.promptEncoder = promptEncoder
        
        do {
            let lib = try device.makeDefaultLibrary(bundle: Bundle.module)
            guard let fn = lib.makeFunction(name: "fuse_memory") else {
                 fatalError("SAM3MemoryEncoder: Missing 'fuse_memory' kernel in library.")
            }
            self.fusePipeline = try device.makeComputePipelineState(function: fn)
        } catch {
             fatalError("SAM3MemoryEncoder: Failed to create pipeline: \(error)")
        }
    }
    
    /// Encodes the memory features for the current frame.
    /// - Parameters:
    ///   - imageEmbeddings: The unconditioned features [1, 64, 64, 256].
    ///   - mask: The predicted mask [1, 1, H, W].
    public func encodeMemory(
        imageEmbeddings: MTLBuffer, 
        mask: MTLTexture,           
        commandBuffer: MTLCommandBuffer
    ) throws -> MTLTexture {
        // 1. Downscale Mask -> Dense Embedding Buffer [64*64*256]
        guard let maskBuf = try promptEncoder.processDenseMask(mask, commandBuffer: commandBuffer) else {
             throw NSError(domain: "SAM3MemoryEncoder", code: 1, userInfo: [NSLocalizedDescriptionKey: "Mask encoding failed"])
        }
        
        // 2. Output Texture
        guard let outputTexture = makeMemoryTexture() else {
             throw NSError(domain: "SAM3MemoryEncoder", code: 2, userInfo: [NSLocalizedDescriptionKey: "Texture alloc failed"])
        }
        
        // 3. Fuse (Compute Kernel)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
             throw NSError(domain: "SAM3MemoryEncoder", code: 3, userInfo: [NSLocalizedDescriptionKey: "Compute Encoder Alloc Failed"])
        }
        
        encoder.setComputePipelineState(fusePipeline)
        encoder.setBuffer(imageEmbeddings, offset: 0, index: 0)
        encoder.setBuffer(maskBuf, offset: 0, index: 1)
        encoder.setTexture(outputTexture, index: 0)
        
        // Grid: (64, 64, 64)
        // 64 slices cover 256 feature channels (RGBA16Float)
        let threadsPerGrid = MTLSize(width: 64, height: 64, depth: 64)
        
        // Threadgroup Size optimization
        let w = fusePipeline.threadExecutionWidth
        let h = fusePipeline.maxTotalThreadsPerThreadgroup / w
        let threadsPerThreadgroup = MTLSize(width: w, height: h, depth: 1)
        
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        return outputTexture
    }
    
    private func makeMemoryTexture() -> MTLTexture? {
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 64, height: 64, mipmapped: false)
        desc.textureType = .type2DArray
        desc.arrayLength = 64
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .private
        return device.makeTexture(descriptor: desc)
    }
}
