import Metal
import MetalPerformanceShaders

/// Manages the visual memory for SAM 3 Video Tracking.
///
/// M3 Optimization:
/// - Uses `MTLTexture` (Array) for storage instead of Buffers.
/// - This leverages the M3 GPU's Tile Memory and Texture L1/L2 caches for efficient spatial access during Memory Attention.
/// - Stores feature maps in Float16 (RGBA16Float) to halve bandwidth usage compared to Float32.
public class MemoryBank {
    public let device: MTLDevice
    
    // Memory Storage
    // We store up to N past frames.
    // Per SAM 2 / SAM 3: "memory bank that encodes the past 6 frames" (roughly).
    // Dimensions: 256 channels, 64x64 or 64x64 spatial.
    // If using Textures, we need RGBA16Float (4 channels). 256 channels = 64 slices.
    // Or we can use a simpler layout if we treat channels as depth.
    
    private var memories: [MTLTexture] = []
    private let maxFrames = 6
    
    // Feature dimensions (matching Neck s2 output)
    private let width = 64
    private let height = 64
    private let channels = 256
    
    public init(device: MTLDevice) {
        self.device = device
    }
    
    public func reset() {
        memories.removeAll()
    }
    
    /// Adds a new memory feature to the bank.
    /// - Parameter memoryFeature: A 64x64x256 tensor (as MTLTexture or Buffer).
    ///   We assume input is MTLTexture (Array=64, RGBA16Float) to match our internal storage.
    public func add(memoryFeature: MTLTexture) {
        if memories.count >= maxFrames {
            memories.removeFirst()
        }
        memories.append(memoryFeature)
        // print("MemoryBank: Added frame. Count: \(memories.count)")
    }
    
    /// Returns the current bank of memories as an array of textures.
    public func getMemories() -> [MTLTexture] {
        return memories
    }
    
    /// Helper to create a storage texture for a new memory.
    public func makeMemoryTexture() -> MTLTexture? {
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false)
        desc.textureType = .type2DArray
        desc.arrayLength = channels / 4 // 256 / 4 = 64 slices
        desc.usage = [.shaderRead, .shaderWrite]
        // M3 Optimization: Private storage for high bandwidth on-chip access
        desc.storageMode = .private 
        return device.makeTexture(descriptor: desc)
    }
}
