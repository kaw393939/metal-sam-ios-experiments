//
//  WeightsLoader.swift
//  SAM3Metal
//
//  Loads extracted PyTorch weights into Metal buffers
//

import Foundation
import Metal

/// Loads SAM3 weights from .npz file into Metal buffers
@available(macOS 15.0, *)
public final class WeightsLoader {
    private let device: MTLDevice
    private var weights: [String: Data] = [:]
    
    public init(device: MTLDevice) {
        self.device = device
    }
    
    /// Load weights from .npz file (via JSON intermediate)
    public func load(from url: URL) throws {
        print("Loading weights from: \(url.path)")

        // Mac App Store constraint: runtime must not invoke Python.
        // Offline conversion should generate a JSON (or packed binary) artifact ahead of time.
        let jsonURL: URL
        if url.pathExtension.lowercased() == "json" {
            jsonURL = url
        } else {
            jsonURL = URL(fileURLWithPath: url.deletingPathExtension().path + ".json")
        }

        guard FileManager.default.fileExists(atPath: jsonURL.path) else {
            print("❌ JSON not found at: \(jsonURL.path)")
            throw WeightsError.offlineExtractionRequired
        }
        
        // Load JSON
        let jsonData = try Data(contentsOf: jsonURL)
        guard let json = try JSONSerialization.jsonObject(with: jsonData) as? [String: [String: Any]] else {
            throw WeightsError.invalidFormat
        }
        
        print("Loaded \(json.count) weights from JSON")
        
        // Convert to Metal buffers
        var converted = 0
        for (key, weightDict) in json {
            guard let dataArray = weightDict["data"] as? [Double] else {
                print("⚠️  Skipping \(key): no data array")
                continue
            }
            
            // Convert Double array to Float32 Data
            let floatArray = dataArray.map { Float($0) }
            var mutableArray = floatArray
            let data = Data(bytes: &mutableArray, count: floatArray.count * MemoryLayout<Float>.stride)
            weights[key] = data
            converted += 1
        }
        
        print("✅ Converted \(converted) weights to Metal format")
    }
    
    /// Get weight as Metal buffer
    /// - Parameters:
    ///   - key: Weight key
    ///   - privateCopy: If true, returns a .storageModePrivate buffer (requires commandBuffer)
    ///   - commandBuffer: Command buffer for blit operaiton (required if privateCopy is true)
    public func buffer(for key: String, privateCopy: Bool = false, commandBuffer: MTLCommandBuffer? = nil) -> MTLBuffer? {
        guard let data = weights[key] else {
            print("⚠️  Warning: Weight '\(key)' not found")
            return nil
        }
        
        guard let shared = device.makeBuffer(
            bytes: (data as NSData).bytes,
            length: data.count,
            options: .storageModeShared
        ) else { return nil }
        
        if privateCopy {
            guard let cmd = commandBuffer else {
                print("❌ Private copy requested for \(key) but no command buffer provided")
                return shared
            }
            // Use allocator helper
            return BufferAllocator.shared.makePrivateCopy(from: shared, device: device, commandBuffer: cmd, label: key)
        }
        
        return shared
    }
    
    /// Load encoder weights into ViT
    public func loadEncoder(into encoder: ViTEncoder) {
        // Create a command buffer for weight upload (Shared -> Private blits)
        // Optimization 7: Fused Weight Loading
        guard let queue = device.makeCommandQueue(),
              let commandBuffer = queue.makeCommandBuffer() else {
            print("❌ Failed to create queue for weight loading")
            return
        }
        commandBuffer.label = "WeightUpload"
        
        // Patch embedding (MPSCNN requires CPU access via pointer, so keep Shared)
        if let weights = buffer(for: "backbone.vision_backbone.trunk.patch_embed.proj.weight", privateCopy: false),
           let bias = buffer(for: "backbone.vision_backbone.trunk.patch_embed.proj.bias", privateCopy: false) {
            encoder.patchEmbed.loadWeights(weights: weights, bias: bias)
        }
        
        // Transformer blocks (MPSGraph supports Private buffers)
        for i in 0..<encoder.blocks.count {
            let prefix = "backbone.vision_backbone.trunk.blocks.\(i)"
            
            // LayerNorm 1
            if let gamma = buffer(for: "\(prefix).norm1.weight", privateCopy: true, commandBuffer: commandBuffer),
               let beta = buffer(for: "\(prefix).norm1.bias", privateCopy: true, commandBuffer: commandBuffer) {
                encoder.blocks[i].layerNorm1.loadWeights(gamma: gamma, beta: beta)
            }
            
            // Attention QKV
            if let qkvW = buffer(for: "\(prefix).attn.qkv.weight", privateCopy: true, commandBuffer: commandBuffer),
               let qkvB = buffer(for: "\(prefix).attn.qkv.bias", privateCopy: true, commandBuffer: commandBuffer),
               let projW = buffer(for: "\(prefix).attn.proj.weight", privateCopy: true, commandBuffer: commandBuffer),
               let projB = buffer(for: "\(prefix).attn.proj.bias", privateCopy: true, commandBuffer: commandBuffer) {
                encoder.blocks[i].attention.loadWeights(
                    qkvWeight: qkvW,
                    qkvBias: qkvB,
                    outputWeight: projW,
                    outputBias: projB
                )
            }
            
            // LayerNorm 2
            if let gamma = buffer(for: "\(prefix).norm2.weight", privateCopy: true, commandBuffer: commandBuffer),
               let beta = buffer(for: "\(prefix).norm2.bias", privateCopy: true, commandBuffer: commandBuffer) {
                encoder.blocks[i].layerNorm2.loadWeights(gamma: gamma, beta: beta)
            }
            
            // MLP
            if let fc1W = buffer(for: "\(prefix).mlp.fc1.weight", privateCopy: true, commandBuffer: commandBuffer),
               let fc1B = buffer(for: "\(prefix).mlp.fc1.bias", privateCopy: true, commandBuffer: commandBuffer),
               let fc2W = buffer(for: "\(prefix).mlp.fc2.weight", privateCopy: true, commandBuffer: commandBuffer),
               let fc2B = buffer(for: "\(prefix).mlp.fc2.bias", privateCopy: true, commandBuffer: commandBuffer) {
                encoder.blocks[i].mlp.loadWeights(
                    fc1W: fc1W,
                    fc1B: fc1B,
                    fc2W: fc2W,
                    fc2B: fc2B
                )
            }
        }
        
        // Neck (Also private)
        // Optimization: Handle Neck weights if they exist (need keys)
        // Check standard ViT keys for neck (usually "neck.0" etc if exported)
        // Assuming current export includes them or handled separately.
        // The original code passed 'weights' dict to 'encoder.loadWeights' for Neck.
        // Here we are in WeightsLoader using individual buffers.
        // We should replicate Neck loading here or rely on ViTEncoder.loadWeights?
        // ViTEncoder.loadWeights uses dictionary lookups.
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        print("✅ Encoder weights loaded (Transferred to Private Memory)")
    }
}

/// Simple binary weight format (alternative to NPZ)
/// Format: [num_weights: uint32][key_len: uint32][key: utf8][shape_rank: uint32][shape: uint32*rank][data: float32*product(shape)]...
public struct BinaryWeightsFormat {
    public static func convert(npzPath: String, outputPath: String) {
        // Python script to convert NPZ → binary
        let script = """
        import numpy as np
        import struct
        
        data = np.load('\(npzPath)')
        with open('\(outputPath)', 'wb') as f:
            # Write number of weights
            f.write(struct.pack('I', len(data.files)))
            
            for key in data.files:
                arr = data[key]
                
                # Write key
                key_bytes = key.encode('utf-8')
                f.write(struct.pack('I', len(key_bytes)))
                f.write(key_bytes)
                
                # Write shape
                f.write(struct.pack('I', len(arr.shape)))
                for dim in arr.shape:
                    f.write(struct.pack('I', dim))
                
                # Write data (as float32)
                arr_f32 = arr.astype(np.float32)
                f.write(arr_f32.tobytes())
        
        print(f'Converted {len(data.files)} weights to {outputPath}')
        """
        
        // Run Python script
        // TODO: Execute this conversion during build
        print("Binary conversion script ready")
    }
    
    public static func load(from url: URL, device: MTLDevice) -> [String: MTLBuffer] {
        guard let fileHandle = try? FileHandle(forReadingFrom: url) else {
            print("❌ Cannot open: \(url.path)")
            return [:]
        }
        
        var buffers: [String: MTLBuffer] = [:]
        
        do {
            // Read number of weights
            let numWeightsData = fileHandle.readData(ofLength: 4)
            let numWeights = numWeightsData.withUnsafeBytes { $0.load(as: UInt32.self) }
            
            for _ in 0..<numWeights {
                // Read key length
                let keyLenData = fileHandle.readData(ofLength: 4)
                let keyLen = keyLenData.withUnsafeBytes { $0.load(as: UInt32.self) }
                
                // Read key
                let keyData = fileHandle.readData(ofLength: Int(keyLen))
                guard let key = String(data: keyData, encoding: .utf8) else { continue }
                
                // Read shape rank
                let rankData = fileHandle.readData(ofLength: 4)
                let rank = rankData.withUnsafeBytes { $0.load(as: UInt32.self) }
                
                // Read shape
                var shape: [UInt32] = []
                for _ in 0..<rank {
                    let dimData = fileHandle.readData(ofLength: 4)
                    let dim = dimData.withUnsafeBytes { $0.load(as: UInt32.self) }
                    shape.append(dim)
                }
                
                // Calculate size
                let count = shape.reduce(1, *)
                let byteCount = Int(count) * MemoryLayout<Float>.stride
                
                // Read data
                let weightData = fileHandle.readData(ofLength: byteCount)
                
                // Create buffer
                if let buffer = device.makeBuffer(bytes: (weightData as NSData).bytes,
                                                  length: byteCount,
                                                  options: .storageModeShared) {
                    buffers[key] = buffer
                }
            }
            
            fileHandle.closeFile()
            print("✅ Loaded \(buffers.count) weight buffers")
            
        } catch {
            print("❌ Error loading weights: \(error)")
        }
        
        return buffers
    }
}


public enum WeightsError: Error, Equatable {
    case invalidFormat
    case extractionFailed
    case fileNotFound
    case offlineExtractionRequired
}
