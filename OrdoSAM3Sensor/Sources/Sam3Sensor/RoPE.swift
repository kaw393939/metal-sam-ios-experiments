import Foundation
import Metal
import MetalPerformanceShaders

/// Rotary Position Embedding (RoPE) Frequency Generator
/// Generates precomputed frequencies for SAM3
/// Layout: [maxSeqLen, dimPerHead] (Float32)
/// which is viewed as [maxSeqLen, dimPerHead/2, 2] by consumers.
public final class RoPE {
    private let device: MTLDevice
    private let numHeads: Int
    private let headDim: Int
    private let maxSeqLen: Int
    
    // Frequency buffer (precomputed)
    private var freqBuffer: MTLBuffer?
    
    public init(device: MTLDevice, numHeads: Int = 16, headDim: Int = 64, maxSeqLen: Int = 5184) {
        self.device = device
        self.numHeads = numHeads
        self.headDim = headDim
        self.maxSeqLen = maxSeqLen
        
        generateFrequencies()
    }
    
    private func generateFrequencies() {
        // Load Library
        var library: MTLLibrary?
        
        // 1. Try to load from the Bundle (SwiftPM)
        if let bundle = Bundle.moduleIfAvailable {
             library = try? device.makeDefaultLibrary(bundle: bundle)
             
             // 2. Fallback: Compile from sources in bundle
             if library == nil {
                 let contents = (try? FileManager.default.contentsOfDirectory(atPath: bundle.bundlePath)) ?? []
                 let metalFiles = contents.filter { $0.hasSuffix(".metal") }
                 if !metalFiles.isEmpty {
                     var source = ""
                     for file in metalFiles {
                         let path = bundle.bundlePath + "/" + file
                         if let content = try? String(contentsOfFile: path, encoding: .utf8) {
                             source += "\n" + content
                         }
                     }
                     library = try? device.makeLibrary(source: source, options: nil)
                 }
             }
        }
        
        // 3. Fallback to system default (Main Bundle)
        if library == nil {
            library = device.makeDefaultLibrary()
        }
        
        guard let lib = library,
              let freqFunc = lib.makeFunction(name: "compute_rope_freqs_2d"),
              let pipeline = try? device.makeComputePipelineState(function: freqFunc) else {
            print("Error: Could not load compute_rope_freqs_2d")
            return
        }
        
        // Allocate buffer for frequencies: [maxSeqLen, headDim] of Float32
        // Matches [maxSeqLen, headDim/2, 2] invariant
        let freqCount = maxSeqLen * headDim
        let bufferSize = freqCount * MemoryLayout<Float>.stride
        
        guard let buffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
            return
        }
        
        self.freqBuffer = buffer
        
        // Generate frequencies on GPU
        guard let commandQueue = device.makeCommandQueue(),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(buffer, offset: 0, index: 0)
        
        // Match Metal RoPEParams: num_heads, dim_per_head, height, width, theta
        let height = Int(sqrt(Double(maxSeqLen)))
        let width = height // Assumes square grid for SAM3 (e.g. 64x64 or 72x72)
        
        // Enforce square assumption to avoid silent truncation/corruption
        precondition(height * width == maxSeqLen, "RoPE Error: maxSeqLen \(maxSeqLen) is not a perfect square. RoPE currently assumes a square grid (height=width).")
        
        var params = RoPEParams(
            num_heads: UInt32(numHeads),
            dim_per_head: UInt32(headDim),
            height: UInt32(height),
            width: UInt32(width),
            theta: 10000.0
        )
        encoder.setBytes(&params, length: MemoryLayout<RoPEParams>.stride, index: 1)
        
        // Use 1D grid for total token count
        let threadGroupSize = MTLSize(width: min(pipeline.threadExecutionWidth, 256), height: 1, depth: 1)
        let threadGroups = MTLSize(
            width: (maxSeqLen + threadGroupSize.width - 1) / threadGroupSize.width,
            height: 1,
            depth: 1
        )
        
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    /// Get precomputed frequencies
    /// Returns buffer of size [maxSeqLen * headDim * 4] bytes
    public func getFrequencies() -> MTLBuffer? {
        return freqBuffer
    }
}

// Parameters structure matching Metal shader
private struct RoPEParams {
    let num_heads: UInt32
    let dim_per_head: UInt32
    let height: UInt32
    let width: UInt32
    let theta: Float
}
