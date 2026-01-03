//
//  SAM3MetalPipeline.swift
//  SAM3Metal
//
//  Main API for full SAM3 inference
//

import Foundation
import Metal
import MetalPerformanceShaders

/// Complete SAM3 Metal pipeline
/// Encoder → Decoder → Tracker
@available(macOS 15.0, *)
public final class SAM3MetalPipeline {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    
    // Components
    internal let encoder: HybridViTEncoder
    internal let maskDecoder: MaskDecoder
    internal let promptEncoder: PromptEncoder
    internal let geometryEncoder: GeometryEncoder
    
    // State
    internal var isLoaded = false
    
    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw SAM3Error.noMetalDevice
        }
        
        guard let queue = device.makeCommandQueue() else {
            throw SAM3Error.cannotCreateQueue
        }
        
        self.device = device
        self.commandQueue = queue
        
        // Initialize components
        self.encoder = try HybridViTEncoder(device: device)
        self.maskDecoder = MaskDecoder(device: device)
        self.promptEncoder = PromptEncoder(device: device, embedDim: 256, imageEmbeddingSize: (64, 64), inputImageSize: (1008, 1008))
        self.geometryEncoder = GeometryEncoder(device: device, embedDim: 256, enableHalfPrecision: true)
        
        print("✅ SAM3Metal initialized on: \(device.name)")
    }
    
    /// Load weights from file
    public func loadWeights(from url: URL) throws {
        // 1. HybridViTEncoder loads directly from URL (safetensors via MLX)
        try encoder.loadWeights(url: url)
        
        // 2. Load other weights using ModelLoader
        let loader = ModelLoader()
        let weights = try loader.load(url: url)
        
        maskDecoder.loadWeights(weights)
        promptEncoder.loadWeights(weights)
        geometryEncoder.loadWeights(weights: weights)
        
        isLoaded = true
        print("✅ Weights loaded")
    }
    
    /// Encode image → features
    public func encode(image: MTLTexture) throws -> MTLBuffer {
        guard isLoaded else {
            throw SAM3Error.weightsNotLoaded("Pipeline weights not loaded - call loadWeights() first")
        }
        
        let commandBuffer = commandQueue.makeCommandBuffer()!
        commandBuffer.label = "SAM3 Encoder"
        
        let features = try encoder.forward(image: image, commandBuffer: commandBuffer)
        
        commandBuffer.commit()
        // Wait here or return promise? For now wait to match API.
        commandBuffer.waitUntilCompleted()
        
        return features
    }
    
    /// Full pipeline: segment with prompts
    public func segment(
        image: MTLTexture,
        points: [SIMD2<Float>],
        labels: [Int32]
    ) throws -> SAM3Result {
        // 1. Encode
        // Note: In real app, encode is cached. Here re-running or user calls encode separately?
        // segment() assumes we want full pass. But often features are precomputed.
        // Let's assume we encode here for simplicity.
        let features = try encode(image: image)
        
        // 2. Prompt Encode
        let sparse: MTLBuffer
        let dense: MTLBuffer
        
        // Convert SIMD points to format expected by PromptEncoder/GeometryEncoder
        // Assuming GeometryEncoder usage if available (matches Predictor logic)
        // But for simplicity in Pipeline, let's use PromptEncoder directly.
        // Or GeometryEncoder if we want "Sam3" features.
        // Let's use PromptEncoder for basic point support.
        
        let promptPoints = points.enumerated().map { (i, p) in
            PromptEncoder.PromptType.point(x: p.x, y: p.y, label: Int(labels[i]))
        }
        let (peSparse, peDense) = try promptEncoder.forward(points: promptPoints, boxes: [], masks: nil)
        sparse = peSparse
        dense = peDense
        
        // 3. Dense PE (Calculated? Or passed? SAM3Predictor calculates it)
        // GeometryEncoder has computeDensePE.
        let peCmd = commandQueue.makeCommandBuffer()!
        let imagePE = geometryEncoder.computeDensePE(gridSize: 64, commandBuffer: peCmd)
        peCmd.commit()
        
        // 4. Decode
        // Use nil for S0/S1 as HybridViTEncoder doesn't produce them yet
        // Pass commandQueue for Reuse (B7)
        let (masks, iou) = try maskDecoder.forward(
            imageEmbeddings: features,
            imagePE: imagePE,
            pointEmbeddings: sparse,
            densePromptEmbeddings: dense,
            highResS0: nil,
            highResS1: nil,
            commandQueue: commandQueue
        )
        // Note: MaskDecoder forward is now async (returns buffers immediately)
        
        // 5. Transfer to Texture (B5 Async Blit)
        let width = 256
        let height = 256
        
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: width, height: height, mipmapped: false)
        desc.textureType = .type2DArray
        desc.arrayLength = 4
        desc.usage = [.shaderRead, .shaderWrite]
        
        guard let outTex = device.makeTexture(descriptor: desc) else {
             throw SAM3Error.bufferAllocationFailed("Output Texture")
        }
        
        guard let blitCmd = commandQueue.makeCommandBuffer(),
              let blitEnc = blitCmd.makeBlitCommandEncoder() else {
             throw SAM3Error.executionFailed("Blit Creation Failed")
        }
        
        let bytesPerRow = width * 4
        let bytesPerImage = height * bytesPerRow
        let sourceSize = MTLSize(width: width, height: height, depth: 1)
        
        for i in 0..<4 {
             let offset = i * bytesPerImage
             blitEnc.copy(from: masks, sourceOffset: offset, sourceBytesPerRow: bytesPerRow, sourceBytesPerImage: bytesPerImage, sourceSize: sourceSize, to: outTex, destinationSlice: i, destinationLevel: 0, destinationOrigin: MTLOriginMake(0, 0, 0))
        }
        blitEnc.endEncoding()
        blitCmd.commit()
        blitCmd.waitUntilCompleted() // Sync for return values
        
        // 6. Read Scores
        var scores: [Float] = []
        let ptr = iou.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<4 {
             let logit = ptr[i]
             let score = Float(1.0 / (1.0 + exp(-Double(logit))))
             scores.append(score)
        }
        
        return SAM3Result(masks: outTex, iouScores: scores)
    }
}

public enum SAM3Error: Error, LocalizedError {
    case noMetalDevice
    case cannotCreateQueue
    case weightsNotLoaded(String)
    case invalidInput(String)
    case graphCompilationFailed(String)
    case bufferAllocationFailed(String)
    case executionFailed(String)
    case deviceError(String)
    case commandBufferCreationFailed
    
    public var errorDescription: String? {
        switch self {
        case .noMetalDevice:
            return "Metal device is unavailable"
        case .cannotCreateQueue:
            return "Failed to create command queue"
        case .weightsNotLoaded(let details):
            return "Weights not loaded: \(details)"
        case .invalidInput(let details):
            return "Invalid input: \(details)"
        case .graphCompilationFailed(let details):
            return "Graph compilation failed: \(details)"
        case .bufferAllocationFailed(let details):
            return "Buffer allocation failed: \(details)"
        case .executionFailed(let details):
            return "Execution failed: \(details)"
        case .deviceError(let details):
            return "Device error: \(details)"
        case .commandBufferCreationFailed:
            return "Failed to create command buffer"
        }
    }
}

/// Performance benchmarking
@available(macOS 15.0, *)
public extension SAM3MetalPipeline {
    func benchmark(iterations: Int = 100) {
        print(String(repeating: "=", count: 60))
        print("SAM3Metal Performance Benchmark")
        print(String(repeating: "=", count: 60))
        
        // Create test image
        let testImage = createTestImage()
        
        // Warmup
        print("\nWarming up...")
        for _ in 0..<10 {
            _ = try? encode(image: testImage)
        }
        
        // Benchmark Encoder
        print("\nBenchmarking Encoder (\(iterations) iterations)...")
        let start = Date()
        
        for _ in 0..<iterations {
            _ = try? encode(image: testImage)
        }
        
        let elapsed = Date().timeIntervalSince(start)
        let avgTime = (elapsed / Double(iterations)) * 1000
        let fps = 1000.0 / avgTime
        
        print("\n📊 Encoder Performance:")
        print("   Average: \(String(format: "%.2f", avgTime)) ms/frame")
        print("   Throughput: \(String(format: "%.1f", fps)) FPS")
        print("   Target: 170ms (50x speedup)")
        
        if avgTime < 170 {
            print("   ✅ TARGET MET!")
        } else {
            let shortfall = avgTime / 170.0
            print("   ⚠️  \(String(format: "%.1f", shortfall))x slower than target")
        }
    }
    
    private func createTestImage() -> MTLTexture {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float,
            width: 1024,
            height: 1024,
            mipmapped: false
        )
        desc.usage = [.shaderRead, .shaderWrite]
        
        return device.makeTexture(descriptor: desc)!
    }
}
