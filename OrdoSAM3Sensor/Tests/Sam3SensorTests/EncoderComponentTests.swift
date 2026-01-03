//
//  EncoderComponentTests.swift
//  SAM3MetalTests
//
//  TDD tests for individual encoder components
//  Testing LayerNorm, MLP, Attention separately before integration
//

import XCTest
import Metal
@testable import Sam3Sensor

final class EncoderComponentTests: XCTestCase {
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    
    override func setUpWithError() throws {
        try super.setUpWithError()
        guard #available(macOS 15.0, *) else {
            throw XCTSkip("Requires macOS 15.0+ (ViTEncoder/MPSGraph)")
        }
        device = MTLCreateSystemDefaultDevice()!
        commandQueue = device.makeCommandQueue()!
    }
    
    // MARK: - LayerNorm Tests
    
    // MARK: - Test Helpers
    
    /// Helper to read contents of a buffer, handling .storageModePrivate via generic blit
    func readBuffer(_ buffer: MTLBuffer, count: Int? = nil) -> [Float] {
        let count = count ?? (buffer.length / MemoryLayout<Float>.stride)
        
        if buffer.storageMode == .shared {
            let ptr = buffer.contents().assumingMemoryBound(to: Float.self)
            return (0..<count).map { ptr[$0] }
        } else {
            // Private buffer: Blit to shared
            let shared = device.makeBuffer(length: buffer.length, options: .storageModeShared)!
            let cmdRef = commandQueue.makeCommandBuffer()!
            let blit = cmdRef.makeBlitCommandEncoder()!
            blit.copy(from: buffer, sourceOffset: 0, to: shared, destinationOffset: 0, size: buffer.length)
            blit.endEncoding()
            cmdRef.commit()
            cmdRef.waitUntilCompleted()
            
            let ptr = shared.contents().assumingMemoryBound(to: Float.self)
            return (0..<count).map { ptr[$0] }
        }
    }
    
    // Float16 version
    func readBufferF16(_ buffer: MTLBuffer, count: Int? = nil) -> [Float] {
        let count = count ?? (buffer.length / MemoryLayout<Float16>.stride)
        
        let shared: MTLBuffer
        if buffer.storageMode == .shared {
             shared = buffer
        } else {
            shared = device.makeBuffer(length: buffer.length, options: .storageModeShared)!
            let cmdRef = commandQueue.makeCommandBuffer()!
            let blit = cmdRef.makeBlitCommandEncoder()!
            blit.copy(from: buffer, sourceOffset: 0, to: shared, destinationOffset: 0, size: buffer.length)
            blit.endEncoding()
            cmdRef.commit()
            cmdRef.waitUntilCompleted()
        }
        
        let ptr = shared.contents().assumingMemoryBound(to: Float16.self)
        return (0..<count).map { Float(ptr[$0]) }
    }
    
    // MARK: - Test Updates
    
    func testLayerNormOutputShape() throws {
        guard #available(macOS 15.0, *) else {
            throw XCTSkip("Requires macOS 15.0+ (ViTEncoder/MPSGraph)")
        }
        // GIVEN: LayerNorm for 1024-dim features
        let layerNorm = LayerNorm(device: device, dim: 1024)
        
        // Create test gamma/beta (ones and zeros)
        let gamma = device.makeBuffer(length: 1024 * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let beta = device.makeBuffer(length: 1024 * MemoryLayout<Float>.stride, options: .storageModeShared)!
        
        // Initialize gamma to 1.0, beta to 0.0
        let gammaPtr = gamma.contents().assumingMemoryBound(to: Float.self)
        let betaPtr = beta.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<1024 {
            gammaPtr[i] = 1.0
            betaPtr[i] = 0.0
        }
        
        layerNorm.loadWeights(gamma: gamma, beta: beta)
        
        // WHEN: Process small patch for unit test (avoid 1.6GB attention matrix OOM)
        // 14x14 = 196 tokens
        let seqLen = 196
        let batch = 1
        let inputSize = seqLen * batch * 1024 * MemoryLayout<Float>.stride
        let input = device.makeBuffer(length: inputSize, options: .storageModeShared)!
        
        // Fill with random values
        let inputPtr = input.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<(seqLen * 1024) {
            inputPtr[i] = Float.random(in: -1...1)
        }
        
        let commandBuffer = commandQueue.makeCommandBuffer()!
        var recycledBuffers: [MTLBuffer] = []
        let output = try layerNorm.forward(input: input, seqLen: seqLen, batch: batch, commandBuffer: commandBuffer, recycledBuffers: &recycledBuffers)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // THEN: Output has same shape as input
        XCTAssertEqual(output.length, input.length)
        XCTAssertEqual(commandBuffer.status, .completed)
        
        // Verify normalized output using helper
        let outputData = readBuffer(output, count: 1024)
        let mean = outputData.reduce(0, +) / Float(outputData.count)
        XCTAssertEqual(mean, 0.0, accuracy: 0.1, "LayerNorm should produce ~zero mean")
    }
    
    // MARK: - MLP Tests
    
    func testMLPOutputShape() throws {
        guard #available(macOS 15.0, *) else {
            throw XCTSkip("Requires macOS 15.0+ (ViTEncoder/MPSGraph)")
        }
        // GIVEN: MLP with 1024 input, 4096 hidden (4x expansion)
        let mlp = MLP(device: device, inputDim: 1024, hiddenDim: 4096)
        
        // Create dummy weights
        let fc1Weight = device.makeBuffer(length: 1024 * 4096 * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let fc1Bias = device.makeBuffer(length: 4096 * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let fc2Weight = device.makeBuffer(length: 4096 * 1024 * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let fc2Bias = device.makeBuffer(length: 1024 * MemoryLayout<Float>.stride, options: .storageModeShared)!
        
        mlp.loadWeights(fc1W: fc1Weight, fc1B: fc1Bias, fc2W: fc2Weight, fc2B: fc2Bias)
        
        // WHEN: Process input
        let seqLen = 5184
        let batch = 1
        let inputSize = seqLen * batch * 1024 * MemoryLayout<Float>.stride
        let input = device.makeBuffer(length: inputSize, options: .storageModeShared)!
        
        let commandBuffer = commandQueue.makeCommandBuffer()!
        var recycledBuffers: [MTLBuffer] = []
        let output = try mlp.forward(input: input, seqLen: seqLen, batch: batch, commandBuffer: commandBuffer, recycledBuffers: &recycledBuffers)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // THEN: Output has same shape as input
        XCTAssertEqual(output.length, input.length)
        XCTAssertEqual(commandBuffer.status, .completed)
    }
    
    // MARK: - Attention Tests
    
    func testAttentionOutputShape() throws {
        guard #available(macOS 15.0, *) else {
            throw XCTSkip("Requires macOS 15.0+ (ViTEncoder/MPSGraph)")
        }
        // GIVEN: Multi-head attention (16 heads, 64 dim per head = 1024 total)
        // Default is Float16 precision
        let attention = MPSAttentionLayer(device: device, numHeads: 16, dimPerHead: 64, useHalfPrecision: true)

        // Create dummy QKV weights (Float16 = 2 bytes)
        let bytesPerElement = 2  // Float16
        let qkvWeight = device.makeBuffer(length: 1024 * 3072 * bytesPerElement, options: .storageModeShared)!
        let qkvBias = device.makeBuffer(length: 3072 * bytesPerElement, options: .storageModeShared)!
        let outWeight = device.makeBuffer(length: 1024 * 1024 * bytesPerElement, options: .storageModeShared)!
        let outBias = device.makeBuffer(length: 1024 * bytesPerElement, options: .storageModeShared)!

        attention.loadWeights(qkvWeight: qkvWeight, qkvBias: qkvBias, outputWeight: outWeight, outputBias: outBias)

        // WHEN: Process small patch for unit test
        // 8x8 = 64 tokens (Minimal size to check logic)
        let seqLen = 64
        let batch = 1
        let inputSize = seqLen * batch * 1024 * bytesPerElement  // Float16 input
        let input = device.makeBuffer(length: inputSize, options: .storageModeShared)!

        // Create dummy RoPE freqs [SeqLen, Dim/2, 2] - always Float32
        // 64 * (64/2) * 2 = 64 * 32 * 2 = 4096 elements
        let ropeSize = 64 * 32 * 2 * MemoryLayout<Float>.stride
        let ropeFreqs = device.makeBuffer(length: ropeSize, options: .storageModeShared)!

        let commandBuffer = commandQueue.makeCommandBuffer()!
        var recycledBuffers: [MTLBuffer] = []
        let output = try attention.forward(
            input: input,
            ropeFreqs: ropeFreqs,
            batch: batch,
            seqLen: seqLen,
            windowed: false,
            commandBuffer: commandBuffer,
            recycledBuffers: &recycledBuffers
        )

        // Commit only if not already committed (MPSGraph might auto-commit on error/completion with defaults)
        if commandBuffer.status == .notEnqueued {
            commandBuffer.commit()
        }
        commandBuffer.waitUntilCompleted()

        // THEN: Output has same shape as input (Float16)
        XCTAssertEqual(output.length, inputSize)
        XCTAssertEqual(commandBuffer.status, .completed)
    }
    
    // MARK: - Integration: Single Transformer Block
    
    func testSingleTransformerBlock() throws {
        guard #available(macOS 15.0, *) else {
            throw XCTSkip("Requires macOS 15.0+ (ViTEncoder/MPSGraph)")
        }
        // GIVEN: One transformer block
        var library: MTLLibrary?
        
        // Try standard bundle loading
        if let bundle = Bundle.moduleIfAvailable {
             if Sam3Debug.rope {
                 print("RoPE DEBUG: Trying bundle \(bundle.bundlePath)")
             }
             if let path = bundle.path(forResource: "default", ofType: "metallib") {
                 library = try? device.makeLibrary(filepath: path)
             } else {
                 library = try? device.makeDefaultLibrary(bundle: bundle)
             }
        }
        
        if library == nil {
            library = device.makeDefaultLibrary()
        }
        
        if library == nil {
             // FALLBACK: Runtime compilation (Same as ViTEncoder)
             let bundle = Bundle(for: ViTEncoder.self)
             fputs("TEST DEBUG: Bundle path: \(bundle.bundlePath)\n", stderr)
             
             if let resources = try? FileManager.default.contentsOfDirectory(atPath: bundle.bundlePath) {
                 let metalFiles = resources.filter { $0.hasSuffix(".metal") }
                 if !metalFiles.isEmpty {
                     var source = ""
                     for file in metalFiles {
                         let filePath = (bundle.bundlePath as NSString).appendingPathComponent(file)
                         if let content = try? String(contentsOfFile: filePath, encoding: .utf8) {
                             source += "\n" + content
                         }
                     }
                     library = try? device.makeLibrary(source: source, options: nil)
                 }
             }
        }
        
        guard let lib = library else {
             throw XCTSkip("Metal library could not be loaded or compiled")
        }
        
        let block = try TransformerBlock(device: device, dim: 1024, numHeads: 16, mlpHiddenDim: 1024 * 4, library: lib)
        
        // WHEN: Process input through block
        // Use reduced seqLen 64 (8x8) to avoid OOM in test runner
        let seqLen = 64
        let batch = 1
        let inputSize = seqLen * batch * 1024 * MemoryLayout<Float>.stride
        let input = device.makeBuffer(length: inputSize, options: .storageModeShared)!
        let ropeFreqs = device.makeBuffer(length: seqLen * 512 * MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared)!
        
        let commandBuffer = commandQueue.makeCommandBuffer()!
        var recycledBuffers: [MTLBuffer] = []
        let output = try block.forward(input: input, ropeFreqs: ropeFreqs, seqLen: seqLen, batch: batch, commandBuffer: commandBuffer, recycledBuffers: &recycledBuffers)
        
        if commandBuffer.status == .notEnqueued {
            commandBuffer.commit()
        }
        commandBuffer.waitUntilCompleted()
        
        // THEN: Output shape matches input (internal conversion happens)
        // Output is same byte count as input due to type coercion
        XCTAssertEqual(output.length, seqLen * batch * 1024 * MemoryLayout<Float>.stride)
    }
    
    func testViTEncoder() throws {
        guard #available(macOS 15.0, *) else {
            throw XCTSkip("Requires macOS 15.0+ (ViTEncoder/MPSGraph)")
        }
        // GIVEN: Full ViT Encoder
        // Create library from bundle for test
        // ViTEncoder internal init uses device.makeDefaultLibrary or Bundle.module
        // We can use the public init
        do {
            let encoder = try ViTEncoder(device: device, numBlocks: 2, useHalfPrecision: true) // Reduced blocks for speed
            
            // GIVEN: Weights are loaded (PatchEmbed requires explicit load)
            // Must match the encoder's config exactly, otherwise MPSCNNConvolution will read
            // out-of-bounds and can crash the test runner (signal 11).
            let patchSize = 14
            let embedDim = 1024
            let inputChannels = 3
            let weightSize = embedDim * inputChannels * patchSize * patchSize
            let weights = device.makeBuffer(length: weightSize * 4, options: .storageModeShared)!
            let bias = device.makeBuffer(length: embedDim * 4, options: .storageModeShared)!
            
            // Fill with random/ones
            memset(weights.contents(), 0, weightSize * 4) // Zeros is fine for shape check, prevents NaN
            memset(bias.contents(), 0, embedDim * 4)
            
            encoder.patchEmbed.loadWeights(weights: weights, bias: bias)
            
            // Create dummy input texture (keep it small but divisible by patchSize)
            let width = 280
            let height = 280
            let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false)
            desc.usage = [.shaderRead, .shaderWrite]
            let inputTexture = device.makeTexture(descriptor: desc)!
            
            let commandBuffer = commandQueue.makeCommandBuffer()!
            
            // WHEN: Forward pass
            let features = try encoder.forward(image: inputTexture, commandBuffer: commandBuffer)
            
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            
            // THEN: Output is [SeqLen, Dim]
            // SeqLen = (W/patchSize)*(H/patchSize)
            // Neck reduces dim to 256 and outputs Float16 in half precision.
            let expectedDim = 256
            // Neck output is now Float16 (Optimization 1)
            let expectedSize = (width / patchSize) * (height / patchSize) * expectedDim * MemoryLayout<Float16>.stride
            XCTAssertEqual(features.length, expectedSize)
        } catch {
             print("ViT Test Failed: \(error)")
             throw error
        }
    }
}

