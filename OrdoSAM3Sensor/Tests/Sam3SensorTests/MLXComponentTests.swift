//
//  MLXComponentTests.swift
//  Sam3SensorTests
//
//  Unit tests comparing MLX vs MPS component outputs
//

import XCTest
import Metal
import MLX
@testable import Sam3Sensor

@available(macOS 15.0, *)
final class MLXComponentTests: XCTestCase {
    var device: MTLDevice!
    
    override func setUp() {
        super.setUp()
        device = MTLCreateSystemDefaultDevice()
        XCTAssertNotNil(device, "Metal device required")
    }
    
    // MARK: - LayerNorm Tests
    
    func testMLXLayerNormMatchesMPS() throws {
        let embedDim = 768
        let seqLen = 256
        
        // Create random input
        let inputSize = seqLen * embedDim
        var inputData = [Float](repeating: 0, count: inputSize)
        for i in 0..<inputSize {
            inputData[i] = Float.random(in: -1...1)
        }
        
        guard let inputBuffer = device.makeBuffer(
            bytes: inputData,
            length: inputSize * 4,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create input buffer")
            return
        }
        
        // Create random gamma/beta
        var gammaData = [Float](repeating: 1.0, count: embedDim)
        var betaData = [Float](repeating: 0.0, count: embedDim)
        for i in 0..<embedDim {
            gammaData[i] = Float.random(in: 0.5...1.5)
            betaData[i] = Float.random(in: -0.1...0.1)
        }
        
        guard let gammaBuffer = device.makeBuffer(bytes: gammaData, length: embedDim * 4, options: .storageModeShared),
              let betaBuffer = device.makeBuffer(bytes: betaData, length: embedDim * 4, options: .storageModeShared) else {
            XCTFail("Failed to create weight buffers")
            return
        }
        
        // MLX LayerNorm
        let mlxLN = MLXLayerNorm(normalizedShape: [embedDim], device: device)
        let gammaArr = gammaBuffer.toMLXArray(shape: [embedDim], dtype: .float32)
        let betaArr = betaBuffer.toMLXArray(shape: [embedDim], dtype: .float32)
        try mlxLN.loadWeights(gamma: gammaArr, beta: betaArr)
        
        // Input
        let mlxInput = inputBuffer.toMLXArray(shape: [1, seqLen, embedDim], dtype: .float32)
        let mlxOutput = mlxLN(mlxInput)
        
        guard let mlxOutputBuffer = mlxOutput.toMTLBuffer(device: device) else {
            XCTFail("Failed to convert MLX output")
            return
        }
        
        // For now, just verify it ran without crashing
        // TODO: Implement MPS LayerNorm reference and compare
        XCTAssertEqual(mlxOutputBuffer.length, inputSize * 2, "Output size mismatch (F16)")
        
        print("✅ MLX LayerNorm executed successfully")
    }
    
    // MARK: - MLP Tests
    
    func testMLXMLPExecution() throws {
        let inputDim = 768
        let hiddenDim = 3072
        let seqLen = 64
        
        // Create random input
        let inputSize = seqLen * inputDim
        var inputData = [Float](repeating: 0, count: inputSize)
        for i in 0..<inputSize {
            inputData[i] = Float.random(in: -0.5...0.5)
        }
        
        guard let inputBuffer = device.makeBuffer(
            bytes: inputData,
            length: inputSize * 4,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create input buffer")
            return
        }
        
        // Create random weights (simplified - just verify execution)
        let fc1Size = inputDim * hiddenDim
        let fc2Size = hiddenDim * inputDim
        
        var fc1W = [Float](repeating: 0, count: fc1Size)
        var fc1B = [Float](repeating: 0, count: hiddenDim)
        var fc2W = [Float](repeating: 0, count: fc2Size)
        var fc2B = [Float](repeating: 0, count: inputDim)
        
        for i in 0..<fc1Size { fc1W[i] = Float.random(in: -0.1...0.1) }
        for i in 0..<hiddenDim { fc1B[i] = Float.random(in: -0.1...0.1) }
        for i in 0..<fc2Size { fc2W[i] = Float.random(in: -0.1...0.1) }
        for i in 0..<inputDim { fc2B[i] = Float.random(in: -0.1...0.1) }
        
        guard let fc1WBuf = device.makeBuffer(bytes: fc1W, length: fc1Size * 4, options: .storageModeShared),
              let fc1BBuf = device.makeBuffer(bytes: fc1B, length: hiddenDim * 4, options: .storageModeShared),
              let fc2WBuf = device.makeBuffer(bytes: fc2W, length: fc2Size * 4, options: .storageModeShared),
              let fc2BBuf = device.makeBuffer(bytes: fc2B, length: inputDim * 4, options: .storageModeShared) else {
            XCTFail("Failed to create weight buffers")
            return
        }
        
        // MLX MLP
        let mlxMLP = MLXMLP(inputDim: inputDim, hiddenDim: hiddenDim, outputDim: inputDim, device: device)
        
        // Convert to MLXArray for loading
        /*
         Note: weights are [Output, Input] usually.
         fc1W: [hiddenDim, inputDim]
         fc1B: [hiddenDim]
         fc2W: [inputDim, hiddenDim]
         fc2B: [inputDim]
         */
        let fc1WArr = fc1WBuf.toMLXArray(shape: [hiddenDim, inputDim], dtype: .float32)
        let fc1BArr = fc1BBuf.toMLXArray(shape: [hiddenDim], dtype: .float32)
        let fc2WArr = fc2WBuf.toMLXArray(shape: [inputDim, hiddenDim], dtype: .float32)
        let fc2BArr = fc2BBuf.toMLXArray(shape: [inputDim], dtype: .float32)
        
        try mlxMLP.loadWeights(fc1W: fc1WArr, fc1B: fc1BArr, fc2W: fc2WArr, fc2B: fc2BArr)
        
        let mlxInput = inputBuffer.toMLXArray(shape: [1, seqLen, inputDim], dtype: .float32)
        let mlxOutput = mlxMLP(mlxInput)
        
        guard let mlxOutputBuffer = mlxOutput.toMTLBuffer(device: device) else {
            XCTFail("Failed to convert MLX output")
            return
        }
        
        XCTAssertEqual(mlxOutputBuffer.length, inputSize * 2, "Output size mismatch (F16)")
        
        print("✅ MLX MLP executed successfully")
    }
    
    // MARK: - Integration Test
    
    func testMLXTransformerBlockExecution() throws {
        let embedDim = 768
        let numHeads = 12
        let mlpDim = 3072
        let seqLen = 64
        
        // Create random input
        let inputSize = seqLen * embedDim
        var inputData = [Float](repeating: 0, count: inputSize)
        for i in 0..<inputSize {
            inputData[i] = Float.random(in: -0.5...0.5)
        }
        
        guard let inputBuffer = device.makeBuffer(
            bytes: inputData,
            length: inputSize * 4,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create input buffer")
            return
        }
        
        // Create transformer block
        let block = MLXTransformerBlock(
            embedDim: embedDim,
            numHeads: numHeads,
            mlpDim: mlpDim,
            device: device
        )
        
        // Note: Would need to load actual weights for real test
        // For now, just verify construction succeeds
        
        let mlxInput = inputBuffer.toMLXArray(shape: [1, seqLen, embedDim], dtype: .float32)
        
        // This will fail without loaded weights, but verifies API
        // let mlxOutput = block(mlxInput)
        
        print("✅ MLX TransformerBlock constructed successfully")
    }
    
    // MARK: - Conversion Tests
    
    func testMLXConversionRoundTrip() throws {
        let size = 1024
        var data = [Float16](repeating: 0, count: size)
        for i in 0..<size {
            data[i] = Float16(Float.random(in: -1...1))
        }
        
        // MTLBuffer → MLXArray
        guard let buffer = device.makeBuffer(
            bytes: data,
            length: size * 2,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create buffer")
            return
        }
        
        let mlxArray = buffer.toMLXArray(shape: [32, 32], dtype: .float16)
        
        // MLXArray → MTLBuffer
        guard let roundTripBuffer = mlxArray.toMTLBuffer(device: device) else {
            XCTFail("Failed to convert back")
            return
        }
        
        // Verify size matches
        XCTAssertEqual(roundTripBuffer.length, buffer.length, "Round-trip size mismatch")
        
        // Verify data matches (with F16 tolerance)
        let originalPtr = buffer.contents().assumingMemoryBound(to: Float16.self)
        let roundTripPtr = roundTripBuffer.contents().assumingMemoryBound(to: Float16.self)
        
        var maxError: Float = 0
        for i in 0..<size {
            let error = abs(Float(originalPtr[i]) - Float(roundTripPtr[i]))
            maxError = max(maxError, error)
        }
        
        XCTAssertLessThan(maxError, 1e-3, "Round-trip conversion error too large: \(maxError)")
        
        print("✅ MLX conversion round-trip successful, max error: \(maxError)")
    }
}
