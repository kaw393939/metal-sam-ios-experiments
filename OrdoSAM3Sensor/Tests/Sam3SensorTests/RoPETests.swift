//
//  RoPETests.swift
//  SAM3MetalTests
//
//  Validates RoPE Metal kernel against PyTorch reference
//

import XCTest
import Metal
@testable import Sam3Sensor

@available(macOS 15.0, *)
final class RoPETests: XCTestCase {
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    var library: MTLLibrary!
    
    override func setUpWithError() throws {
        device = MTLCreateSystemDefaultDevice()!
        commandQueue = device.makeCommandQueue()!
        
        // Robust library loading
        if let defaultLib = device.makeDefaultLibrary() {
            library = defaultLib
        } else {
            let bundle = Bundle(for: ViTEncoder.self)
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
        
        if library == nil {
             throw XCTSkip("Metal library not available")
        }
    }
    
    func testFrequencyGeneration() throws {
        print("Testing RoPE frequency generation...")
        
        let height: UInt = 64
        let width: UInt = 64
        let numHeads: UInt = 12
        let dimPerHead: UInt = 64
        
        // Create output buffer
        let freqsCount = Int(height * width * (dimPerHead / 2))
        let freqsBuffer = device.makeBuffer(
            length: freqsCount * MemoryLayout<SIMD2<Float>>.stride,
            options: .storageModeShared
        )!
        
        // Parameters
        var params = RoPEParams(
            num_heads: numHeads,
            dim_per_head: dimPerHead,
            height: height,
            width: width,
            theta: 10000.0
        )
        let paramsBuffer = device.makeBuffer(
            bytes: &params,
            length: MemoryLayout<RoPEParams>.stride,
            options: .storageModeShared
        )!
        
        // Create compute pipeline
        let function = library.makeFunction(name: "compute_rope_freqs_2d")!
        let pipeline = try device.makeComputePipelineState(function: function)
        
        // Encode
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(freqsBuffer, offset: 0, index: 0)
        encoder.setBuffer(paramsBuffer, offset: 0, index: 1)
        
        let threadgroupSize = MTLSize(width: 8, height: 8, depth: 1)
        let threadgroups = MTLSize(
            width: (Int(width) + 7) / 8,
            height: (Int(height) + 7) / 8,
            depth: 1
        )
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Verify
        let freqsPointer = freqsBuffer.contents().assumingMemoryBound(to: SIMD2<Float>.self)
        
        // Check first few values (sanity)
        print("Sample frequencies:")
        for i in 0..<min(4, freqsCount) {
            let freq = freqsPointer[i]
            print("  [\(i)]: (\(freq.x), \(freq.y))")
        }
        
        // Basic validation: frequencies should be unit vectors (cos² + sin² = 1)
        for i in 0..<min(100, freqsCount) {
            let freq = freqsPointer[i]
            let magnitude = sqrt(freq.x * freq.x + freq.y * freq.y)
            XCTAssertEqual(magnitude, 1.0, accuracy: 0.01, "Frequency \(i) should be unit vector")
        }
        
        print("✅ Frequency generation passed")
    }
    
    func testRoPEApplication() throws {
        print("Testing RoPE application...")
        
        // TODO: Load PyTorch reference from /tmp/rope_reference.npz
        // Compare Metal output vs PyTorch
        // Assert MSE < 1e-5
        
        print("⚠️  PyTorch comparison not yet implemented")
    }
    
    func benchmarkRoPE() throws {
        print("Benchmarking RoPE performance...")
        
        let height: UInt = 64
        let width: UInt = 64
        let numHeads: UInt = 12
        let dimPerHead: UInt = 64
        let iterations = 100
        
        // Setup (same as testFrequencyGeneration)
        let freqsCount = Int(height * width * (dimPerHead / 2))
        let freqsBuffer = device.makeBuffer(
            length: freqsCount * MemoryLayout<SIMD2<Float>>.stride,
            options: .storageModeShared
        )!
        
        var params = RoPEParams(
            num_heads: numHeads,
            dim_per_head: dimPerHead,
            height: height,
            width: width,
            theta: 10000.0
        )
        let paramsBuffer = device.makeBuffer(
            bytes: &params,
            length: MemoryLayout<RoPEParams>.stride,
            options: .storageModeShared
        )!
        
        let function = library.makeFunction(name: "compute_rope_freqs_2d")!
        let pipeline = try device.makeComputePipelineState(function: function)
        
        // Warmup
        for _ in 0..<10 {
            let commandBuffer = commandQueue.makeCommandBuffer()!
            let encoder = commandBuffer.makeComputeCommandEncoder()!
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(freqsBuffer, offset: 0, index: 0)
            encoder.setBuffer(paramsBuffer, offset: 0, index: 1)
            
            let threadgroupSize = MTLSize(width: 8, height: 8, depth: 1)
            let threadgroups = MTLSize(
                width: (Int(width) + 7) / 8,
                height: (Int(height) + 7) / 8,
                depth: 1
            )
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        
        // Benchmark
        let start = Date()
        for _ in 0..<iterations {
            let commandBuffer = commandQueue.makeCommandBuffer()!
            let encoder = commandBuffer.makeComputeCommandEncoder()!
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(freqsBuffer, offset: 0, index: 0)
            encoder.setBuffer(paramsBuffer, offset: 0, index: 1)
            
            let threadgroupSize = MTLSize(width: 8, height: 8, depth: 1)
            let threadgroups = MTLSize(
                width: (Int(width) + 7) / 8,
                height: (Int(height) + 7) / 8,
                depth: 1
            )
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        let elapsed = Date().timeIntervalSince(start)
        
        let avgTime = (elapsed / Double(iterations)) * 1000 // ms
        print("📊 RoPE frequency generation: \(String(format: "%.3f", avgTime)) ms/iter")
        print("📊 Throughput: \(String(format: "%.1f", 1000.0 / avgTime)) ops/sec")
    }
}

// Helper struct matching Metal definition
struct RoPEParams {
    var num_heads: UInt
    var dim_per_head: UInt
    var height: UInt
    var width: UInt
    var theta: Float
}
