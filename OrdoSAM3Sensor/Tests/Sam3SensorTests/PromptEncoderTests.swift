//
//  PromptEncoderTests.swift
//  SAM3MetalTests
//
//  Created by User on 12/30/2025.
//

import XCTest
import Metal
import MetalPerformanceShadersGraph
@testable import Sam3Sensor

final class PromptEncoderTests: XCTestCase {
    var device: MTLDevice!
    var encoder: PromptEncoder!
    
    override func setUpWithError() throws {
        device = MTLCreateSystemDefaultDevice()
        if device == nil {
            throw XCTSkip("Metal device not available")
        }
        
        // Initialize with standard SAM dimensions
        // embedDim=256, imageEmbeddingSize=(64, 64), inputImageSize=(1024, 1024)
        encoder = PromptEncoder(device: device, embedDim: 256, imageEmbeddingSize: (64, 64), inputImageSize: (1024, 1024))
    }
    
    func testStructureInitialization() throws {
        XCTAssertNotNil(encoder)
    }
    
    func testPointEmbeddingShape() throws {
        // Test embedding a single point
        // Input: 1 point
        // Output: Sparse embeddings [1, 2, 256] -> (Point + Padding)
        // SAM convention: Points are padded to fixed length or processed dynamically
        
        /*
         SAM Logic:
         - Points are embedded as (N, 256)
         - Added to a "point label" embedding
        */
        
        let point = PromptEncoder.PromptType.point(x: 512, y: 512, label: 1)
        
        // This method doesn't exist yet
        let (sparse, dense) = try encoder.forward(points: [point], boxes: [], masks: nil)
        
        // Check Sparse Shape: [Batch, Tokens, Dim]
        // Tokens = Number of points + 1 (usually) or just N. 
        // SAM usually pads to a max number of points, but for single inference we might likely see [1, N, 256]
        
        XCTAssertNotNil(sparse)
        // With 1 point, we expect 1 token of size 256 * 4 bytes
        XCTAssertEqual(sparse.length, 1 * 256 * 4) 
        // Actually, let's just assert it runs first, then refine shape
    }
    
    func testLabelDifferentiation() throws {
        // Output for Label 1 (Foreground)
        let point1 = PromptEncoder.PromptType.point(x: 512, y: 512, label: 1)
        let (sparse1, _) = try encoder.forward(points: [point1], boxes: [], masks: nil)
        
        // Output for Label 0 (Background) -> Should differ due to label embedding
        let point0 = PromptEncoder.PromptType.point(x: 512, y: 512, label: 0)
        let (sparse0, _) = try encoder.forward(points: [point0], boxes: [], masks: nil)
        
        // Compare buffers
        // Just check first few bytes
        let ptr1 = sparse1.contents().bindMemory(to: Float.self, capacity: 256)
        let ptr0 = sparse0.contents().bindMemory(to: Float.self, capacity: 256)
        
        var diff = false
        for i in 0..<256 {
            if ptr1[i] != ptr0[i] {
                diff = true
                break
            }
        }
        
        XCTAssertTrue(diff, "Embeddings should differ for different labels")
    }

    func testDenseMaskEmbedding() throws {
        // Create dummy mask: 1x256x256 (typical mask input)
        // SAM mask input size is usually 256x256 (low res mask)
        let maskDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: 256, height: 256, mipmapped: false)
        maskDesc.usage = [.shaderRead, .shaderWrite]
        let mask = device.makeTexture(descriptor: maskDesc)!
        
        // Forward with mask
        let point = PromptEncoder.PromptType.point(x: 512, y: 512, label: 1)
        let (_, dense) = try encoder.forward(points: [point], boxes: [], masks: mask)
        
        XCTAssertNotNil(dense, "Dense embedding should not be nil when mask is provided")
        
        XCTAssertNotNil(dense, "Dense embedding should not be nil")
        
        // Expected size: 64*64*256 * 4 bytes
        let expectedBytes = 64 * 64 * 256 * 4
        XCTAssertEqual(dense.length, expectedBytes, "Dense buffer size mismatch") 

    }
}

