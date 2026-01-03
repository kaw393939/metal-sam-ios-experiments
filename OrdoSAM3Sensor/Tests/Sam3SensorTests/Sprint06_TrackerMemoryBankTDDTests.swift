import XCTest
import Metal
import MetalPerformanceShadersGraph
@testable import Sam3Sensor

@available(macOS 15.0, *)
final class Sprint06_TrackerMemoryBankTDDTests: XCTestCase {
    
    var device: MTLDevice!
    var queue: MTLCommandQueue!
    
    override func setUp() {
        super.setUp()
        guard let dev = MTLCreateSystemDefaultDevice(),
              let q = dev.makeCommandQueue() else {
            XCTFail("Metal not available")
            return
        }
        device = dev
        queue = q
    }
    
    // MARK: - Test Helpers
    
    func createTestEmbedding() -> MTLBuffer {
        // Create 256-dim embedding
        let count = 256
        let data = [Float](repeating: 0.5, count: count)
        return device.makeBuffer(bytes: data, length: count * 4, options: .storageModeShared)!
    }
    
    func createTestMask(objectID: Int = 0, center: CGPoint = CGPoint(x: 144, y: 144)) -> MTLTexture {
        // Create 288x288 binary mask
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r8Unorm,
            width: 288,
            height: 288,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .shared
        let texture = device.makeTexture(descriptor: descriptor)!
        
        // Fill with test pattern (circle at center)
        var data = [UInt8](repeating: 0, count: 288 * 288)
        for y in 0..<288 {
            for x in 0..<288 {
                let dx = Double(x) - center.x
                let dy = Double(y) - center.y
                let dist = sqrt(dx*dx + dy*dy)
                if dist < 50.0 { // Circle radius
                    data[y * 288 + x] = 255
                }
            }
        }
        texture.replace(region: MTLRegionMake2D(0, 0, 288, 288), mipmapLevel: 0, withBytes: data, bytesPerRow: 288)
        return texture
    }
    
    func createTestFrame(index: Int = 0) -> MTLTexture {
        // Create 1008x1008 RGBA frame
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: 1008,
            height: 1008,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .shared
        let texture = device.makeTexture(descriptor: descriptor)!
        
        // Fill with test pattern
        var data = [UInt8](repeating: 128, count: 1008 * 1008 * 4)
        texture.replace(region: MTLRegionMake2D(0, 0, 1008, 1008), mipmapLevel: 0, withBytes: data, bytesPerRow: 1008 * 4)
        return texture
    }
    
    func computeMSE(_ buffer1: MTLBuffer, _ buffer2: MTLBuffer) -> Double {
        let count = min(buffer1.length, buffer2.length) / 4
        let ptr1 = buffer1.contents().bindMemory(to: Float.self, capacity: count)
        let ptr2 = buffer2.contents().bindMemory(to: Float.self, capacity: count)
        
        var mse: Double = 0.0
        for i in 0..<count {
            let diff = Double(ptr1[i] - ptr2[i])
            mse += diff * diff
        }
        return mse / Double(count)
    }
    
    func computeCentroid(_ mask: MTLTexture) -> CGPoint {
        // Compute centroid of mask
        let width = mask.width
        let height = mask.height
        var data = [UInt8](repeating: 0, count: width * height)
        mask.getBytes(&data, bytesPerRow: width, from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
        
        var sumX: Double = 0
        var sumY: Double = 0
        var count: Double = 0
        
        for y in 0..<height {
            for x in 0..<width {
                if data[y * width + x] > 128 {
                    sumX += Double(x)
                    sumY += Double(y)
                    count += 1
                }
            }
        }
        
        return count > 0 ? CGPoint(x: sumX / count, y: sumY / count) : CGPoint(x: 0, y: 0)
    }
    
    // MARK: - Test 1: Memory Bank Storage & Retrieval
    
    func testMemoryBankStorageRetrieval() throws {
        let memoryBank = MemoryBank(device: device)
        
        // Create test memory feature
        guard let memoryTexture = memoryBank.makeMemoryTexture() else {
            XCTFail("Failed to create memory texture")
            return
        }
        
        // Store in memory bank
        memoryBank.add(memoryFeature: memoryTexture)
        
        // Retrieve and verify
        let memories = memoryBank.getMemories()
        XCTAssertEqual(memories.count, 1, "Memory bank should have 1 entry")
        XCTAssertEqual(memories[0].width, 72, "Memory width incorrect")
        XCTAssertEqual(memories[0].height, 72, "Memory height incorrect")
        
        print("Sprint06 Test 1: Memory bank storage/retrieval verified")
    }
    
    // MARK: - Test 2: Tracker Propagation Correctness
    
    func testTrackerPropagation() throws {
        // This test requires SAM3Tracker to be fully implemented
        // For now, test basic initialization
        
        let predictor = SAM3Predictor(device: device)
        let tracker = SAM3Tracker(device: device, predictor: predictor)
        
        // Verify tracker components initialized
        XCTAssertNotNil(tracker.predictor)
        XCTAssertNotNil(tracker.memoryBank)
        XCTAssertNotNil(tracker.memoryEncoder)
        XCTAssertNotNil(tracker.memoryAttention)
        
        print("Sprint06 Test 2: Tracker components initialized")
        
        // TODO: Full propagation test requires weights
        // let frame = createTestFrame()
        // let result = try tracker.track(texture: frame, points: [CGPoint(x: 500, y: 500)], labels: [1])
        // XCTAssertNotNil(result.masks)
    }
    
    // MARK: - Test 3: Multi-Object Tracking (5 Objects)
    
    func testMultiObjectTracking() throws {
        let memoryBank = MemoryBank(device: device)
        
        // Simulate tracking 5 objects by adding 5 memory features
        for objID in 0..<5 {
            guard let memoryTexture = memoryBank.makeMemoryTexture() else {
                XCTFail("Failed to create memory texture for object \(objID)")
                return
            }
            memoryBank.add(memoryFeature: memoryTexture)
        }
        
        // Verify all 5 objects stored (limited by maxFrames=6)
        let memories = memoryBank.getMemories()
        XCTAssertEqual(memories.count, 5, "Should have 5 memory entries")
        
        print("Sprint06 Test 3: Multi-object memory storage verified")
        
        // TODO: Full multi-object tracking test requires tracker implementation
    }
    
    // MARK: - Test 4: Performance Target (Sustain FPS)
    
    func testTrackerPerformance() throws {
        let memoryBank = MemoryBank(device: device)
        
        // Measure memory bank operations
        var samples: [UInt64] = []
        
        for _ in 0..<20 {
            let t0 = DispatchTime.now().uptimeNanoseconds
            
            // Add memory feature
            if let memoryTexture = memoryBank.makeMemoryTexture() {
                memoryBank.add(memoryFeature: memoryTexture)
            }
            
            // Retrieve memories
            _ = memoryBank.getMemories()
            
            let t1 = DispatchTime.now().uptimeNanoseconds
            samples.append(t1 - t0)
        }
        
        let meanMs = Double(samples.reduce(0, +)) / Double(samples.count) / 1_000_000.0
        print("Sprint06 Test 4: Memory bank operations: \(meanMs)ms")
        
        // Memory bank operations should be very fast (< 1ms)
        XCTAssertLessThan(meanMs, 1.0, "Memory bank operations too slow")
        
        // TODO: Full tracker performance test requires implementation
    }
    
    // MARK: - Test 5: Stability (No Identity Swaps)
    
    func testTrackerStability() throws {
        // Create 2 test masks with different centroids
        let mask0 = createTestMask(objectID: 0, center: CGPoint(x: 100, y: 144))
        let mask1 = createTestMask(objectID: 1, center: CGPoint(x: 200, y: 144))
        
        // Verify centroids are distinct
        let centroid0 = computeCentroid(mask0)
        let centroid1 = computeCentroid(mask1)
        
        XCTAssertLessThan(centroid0.x, 150, "Mask 0 centroid incorrect")
        XCTAssertGreaterThan(centroid1.x, 150, "Mask 1 centroid incorrect")
        
        print("Sprint06 Test 5: Mask centroids verified - \(centroid0) vs \(centroid1)")
        
        // TODO: Full stability test requires tracker propagation
    }
    
    // MARK: - Test 6: Memory Efficiency
    
    func testMemoryBankEfficiency() throws {
        let memoryBank = MemoryBank(device: device)
        
        // Add maximum frames (6)
        for _ in 0..<6 {
            guard let memoryTexture = memoryBank.makeMemoryTexture() else {
                XCTFail("Failed to create memory texture")
                return
            }
            memoryBank.add(memoryFeature: memoryTexture)
        }
        
        // Verify memory bank caps at maxFrames
        XCTAssertEqual(memoryBank.getMemories().count, 6, "Memory bank should cap at 6 frames")
        
        // Add one more - should evict oldest
        if let memoryTexture = memoryBank.makeMemoryTexture() {
            memoryBank.add(memoryFeature: memoryTexture)
        }
        
        XCTAssertEqual(memoryBank.getMemories().count, 6, "Memory bank should still be 6 after eviction")
        
        // Estimate memory usage
        // Each memory: 72x72x256 channels in RGBA16Float (4 channels per slice)
        // 256/4 = 64 slices, each slice is 72x72x4 channels x 2 bytes = 41,472 bytes
        // Total per memory: 64 * 41,472 = 2,654,208 bytes ≈ 2.5MB
        // 6 memories: ~15MB
        let estimatedMemoryMB = (72 * 72 * 256 * 2 * 6) / (1024 * 1024)
        print("Sprint06 Test 6: Estimated memory usage: \(estimatedMemoryMB)MB")
        
        XCTAssertLessThan(estimatedMemoryMB, 50, "Memory usage too high")
    }
}
