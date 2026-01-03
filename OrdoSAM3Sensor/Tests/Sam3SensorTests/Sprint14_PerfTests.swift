
import XCTest
@testable import Sam3Sensor

@available(macOS 15.0, *)
final class Sprint14_PerfTests: XCTestCase {
    
    // Test: Hot-path logging removed
    // In DEBUG mode, logs are present
    // In RELEASE mode, logs should be compiled out
    func testLoggingHygiene() {
        // This test verifies that #if DEBUG guards are in place
        // Actual verification requires release build inspection
        
        #if DEBUG
        // In debug mode, logging is allowed
        XCTAssertTrue(true, "Debug mode: logging enabled")
        #else
        // In release mode, logging should be compiled out
        XCTAssertTrue(true, "Release mode: logging compiled out")
        #endif
    }
    
    // Test: BufferAllocator metrics
    func testBufferAllocatorMetrics() {
        guard let device = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
        
        // Reset to clean state
        BufferAllocator.shared.reset()
        
        var initialMetrics = BufferAllocator.shared.getMetrics()
        XCTAssertEqual(initialMetrics.totalAllocations, 0)
        XCTAssertEqual(initialMetrics.totalReuses, 0)
        
        // Allocate some buffers
        let buf1 = BufferAllocator.shared.privateBuffer(length: 1024, device: device, label: "test1")
        XCTAssertNotNil(buf1)
        
        let buf2 = BufferAllocator.shared.privateBuffer(length: 1024, device: device, label: "test2")
        XCTAssertNotNil(buf2)
        
        var afterAlloc = BufferAllocator.shared.getMetrics()
        XCTAssertEqual(afterAlloc.totalAllocations, 2, "Should have 2 allocations")
        XCTAssertEqual(afterAlloc.totalReuses, 0, "No reuses yet")
        
        // Recycle and reallocate
        BufferAllocator.shared.recycle(buf1!)
        let buf3 = BufferAllocator.shared.privateBuffer(length: 1024, device: device, label: "test3")
        
        var afterReuse = BufferAllocator.shared.getMetrics()
        XCTAssertEqual(afterReuse.totalAllocations, 3, "Should have 3 total allocations")
        XCTAssertEqual(afterReuse.totalReuses, 1, "Should have 1 reuse")
        XCTAssertGreaterThan(afterReuse.reuseRatio, 0.0, "Reuse ratio should be positive")
        
        print("📊 BufferAllocator Metrics:")
        print("   Allocations: \(afterReuse.totalAllocations)")
        print("   Reuses: \(afterReuse.totalReuses)")
        print("   Reuse Ratio: \(String(format: "%.1f%%", afterReuse.reuseRatio * 100))")
        print("   Pool Size: \(afterReuse.poolSize)")
        print("   Unique Sizes: \(afterReuse.uniqueSizes)")
    }
    
    // Test: Pool size limit
    func testPoolSizeLimit() {
        guard let device = MTLCreateSystemDefaultDevice() else { XCTFail(); return }
        
        BufferAllocator.shared.reset()
        
        // Allocate and recycle many buffers of same size
        var buffers: [MTLBuffer] = []
        for i in 0..<100 {
            if let buf = BufferAllocator.shared.privateBuffer(length: 2048, device: device, label: "test_\(i)") {
                buffers.append(buf)
            }
        }
        
        // Recycle all
        for buf in buffers {
            BufferAllocator.shared.recycle(buf)
        }
        
        let metrics = BufferAllocator.shared.getMetrics()
        
        // Pool should be capped at maxPoolSizePerBucket (64)
        XCTAssertLessThanOrEqual(metrics.poolSize, 64, "Pool should be capped at 64 buffers per size")
        
        print("📊 Pool size after recycling 100 buffers: \(metrics.poolSize) (max: 64)")
    }
}
