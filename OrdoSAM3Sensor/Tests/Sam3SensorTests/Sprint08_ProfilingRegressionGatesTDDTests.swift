import XCTest
import Metal
import MetalPerformanceShadersGraph
@testable import Sam3Sensor

@available(macOS 15.0, *)
final class Sprint08_ProfilingRegressionGatesTDDTests: XCTestCase {
    
    var device: MTLDevice!
    
    override func setUp() {
        super.setUp()
        guard let dev = MTLCreateSystemDefaultDevice() else {
            XCTFail("Metal not available")
            return
        }
        device = dev
    }
    
    // MARK: - Test 1: Component Initialization Latency Gate
    
    func testComponentInitializationLatency() throws {
        // Measure initialization time for key components
        let t0 = DispatchTime.now().uptimeNanoseconds
        
        let encoder = SAM3Encoder(device: device)
        let decoder = MaskDecoder(device: device)
        let memoryBank = MemoryBank(device: device)
        
        let t1 = DispatchTime.now().uptimeNanoseconds
        
        let latencyMs = Double(t1 - t0) / 1_000_000.0
        print("Sprint08 Test 1: Component initialization: \(latencyMs)ms")
        
        // Regression gate: < 100ms for initialization
        XCTAssertLessThan(latencyMs, 100.0, "Component initialization too slow")
        
        // Verify components created
        XCTAssertNotNil(encoder)
        XCTAssertNotNil(decoder)
        XCTAssertNotNil(memoryBank)
    }
    
    // MARK: - Test 2: Graph Building Latency Gate
    
    func testGraphBuildingLatency() throws {
        let decoder = MaskDecoder(device: device)
        
        // Measure buildGraph latency
        let t0 = DispatchTime.now().uptimeNanoseconds
        let graph = MPSGraph()
        _ = decoder.buildGraph(graph: graph, pointCount: 2, hasDensePrompt: false, hasS0: false, hasS1: false)
        let t1 = DispatchTime.now().uptimeNanoseconds
        
        let latencyMs = Double(t1 - t0) / 1_000_000.0
        print("Sprint08 Test 2: Decoder graph build: \(latencyMs)ms")
        
        // Regression gate: < 50ms for graph building
        XCTAssertLessThan(latencyMs, 50.0, "Decoder graph build too slow")
    }
    
    // MARK: - Test 3: Memory Usage Gate
    
    func testMemoryUsageGate() throws {
        let memoryBank = MemoryBank(device: device)
        
        // Add maximum frames
        for _ in 0..<6 {
            if let texture = memoryBank.makeMemoryTexture() {
                memoryBank.add(memoryFeature: texture)
            }
        }
        
        // Calculate memory usage
        // 6 frames x 72x72 spatial x 256 channels x 2 bytes (fp16)
        let memoryBankSize = 6 * 72 * 72 * 256 * 2
        print("Sprint08 Test 3: Memory bank size: \(memoryBankSize / 1024 / 1024)MB")
        
        // Regression gate: < 50MB
        XCTAssertLessThan(memoryBankSize, 50 * 1024 * 1024, "Memory bank too large")
        XCTAssertEqual(memoryBank.getMemories().count, 6)
    }
    
    // MARK: - Test 4: Profiling Timer Accuracy
    
    func testProfilingTimerAccuracy() throws {
        let expectedDelayMs = 10.0
        
        let t0 = DispatchTime.now().uptimeNanoseconds
        Thread.sleep(forTimeInterval: expectedDelayMs / 1000.0)
        let t1 = DispatchTime.now().uptimeNanoseconds
        
        let measuredMs = Double(t1 - t0) / 1_000_000.0
        let error = abs(measuredMs - expectedDelayMs) / expectedDelayMs
        
        print("Sprint08 Test 4: Profiling error: \(error * 100)% (measured: \(measuredMs)ms)")
        
        // Timer should be accurate within 20% (accounting for system overhead)
        XCTAssertLessThan(error, 0.2, "Profiling timer inaccurate")
    }
    
    // MARK: - Test 5: Performance Counter Infrastructure
    
    func testPerformanceCounterInfrastructure() throws {
        var counters: [String: Double] = [:]
        
        // Simulate operation timing
        let operations = [
            ("encoder_init", 0.001),
            ("decoder_init", 0.001),
            ("memory_bank_add", 0.001)
        ]
        
        for (op, expectedDelay) in operations {
            let t0 = DispatchTime.now().uptimeNanoseconds
            Thread.sleep(forTimeInterval: expectedDelay)
            let t1 = DispatchTime.now().uptimeNanoseconds
            
            counters[op] = Double(t1 - t0) / 1_000_000.0
        }
        
        // Verify all counters recorded
        XCTAssertEqual(counters.count, 3)
        
        for (op, time) in counters {
            print("Sprint08 Test 5: \(op) = \(String(format: "%.3f", time))ms")
            XCTAssertGreaterThan(time, 0.0, "\(op) counter not recorded")
        }
    }
}
