import XCTest
import Metal
@testable import Sam3Sensor

@available(macOS 15.0, *)
final class MinimalPredictorTest: XCTestCase {
    
    func testCreateDevice() {
        print("Test 1: Creating Metal device...")
        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("No Metal device")
            return
        }
        print("✅ Metal device created: \(device.name)")
        XCTAssert(true)
    }
    
    func testCreateCommandQueue() {
        print("Test 2: Creating command queue...")
        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("No Metal device")
            return
        }
        guard let queue = device.makeCommandQueue() else {
            XCTFail("No command queue")
            return
        }
        print("✅ Command queue created")
        XCTAssert(true)
    }
    
    func testCreateEncoder() {
        print("Test 3: Creating SAM3Encoder...")
        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("No Metal device")
            return
        }
        
        do {
            let encoder = SAM3Encoder(device: device)
            print("✅ SAM3Encoder created")
            XCTAssert(true)
        } catch {
            print("❌ SAM3Encoder creation failed: \(error)")
            XCTFail("Encoder creation failed")
        }
    }
    
    func testCreatePredictor() {
        print("Test 4: Creating SAM3Predictor...")
        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("No Metal device")
            return
        }
        
        print("  - Creating predictor...")
        let predictor = SAM3Predictor(device: device)
        print("✅ SAM3Predictor created")
        XCTAssert(true)
    }
}
