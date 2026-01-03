import XCTest
import Metal
@testable import Sam3Sensor


@available(macOS 15.0, *)
final class IsolatedWeightTest: XCTestCase {
    
    func testJustLoadWeights() throws {
        print("\n🔍 Starting isolated weight loading test...")
        
        let device = MTLCreateSystemDefaultDevice()!
        print("✅ Device: \(device.name)")
        
        let wtsPath = "/Users/kwilliams/Projects/Sam3/OrdoSAM3Sensor/Tests/Artifacts/sam3_weights.wts"
        guard FileManager.default.fileExists(atPath: wtsPath) else {
            throw XCTSkip("No .wts file")
        }
        print("✅ Found .wts file")
        
        let predictor = SAM3Predictor(device: device, enableHalfPrecision: true)
        print("✅ Predictor created")
        
        print("📦 Loading weights...")
        let npzPath = wtsPath.replacingOccurrences(of: ".wts", with: ".npz")
        try predictor.loadWeights(from: URL(fileURLWithPath: npzPath))
        print("✅ WEIGHTS LOADED!")
        
        XCTAssert(true, "Success!")
    }
}
