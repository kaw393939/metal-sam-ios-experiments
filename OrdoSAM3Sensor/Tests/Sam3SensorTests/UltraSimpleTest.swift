import XCTest
import Metal
@testable import Sam3Sensor

// NO @available check - let's see if that's the issue
final class UltraSimpleTest: XCTestCase {
    
    func testBasicMetal() {
        print("🔍 Ultra simple test starting...")
        let device = MTLCreateSystemDefaultDevice()
        XCTAssertNotNil(device)
        print("✅ Test passed!")
    }
}
