
import XCTest
@testable import Sam3Sensor

final class GeometryKeysTest: XCTestCase {
    func testDumpGeometryKeys() throws {
        let path = "/Users/kwilliams/Projects/ordo/Sensors/SAM3Metal/Tests/Artifacts/sam3_weights.npz"
        let loader = ModelLoader()
        let weights = try loader.load(url: URL(fileURLWithPath: path))
        
        // Filter for geometry_encoder
        let keys = weights.keys.filter { $0.hasPrefix("geometry_encoder.") }
        print("DEBUG: Geometry Keys Total: \(keys.count)")
        for k in keys.sorted() {
            print(k)
        }
        
        // Also check if there are any other 'encoder' related keys we missed
        let encKeys = weights.keys.filter { $0.contains("encoder") && !$0.contains("backbone") && !$0.contains("geometry") }
        print("DEBUG: Other Encoder Keys:")
        for k in encKeys.sorted() {
            print(k)
        }
    }
}
