
import XCTest
@testable import Sam3Sensor

final class GeometryShapesTest: XCTestCase {
    func testDumpShapes() throws {
        let path = "/Users/kwilliams/Projects/ordo/Sensors/SAM3Metal/Tests/Artifacts/sam3_weights.npz"
        let loader = ModelLoader()
        let weights = try loader.load(url: URL(fileURLWithPath: path))
        
        let keysToCheck = [
            "geometry_encoder.points_direct_project.weight",
            "geometry_encoder.points_pos_enc_project.weight",
            "geometry_encoder.label_embed.weight",
            "geometry_encoder.encode.0.self_attn.in_proj_weight",
            "geometry_encoder.encode.0.cross_attn_image.in_proj_weight",
            "transformer.encoder.layers.0.self_attn.in_proj_weight"
        ]
        
        print("DEBUG: Shapes:")
        for k in keysToCheck {
            if let data = weights[k] {
                // Heuristic: assume float32. 
                let count = data.count / 4
                print("\(k) -> Elements: \(count)")
            } else {
                print("\(k) -> MISSING")
            }
        }
    }
}
