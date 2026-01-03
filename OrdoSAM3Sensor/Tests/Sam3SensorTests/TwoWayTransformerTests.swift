
import XCTest
import Metal
@testable import Sam3Sensor

final class TwoWayTransformerTests: XCTestCase {
    var device: MTLDevice!
    var transformer: TwoWayTransformer!
    
    override func setUpWithError() throws {
        device = MTLCreateSystemDefaultDevice()
        if device == nil {
            throw XCTSkip("Metal device not available")
        }
        transformer = TwoWayTransformer(device: device)
    }
    
    func testInit() {
        XCTAssertNotNil(transformer)
    }
    
    func testForwardShape() throws {
        // Mock inputs
        // Image: 64x64 = 4096 tokens, dim 256
        let imageSeqLen = 4096
        let dim = 256
        let pointCount = 5
        
        // Random data
        let imageBytes = imageSeqLen * dim * 4
        let pointBytes = pointCount * dim * 4
        
        let imageBuf = device.makeBuffer(length: imageBytes, options: .storageModeShared)!
        let pointBuf = device.makeBuffer(length: pointBytes, options: .storageModeShared)!
        
        let imagePEBuf = device.makeBuffer(length: 1 * 5184 * 256 * 4, options: .storageModeShared)!
        let (outPoints, outImage) = try transformer.forward(imageEmbeddings: imageBuf, imagePE: imagePEBuf, pointEmbeddings: pointBuf)
        
        XCTAssertNotNil(outPoints)
        XCTAssertNotNil(outImage)
        XCTAssertEqual(outPoints.length, pointBytes)
        XCTAssertEqual(outImage.length, imageBytes)
    }
}
