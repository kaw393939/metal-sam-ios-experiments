
import XCTest
import Metal
@testable import Sam3Sensor

final class MaskDecoderTests: XCTestCase {
    var device: MTLDevice!
    var decoder: MaskDecoder!
    
    override func setUpWithError() throws {
        device = MTLCreateSystemDefaultDevice()
        if device == nil {
            throw XCTSkip("Metal device not available")
        }
        decoder = MaskDecoder(device: device)
    }
    
    func testInit() {
        XCTAssertNotNil(decoder)
    }
    
    func testForwardShape() throws {
        let imageSeqLen = 5184 // 72x72
        let dim = 256
        let pointCount = 5

        let bytesPerElement = 2 // Float16 (default path)
        let imageBytes = imageSeqLen * dim * bytesPerElement
        let pointBytes = pointCount * dim * bytesPerElement
        
        let imageBuf = device.makeBuffer(length: imageBytes, options: .storageModeShared)!
        let pointBuf = device.makeBuffer(length: pointBytes, options: .storageModeShared)!

        let imagePEBuf = device.makeBuffer(length: 1 * 5184 * 256 * bytesPerElement, options: .storageModeShared)!
        let (masks, iou) = try decoder.forward(imageEmbeddings: imageBuf, imagePE: imagePEBuf, pointEmbeddings: pointBuf, commandQueue: device.makeCommandQueue()!)
        
        XCTAssertNotNil(masks)
        XCTAssertNotNil(iou)
    }
}
