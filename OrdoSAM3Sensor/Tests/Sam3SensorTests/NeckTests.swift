import XCTest
import Metal
import MetalPerformanceShadersGraph
@testable import Sam3Sensor

@available(macOS 15.0, *)
final class NeckTests: XCTestCase {
    
    var device: MTLDevice!
    var neck: SAM3Neck!
    
    override func setUp() {
        device = MTLCreateSystemDefaultDevice()
        neck = SAM3Neck(device: device)
    }
    
    func testNeckProjection() {
        let graph = MPSGraph()
        
        // Input: [1, 72, 72, 1024] (SAM3 Output)
        let input = graph.placeholder(
            shape: [1, 72, 72, 1024],
            dataType: .float32,
            name: "backbone_out"
        )
        
        let (s0, s1, s2, placeholders) = neck.buildGraph(input: input, graph: graph)
        _ = s0
        _ = s1
        
        XCTAssertNotNil(s2)
        // Check placeholders key existence?
        // e.g. placeholders["neck/s4/d0/w"] but names changed. Skiping specific name check or updating it?
        // Old test checked "neck.conv1.weight". New names are "neck/s4/..."
        
        // Run with random data
        guard let queue = device.makeCommandQueue() else { return }
        
        var feeds: [MPSGraphTensor: MPSGraphTensorData] = [:]
        
        // Random input
        let inFloats = [Float](repeating: 0.1, count: 1 * 5184 * 1024) // Was 72*72*1024 in test, input shape [1, 72, 72, 1024]
        feeds[input] = MPSGraphTensorData(
            device: MPSGraphDevice(mtlDevice: device),
            data: Data(bytes: inFloats, count: inFloats.count * 4),
            shape: [1, 72, 72, 1024],
            dataType: .float32
        )
        
        // Random weights
        for (_, ph) in placeholders {
            let shape = ph.shape!.map { $0.intValue }
            let count = shape.reduce(1, *)

            switch ph.dataType {
            case .float16:
                let values = [Float16](repeating: Float16(0.01), count: count)
                let data = values.withUnsafeBytes { raw in
                    Data(bytes: raw.baseAddress!, count: raw.count)
                }
                feeds[ph] = MPSGraphTensorData(
                    device: MPSGraphDevice(mtlDevice: device),
                    data: data,
                    shape: ph.shape!,
                    dataType: .float16
                )
            default:
                let values = [Float](repeating: 0.01, count: count)
                let data = values.withUnsafeBytes { raw in
                    Data(bytes: raw.baseAddress!, count: raw.count)
                }
                feeds[ph] = MPSGraphTensorData(
                    device: MPSGraphDevice(mtlDevice: device),
                    data: data,
                    shape: ph.shape!,
                    dataType: .float32
                )
            }
        }
        
        let results = graph.run(
            with: queue,
            feeds: feeds,
            targetTensors: [s2],
            targetOperations: nil
        )
        
        guard let outData = results[s2] else {
            XCTFail("No output")
            return
        }
        
        // Check output shape: [1, 72, 72, 256]
        let outShape = outData.shape.map { $0.intValue }
        print("Neck Out: \(outShape)")
        XCTAssertEqual(outShape, [1, 72, 72, 256])
    }
}
