import XCTest
import Metal
@testable import Sam3Sensor

final class Sprint07_QuantizationTDDTests: XCTestCase {
    
    var device: MTLDevice!
    
    override func setUp() {
        super.setUp()
        guard let dev = MTLCreateSystemDefaultDevice() else {
            XCTFail("Metal not available")
            return
        }
        device = dev
    }
    
    // MARK: - Test 1: Quantization Correctness
    
    func testQuantizationCorrectness() {
        // Create test data
        let count = 1000
        var original = [Float](repeating: 0, count: count)
        for i in 0..<count {
            original[i] = Float.random(in: -1.0...1.0)
        }
        
        // Quantize to 8-bit
        let (quantized, scale, zeroPoint) = quantize8bit(original)
        
        // Dequantize
        let dequantized = dequantize8bit(quantized, scale: scale, zeroPoint: zeroPoint)
        
        // Compute error
        var mse: Double = 0.0
        for i in 0..<count {
            let diff = Double(original[i] - dequantized[i])
            mse += diff * diff
        }
        mse /= Double(count)
        
        print("Sprint07 Test 1: Quantization MSE: \(mse)")
        XCTAssertLessThan(mse, 0.01, "Quantization error too high")
    }
    
    // MARK: - Test 2: Memory Reduction
    
    func testMemoryReduction() {
        let count = 10000
        let fp32Data = [Float](repeating: 0.5, count: count)
        let fp32Size = count * 4 // 4 bytes per float
        
        let (int8Data, _, _) = quantize8bit(fp32Data)
        let int8Size = int8Data.count * 1 // 1 byte per int8
        
        let reduction = Double(fp32Size - int8Size) / Double(fp32Size)
        
        print("Sprint07 Test 2: Memory reduction: \(reduction * 100)%")
        XCTAssertGreaterThan(reduction, 0.7, "Should achieve >70% reduction")
    }
    
    // MARK: - Test 3: Quantization Range
    
    func testQuantizationRange() {
        // Test edge cases
        let testValues: [Float] = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        for value in testValues {
            let (quantized, scale, zeroPoint) = quantize8bit([value])
            let dequantized = dequantize8bit(quantized, scale: scale, zeroPoint: zeroPoint)
            
            let error = abs(value - dequantized[0])
            XCTAssertLessThan(error, 0.01, "Edge case \(value) error too high")
        }
        
        print("Sprint07 Test 3: Edge cases verified")
    }
    
    // MARK: - Test 4: Batch Quantization
    
    func testBatchQuantization() {
        // Test quantizing large batch
        let count = 100000
        var data = [Float](repeating: 0, count: count)
        for i in 0..<count {
            data[i] = Float.random(in: -1.0...1.0)
        }
        
        let t0 = DispatchTime.now().uptimeNanoseconds
        let (quantized, scale, zeroPoint) = quantize8bit(data)
        let t1 = DispatchTime.now().uptimeNanoseconds
        
        let timeMs = Double(t1 - t0) / 1_000_000.0
        print("Sprint07 Test 4: Quantized \(count) values in \(timeMs)ms")
        
        XCTAssertLessThan(timeMs, 10.0, "Quantization too slow")
        XCTAssertEqual(quantized.count, count)
    }
    
    // MARK: - Test 5: Symmetric vs Asymmetric
    
    func testSymmetricQuantization() {
        let data: [Float] = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        // Symmetric quantization (zero point = 0)
        let (quantized, scale, zeroPoint) = quantize8bitSymmetric(data)
        
        XCTAssertEqual(zeroPoint, 0, "Symmetric should have zero point = 0")
        
        let dequantized = dequantize8bit(quantized, scale: scale, zeroPoint: zeroPoint)
        
        // Verify symmetry
        for i in 0..<data.count {
            let error = abs(data[i] - dequantized[i])
            XCTAssertLessThan(error, 0.02, "Symmetric quantization error")
        }
        
        print("Sprint07 Test 5: Symmetric quantization verified")
    }
    
    // MARK: - Helper Functions
    
    func quantize8bit(_ data: [Float]) -> (quantized: [Int8], scale: Float, zeroPoint: Int8) {
        guard !data.isEmpty else { return ([], 0, 0) }
        
        let minVal = data.min()!
        let maxVal = data.max()!
        
        // Asymmetric quantization
        let scale = (maxVal - minVal) / 255.0
        let zeroPoint = Int8(-minVal / scale)
        
        let quantized = data.map { value -> Int8 in
            let q = Int((value / scale) + Float(zeroPoint))
            return Int8(max(-128, min(127, q)))
        }
        
        return (quantized, scale, zeroPoint)
    }
    
    func quantize8bitSymmetric(_ data: [Float]) -> (quantized: [Int8], scale: Float, zeroPoint: Int8) {
        guard !data.isEmpty else { return ([], 0, 0) }
        
        let absMax = data.map { abs($0) }.max()!
        let scale = absMax / 127.0
        
        let quantized = data.map { value -> Int8 in
            let q = Int(value / scale)
            return Int8(max(-128, min(127, q)))
        }
        
        return (quantized, scale, 0)
    }
    
    func dequantize8bit(_ quantized: [Int8], scale: Float, zeroPoint: Int8) -> [Float] {
        return quantized.map { q in
            (Float(q) - Float(zeroPoint)) * scale
        }
    }
}
