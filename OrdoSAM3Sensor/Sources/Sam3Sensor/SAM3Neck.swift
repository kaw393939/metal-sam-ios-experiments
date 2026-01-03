//
//  SAM3Neck.swift
//  SAM3Metal
//
//  Implements Sam3DualViTDetNeck (SimpleFPN) logic
//  Upsamples ViT output to generate high-resolution feature maps
//

import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

@available(macOS 15.0, *)
public class SAM3Neck {
    let device: MTLDevice

    private let config: SAM3EncoderConfig
    
    // Weights
    // convs[0] (Scale 4.0): DConv(s2) -> GELU -> DConv(s2) -> 1x1 -> 3x3
    // convs[1] (Scale 2.0): DConv(s2) -> 1x1 -> 3x3
    // convs[2] (Scale 1.0): 1x1 -> 3x3
    
    // Scale 4.0 (Block 0)
    var s4_dconv0_w: MTLBuffer?
    var s4_dconv0_b: MTLBuffer?
    var s4_dconv1_w: MTLBuffer?
    var s4_dconv1_b: MTLBuffer?
    var s4_conv1_w: MTLBuffer?
    var s4_conv1_b: MTLBuffer?
    var s4_conv3_w: MTLBuffer?
    var s4_conv3_b: MTLBuffer?
    
    // Scale 2.0 (Block 1)
    var s2_dconv0_w: MTLBuffer?
    var s2_dconv0_b: MTLBuffer?
    var s2_conv1_w: MTLBuffer?
    var s2_conv1_b: MTLBuffer?
    var s2_conv3_w: MTLBuffer?
    var s2_conv3_b: MTLBuffer?
    
    // Scale 1.0 (Block 2) (Low Res Feature)
    var s1_conv1_w: MTLBuffer?
    var s1_conv1_b: MTLBuffer?
    var s1_conv3_w: MTLBuffer?
    var s1_conv3_b: MTLBuffer?
    
    private let enableHalfPrecision: Bool

    public convenience init(device: MTLDevice, enableHalfPrecision: Bool = true) {
        self.init(device: device, config: .sam3Checkpoint, enableHalfPrecision: enableHalfPrecision)
    }

    public init(device: MTLDevice, config: SAM3EncoderConfig, enableHalfPrecision: Bool = true) {
        self.device = device
        self.config = config
        self.enableHalfPrecision = enableHalfPrecision
    }
    
    public func loadWeights(_ weights: [String: Data]) {
        let prefix = "backbone.vision_backbone.convs"
        
        // Scale 4.0 (Index 0)
        // dconv_2x2_0
        s4_dconv0_w = weights.buffer(for: "\(prefix).0.dconv_2x2_0.weight", device: device)
        s4_dconv0_b = weights.buffer(for: "\(prefix).0.dconv_2x2_0.bias", device: device)
        // dconv_2x2_1
        s4_dconv1_w = weights.buffer(for: "\(prefix).0.dconv_2x2_1.weight", device: device)
        s4_dconv1_b = weights.buffer(for: "\(prefix).0.dconv_2x2_1.bias", device: device)
        // conv_1x1
        s4_conv1_w = weights.buffer(for: "\(prefix).0.conv_1x1.weight", device: device)
        s4_conv1_b = weights.buffer(for: "\(prefix).0.conv_1x1.bias", device: device)
        // conv_3x3
        s4_conv3_w = weights.buffer(for: "\(prefix).0.conv_3x3.weight", device: device)
        s4_conv3_b = weights.buffer(for: "\(prefix).0.conv_3x3.bias", device: device)
        
        // Scale 2.0 (Index 1)
        // dconv_2x2
        s2_dconv0_w = weights.buffer(for: "\(prefix).1.dconv_2x2.weight", device: device)
        s2_dconv0_b = weights.buffer(for: "\(prefix).1.dconv_2x2.bias", device: device)
        // conv_1x1
        s2_conv1_w = weights.buffer(for: "\(prefix).1.conv_1x1.weight", device: device)
        s2_conv1_b = weights.buffer(for: "\(prefix).1.conv_1x1.bias", device: device)
        // conv_3x3
        s2_conv3_w = weights.buffer(for: "\(prefix).1.conv_3x3.weight", device: device)
        s2_conv3_b = weights.buffer(for: "\(prefix).1.conv_3x3.bias", device: device)
        
        // Scale 1.0 (Index 2)
        // conv_1x1
        s1_conv1_w = weights.buffer(for: "\(prefix).2.conv_1x1.weight", device: device)
        s1_conv1_b = weights.buffer(for: "\(prefix).2.conv_1x1.bias", device: device)
        // conv_3x3
        s1_conv3_w = weights.buffer(for: "\(prefix).2.conv_3x3.weight", device: device)
        s1_conv3_b = weights.buffer(for: "\(prefix).2.conv_3x3.bias", device: device)
        
        print("SAM3Neck: ✅ Loaded weights (S4: \(s4_conv3_w != nil), S2: \(s2_conv3_w != nil), S1: \(s1_conv3_w != nil))")
    }
    
    public func buildGraph(
        input: MPSGraphTensor,
        graph: MPSGraph,
        computeDT: MPSDataType = .float32
    ) -> (featS0: MPSGraphTensor, featS1: MPSGraphTensor, featS2: MPSGraphTensor, placeholders: [String: MPSGraphTensor]) {
        var phs: [String: MPSGraphTensor] = [:]

        let gridSize = config.checkpointGridSize
        let embedDim = config.embedDim
        let s2Size = gridSize * 2
        let s4Size = gridSize * 4
        
        // Helper: GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        func gelu(_ x: MPSGraphTensor, name: String) -> MPSGraphTensor {
            let pointFive = graph.constant(0.5, dataType: computeDT)
            let one = graph.constant(1.0, dataType: computeDT)
            let sqrtTwo = graph.constant(1.41421356, dataType: computeDT)
            let div = graph.division(x, sqrtTwo, name: "\(name)_div")
            let erf = graph.erf(with: div, name: "\(name)_erf")
            let onePlusErf = graph.addition(one, erf, name: "\(name)_add")
            let halfX = graph.multiplication(pointFive, x, name: "\(name)_half")
            return graph.multiplication(halfX, onePlusErf, name: "\(name)/gelu")
        }
        
        let desc = MPSGraphConvolution2DOpDescriptor(strideInX: 2, strideInY: 2, dilationRateInX: 1, dilationRateInY: 1, groups: 1, paddingLeft: 0, paddingRight: 0, paddingTop: 0, paddingBottom: 0, paddingStyle: .TF_SAME, dataLayout: .NHWC, weightsLayout: .OIHW)!
        let desc1x1 = MPSGraphConvolution2DOpDescriptor(strideInX: 1, strideInY: 1, dilationRateInX: 1, dilationRateInY: 1, groups: 1, paddingLeft: 0, paddingRight: 0, paddingTop: 0, paddingBottom: 0, paddingStyle: .explicit, dataLayout: .NHWC, weightsLayout: .OIHW)!
        let desc3x3 = MPSGraphConvolution2DOpDescriptor(strideInX: 1, strideInY: 1, dilationRateInX: 1, dilationRateInY: 1, groups: 1, paddingLeft: 1, paddingRight: 1, paddingTop: 1, paddingBottom: 1, paddingStyle: .explicit, dataLayout: .NHWC, weightsLayout: .OIHW)!

        // Input Reshape
        // Input Reshape
        // Input is Float32. No cast needed.
        let xInput = input
        
        // Helper: Create placeholder (F16/F32)
        func loadWeight(_ name: String, shape: [NSNumber]) -> MPSGraphTensor {
            let ph = graph.placeholder(shape: shape, dataType: computeDT, name: name)
            phs[name] = ph
            return ph
        }

        let inputSpatial = graph.reshape(
            xInput,
            shape: [1, NSNumber(value: gridSize), NSNumber(value: gridSize), NSNumber(value: embedDim)],
            name: "neck/input_reshape"
        )
        
        // --- Scale 4.0 (S0) ---
        // 1. DConv0: 1024 -> 512
        let s4_d0_w_ph = loadWeight("neck/s4/d0/w", shape: [NSNumber(value: embedDim), 512, 2, 2])
        let s4_d0_b_ph = loadWeight("neck/s4/d0/b", shape: [1, 1, 1, 512])
        
        // Note outputShape calculation: grid*2
        var x4 = graph.convolutionTranspose2D(
            inputSpatial,
            weights: s4_d0_w_ph,
            outputShape: [1, NSNumber(value: s2Size), NSNumber(value: s2Size), 512],
            descriptor: desc,
            name: "neck/s4/d0"
        )
        x4 = graph.addition(x4, s4_d0_b_ph, name: "neck/s4/d0/add")
        x4 = gelu(x4, name: "neck/s4/d0/gelu")
        
        // DConv 2: 512->256
        let s4_d1_w_ph = loadWeight("neck/s4/d1/w", shape: [512, 256, 2, 2])
        let s4_d1_b_ph = loadWeight("neck/s4/d1/b", shape: [1, 1, 1, 256])
        
        // 2. DConv1: 512 -> 256

        
        // Output Shape of S4 DConv1: grid*4
        x4 = graph.convolutionTranspose2D(
            x4,
            weights: s4_d1_w_ph,
            outputShape: [1, NSNumber(value: s4Size), NSNumber(value: s4Size), 256],
            descriptor: desc,
            name: "neck/s4/d1"
        )
        x4 = graph.addition(x4, s4_d1_b_ph, name: "neck/s4/d1/add")
        
        // 3. Conv 1x1: 256 -> 256
        // 3. Conv 1x1: 256 -> 256
        let s4_c1_w_ph = loadWeight("neck/s4/c1/w", shape: [256, 256, 1, 1])
        let s4_c1_b_ph = loadWeight("neck/s4/c1/b", shape: [1, 1, 1, 256])
        
        x4 = graph.convolution2D(x4, weights: s4_c1_w_ph, descriptor: desc1x1, name: "neck/s4/c1")
        x4 = graph.addition(x4, s4_c1_b_ph, name: "neck/s4/c1/add")
        
        // 4. Conv 3x3: 256 -> 256
        let s4_c3_w_ph = loadWeight("neck/s4/c3/w", shape: [256, 256, 3, 3])
        let s4_c3_b_ph = loadWeight("neck/s4/c3/b", shape: [1, 1, 1, 256])
        
        x4 = graph.convolution2D(x4, weights: s4_c3_w_ph, descriptor: desc3x3, name: "neck/s4/c3")
        let featS0 = graph.addition(x4, s4_c3_b_ph, name: "neck/s4/c3/add")
        
        // Scale 2.0 (S1)
        // DConv: 1024 -> 512
        // Scale 2.0 (S1)
        // DConv: 1024 -> 512
        let s2_d0_w_ph = loadWeight("neck/s2/d0/w", shape: [NSNumber(value: embedDim), 512, 2, 2])
        let s2_d0_b_ph = loadWeight("neck/s2/d0/b", shape: [1, 1, 1, 512])
        
        var x2 = graph.convolutionTranspose2D(
            inputSpatial,
            weights: s2_d0_w_ph,
            outputShape: [1, NSNumber(value: s2Size), NSNumber(value: s2Size), 512],
            descriptor: desc,
            name: "neck/s2/d0"
        )
        x2 = graph.addition(x2, s2_d0_b_ph, name: "neck/s2/d0/add")
        
        // Conv 1x1: 512 -> 256
        let s2_c1_w_ph = loadWeight("neck/s2/c1/w", shape: [256, 512, 1, 1])
        let s2_c1_b_ph = loadWeight("neck/s2/c1/b", shape: [1, 1, 1, 256])
        
        x2 = graph.convolution2D(x2, weights: s2_c1_w_ph, descriptor: desc1x1, name: "neck/s2/c1")
        x2 = graph.addition(x2, s2_c1_b_ph, name: "neck/s2/c1/add")
        
        // Conv 3x3: 256 -> 256
        let s2_c3_w_ph = loadWeight("neck/s2/c3/w", shape: [256, 256, 3, 3])
        let s2_c3_b_ph = loadWeight("neck/s2/c3/b", shape: [1, 1, 1, 256])
        
        x2 = graph.convolution2D(x2, weights: s2_c3_w_ph, descriptor: desc3x3, name: "neck/s2/c3")
        let featS1 = graph.addition(x2, s2_c3_b_ph, name: "neck/s2/c3/add")
        
        // Scale 1.0 (S2)
        // Conv 1x1: 1024 -> 256
        // Scale 1.0 (S2)
        // Conv 1x1: 1024 -> 256
        let s1_c1_w_ph = loadWeight("neck/s1/c1/w", shape: [256, NSNumber(value: embedDim), 1, 1])
        let s1_c1_b_ph = loadWeight("neck/s1/c1/b", shape: [1, 1, 1, 256])
        
        var x1 = graph.convolution2D(inputSpatial, weights: s1_c1_w_ph, descriptor: desc1x1, name: "neck/s1/c1")
        x1 = graph.addition(x1, s1_c1_b_ph, name: "neck/s1/c1/add")
        
        // Conv 3x3: 256 -> 256
        let s1_c3_w_ph = loadWeight("neck/s1/c3/w", shape: [256, 256, 3, 3])
        let s1_c3_b_ph = loadWeight("neck/s1/c3/b", shape: [1, 1, 1, 256])
        
        x1 = graph.convolution2D(x1, weights: s1_c3_w_ph, descriptor: desc3x3, name: "neck/s1/c3")
        let featS2 = graph.addition(x1, s1_c3_b_ph, name: "neck/s1/c3/add")
        
        return (featS0, featS1, featS2, phs)
    }
    
    public func addFeeds(placeholders: [String: MPSGraphTensor], feeds: inout [MPSGraphTensor: MPSGraphTensorData]) {
        let embedDim = config.embedDim
        func add(_ phName: String, _ buffer: MTLBuffer?, shape: [NSNumber]) {
            if let b = buffer, let ph = placeholders[phName] {
                feeds[ph] = MPSGraphTensorData(b, shape: shape, dataType: enableHalfPrecision ? .float16 : .float32)
            }
        }
        
        // Scale 4.0
        add("neck/s4/d0/w", s4_dconv0_w, shape: [NSNumber(value: embedDim), 512, 2, 2])
        add("neck/s4/d0/b", s4_dconv0_b, shape: [1, 1, 1, 512])
        add("neck/s4/d1/w", s4_dconv1_w, shape: [512, 256, 2, 2])
        add("neck/s4/d1/b", s4_dconv1_b, shape: [1, 1, 1, 256])
        add("neck/s4/c1/w", s4_conv1_w, shape: [256, 256, 1, 1])
        add("neck/s4/c1/b", s4_conv1_b, shape: [1, 1, 1, 256])
        add("neck/s4/c3/w", s4_conv3_w, shape: [256, 256, 3, 3])
        add("neck/s4/c3/b", s4_conv3_b, shape: [1, 1, 1, 256])
        
        // Scale 2.0
        add("neck/s2/d0/w", s2_dconv0_w, shape: [NSNumber(value: embedDim), 512, 2, 2])
        add("neck/s2/d0/b", s2_dconv0_b, shape: [1, 1, 1, 512])
        add("neck/s2/c1/w", s2_conv1_w, shape: [256, 512, 1, 1])
        add("neck/s2/c1/b", s2_conv1_b, shape: [1, 1, 1, 256])
        add("neck/s2/c3/w", s2_conv3_w, shape: [256, 256, 3, 3])
        add("neck/s2/c3/b", s2_conv3_b, shape: [1, 1, 1, 256])
        
        // Scale 1.0
        add("neck/s1/c1/w", s1_conv1_w, shape: [256, NSNumber(value: embedDim), 1, 1])
        add("neck/s1/c1/b", s1_conv1_b, shape: [1, 1, 1, 256])
        add("neck/s1/c3/w", s1_conv3_w, shape: [256, 256, 3, 3])
        add("neck/s1/c3/b", s1_conv3_b, shape: [1, 1, 1, 256])
    }
    
    public func randomInitialize() {
        let embedDim = config.embedDim
        func alloc(_ shape: [Int]) -> MTLBuffer {
            let count = shape.reduce(1, *)
            let bytes = count * (enableHalfPrecision ? 2 : 4)
            return device.makeBuffer(length: bytes, options: .storageModeShared)!
        }
        
        s4_dconv0_w = alloc([embedDim, 512, 2, 2])
        s4_dconv0_b = alloc([1, 1, 1, 512])
        s4_dconv1_w = alloc([512, 256, 2, 2])
        s4_dconv1_b = alloc([1, 1, 1, 256])
        s4_conv1_w = alloc([256, 256, 1, 1])
        s4_conv1_b = alloc([1, 1, 1, 256])
        s4_conv3_w = alloc([256, 256, 3, 3])
        s4_conv3_b = alloc([1, 1, 1, 256])
        
        s2_dconv0_w = alloc([embedDim, 512, 2, 2])
        s2_dconv0_b = alloc([1, 1, 1, 512])
        s2_conv1_w = alloc([256, 512, 1, 1])
        s2_conv1_b = alloc([1, 1, 1, 256])
        s2_conv3_w = alloc([256, 256, 3, 3])
        s2_conv3_b = alloc([1, 1, 1, 256])
        
        s1_conv1_w = alloc([256, embedDim, 1, 1])
        s1_conv1_b = alloc([1, 1, 1, 256])
        s1_conv3_w = alloc([256, 256, 3, 3])
        s1_conv3_b = alloc([1, 1, 1, 256])
    }
}
