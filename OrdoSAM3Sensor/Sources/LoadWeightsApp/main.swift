
import Foundation
import Metal
import MetalPerformanceShaders
import Sam3Sensor

@available(macOS 15.0, *)
func runBenchmark() {
    fputs("=== HYBRID ViT ENCODER BENCHMARK APP ===\n", stderr)
    
    guard let device = MTLCreateSystemDefaultDevice() else {
        fputs("Error: No Metal Device\n", stderr)
        exit(1)
    }
    fputs("DEBUG: Device created\n", stderr)
    
    guard let commandQueue = device.makeCommandQueue() else {
        fputs("Error: No Command Queue\n", stderr)
        exit(1)
    }
    fputs("DEBUG: Queue created\n", stderr)
    let weightsPath = "weights.safetensors"
    
    do {
        fputs("Creating Encoder...\n", stderr)
        // Using "Base" variant params for stress test (ViT-B)
        let encoder = try HybridViTEncoder(
            device: device,
            embedDim: 768,
            depth: 12,
            numHeads: 12,
            mlpDim: 3072,
            patchSize: 16,
            imageSize: 1024
        )
        fputs("DEBUG: Encoder created\n", stderr)
        
        if FileManager.default.fileExists(atPath: weightsPath) {
            fputs("Loading weights from \(weightsPath)...\n", stderr)
            try encoder.loadWeights(url: URL(fileURLWithPath: weightsPath))
        } else {
            fputs("Weights not found, using random init...\n", stderr)
            encoder.randomInitialize()
        }
        fputs("DEBUG: Weights initialized\n", stderr)
        
        fputs("Creating Input Texture...\n", stderr)
        let textureDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: 1024, height: 1024, mipmapped: false)
        textureDesc.usage = [.shaderRead, .shaderWrite]
        guard let inputTexture = device.makeTexture(descriptor: textureDesc) else {
            fputs("Error: Texture creation failed\n", stderr)
            exit(1)
        }
        
        // Warmup
        fputs("Warmup (3 iterations)...\n", stderr)
        for i in 0..<3 {
            guard let commandBuffer = commandQueue.makeCommandBuffer() else { exit(1) }
            _ = try encoder.forward(image: inputTexture, commandBuffer: commandBuffer)
            // Buffer already committed inside encoder due to MLX sync
            fputs("  Warmup \(i+1) done\n", stderr)
        }
        
        // Measure
        fputs("Measuring (10 iterations)...\n", stderr)
        let start = CFAbsoluteTimeGetCurrent()
        let iterations = 10
        
        for i in 0..<iterations {
            guard let commandBuffer = commandQueue.makeCommandBuffer() else { exit(1) }
            _ = try encoder.forward(image: inputTexture, commandBuffer: commandBuffer)
            if (i+1) % 5 == 0 { fputs("  Iter \(i+1) done\n", stderr) }
        }
        
        let duration = CFAbsoluteTimeGetCurrent() - start
        let avgTime = duration / Double(iterations)
        let fps = 1.0 / avgTime
        
        fputs("\nRESULTS:\n", stderr)
        fputs("Average Time: \(String(format: "%.4f", avgTime)) s\n", stderr)
        fputs("FPS: \(String(format: "%.2f", fps))\n", stderr)
        fputs("Total Time: \(String(format: "%.4f", duration)) s\n", stderr)
        
    } catch {
        fputs("Error: \(error)\n", stderr)
        exit(1)
    }
}

if #available(macOS 15.0, *) {
    fputs("DEBUG: App Start\n", stderr)
    // Check if MLX import affects it
    fputs("DEBUG: Imported MLX\n", stderr)
    runBenchmark()
} else {
    print("Requires macOS 15.0")
}
