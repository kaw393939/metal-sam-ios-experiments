#!/usr/bin/env swift

// ANE Latency Benchmark for SAM3 Early Blocks
// Usage: swift Scripts/benchmark_ane_latency.swift

import Foundation
import CoreML

print("🔥 SAM3 ANE Latency Benchmark")
print(String(repeating: "=", count: 60))

let modelPath = "/Users/kwilliams/Projects/Sam3/OrdoSAM3Sensor/CoreML/SAM3EarlyBlocks.mlpackage"
let modelURL = URL(fileURLWithPath: modelPath)

do {
    // Compile model
    print("📦 Compiling Core ML model...")
    let compiledURL = try MLModel.compileModel(at: modelURL)
    print("✅ Compiled to: \(compiledURL.path)")
    
    // Load model with ANE targeting
    let config = MLModelConfiguration()
    config.computeUnits = .cpuAndNeuralEngine
    let model = try MLModel(contentsOf: compiledURL, configuration: config)
    print("✅ Model loaded for ANE")
    print()
    
    // Create test input: [1, 5184, 1024] FP16
    let inputArray = try MLMultiArray(shape: [1, 5184, 1024], dataType: .float16)
    let inputDict: [String: Any] = ["input": inputArray]
    
    // Warmup (3 iterations)
    print("🔥 Warming up ANE (3 iterations)...")
    for i in 1...3 {
        let start = Date()
        _ = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: inputDict))
        let elapsed = Date().timeIntervalSince(start)
        print("  Warmup \(i): \(String(format: "%.1f", elapsed * 1000))ms")
    }
    print()
    
    // Benchmark (20 iterations)
    print("📊 Benchmarking ANE inference (20 iterations)...")
    var latencies: [Double] = []
    
    for i in 1...20 {
        let start = Date()
        _ = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: inputDict))
        let elapsed = Date().timeIntervalSince(start)
        latencies.append(elapsed * 1000) // Convert to ms
        
        if i % 5 == 0 {
            print("  Progress: \(i)/20 iterations complete")
        }
    }
    
    // Calculate statistics
    let avgLatency = latencies.reduce(0, +) / Double(latencies.count)
    let minLatency = latencies.min() ?? 0
    let maxLatency = latencies.max() ?? 0
    let sortedLatencies = latencies.sorted()
    let p50 = sortedLatencies[sortedLatencies.count / 2]
    let p95 = sortedLatencies[Int(Double(sortedLatencies.count) * 0.95)]
    
    print()
    print(String(repeating: "=", count: 60))
    print("📈 RESULTS (M3 Air ANE)")
    print(String(repeating: "=", count: 60))
    print("Average Latency: \(String(format: "%.1f", avgLatency))ms")
    print("Minimum Latency: \(String(format: "%.1f", minLatency))ms")
    print("Maximum Latency: \(String(format: "%.1f", maxLatency))ms")
    print("Median (P50):    \(String(format: "%.1f", p50))ms")
    print("P95:             \(String(format: "%.1f", p95))ms")
    print(String(repeating: "=", count: 60))
    
    // Performance assessment
    let target = 20.0 // ms
    print()
    if avgLatency < target {
        print("✅ PASSED: Blocks 0-8 latency is \(String(format: "%.1f", avgLatency))ms (target: < \(target)ms)")
        let speedup = 45.0 / avgLatency // Estimated GPU baseline is ~45ms
        print("🚀 Estimated speedup vs GPU: \(String(format: "%.1f", speedup))x")
    } else {
        print("⚠️  WARNING: Latency \(String(format: "%.1f", avgLatency))ms exceeds target of \(target)ms")
    }
    
    // Save results
    let resultsPath = "/Users/kwilliams/Projects/Sam3/OrdoSAM3Sensor/ane_latency_results.txt"
    let resultsContent = """
    SAM3 ANE Latency Benchmark Results
    Generated: \(Date())
    
    Model: SAM3EarlyBlocks (blocks 0-8)
    Device: M3 Air
    Compute: Apple Neural Engine
    Input: [1, 5184, 1024] FP16
    
    Statistics (20 iterations):
    - Average: \(String(format: "%.1f", avgLatency))ms
    - Minimum: \(String(format: "%.1f", minLatency))ms
    - Maximum: \(String(format: "%.1f", maxLatency))ms
    - Median:  \(String(format: "%.1f", p50))ms
    - P95:     \(String(format: "%.1f", p95))ms
    
    Individual measurements:
    \(latencies.enumerated().map { "  [\($0.offset + 1)]: \(String(format: "%.1f", $0.element))ms" }.joined(separator: "\n"))
    """
    
    try resultsContent.write(toFile: resultsPath, atomically: true, encoding: .utf8)
    print()
    print("💾 Results saved to: \(resultsPath)")
    
} catch {
    print("❌ Benchmark failed: \(error)")
    exit(1)
}
