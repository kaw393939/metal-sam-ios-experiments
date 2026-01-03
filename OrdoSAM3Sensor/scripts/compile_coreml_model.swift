#!/usr/bin/env swift

// Compile Core ML model for deployment
// Usage: swift Scripts/compile_coreml_model.swift

import Foundation
import CoreML

let modelPath = "/Users/kwilliams/Projects/Sam3/OrdoSAM3Sensor/CoreML/SAM3EarlyBlocks.mlpackage"
let modelURL = URL(fileURLWithPath: modelPath)

print("Compiling Core ML model...")
print("Source: \(modelPath)")

do {
    let compiledURL = try MLModel.compileModel(at: modelURL)
    print("✅ Compiled successfully!")
    print("Output: \(compiledURL.path)")
} catch {
    print("❌ Compilation failed: \(error)")
    exit(1)
}
