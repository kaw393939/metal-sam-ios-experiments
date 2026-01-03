//
//  TokenPruner.swift
//  Sam3Sensor
//
//  Dynamic Token Pruning (Optimization 6)
//  Selects Top-K most significant tokens after block 2 to reduce sequence length.
//

import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

@available(macOS 15.0, *)
public final class TokenPruner {
    private let device: MTLDevice
    private let keepKInternal: Int
    private let dimInternal: Int
    private let ropeDimInternal: Int

    public var keepK: Int { keepKInternal }
    public var dim: Int { dimInternal }
    public var ropeDim: Int { ropeDimInternal }
    
    public init(device: MTLDevice, keepK: Int = 1024, dim: Int = WeightMapper.embedDim, ropeDim: Int = 64) {
        self.device = device
        self.keepKInternal = keepK
        self.dimInternal = dim
        self.ropeDimInternal = ropeDim
    }
    
    /// Prunes input based on magnitude (simple score)
    public func prune(
        input: MTLBuffer,
        ropeFreqs: MTLBuffer,
        seqLen: Int,
        batch: Int,
        commandBuffer: MTLCommandBuffer,
        recycledBuffers: inout [MTLBuffer]
    ) throws -> (MTLBuffer, MTLBuffer, MTLBuffer)? {

        let ropeDim = ropeDimInternal
        let cacheKey = "Prune_\(batch)_\(seqLen)_\(keepKInternal)_\(dimInternal)_\(ropeDim)"
        
        let (graph, placeholders, resultsTensors, executable) = CompiledGraphCache.shared.getOrCompileMulti(key: cacheKey, device: device) {
            let graph = MPSGraph()
            
            // 1. Inputs
            let inputShape: [NSNumber] = [NSNumber(value: batch), NSNumber(value: seqLen), NSNumber(value: dimInternal)]
            let inputTensor = graph.placeholder(shape: inputShape, dataType: .float16, name: "input")
            
            let ropeTensor = graph.placeholder(shape: [NSNumber(value: seqLen), NSNumber(value: ropeDim)], dataType: .float32, name: "rope")
            
            // 2. Score Tokens
            let absInput = graph.absolute(with: inputTensor, name: "abs")
            let summed = graph.reductionSum(with: absInput, axes: [2], name: "sum_scores") 
            let scores = graph.reshape(summed, shape: [NSNumber(value: batch), NSNumber(value: seqLen)], name: "scores_flat")
            
            // 3. Top K
            let topKResult = graph.topK(scores, k: keepKInternal, name: "topK")
            let topIndices = topKResult[1] 
            
            // 4. Gather Features
            let prunedFeatures = graph.gather(withUpdatesTensor: inputTensor, indicesTensor: topIndices, axis: 1, batchDimensions: 1, name: "gather_features")
            
            // 5. Gather RoPE
            // Use Gather with explicit batch dimension support
            let ropeExpanded = graph.expandDims(ropeTensor, axes: [0], name: "rope_expand")
            let ropeBatched = graph.broadcast(ropeExpanded, shape: [NSNumber(value: batch), NSNumber(value: seqLen), NSNumber(value: ropeDim)], name: "rope_broadcast")
            let prunedRoPE = graph.gather(withUpdatesTensor: ropeBatched, indicesTensor: topIndices, axis: 1, batchDimensions: 1, name: "gather_rope")
            
            return (graph, ["input": inputTensor, "rope": ropeTensor], [prunedFeatures, prunedRoPE, topIndices])
        }
        
        // --- Execution ---
        let inputShape: [NSNumber] = [NSNumber(value: batch), NSNumber(value: seqLen), NSNumber(value: dimInternal)]
        let dt: MPSDataType = .float16
        
        // Validate inputs
        guard input.length >= batch * seqLen * dimInternal * 2 else {
             print("TokenPruner: Input buffer too small")
             return nil
        }
        
        let inputData = MPSGraphTensorData(input, shape: inputShape, dataType: dt)
        let ropeData = MPSGraphTensorData(ropeFreqs, shape: [NSNumber(value: seqLen), NSNumber(value: ropeDim)], dataType: .float32)
        
        // Buffers
        guard let featsOut = BufferAllocator.shared.privateBuffer(length: batch * keepKInternal * dimInternal * 2, device: device, label: "PrunedFeats"),
              let ropeOut = BufferAllocator.shared.privateBuffer(length: batch * keepKInternal * ropeDim * 4, device: device, label: "PrunedRoPE"),
              let indicesOut = BufferAllocator.shared.privateBuffer(length: batch * keepKInternal * 4, device: device, label: "PruneIndices") else {
            return nil
        }
        
        recycledBuffers.append(featsOut)
        recycledBuffers.append(ropeOut)
        recycledBuffers.append(indicesOut)
        
        var feeds: [MPSGraphTensor : MPSGraphTensorData] = [:]
        if let p = placeholders["input"] { feeds[p] = inputData }
        if let p = placeholders["rope"] { feeds[p] = ropeData }
        
        let queue = commandBuffer.commandQueue
        
        // Sprint 01: Fail fast if compilation failed
        guard executable != nil else {
            throw SAM3Error.graphCompilationFailed("TokenPruner.prune: \(cacheKey)")
        }
        let results = try CompiledGraphCache.shared.runExecutable(key: cacheKey, queue: queue, feeds: feeds, targetTensors: resultsTensors)
        
        let prunedFeatures = resultsTensors[0]
        let prunedRoPE = resultsTensors[1]
        let topIndices = resultsTensors[2]
        
        // Export results
        if let data = results[prunedFeatures] {
            data.mpsndarray().exportData(with: commandBuffer, to: featsOut, destinationDataType: dt, offset: 0, rowStrides: nil)
        }
        if let data = results[prunedRoPE] {
            data.mpsndarray().exportData(with: commandBuffer, to: ropeOut, destinationDataType: .float32, offset: 0, rowStrides: nil)
        }
        if let data = results[topIndices] {
             data.mpsndarray().exportData(with: commandBuffer, to: indicesOut, destinationDataType: .int32, offset: 0, rowStrides: nil)
        }
        
        return (featsOut, ropeOut, indicesOut)
    }
    
    /// Restores spatial arrangement using retained indices (Optimization 6)
    /// Uses scatterND for macOS 26 compatibility instead of scatter
    public func restoreSpatial(
        pruned: MTLBuffer,
        indices: MTLBuffer,
        originalSeqLen: Int,
        batch: Int,
        commandBuffer: MTLCommandBuffer,
        recycledBuffers: inout [MTLBuffer]
    ) throws -> MTLBuffer {
        let cacheKey = "Restore_\(batch)_\(originalSeqLen)_\(keepKInternal)_\(dimInternal)_v2"
        let dt: MPSDataType = .float16

        let (graph, placeholders, resultsTensors, executable) = CompiledGraphCache.shared.getOrCompileMulti(key: cacheKey, device: device) {
            let graph = MPSGraph()

            // For macOS 26 compatibility, handle batch=1 case explicitly
            // Use a simpler gather-based approach: create zeros, then use scatterND

            let prunedShape: [NSNumber] = [NSNumber(value: batch), NSNumber(value: keepKInternal), NSNumber(value: dimInternal)]
            let prunedT = graph.placeholder(shape: prunedShape, dataType: dt, name: "pruned")

            // Indices: [batch, keepK] but we need [batch, keepK, 1] for scatterND
            let indicesT = graph.placeholder(shape: [NSNumber(value: batch), NSNumber(value: keepKInternal)], dataType: .int32, name: "indices")
            let indicesExpanded = graph.expandDims(indicesT, axis: 2, name: "indices_expand")

            // Create output shape
            let outShape: [NSNumber] = [NSNumber(value: batch), NSNumber(value: originalSeqLen), NSNumber(value: dimInternal)]

            // Use scatterND instead of scatter for better macOS 26 compatibility
            // Determinism: scatterND is generally deterministic if indices are unique. TopK guarantees unique indices.
            let restored = graph.scatterND(
                withUpdatesTensor: prunedT,
                indicesTensor: indicesExpanded,
                shape: outShape,
                batchDimensions: 1,
                mode: .add, // Add to zeros
                name: "restore_nd"
            )

            return (graph, ["pruned": prunedT, "indices": indicesT], [restored])
        }

        let prunedData = MPSGraphTensorData(pruned, shape: [NSNumber(value: batch), NSNumber(value: keepKInternal), NSNumber(value: dimInternal)], dataType: dt)
        let indicesData = MPSGraphTensorData(indices, shape: [NSNumber(value: batch), NSNumber(value: keepKInternal)], dataType: .int32)

        let outLength = batch * originalSeqLen * dimInternal * 2
        guard let outBuffer = BufferAllocator.shared.privateBuffer(length: outLength, device: device, label: "RestoredSpatial") else {
            throw SAM3Error.bufferAllocationFailed("TokenPruner.restoreSpatial: Restored buffer")
        }
        recycledBuffers.append(outBuffer)

        var feeds: [MPSGraphTensor : MPSGraphTensorData] = [:]
        if let p = placeholders["pruned"] { feeds[p] = prunedData }
        if let p = placeholders["indices"] { feeds[p] = indicesData }

        let queue = commandBuffer.commandQueue

        guard executable != nil else {
            throw SAM3Error.graphCompilationFailed("TokenPruner.restoreSpatial: \(cacheKey)")
        }
        let results = try CompiledGraphCache.shared.runExecutable(key: cacheKey, queue: queue, feeds: feeds, targetTensors: resultsTensors)

        if let data = results[resultsTensors[0]] {
            data.mpsndarray().exportData(with: commandBuffer, to: outBuffer, destinationDataType: dt, offset: 0, rowStrides: nil)
        }

        return outBuffer
    }
}
