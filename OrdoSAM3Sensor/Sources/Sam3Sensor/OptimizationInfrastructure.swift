//
//  OptimizationInfrastructure.swift
//  Sam3Sensor
//
//  Optimization utilities for SAM3 Metal performance:
//  - Graph Executable Caching (Optimization 1)
//  - Private Memory Storage (Optimization 2)
//  - Triple Buffering (Optimization 4)
//

import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

// MARK: - Graph Executable Caching (Optimization 1)

/// Thread-safe cache for compiled MPSGraph executables.
/// Avoids per-forward graph construction overhead (15-20% latency reduction).
public final class CompiledGraphCache {
    public static let shared = CompiledGraphCache()

    /// Cache entry containing compiled graph and metadata
    private struct CacheEntry {
        let graph: MPSGraph
        let executable: MPSGraphExecutable?
        let inputPlaceholders: [String: MPSGraphTensor]
        let orderedFeedTensors: [MPSGraphTensor]  // Order of feeds for executable
        let targetTensors: [MPSGraphTensor]
    }

    private var cache: [String: CacheEntry] = [:]
    private let lock = NSLock()

    private init() {}

    /// Get or compile a graph executable.
    public func getOrCompile(
        key: String,
        device: MTLDevice,
        builder: () -> (graph: MPSGraph, placeholders: [String: MPSGraphTensor], output: MPSGraphTensor)
    ) -> (graph: MPSGraph, placeholders: [String: MPSGraphTensor], output: MPSGraphTensor, executable: MPSGraphExecutable?) {
        // 1. Fast Path: Read Check
        lock.lock()
        if let entry = cache[key] {
            lock.unlock()
            return (entry.graph, entry.inputPlaceholders, entry.targetTensors[0], entry.executable)
        }
        lock.unlock()

        // 2. Slow Path: Compile (Unlocked)
        let (graph, placeholders, output) = builder()
        let mpsDevice = MPSGraphDevice(mtlDevice: device)

        // Build shaped feeds dictionary for compile.
        // Use sorted keys to ensure consistent ordering for fallback paths.
        let sortedKeys = placeholders.keys.sorted()
        var shapedFeeds: [MPSGraphTensor : MPSGraphShapedType] = [:]
        var fallbackOrderedFeedTensors: [MPSGraphTensor] = []

        for key in sortedKeys {
            let t = placeholders[key]!
            if let shape = t.shape {
                shapedFeeds[t] = MPSGraphShapedType(shape: shape, dataType: t.dataType)
                fallbackOrderedFeedTensors.append(t)
            }
        }

        let targetTensors = [output]
        let executable = graph.compile(with: mpsDevice,
                                      feeds: shapedFeeds,
                                      targetTensors: targetTensors,
                                      targetOperations: nil,
                                      compilationDescriptor: nil)

        let orderedFeedTensors = executable.feedTensors ?? fallbackOrderedFeedTensors
        let orderedTargetTensors = executable.targetTensors ?? targetTensors
        
        // 3. Commit Path: Write Check
        lock.lock()
        defer { lock.unlock() }

        // Double-check (Race condition protection aka "The Double Check")
        if let entry = cache[key] {
            return (entry.graph, entry.inputPlaceholders, entry.targetTensors[0], entry.executable)
        }

        let entry = CacheEntry(
            graph: graph,
            executable: executable,
            inputPlaceholders: placeholders,
            orderedFeedTensors: orderedFeedTensors,
            targetTensors: orderedTargetTensors
        )
        cache[key] = entry

        return (graph, placeholders, output, executable)
    }
    
    /// Multi-output version
    public func getOrCompileMulti(
        key: String,
        device: MTLDevice,
        builder: () -> (graph: MPSGraph, placeholders: [String: MPSGraphTensor], outputs: [MPSGraphTensor])
    ) -> (graph: MPSGraph, placeholders: [String: MPSGraphTensor], outputs: [MPSGraphTensor], executable: MPSGraphExecutable?) {
        // 1. Fast Path: Read Check
        lock.lock()
        if let entry = cache[key] {
            lock.unlock()
            return (entry.graph, entry.inputPlaceholders, entry.targetTensors, entry.executable)
        }
        lock.unlock()

        // 2. Slow Path: Compile (Unlocked)
        let (graph, placeholders, outputs) = builder()
        let mpsDevice = MPSGraphDevice(mtlDevice: device)

        // Build shaped feeds with consistent ordering for fallback paths.
        let sortedKeys = placeholders.keys.sorted()
        var shapedFeeds: [MPSGraphTensor : MPSGraphShapedType] = [:]
        var fallbackOrderedFeedTensors: [MPSGraphTensor] = []

        for key in sortedKeys {
            let t = placeholders[key]!
            if let shape = t.shape {
                shapedFeeds[t] = MPSGraphShapedType(shape: shape, dataType: t.dataType)
                fallbackOrderedFeedTensors.append(t)
            }
        }

        let executable = graph.compile(with: mpsDevice,
                                      feeds: shapedFeeds,
                                      targetTensors: outputs,
                                      targetOperations: nil,
                                      compilationDescriptor: nil)

        let orderedFeedTensors = executable.feedTensors ?? fallbackOrderedFeedTensors
        let orderedTargetTensors = executable.targetTensors ?? outputs

        // 3. Commit Path: Write Check
        lock.lock()
        defer { lock.unlock() }
        
        if let entry = cache[key] {
            return (entry.graph, entry.inputPlaceholders, entry.targetTensors, entry.executable)
        }

        let entry = CacheEntry(
            graph: graph,
            executable: executable,
            inputPlaceholders: placeholders,
            orderedFeedTensors: orderedFeedTensors,
            targetTensors: orderedTargetTensors
        )
        cache[key] = entry

        return (graph, placeholders, outputs, executable)
    }

    /// Execute an executable using a dictionary of feeds (Tensor keyed).
    public func runExecutable(
        key: String,
        queue: MTLCommandQueue,
        feeds: [MPSGraphTensor : MPSGraphTensorData],
        targetTensors: [MPSGraphTensor]
    ) throws -> [MPSGraphTensor : MPSGraphTensorData] {
        lock.lock()
        let entry = cache[key]
        lock.unlock()

        guard let entry = entry, let executable = entry.executable else {
            throw SAM3Error.executionFailed("Executable not found for key: \(key)")
        }

        // Use executable-provided feed ordering to build input array.
        var orderedInputs: [MPSGraphTensorData] = []
        for tensor in entry.orderedFeedTensors {
            guard let data = feeds[tensor] else {
                throw SAM3Error.executionFailed("Missing feed for tensor: \(String(describing: tensor)) (key: \(key))")
            }
            orderedInputs.append(data)
        }

        // Execute with ordered inputs array
        let resultDataArray = executable.run(with: queue, inputs: orderedInputs, results: nil, executionDescriptor: nil)

        var results: [MPSGraphTensor : MPSGraphTensorData] = [:]
        for (i, data) in resultDataArray.enumerated() {
            if i < entry.targetTensors.count {
                results[entry.targetTensors[i]] = data
            }
        }
        return results
    }
    
    public func clear() {
        lock.lock()
        defer { lock.unlock() }
        cache.removeAll()
    }
}

// MARK: - Buffer Allocator (Optimization 2)
public final class BufferAllocator {
    public static let shared = BufferAllocator()
    private var pool: [Int: [MTLBuffer]] = [:]
    private let poolLock = NSLock()

    private var heapByDevice: [ObjectIdentifier: MTLHeap] = [:]
    private let heapLock = NSLock()
    private let minimumHeapSizeBytes = 32 * 1024 * 1024
    
    // Sprint 14: Metrics and pool management
    private var totalAllocations: Int = 0
    private var totalReuses: Int = 0
    private let maxPoolSizePerBucket: Int = 64  // Max buffers per size bucket
    
    private init() {}
    
    /// Metrics for monitoring buffer allocation behavior
    public struct Metrics {
        public let totalAllocations: Int
        public let totalReuses: Int
        public let reuseRatio: Double
        public let poolSize: Int
        public let uniqueSizes: Int
    }
    
    /// Get current allocation metrics
    public func getMetrics() -> Metrics {
        poolLock.lock()
        defer { poolLock.unlock() }
        
        let poolSize = pool.values.reduce(0) { $0 + $1.count }
        let reuseRatio = totalAllocations > 0 ? Double(totalReuses) / Double(totalAllocations) : 0.0
        
        return Metrics(
            totalAllocations: totalAllocations,
            totalReuses: totalReuses,
            reuseRatio: reuseRatio,
            poolSize: poolSize,
            uniqueSizes: pool.keys.count
        )
    }
    
    /// Reset pool and metrics (use between sessions to prevent unbounded growth)
    public func reset() {
        poolLock.lock()
        defer { poolLock.unlock() }
        
        pool.removeAll()
        totalAllocations = 0
        totalReuses = 0
    }

    private func getOrCreatePrivateHeap(device: MTLDevice, minimumSize: Int) -> MTLHeap? {
        let key = ObjectIdentifier(device)

        heapLock.lock()
        defer { heapLock.unlock() }

        if let existing = heapByDevice[key] {
            return existing
        }

        let desc = MTLHeapDescriptor()
        desc.storageMode = .private
        desc.cpuCacheMode = .defaultCache
        desc.size = max(minimumHeapSizeBytes, minimumSize)

        guard let heap = device.makeHeap(descriptor: desc) else {
            return nil
        }

        heap.label = "Sam3Sensor.PrivateHeap"
        heapByDevice[key] = heap
        return heap
    }
    
    public func privateBuffer(length: Int, device: MTLDevice, label: String? = nil) -> MTLBuffer? {
        poolLock.lock()
        defer { poolLock.unlock() }
        
        totalAllocations += 1
        
        if var buffers = pool[length], !buffers.isEmpty {
            let buf = buffers.removeLast()
            pool[length] = buffers
            buf.label = label
            totalReuses += 1
            return buf
        }

        // Sprint 02: Prefer heap-backed allocations for GPU-private buffers.
        // If heap allocation fails (e.g., heap too small), fall back to direct device allocation.
          if let heap = getOrCreatePrivateHeap(device: device, minimumSize: length * 8),
              let buf = heap.makeBuffer(length: length, options: .storageModePrivate) {
            buf.label = label
            return buf
        }

        let fallback = device.makeBuffer(length: length, options: .storageModePrivate)
        fallback?.label = label
        return fallback
    }
    
    public func recycle(_ buffer: MTLBuffer) {
        poolLock.lock()
        defer { poolLock.unlock() }
        
        var buffers = pool[buffer.length] ?? []
        
        // Sprint 14: Enforce max pool size to prevent unbounded growth
        guard buffers.count < maxPoolSizePerBucket else {
            // Pool full for this size - don't recycle, let buffer be deallocated
            return
        }
        
        buffers.append(buffer)
        pool[buffer.length] = buffers
    }
    
    public func makePrivateCopy(from source: MTLBuffer, device: MTLDevice, commandBuffer: MTLCommandBuffer, label: String? = nil) -> MTLBuffer? {
        guard let dest = privateBuffer(length: source.length, device: device, label: label) else { return nil }
        dest.label = label
        if let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.copy(from: source, sourceOffset: 0, to: dest, destinationOffset: 0, size: source.length)
            blit.endEncoding()
        }
        return dest
    }
}

// MARK: - Pipelining Infrastructure (Optimization 4)
public final class RingBuffer<T> {
    private var items: [T]
    private var index = 0
    public init(_ items: [T]) { self.items = items }
    public func next() -> T {
        let item = items[index]
        index = (index + 1) % items.count
        return item
    }
}

public final class FrameSynchronizer {
    private let semaphore: DispatchSemaphore
    public init(maxFrames: Int) { self.semaphore = DispatchSemaphore(value: maxFrames) }
    public func waitForFrame() { semaphore.wait() }
    public func signalFrame() { semaphore.signal() }
}
