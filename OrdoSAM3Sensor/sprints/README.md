# OrdoSAM3Sensor Sprint Backlog (Performance + Accuracy, No Back-Compat)

This folder is an execution-focused backlog for porting SAM3 to Apple Silicon with a **performance-first** philosophy:

- Target: highest accuracy possible under tight latency and memory constraints.
- Constraint: **zero backward compatibility** (we can raise minimum OS, require M-series GPU/ANE capabilities, and break APIs freely).
- Principle: prefer *fewer, larger fused kernels* and *fewer graph boundaries*.
- Validation: every sprint must ship with a correctness gate (numerical / mask IoU / parity with reference intermediates) and a performance gate (latency + peak memory).

## Current repo signals (from code audit)

The current Swift implementation shows repeat patterns that dominate end-to-end latency:

- Widespread synchronous execution via `graph.run(...)` and `commandBuffer.waitUntilCompleted()`.
- Comments indicate `graph.encode(async)` attempts crash; several modules fall back to synchronous paths.
- There are TODOs for boxes, decoder completion, tracker completion, and weight conversion build-time.

These sprints are written to systematically remove those bottlenecks.

## Sprint ordering

Start with execution model + memory (Sprints 01–02). Kernel fusion and model blocks come next (03–06). Quantization and hybridization are later (07–09). Tooling is continuous but gets a focused sprint (08).

- [Sprint 01](Sprint-01_Async-Execution-and-Graph-Compilation.md)
- [Sprint 02](Sprint-02_GPU-Memory-Heaps-ArgBuffers-Streaming.md)
- [Sprint 03](Sprint-03_Attention-Fusion-FlashAttention-Metal.md)
- [Sprint 04](Sprint-04_ViT-Encoder-EndToEnd.md)
- [Sprint 05](Sprint-05_Decoder-RoIAlign-TwoWayTransformer.md)
- [Sprint 06](Sprint-06_Video-Tracker-MemoryBank.md)
- [Sprint 07](Sprint-07_Quantization-Palettization-WeightFormats.md)
- [Sprint 08](Sprint-08_Profiling-Counters-Regression-Gates.md)
- [Sprint 09](Sprint-09_CoreML-ANE-Hybrid-Optional.md)
- [Sprint 10](Sprint-10_EncoderConfig-Precision-Contracts.md)
- [Sprint 11](Sprint-11_AsyncGraphCache-FeedOrdering.md)
- [Sprint 12](Sprint-12_PatchEmbed-Layout-RoPE-Invariants.md)
- [Sprint 13](Sprint-13_Prune-Restore-Windowing.md)
- [Sprint 14](Sprint-14_PerfHygiene-HeapAllocator-ProdHardening.md)

## Doc pointers (2025-era platform levers)

- Metal: resource loading, memory heaps, synchronization, and **Metal 4** APIs: https://developer.apple.com/documentation/metal
- MPSGraph: compile to executable, serialize to `.mpsgraphpackage`, execution descriptors, reduced-precision fast-math: https://developer.apple.com/documentation/metalperformanceshadersgraph
- Core ML: compute devices, compute plan, and `MLTensor` compute policy: https://developer.apple.com/documentation/coreml
- Accelerate/BNNS: high-performance CPU fallback or pre/post processing: https://developer.apple.com/documentation/accelerate
