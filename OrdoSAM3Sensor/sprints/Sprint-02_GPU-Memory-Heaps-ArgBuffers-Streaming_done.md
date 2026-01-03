# Sprint 02 — GPU Memory Heaps + Argument Buffers + Streaming Weights

## Goal
Minimize memory overhead and CPU↔GPU transfer by:
- Keeping weights and long-lived activations in GPU-private memory.
- Reducing allocation churn.
- Streaming/packing weights so startup is fast and steady-state is allocation-free.

## Platform levers
Metal explicitly supports:
- Memory heaps for manual GPU memory management.
- Resource loading via dedicated IO queue.

Reference: https://developer.apple.com/documentation/metal (see “Memory heaps” and “Resource loading”).

## Current issues suggested by code
- Many buffers are created ad-hoc; some exports allocate new buffers.
- Weights conversion to Float16 must not happen at runtime (now enforced in `SAM3Predictor.loadWeights` by throwing if backbone/neck weights are not already Float16 when `enableHalfPrecision == true`).
- Several `waitUntilCompleted()` calls exist during setup/loading.

## Deliverables
1. A single “GPU arena” allocator:
   - Uses `MTLHeap` for persistent weights and recurring activation buffers.
   - Provides sub-allocation strategy for fixed-size tensors (encoder outputs, prompt embeddings, etc.).
2. Argument-buffered binding strategy:
   - Bind large sets of weights once (or per-layer once), reuse across dispatches.
3. Streaming weight load:
   - Prefer a packed binary format that maps cleanly to GPU buffers.
   - Optionally use compressed storage on disk, decompress on CPU (Accelerate/Compression) only during install/build.

## Tasks
- Convert runtime weight casting to build-time:
  - Generate Float16 (or quantized) weights offline.
  - Avoid Float32→Float16 conversion in Swift hot paths.
- Adopt packed weights for runtime:
  - Generate a `.wts` packed weights file offline (preferred runtime artifact).
  - Runtime must not unzip/parse `.npz` (no `/usr/bin/unzip`); `ModelLoader` requires a sibling `.wts`.
- Adopt GPU-private storage everywhere feasible:
  - `storageModePrivate` for weights and activations.
  - Only copy to shared memory for debugging/validation.
- Introduce a predictable buffer lifetime model:
  - Persistent (weights, constant tensors)
  - Per-frame (encoder embeddings, high-res features)
  - Per-prompt (prompt/geometry embeddings)
- Introduce a texture strategy:
  - For image inputs, prefer textures for read-only sampling and pre-processing.
  - For large activation maps, prefer `MTLBuffer` unless texture hardware paths are beneficial.

## Acceptance criteria
- Startup:
  - No per-weight runtime conversions in Swift.
  - Weight load time reduced and stable.
- Runtime I/O:
  - Weights load from a packed `.wts` file (memory-mapped) rather than `.npz` unzip/NPY parsing.
- Steady state:
  - Zero allocations in the per-frame hot loop (instrumentation verified).
  - Peak memory usage is bounded and documented per input size.

## Risks
- Heaps complicate debugging; keep a “debug allocator” flag for shared memory + validation.

## Notes / next
This sprint makes later kernel fusion pay off: fused kernels are only fast if you aren’t constantly allocating/copying buffers.