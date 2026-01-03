# Sprint 07 — Quantization / Palettization / Weight Formats (Speed without Accuracy Collapse)

## Goal
Reduce bandwidth and increase throughput via weight-only compression and/or mixed precision, while preserving mask quality.

## Why
Transformers on Apple GPUs can be bandwidth-bound. Weight compression often gives a large win even if compute stays the same.

## Deliverables
1. A single packed weight format optimized for:
   - sequential reads
   - GPU-private residency
2. At least one compression mode:
   - FP16 baseline
   - optional weight-only int8 / palettized blocks

## Tasks
- Offline tooling (Python) to export:
  - per-tensor scales/zero points if int8
  - palette tables if palettized
- Swift loader:
  - zero-copy mapping into `MTLBuffer`
- Kernel changes:
  - dequantize on the fly inside fused GEMM/attention kernels

## Acceptance criteria
- Accuracy gates:
  - mask IoU drop within a small budget on the asset suite.
- Performance:
  - measurable latency reduction or power reduction.

## Risks
- Per-layer sensitivity varies; you may need mixed precision policies (keep some layers FP16/FP32).

## Notes
This sprint should be driven by measurements: profile whether you are compute-bound or bandwidth-bound first.