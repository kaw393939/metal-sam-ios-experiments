# Sprint 04 — ViT Encoder End-to-End (24 blocks + Neck) with Zero Copies

## Goal
Ship a full, validated ViT encoder (patch embed → 24 transformer blocks → neck outputs) that:
- Runs without CPU round-trips.
- Produces embeddings identical (within tolerance) to reference.

## Why
Your README targets real-time encoder performance; but the current system still has synchronous graph runs and multiple export steps.

## Deliverables
1. End-to-end encoder path producing:
   - image embeddings (72×72×256)
   - high-res features (288×288×256 and 144×144×256)
2. A deterministic benchmarking harness:
   - fixed input, warmup, p50/p90 latency
3. Memory plan (from Sprint 02) applied.

## Tasks
- Patch embedding:
  - Ensure optimal texture format, sampling strategy, and normalization pipeline.
- Transformer blocks:
  - Integrate fused attention (Sprint 03) + MLP + LayerNorm.
  - Confirm RoPE correctness.
- Neck:
  - Ensure down/up-sampling uses the GPU efficiently; avoid intermediate texture-buffer copies.

## Acceptance criteria
- Correctness:
  - Compare encoder outputs against PyTorch reference intermediates.
- Performance:
  - Encoder latency improvements sprint-over-sprint.
  - No steady-state sync points.

## Risks
- LayerNorm and GELU approximations can cause drift; validate per-layer and allow controlled approximations only with explicit accuracy metrics.

## Notes
If you decide to keep some encoder portions in MPSGraph, this sprint still enforces: executable + encode path, no blocking.