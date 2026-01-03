# Sprint 05 — Decoder: RoIAlign + TwoWayTransformer + Mask Head

## Goal
Implement and optimize the decoder (prompt conditioning → mask prediction) with high accuracy and low latency.

## Why
SAM3’s perceived quality is dominated by decoder fidelity: small mask errors are obvious even if encoder is fast.

## Deliverables
1. Fully working decoder path:
   - Prompt encoding (points/boxes/masks)
   - Two-way transformer
   - Mask upscaling and IoU scoring
2. Box support completed (there is an explicit TODO for boxes).

## Tasks
- Finish prompt paths:
  - Points + labels (already present)
  - Boxes (+/- exemplars) for concept segmentation workflows
  - Optional mask prompt refinement
- RoIAlign:
  - Ensure correct alignment behavior and match PyTorch reference.
- TwoWayTransformer:
  - Identify hotspots and substitute fused attention where needed.
- Output heads:
  - Ensure mask logits scaling/activation matches reference.

## Acceptance criteria
- Parity:
  - Mask IoU vs reference above threshold on the test asset suite.
- Latency:
  - Decoder stays within a tight budget (ms-level), measured after Sprint 01/02 execution fixes.

## Risks
- RoIAlign edge conditions (sampling, align corners) are common correctness traps.

## Notes
This sprint gates interactive usability: point/box prompts must feel instantaneous.