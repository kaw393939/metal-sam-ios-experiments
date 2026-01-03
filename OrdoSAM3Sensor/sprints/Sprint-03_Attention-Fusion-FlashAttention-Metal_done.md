# Sprint 03 — Attention Fusion (FlashAttention-style) in Metal

## Goal

Replace “many small ops” attention with a single fused kernel (or minimal kernels) for:

- QKV projection
- Scaled dot-product attention + softmax
- V projection

This is the dominant FLOP + bandwidth consumer in ViT blocks and TwoWayTransformer.

## Why

Even if MPSGraph fuses some ops, Transformer latency is often dominated by:

- Repeated reads/writes of Q, K, V, scores, probs.
- Softmax numeric stabilization.
- Non-optimal tiling for Apple GPU memory hierarchy.

## Approach

- Implement a FlashAttention-like tiling:
  - Tile over sequence dimension.
  - Maintain running row-wise max and sum to compute stable softmax without storing full score matrix.
  - Accumulate output in FP32 (or mixed) while storing FP16.

Math sketch (per row):

- Maintain $m = \max_j s_{ij}$ and $l = \sum_j \exp(s_{ij} - m)$ incrementally across tiles.

## Deliverables

1. A fused attention kernel for the common shapes in your model:
   - ViT self-attn shapes (seqLen tied to patch grid).
   - Decoder/TwoWayTransformer cross-attn shapes.
2. A correctness harness:
   - Compare to MPSGraph/MPS attention reference within tolerance.
3. A performance benchmark:
   - Report ms per attention op and end-to-end block latency.

## Tasks

- Decide the primitive:
  - Metal compute kernel with threadgroup tiling and simdgroup operations.
  - Or leverage MPS matrix primitives only if they don’t force intermediate materialization.
- Add RoPE integration:
  - Fold RoPE into Q/K generation if possible to avoid extra passes.
- Add attention masking support needed by decoder/tracker.

## Acceptance criteria

- Latency reduction for a single ViT block vs current implementation (target: significant drop, measured on-device).
- Numerics within tolerance for:
  - attention output
  - downstream block output

## Benchmarking (Current)

Run:

- `swift run -q OrdoCli bench-attn --b 1 --h 16 --n 256 --d 64 --warmup 25 --iters 200`
- `swift run -q OrdoCli bench-attn --b 1 --h 16 --n 1024 --d 64 --warmup 10 --iters 50`

Example results on one dev machine (wall-time around `MPSGraphExecutable.run`):

- `B=1 H=16 N=256 D=64`
  - Reference (matmul+softmax): mean ~0.824ms (p50 ~0.726ms, p90 ~1.014ms)
  - SDPA (fused): mean ~1.390ms (p50 ~1.282ms, p90 ~1.403ms)
- `B=1 H=16 N=1024 D=64`
  - Reference: mean ~5.236ms (p50 ~5.221ms, p90 ~5.795ms)
  - SDPA: mean ~4.215ms (p50 ~4.151ms, p90 ~4.349ms)

Interpretation: SDPA starts to win at larger sequence lengths; keep validating at SAM3-real seqLens.

## IoU (Mask Accuracy)

Compute mean IoU for predicted-vs-ground-truth binary masks (matched by basename):

- `swift run -q OrdoCli eval-iou --pred <pred_dir> --gt <gt_dir> --threshold 128`

### Offline extraction (reference SAM3 veval JSON → PNG)

The reference repo under `research/sam3` includes toy veval artifacts where masks are stored as per-frame COCO compressed RLE.
To score those with `eval-iou`, export paired PNG masks first:

- `python scripts/python_extraction/export_veval_rle_to_png.py \
  --gt ../research/sam3/assets/veval/toy_gt_and_pred/toy_saco_veval_sav_test_gt.json \
  --pred ../research/sam3/assets/veval/toy_gt_and_pred/toy_saco_veval_sav_test_pred.json \
  --gt-out /tmp/veval_export/gt \
  --pred-out /tmp/veval_export/pred`

Then run:

- `swift run -q OrdoCli eval-iou --pred /tmp/veval_export/pred --gt /tmp/veval_export/gt --threshold 128`

Example result (toy veval export, matching GT↔pred by `(video_id, category_id)` and picking best-score pred per category):

- Mean IoU: ~0.1636 over 2706 paired frames

Note: the exporter writes collision-free filenames (includes GT annotation id and pred index) so multiple GT tracks sharing a category id do not overwrite each other.

## Risks

- Precision: FP16 softmax can destabilize. Use stable softmax + FP32 accumulations.
- Engineering: kernel complexity is high; lock down shapes first (no generality tax).

## Notes

This sprint is where “machine learning passes” / GPU-native workflow (Metal) pays off.
Reference: <https://developer.apple.com/documentation/metal> (see “Compute passes” and “Machine-learning passes”).
