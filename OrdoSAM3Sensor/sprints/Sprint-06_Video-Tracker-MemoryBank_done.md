# Sprint 06 — Video Tracker + Memory Bank (Masklets at Scale)

## Goal
Implement the tracker + memory bank so video PCS is stable and fast for multiple objects.

## Why
Tracker complexity grows with number of objects and frames. If memory bank writes/reads aren’t optimized, performance collapses.

## Deliverables
1. Functional tracker:
   - per-frame propagation
   - memory bank update policy
   - object confirmation / suppression policy
2. A “5 concurrent objects” real-time target benchmark.

## Tasks
- Memory bank representation:
  - Choose layout that minimizes scattered reads.
  - Prefer contiguous per-object slots.
- Temporal stride + update policy:
  - Store only frames that help (confidence-gated).
- Mask post-processing:
  - hole filling / non-overlap constraints implemented as GPU kernels where possible.
- Scheduling:
  - Pipeline detector refresh + tracker propagation.

## Acceptance criteria
- Stability:
  - No identity swaps on curated test clips (define a minimal regression set).
- Performance:
  - Sustains target FPS for ~5 objects without blowing memory.

## Risks
- Tracking-by-detection refresh policies can cause flicker; tune with deterministic rules and tests.

## Notes
This is where your MemoryBank.swift + attention implementation must be aggressively profiled and fused.