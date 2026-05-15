# Metal Sam iOS Experiments

<!-- portfolio-curation -->
## Portfolio Overview
Swift experiment workspace for organizing Metal, media, and iOS prototype work.

## What This Demonstrates
- iOS prototyping
- experiment organization
- graphics/media exploration

## Stack
Swift

## Portfolio Status
This repository is part of Keith Williams' curated public portfolio. The README has been updated to explain the project purpose, technical focus, and why the work is worth reviewing.
<!-- /portfolio-curation -->

---

## Original Notes

# Experiment Folder Organization

## Structure

```
experiment/
├── SAM3Metal/              # SAM3 Metal implementation (Swift + Metal)
├── JEPAMetal/              # JEPA Metal implementation (Swift + Metal)
├── papers/                 # Research papers
│   ├── sam3.txt           # SAM3 paper
│   └── jepa.txt           # JEPA paper
├── research/               # Implementation research & results
│   ├── sam3_metal_results.md
│   ├── jepa_metal_feasibility.md
│   ├── jepa_metal_results.md
│   └── optimization_progress.md
├── runs/                   # Experimental validation runs
│   └── run_YYYYMMDD_HHMMSS_*/
├── letter.md              # Correspondence
├── master_inventory.md    # Asset inventory
└── README.md             # This file
```

## Organization
- **SAM3Metal/** - SAM3 Metal implementation with tests
- **JEPAMetal/** - JEPA Metal implementation
- **papers/** - Original research papers for reference
- **research/** - Implementation analysis and results
- **runs/** - Timestamped experimental runs with benchmarks

