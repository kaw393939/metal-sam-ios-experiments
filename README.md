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
