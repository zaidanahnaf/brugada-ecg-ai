┌─────────────────────────────────────────────────────────────────────┐
│ ABLATION STUDY STRUCTURE │
│ │
│ AXIS 1: Feature Group Ablation │
│ → Which feature groups are necessary? │
│ → Additive build-up AND subtractive knockout │
│ │
│ AXIS 2: Lead Ablation │
│ → Which leads contribute signal? │
│ → V1–V3 vs. all leads vs. no V1–V3 │
│ │
│ AXIS 3: Model Ablation │
│ → Are gains model-specific or general? │
│ → Tested across LR, RF, and best model │
│ │
│ AXIS 4: Hybrid Ablation │
│ → Handcrafted vs. CNN vs. Fusion │
│ → Requires CNN embeddings from Person 2 │
│ │
│ AXIS 5: Data / Metadata Experiments │
│ → Metadata-only (sanity check) │
│ → ECG + metadata (exploratory, clearly flagged) │
│ │
│ ALL AXES use the SAME fold_assignments.csv │
│ ALL AXES evaluated with the SAME metrics suite │
│ ALL AXES use the SAME best classical model config │
└─────────────────────────────────────────────────────────────────────┘
