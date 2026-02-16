# Adaptive Activation Regularisation

A framework for improving safety alignment in language models through adaptive activation-based regularization and harmfulness detection.

## Overview

This project leverages model activation patterns to dynamically regularize harmful behaviors during fine-tuning, ensuring robust safety alignment against adversarial attacks.

## Repository Structure

```
adaptive-activation-regularisation/
├── scripts/
│   ├── harmful_tuning.py          # Fine-tune models with KL regularization
│   ├── probe_harm_tuning.py       # Train with probe-based harm detection
│   ├── probe_utils.py             # Utilities for activation probes
│   ├── prompts.py                 # Safety evaluation prompts
│   └── utils.py                   # Evaluation utilities
├── README.md
└── LICENSE
```

## Features

- **Harmfulness Detection**: Probe-based approach to identify harmful activations
- **Adaptive Regularization**: Dynamic regularization based on activation patterns
- **Safety Evaluation**: LLM-based safety scoring framework
- **Multiple Fine-tuning Methods**: SFT with KL regularization and probe-informed training
- **Multi-layer Analysis**: Analysis across transformer layers
