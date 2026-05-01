# MoE Trading Docs

This folder documents the current infrastructure for the US100/US500 multi-asset Mixture-of-Experts trading model implemented in `src/moe_trading`.

The codebase is organized as a research and execution stack, not a single monolithic model. Each layer has its own contract:

1. Data ingestion and synchronization.
2. Leak-safe feature engineering.
3. Setup-specific labeling.
4. Sequence dataset construction.
5. Model architecture: shared encoder, TCN experts, manager, calibrator.
6. Training and walk-forward evaluation.
7. Backtesting and live inference.
8. Experiment tracking, checkpointing, and reproducibility.

Use these docs first in the next sessions:

- [System Overview](./system_overview.md)
- [Data And Features](./data_and_features.md)
- [Labels And Targets](./labels_and_targets.md)
- [Model Stack](./model_stack.md)
- [Training, Backtesting, And Live Inference](./training_backtesting_live.md)
- [Readiness And Known Gaps](./readiness_and_gaps.md)
