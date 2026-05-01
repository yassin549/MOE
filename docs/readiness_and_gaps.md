# Readiness And Known Gaps

## Current Readiness Status

The project is structurally ready for controlled research runs, but not yet ready for an efficient full-history training cycle on the current machine.

That distinction matters:

- architecturally: yes
- operationally: not yet for the full multi-year minute dataset

## What Is Already Ready

- Python package structure
- typed configuration
- synchronized dual-asset data loading
- feature generation
- setup labeling framework
- MoE + TCN implementation
- walk-forward split logic
- backtesting path
- live inference path
- unit tests for critical structural pieces

## What Has Been Verified

- `python -m pytest -q` passes
- feature generation works on real slices
- labeling works on real slices
- no CUDA GPU is available in the current environment
- the machine currently exposes 4 CPU cores

## Current Blocking Constraints

### 1. Preprocessing Speed

Observed benchmark on this machine:

- building a 10,000-row research frame took about 46.75 seconds

That is too slow for comfortable iteration on a dataset with roughly 1.68 million synchronized rows.

### 2. Feature Builder Efficiency

`features/engineering.py` currently adds many columns incrementally and pandas warns about frame fragmentation.

Impact:

- unnecessary memory churn
- slower preprocessing
- poor scaling to full-history runs

### 3. Label Generation Efficiency

The labeler loops through candidate bars and simulates trade outcomes in Python.

Impact:

- likely the dominant wall-clock cost on full data
- difficult to run repeatedly during model iteration without caching

### 4. Training Was Not Yet Completed End To End On Full Data

The code is wired for training, but a realistic full-history run has not been completed in this environment yet.

That means:

- we should treat the next run as an engineering validation run, not a final research run

## Are We Ready To Start Training?

Yes, with the following interpretation:

- ready to start a bounded training run for pipeline validation: yes
- ready to start a full multi-year research run on this machine: technically yes, practically not advisable before preprocessing optimization or caching

## Recommended Next Training Sequence

1. Run a small bounded training job with `data.max_rows` set to a modest slice.
2. Verify checkpoint, scaler, and backtest artifacts.
3. Add cached feature and label artifacts.
4. Optimize feature generation and label simulation.
5. Start larger walk-forward runs.

## Time Estimate

This estimate is based on current observed runtime, current hardware, and the absence of a GPU.

### Current Hardware Context

- GPU: none detected by PyTorch
- CPU cores: 4
- dataset size after alignment: about 1.68 million rows

### Rough Preprocessing Extrapolation

Observed:

- about 46.75 seconds for roughly 9,940 labeled/usable rows from a 10,000-row slice

A naive linear extrapolation to the full aligned dataset implies many hours just to build the research frame, and the real full run could be worse because:

- pandas fragmentation gets more painful at larger scale
- label loops compound the cost
- walk-forward training repeats data preparation work if uncached

### Practical Estimate Right Now

On the current machine, without refactoring or caching:

- small validation run on 10k to 25k rows: about 5 to 20 minutes
- medium run on 100k to 150k rows: likely 1 to 4 hours
- full aligned multi-year run with walk-forward splits: likely well beyond 10 hours, and possibly materially longer

This is an engineering estimate, not a guarantee.

### Recommended Interpretation

We are ready to start training the model in a staged way.

We are not ready for an efficient full-history training campaign until we:

1. cache the research frame or split-level artifacts
2. remove pandas fragmentation in feature generation
3. speed up or vectorize label generation

## Next Session Priorities

If the goal is to start serious training, the next best work is:

1. add feature-cache and label-cache artifacts keyed by config
2. refactor feature generation to batch-concatenate columns
3. optimize or vectorize the label simulator
4. then run the first meaningful walk-forward training job
