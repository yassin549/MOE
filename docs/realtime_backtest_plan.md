# Realtime Backtest Plan

## Goal

Replace the removed backtesting engine with a simulator that evaluates the model exactly as if it were trading live:

- replay synchronized closed candles in timestamp order
- update feature state incrementally
- run model inference at each decision step
- apply order, fill, and position logic sequentially
- enforce live-like portfolio and risk constraints
- remain fast enough for long minute-bar windows

The simulator must not precompute a full year of entry signals and then replay them blindly. The model and simulator must interact step-by-step.

## Non-Negotiable Architecture

### 1. Replay Kernel

Single monotonic event loop over aligned US100 and US500 bars.

Inputs:

- timestamp arrays
- OHLCV arrays
- session/calendar arrays

Responsibilities:

- advance the clock by one closed candle
- update rolling feature caches
- expose the latest model input window
- process open-position exits before new entries

### 2. Incremental Feature State

Feature engineering must be online, not batch-rebuilt per step.

Implementation target:

- structure-of-arrays layout
- rolling ring buffers for lookback windows
- constant-time updates per new candle where possible

The simulator should only recompute values touched by the newest bar.

### 3. Model Adapter

The model must be wrapped behind a stepwise interface:

- `warmup(...)`
- `on_bar(bar_index) -> list[ModelSignal]`

This adapter is responsible for:

- maintaining the current sequence window
- applying scaling
- calling the MoE model
- converting raw outputs into ranked trade intents

### 4. Execution Simulator

The execution layer must be independent from the model.

Responsibilities:

- entry approval
- spread/slippage/commission
- long and short handling
- stop loss and take profit evaluation
- max holding exit
- one-position-per-asset and portfolio limits
- equity and drawdown tracking

### 5. Result Writer

The simulator should emit:

- fills/trades
- equity curve
- rejection counters
- latency/performance counters

This layer stays outside the hot path.

## Performance Strategy

### Phase 1: Fast Python Core

Build the first correct version with:

- NumPy arrays only
- no pandas in the hot path
- preallocated buffers
- explicit scalar state

This version should already be materially faster than the legacy pandas loop.

### Phase 2: Compiled Hot Loop

Compile the replay and execution kernel with one of:

1. Numba
2. Cython
3. C shared library via `ctypes`/`cffi`
4. Rust shared library if preferred over C

Recommended order:

1. Python + NumPy scaffold
2. Numba prototype for quick validation
3. Native C or Rust core only if profiling proves Python/Numba is still the bottleneck

### Phase 3: Native Engine Layout

If native code is required, the core loop should move into a single compiled module with:

- contiguous `double` / `int32` / `int64` arrays
- no heap allocation inside the main loop
- SoA memory layout
- explicit trade/state structs

Python should remain orchestration only.

## Implementation Order

### Step 1. Contracts

Create and freeze the simulator contracts:

- candle batch
- model signal
- fill event
- portfolio state
- replay config

### Step 2. Data Feed

Build a feed that exposes aligned arrays for:

- timestamp
- open/high/low/close/volume
- any extra live-required context columns

### Step 3. Online Feature Engine

Replace batch feature generation with rolling state objects and ring buffers.

### Step 4. Model Adapter

Implement the stepwise inference wrapper around the MoE model.

### Step 5. Execution Kernel

Implement deterministic sequential fills and position state transitions.

### Step 6. Portfolio Rules

Add:

- per-asset position limits
- aggregate open risk
- challenge/funded account rules if still required

### Step 7. Compiled Optimization

Profile first, then move only the real bottleneck into compiled code.

## Validation Plan

### Correctness

Validate:

- no lookahead
- no future feature access
- deterministic repeated runs
- correct exit ordering
- correct fee/slippage application
- correct weekend/session handling

### Realism

Validate that the model is invoked only with information available at each closed candle.

### Performance

Track separately:

- feature update time
- model inference time
- execution loop time

The next engine should make it obvious whether the bottleneck is inference or simulation.

## Explicit Anti-Goals

- no pandas row loop
- no full-window signal precomputation as the primary mode
- no monolithic notebook-style pipeline
- no hidden coupling between model inference and execution state

## Deliverable Shape

The replacement should be built as:

- `src/moe_trading/backtesting/realtime.py`
- incremental feature-state module
- model adapter module
- compiled execution kernel module when profiling justifies it

The current repo state intentionally removes the old engine and leaves a scaffold so the new implementation can start cleanly.
