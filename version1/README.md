# Market Maker RL Environment

> **Gymnasium custom environment for a Market Making agent**  
> Grado en Ingeniería Informática · Proyecto de Inteligencia Artificial · Entrega 1

---

## Table of Contents

1. [What is a Market Maker?](#what-is-a-market-maker)
2. [Environment Overview](#environment-overview)
3. [AI Concepts Applied](#ai-concepts-applied)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Running the Code](#running-the-code)
7. [Configuration & Hyperparameters](#configuration--hyperparameters)
8. [Extending to Entrega 2](#extending-to-entrega-2)
9. [References](#references)

---

## What is a Market Maker?

A **Market Maker (MM)** is a financial agent that continuously provides *two-sided quotes* to the market:

| Quote side | Direction | Profit mechanism |
|------------|-----------|------------------|
| **Bid**    | Buy order posted *below* mid-price | Buys cheap |
| **Ask**    | Sell order posted *above* mid-price | Sells expensive |

The difference between the ask and the bid is the **spread**. Every time both sides fill, the MM earns the spread. The fundamental challenge is **inventory risk**: if the price moves against an accumulated position, losses can exceed the captured spread.

```
   Mid price
       │
 bid ──┤── ask
  ↑    │    ↑
fill   │   fill
 +1    │    -1
 inv   │   inv
```

The optimal strategy balances:
- **Spread capture** – post quotes close to mid for high fill rates.
- **Inventory control** – avoid large directional exposure.
- **Volatility adaptation** – widen quotes when the market is noisy.

This is precisely the Avellaneda-Stoikov (2008) problem, which we solve here with Reinforcement Learning.

---

## Environment Overview

### Observation Space `Box(7,)`

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0 | `mid_price_norm` | [-1, 1] | Price drift from initial price |
| 1 | `bid_offset_proxy` | [0, 1] | Normalised bid distance from mid |
| 2 | `ask_offset_proxy` | [0, 1] | Normalised ask distance from mid |
| 3 | `inventory_norm` | [-1, 1] | Signed inventory / max_inventory |
| 4 | `volatility_norm` | [0, 1] | Rolling 20-step volatility |
| 5 | `pnl_norm` | [-10, 10] | Cumulative PnL normalised |
| 6 | `time_remaining` | [0, 1] | Fraction of episode left |

### Action Space `Box(2,)`

| Index | Action | Meaning |
|-------|--------|---------|
| 0 | `bid_offset` | How far below mid to post the buy order ∈ [0, 1] |
| 1 | `ask_offset` | How far above mid to post the sell order ∈ [0, 1] |

The actual price offset is `action[i] × max_spread × fuzzy_multiplier`.

### Reward Function

```
r_t = ΔPnL_t  −  λ · |inventory_t|  +  fuzzy_alignment_bonus_t
```

| Term | Role |
|------|------|
| `ΔPnL_t` | Step-level mark-to-market profit/loss |
| `λ · \|inventory\|` | Penalises large directional exposure |
| `fuzzy_alignment_bonus` | Rewards the agent when its spread choice aligns with the Fuzzy Inference System recommendation |

### Episode lifecycle

```
reset() ──► random start in price series
  │
  ▼
step() × episode_length
  │  ├─ Fuzzy multiplier computed
  │  ├─ Quotes placed: bid = mid − offset, ask = mid + offset
  │  ├─ Fills sampled stochastically (depth-adjusted probability)
  │  ├─ Inventory and cash updated
  │  └─ Reward returned
  │
  ▼
terminated = True
```

---

## AI Concepts Applied

### 1 · Fuzzy Logic (`FuzzySpreadController`)

The `FuzzySpreadController` implements a **Mamdani Fuzzy Inference System** (FIS) *from scratch*, with no external fuzzy library, to dynamically adjust the spread multiplier.

**Inputs**
- `vol_norm` – normalised rolling volatility ∈ [0, 1]
- `inv_norm` – normalised |inventory| ∈ [0, 1]

**Output**
- `spread_multiplier` ∈ [0.6 (tight), 1.0 (normal), 1.8 (wide)]


**9-rule base**

| Volatility | Inventory | → Output |
|:----------:|:---------:|:--------:|
| LOW | BALANCED | **TIGHT** |
| LOW | MODERATE | NORMAL |
| LOW | EXTREME | WIDE |
| MEDIUM | BALANCED | NORMAL |
| MEDIUM | MODERATE | NORMAL |
| MEDIUM | EXTREME | WIDE |
| HIGH | any | **WIDE** |

Defuzzification uses **centroid method** (weighted average of singleton outputs).

The FIS acts in *two* roles:
1. **Spread multiplier** applied before placing orders.
2. **Reward shaping**: a `fuzzy_alignment_bonus` rewards the RL agent when its chosen offsets match the FIS recommendation, providing a shaped gradient that guides exploration.

---

### 2 · State-Space Search (`InventoryLiquidationPlanner`)

A **BFS (Breadth-First Search)** planner pre-computes, offline at `reset()`, the minimum number of actions needed to reduce any inventory level to zero.

```
States:   integer inventory ∈ [-max_inv, max_inv]
Actions:  {-1: aggressive_sell, 0: balanced, +1: aggressive_buy}
Goal:     inventory == 0
```

The BFS searches backwards from the goal state and builds a look-up table:
```python
plan[inventory] → optimal_first_action
```

This table is used by the `HeuristicMarketMakerAgent` to **skew quotes** when inventory is large (e.g., tighten ask and widen bid to encourage sells when long), and it can serve as a **reward bonus** or **action mask** for RL agents in Entrega 2.

---

### 3 · `itertools` – Hyperparameter Grid

`itertools.product` generates the full Cartesian product of:
- `base_spread` ∈ {0.01, 0.02, 0.05}
- `inventory_penalty` ∈ {0.01, 0.1, 0.5}
- `fill_probability` ∈ {0.3, 0.5, 0.7}

→ **27 combinations** for grid search, or as the initial population for PSO (`pyswarm`) in Entrega 2.

---

### 4 · `logging` – Structured Tracing

All modules use Python's `logging` library with a common format:

```
10:42:31 | MarketMakerEnv | INFO  | ENV RESET | start_idx=412 mid_price=99.7831
10:42:31 | MarketMakerEnv | DEBUG | STEP 0001 | mid=99.79 bid=99.77 ask=99.81 ...
10:42:31 | MarketMakerEnv | DEBUG | FuzzySpread | vol_norm=0.241 inv_norm=0.000 => multiplier=0.718
```

Log levels:
- `INFO`  – episode-level events (reset, final metrics)
- `DEBUG` – step-level decisions (use `logging.DEBUG` to enable)
- `WARNING` – data loading fallbacks

---

## Project Structure

```
market_maker/
├── market_maker_env.py   ← Main environment (this file)
├── example_agent.py      ← Demo: Random / TightSpread / Heuristic policies
├── README.md             ← This file
├── requirements.txt      ← Python dependencies
└── policy_comparison.png ← Generated plot (after running example_agent.py)
```

---

## Installation

### Prerequisites

- Python ≥ 3.10
- pip

### Steps

```bash
# 1. Clone / download the project
cd market_maker

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`

```
gymnasium>=0.29.0
numpy>=1.26.0
pandas>=2.1.0
matplotlib>=3.8.0
pyswarm>=0.6          # optional – PSO optimisation (Entrega 2)
```

---

## Running the Code

### Quick demo (heuristic vs random vs tight-spread)

```bash
python example_agent.py
```

Expected output:

```
▶  Running policy: Random
   Done in 0.34s
▶  Running policy: TightSpread
   Done in 0.29s
▶  Running policy: Heuristic
   Done in 0.31s

============================================================
Policy         Mean Reward    Mean PnL   Mean |Inv|
------------------------------------------------------------
Random          -0.2341      -0.1832          3.40
TightSpread     +0.0912      +0.0741          1.20
Heuristic       +0.1573      +0.1301          0.80
============================================================

✔  Plot saved to policy_comparison.png
```

### Run the environment standalone

```bash
python market_maker_env.py
```

This calls `run_demo()` which runs 3 episodes with the heuristic agent and prints the hyperparameter grid sample.

### Enable DEBUG logging

```python
import logging
logging.getLogger("MarketMakerEnv").setLevel(logging.DEBUG)
```

### Use real price data

```python
env = MarketMakerEnv(data_path="path/to/prices.csv", episode_length=500)
```

The CSV must contain a column named `mid_price`. Any frequency is supported (tick, 1-min, daily).

---

## Configuration & Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_price` | 100.0 | Starting mid-price for synthetic data |
| `max_spread` | 0.10 | Maximum half-spread in price units |
| `base_spread` | 0.02 | Default half-spread |
| `fill_probability` | 0.5 | Base probability an order is filled |
| `max_inventory` | 10 | Hard inventory limit (±) |
| `inventory_penalty` | 0.05 | λ: per-unit inventory cost in reward |
| `episode_length` | 500 | Steps per episode |
| `n_synthetic_steps` | 1000 | Length of synthetic price series |

---

## Extending to Entrega 2

The modular design makes the following extensions straightforward:

| Feature | Extension point |
|---------|----------------|
| RL agent (PPO, SAC) | Replace `HeuristicMarketMakerAgent` with `stable-baselines3` |
| PSO hyperparameter search | Feed `generate_hyperparam_grid()` output to `pyswarm.pso` |
| Real LOB data | Set `data_path` to a tick-data CSV |
| Multi-asset MM | Stack multiple `MarketMakerEnv` instances or extend observation space |
| Neural FIS | Replace `FuzzySpreadController` with a learned fuzzy layer (ANFIS) |
| Curriculum learning | Gradually increase `inventory_penalty` and volatility as agent improves |

---

## References

1. Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book*. Quantitative Finance.
2. Gymnasium documentation – https://gymnasium.farama.org
3. Mamdani, E. H. (1974). *Application of fuzzy algorithms for control of simple dynamic plant*. IEE Proceedings.
4. Russell, S. & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach*, 4th ed. (BFS Chapter 3).

---
# Market Maker RL — Version 2

> Environment with Gymnasium Wrappers  
> BSc in Computer Engineering · Artificial Intelligence Project

---

## What’s New in Version 2?

Version 2 builds on the same environment developed in Deliverable 1 and introduces a **Gymnasium wrapper** that stabilizes agent training by controlling reward scale.

| | Version 1 | Version 2 |
|---|---|---|
| Gymnasium Environment | ✅ | ✅ |
| Fuzzy Logic | ✅ | ✅ |
| BFS Search | ✅ | ✅ |
| Heuristic Agent | ✅ | ✅ |
| **ClipRewardWrapper** | ❌ | ✅ |

---

A wrapper is a layer that sits on top of the original environment without modifying it. It intercepts calls to `step()` and `reset()` in order to transform observations, actions, or rewards.


