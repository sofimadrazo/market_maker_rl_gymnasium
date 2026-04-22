# Market Maker RL — Version 3

> Corrected environment + Reinforcement Learning agents  
> Grado en Ingeniería Informática · Proyecto de Inteligencia Artificial · Entrega 3

---

## Table of Contents

1. [What Changed from Version 2](#what-changed-from-version-2)
2. [Bugs Fixed](#bugs-fixed)
3. [New Reward Function](#new-reward-function)
4. [RL Algorithms Implemented](#rl-algorithms-implemented)
5. [Project Structure](#project-structure)
6. [How to Run](#how-to-run)
7. [Logging & Monitoring](#logging--monitoring)
8. [Configuration](#configuration)
9. [References](#references)

---

## What Changed from Version 2

| Feature | v1 | v2 | v3 |
|---------|----|----|-----|
| Gymnasium Environment | ✅ | ✅ | ✅ (corregido) |
| Fuzzy Logic (FIS) | ✅ | ✅ | ✅ |
| BFS Inventory Planner | ✅ | ✅ | ✅ |
| Heuristic Agent | ✅ | ✅ | ✅ |
| ClipRewardWrapper | ❌ | ✅ | ✅ (integrado en training) |
| **Inventario correcto en fills** | ❌ | ❌ | ✅ |
| **obs[1]/obs[2] dinámicos** | ❌ | ❌ | ✅ |
| **Fill prob independiente por lado** | ❌ | ❌ | ✅ |
| **Reward por spread capturado** | ❌ | ❌ | ✅ |
| **REINFORCE (Policy Gradient)** | ❌ | ❌ | ✅ |
| **Actor-Critic** | ❌ | ❌ | ✅ |
| **DQN** | ❌ | ❌ | ✅ |

---

## Bugs Fixed

Versiones 1 y 2 tenían cuatro bugs críticos que hacían imposible el aprendizaje.

### BUG 1 — Inventario invertido en los fills `[CRÍTICO]`

**Archivo:** `envs/market_maker_env.py` — función `step()`

En market making estándar:
- `bid_filled` = cliente vende al MM → MM **compra** → `inventory += 1`, `cash -= bid_price`
- `ask_filled` = cliente compra al MM → MM **vende** → `inventory -= 1`, `cash += ask_price`

Las versiones anteriores hacían exactamente lo contrario. Consecuencia: cada fill generaba
un cambio de PnL de ±2×precio en lugar de ±spread, produciendo oscilaciones de ±2000 unidades
en la curva de PnL — señal completamente inútil para cualquier agente.

```python
# INCORRECTO (v1/v2)
if bid_filled:
    self._inventory -= 1   # MM supuestamente vende al hacer bid??
    self._cash      -= bid_price

# CORRECTO (v3)
if bid_filled and self._inventory < self.max_inventory:
    self._inventory += 1   # MM compra (bid fill)
    self._cash      -= bid_price
```

---

### BUG 2 — `obs[1]` y `obs[2]` eran constantes `[CRÍTICO]`

**Archivo:** `envs/market_maker_env.py` — función `_obs()`

```python
# INCORRECTO (v1/v2): siempre devuelve 0.2
obs[1] = self.base_spread / self.max_spread   # 0.02 / 0.10 = 0.2 siempre
obs[2] = self.base_spread / self.max_spread   # ídem

# CORRECTO (v3): refleja la última acción del agente
obs[1] = self._last_bid_norm   # bid_offset normalizado del paso anterior
obs[2] = self._last_ask_norm   # ask_offset normalizado del paso anterior
```

Sin esto, el agente RL nunca veía el efecto de sus propias acciones en el estado.
El bucle de retroalimentación estaba completamente roto.

---

### BUG 3 — Fill probability compartida para bid y ask `[IMPORTANTE]`

**Archivo:** `envs/market_maker_env.py` — función `step()`

```python
# INCORRECTO (v1/v2): ambas órdenes usan la misma profundidad
depth     = 1.0 - (bid_offset + ask_offset) / (2 * max_spread + 1e-9)
fill_prob = fill_probability * max(0.1, depth)
bid_filled = rng.random() < fill_prob   # misma prob
ask_filled = rng.random() < fill_prob   # misma prob

# CORRECTO (v3): cada lado tiene su propia profundidad
bid_depth     = 1.0 - bid_offset / (max_spread * spread_mult + 1e-9)
ask_depth     = 1.0 - ask_offset / (max_spread * spread_mult + 1e-9)
bid_fill_prob = fill_probability * max(0.1, bid_depth)
ask_fill_prob = fill_probability * max(0.1, ask_depth)
```

Ahora el agente puede controlar de forma independiente la agresividad de cada lado del libro.

---

### BUG 4 — Función de recompensa ruidosa `[MODERADO]`

**Archivo:** `envs/market_maker_env.py` — función `_reward()`

```python
# INCORRECTO (v1/v2): mark-to-market muy ruidoso + bonus fuzzy insignificante (0.005)
reward = pnl_step - inv_cost + 0.005 * alignment

# CORRECTO (v3): spread capturado por fill (señal limpia y densa)
realized = bid_offset * float(bid_filled) + ask_offset * float(ask_filled)
inv_cost  = inventory_penalty * abs(inventory) + 0.001 * inventory**2
reward    = realized - inv_cost
```

Cada fill ahora genera una recompensa proporcional al spread capturado. La señal es
densa (cada paso tiene información útil) y tiene varianza mucho menor.

---

## New Reward Function

```
r_t = spread_capturado_t  −  λ·|inventory_t|  −  0.001·inventory_t²
```

| Término | Rango típico | Propósito |
|---------|-------------|-----------|
| `bid_offset × bid_filled` | [0, 0.18] | Beneficio por fill en lado bid |
| `ask_offset × ask_filled` | [0, 0.18] | Beneficio por fill en lado ask |
| `λ · \|inventory\|` | [0, 0.3×10=3] | Penaliza acumulación de posición |
| `0.001 · inventory²` | [0, 0.1] | Penalización cuadrática (más agresiva para inventarios grandes) |

El `ClipRewardWrapper` con `r_max=0.5` se aplica durante entrenamiento y evaluación
para estabilizar los gradientes del agente RL.

---

## RL Algorithms Implemented

Los tres algoritmos implementados pertenecen al paradigma de **Reinforcement Learning**:

```
Reinforcement Learning
├── Policy Gradient Methods
│   ├── REINFORCE          agents/reinforce_agent.py
│   └── Actor-Critic       agents/actor_critic_agent.py   ← recomendado
└── Value-Based
    └── DQN                agents/dqn_agent.py
```

### REINFORCE (`agents/reinforce_agent.py`)

Algoritmo de Policy Gradient Monte Carlo puro. Actualiza la política al final de
cada episodio usando el retorno descontado bruto como señal.

- **Red:** `obs(7) → Dense(32, tanh) → Dense(16, tanh) → Dense(2, sigmoid) = μ`
- **Exploración:** política Gaussiana `a ~ N(μ, σ²)`; σ entrenado por separado
- **Actualización:** `θ ← θ + α · ∇θ log π(a|s) · G_t`
- **Problema:** alta varianza → dificultad para converger

### Actor-Critic (`agents/actor_critic_agent.py`) ← recomendado

Extiende REINFORCE añadiendo una **red de valor V(s)** (el crítico) que estima el
retorno esperado de cada estado. El actor se actualiza con el **advantage**:

```
A_t = G_t − V(s_t)     ← cuánto mejor fue este paso que lo esperado
```

- **Actor:** misma arquitectura que REINFORCE  
- **Crítico:** `obs(7) → Dense(32, tanh) → Dense(16, tanh) → Dense(1, lineal) = V(s)`
- **Estabilidad numérica:** advantages normalizados (μ=0, σ=1), gradientes divididos por T, gradient clipping (max_norm=1.0)
- **Learning rates separados:** `actor_lr=1e-4`, `critic_lr=5e-4`

### DQN (`agents/dqn_agent.py`)

Deep Q-Network. Aprende la función de valor acción `Q(s,a)` con una red neuronal.
Requiere **discretizar el espacio de acciones** (5×5 = 25 combinaciones bid/ask).

- **Red Q:** `obs(7) → Dense(64, ReLU) → Dense(64, ReLU) → Dense(25, lineal)`
- **Red objetivo (target network):** copia congelada, sincronizada cada N episodios
- **Replay buffer:** almacena `(s, a, r, s', done)`, muestreo aleatorio por lotes
- **Exploración:** ε-greedy con decaimiento exponencial (1.0 → 0.05)
- **Ecuación de Bellman:** `Q(s,a) ← r + γ · max_a' Q_target(s', a')`

---

## Project Structure

```
version3/
├── envs/
│   ├── market_maker_env.py      # Entorno corregido (bugs 1-4)
│   └── wrappers.py              # ClipRewardWrapper
├── agents/
│   ├── heuristic_agent.py       # Agente baseline (Fuzzy + BFS)
│   ├── reinforce_agent.py       # Policy Gradient (REINFORCE)
│   ├── actor_critic_agent.py    # Actor-Critic con advantage
│   └── dqn_agent.py             # Deep Q-Network
├── core/
│   ├── fuzzy_controller.py      # FIS Mamdani (9 reglas)
│   ├── inventory_planner.py     # BFS planner
│   └── data_generator.py        # Precios GBM sintéticos
├── utils/
│   └── hyperparam_grid.py       # Grid 27 combinaciones (itertools)
├── train_actor_critic.py        # Entrenamiento Actor-Critic
├── train_reinforce.py           # Entrenamiento REINFORCE
├── main.py                      # Comparativa 4 políticas
└── README.md                    # Este archivo
```

---

## How to Run

### 1. Entrenar el agente (Actor-Critic recomendado)

```bash
cd market_maker_rl_gymnasium/version3
python train_actor_critic.py
```

Salida esperada:
```
============================================================
  Actor-Critic Training -- MarketMakerEnv v3
============================================================
  actor_lr=0.0001  critic_lr=0.0005  episodes=3000
  ClipRewardWrapper active | r_max=0.5

  Ep   100/3000 | Reward:  -87.10 | PnL:   +4.48 | Adv:  -22.58 | ValErr:  22.58 | 21s
                  Inv:  7.2 | Fill: 48.3% | BidOff: 0.421 | AskOff: 0.438 | Entropy: -1.823
  Ep   200/3000 | Reward:  -85.51 | PnL:   +8.31 | Adv:  -21.94 | ValErr:  21.94 | 43s
                  Inv:  6.8 | Fill: 49.1% | BidOff: 0.405 | AskOff: 0.412 | Entropy: -1.901
  ...

[OK] Training complete. Weights saved -> actor_critic_weights.npz
     Final avg reward (last 200): -XX.XX
     Final avg PnL    (last 200): +XX.XX
     Final avg inv    (last 200):   X.XX
     Final fill rate  (last 200):  XX.X%
     Final entropy    (last 200): -X.XXXX
```

### 2. Comparar las 4 políticas

```bash
python main.py
```

Genera `policy_comparison_v3.png` comparando:
`Random` · `TightSpread` · `Heuristic` · `Actor-Critic`

### 3. Activar logging detallado por pasos

El script de entrenamiento ya usa `logging.basicConfig(level=logging.INFO)` por defecto,
mostrando las métricas estructuradas cada 100 episodios. Para ver cada step individual:

```bash
# Windows — activa nivel DEBUG en todos los módulos
set PYTHONLOGLEVEL=DEBUG && python train_actor_critic.py
```

O desde código:
```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

Con nivel DEBUG el entorno imprime por cada step:
```
STEP 0001 | mid=100.0123 bid=99.9823 ask=100.0423 inv=+1 pnl=0.0300 r=0.0123
```

---

## Logging & Monitoring

Todos los módulos usan el sistema estándar de Python `logging`.

### Niveles disponibles

| Nivel | Activado con | Qué muestra |
|-------|-------------|-------------|
| `WARNING` | siempre | Errores de carga de datos |
| `INFO` | por defecto en training | Métricas agregadas cada 100 eps, reset de episodio, guardado de pesos |
| `DEBUG` | explícito | Cada step: mid, bid, ask, inventory, reward |

### Métricas del bloque de entrenamiento (cada 100 episodios)

Línea 1 — rendimiento del agente RL:

| Campo | Descripción | Señal esperada |
|-------|-------------|----------------|
| `Reward` | Recompensa acumulada media (con ClipRewardWrapper) | Debe subir gradualmente |
| `PnL` | PnL mark-to-market final (sin clip, valor real) | Valores positivos indican beneficio neto |
| `Adv` | Media del advantage `G_t − V(s_t)` | Se acerca a 0 cuando el crítico aprende bien |
| `ValErr` | Error absoluto medio del crítico `|G_t − V(s_t)|` | Debe decrecer con el tiempo |

Línea 2 — comportamiento de la política:

| Campo | Descripción | Señal esperada |
|-------|-------------|----------------|
| `Inv` | Inventario absoluto medio por step | Debe bajar: el agente aprende a no acumular |
| `Fill` | % de steps con al menos un fill | ~40–60% es saludable; muy bajo = quotes demasiado alejados |
| `BidOff` | Media del offset de bid elegido (0–1 normalizado) | Se estabiliza en valores que equilibran fills y spread |
| `AskOff` | Media del offset de ask elegido (0–1 normalizado) | Idealmente simétrico con BidOff |
| `Entropy` | Entropía diferencial de la política Gaussiana | Baja lentamente: la política se vuelve menos aleatoria |

---

## Configuration

### Entorno (`MarketMakerEnv`)

| Parámetro | v3 default | Descripción |
|-----------|-----------|-------------|
| `initial_price` | 100.0 | Precio inicial de los datos sintéticos |
| `max_spread` | 0.10 | Máximo half-spread en unidades de precio |
| `fill_probability` | 0.50 | Probabilidad base de fill por orden |
| `max_inventory` | 10 | Límite duro de inventario (±) |
| `inventory_penalty` | **0.30** | λ: coste por unidad de inventario (**×6 vs v2**) |
| `episode_length` | **500** | Steps por episodio (**+200 vs v2**) |

### Actor-Critic (`train_actor_critic.py`)

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `N_EPISODES` | 3000 | Episodios de entrenamiento |
| `ACTOR_LR` | 1e-4 | Learning rate del actor |
| `CRITIC_LR` | 5e-4 | Learning rate del crítico |
| `GAMMA` | 0.99 | Factor de descuento |
| `CLIP_R_MAX` | 0.5 | Límite del ClipRewardWrapper |

---

## References

1. Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book*. Quantitative Finance.
2. Williams, R.J. (1992). *Simple statistical gradient-following algorithms for connectionist reinforcement learning*. Machine Learning. (REINFORCE)
3. Sutton, R.S. & Barto, A.G. (2018). *Reinforcement Learning: An Introduction*, 2nd ed. (Actor-Critic, DQN)
4. Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning*. Nature. (DQN)
5. Gymnasium documentation — https://gymnasium.farama.org
