# Workspace Guide: FedNS-Traffic

A practical reference for developers working on this repository.  
For the full research vision see [README.md](README.md) and [RESEARCH.md](RESEARCH.md).

---

## What Is This Project?

**FedNS-Traffic** is a research codebase for an AI-based traffic-signal control system.  
The long-term goal is a three-layer architecture that combines:

| Layer | Technology | Purpose |
|-------|-----------|---------|
| 1 — Perception | YOLOv8-nano + ByteTrack | Real-time vehicle detection & counting from edge cameras |
| 2 — Decision | Federated Multi-Agent RL (Dueling DQN + Graph Attention) | Privacy-preserving distributed signal timing |
| 3 — Safety | Symbolic Rule Engine | Hard safety constraints + human-readable explanations |

**Current status:** Phase 1 (foundation) is implemented and tested.  
Phases 2–8 are planned but not yet coded (see [docs/roadmap.md](docs/roadmap.md)).

---

## Repository Layout

```
research/
│
├── README.md               # Project overview & key-innovation summary
├── RESEARCH.md             # Full literature review & technical design (56 KB)
├── WORKSPACE.md            # This file — developer quick-start
│
├── docs/
│   ├── architecture.md     # Detailed system architecture diagrams
│   └── roadmap.md          # 30-week phase-by-phase implementation plan
│
├── configs/
│   └── default.yaml        # All hyper-parameters (simulation, agent, reward …)
│
├── src/                    # Installable Python package ("fedns-traffic")
│   ├── __init__.py
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── sumo_env.py          # Gymnasium wrapper for SUMO single intersection
│   │   ├── cityflow_env.py      # Gymnasium wrapper for CityFlow (faster training)
│   │   └── state_representation.py  # TrafficState dataclass + StateEncoder
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── dqn_agent.py         # Dueling DQN + Prioritized Replay Buffer
│   │   └── baselines.py         # FixedTimingBaseline, MaxPressureBaseline
│   └── utils/
│       ├── __init__.py
│       ├── config.py            # load_config(), merge_configs(), get_device_str()
│       └── metrics.py           # EpisodeMetrics dataclass + improvement helpers
│
├── scripts/
│   ├── train.py            # Training entry point (CLI)
│   └── evaluate.py         # Evaluation / comparison entry point (CLI)
│
├── tests/
│   ├── __init__.py
│   ├── test_agent.py        # DQN network, SumTree, PER buffer, agent, baselines
│   ├── test_environment.py  # StateEncoder, StateConfig, TrafficState
│   └── test_utils.py        # load_config, merge_configs, EpisodeMetrics
│
├── docker/
│   ├── Dockerfile.edge      # Edge unit (Jetson Orin) image
│   ├── Dockerfile.server    # Regional aggregation server image
│   └── docker-compose.yml   # Local multi-container dev stack
│
├── pyproject.toml           # Build config, deps, pytest & ruff settings
└── requirements.txt         # Pinned deps for reproducibility
```

---

## Setup

### Prerequisites

- Python ≥ 3.10
- (Optional) SUMO ≥ 1.18 for the full simulation environment — install from [sumo.dlr.de](https://sumo.dlr.de/docs/Downloads.php)
- (Optional) NVIDIA GPU + CUDA for accelerated training

### Install (editable)

```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install the package and its dependencies
pip install -e .

# 3. Install development extras (pytest, ruff, black, pre-commit)
pip install -e ".[dev]"

# 4. (Optional) Install SUMO Python bindings
pip install -e ".[sumo]"    # requires SUMO to already be installed on PATH
```

### Install PyTorch (CPU-only, for CI / machines without a GPU)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Running the Tests

```bash
# All tests with coverage report
pytest tests/

# A single test module
pytest tests/test_agent.py -v

# Skip coverage and run fast
pytest tests/ --no-cov -q
```

The project requires **≥ 65% code coverage** (configured in `pyproject.toml`).  
`sumo_env.py` and `cityflow_env.py` are excluded from coverage because they
require the SUMO / CityFlow binaries to be available.

---

## Training

Training requires a SUMO network and route file (not included in the repo — see
[SUMO tutorials](https://sumo.dlr.de/docs/Tutorials.html) to generate them).

```bash
python scripts/train.py \
    --config  configs/default.yaml \
    --net     path/to/intersection.net.xml \
    --routes  path/to/intersection.rou.xml \
    --episodes 500

# With SUMO GUI (for debugging)
python scripts/train.py --net … --routes … --gui

# Force CPU training
python scripts/train.py --net … --routes … --device cpu
```

Checkpoints are written to `checkpoints/` and logs to `logs/`.

---

## Configuration

`configs/default.yaml` is the single source of truth for all hyper-parameters.
Key sections:

| Section | Controls |
|---------|---------|
| `simulation` | Simulator choice (`sumo`/`cityflow`), episode length, seed |
| `state` | Number of lanes, normalisation limits, embedding dimension |
| `action` | Number of signal phases, duration buckets |
| `safety` | Min/max green times, pedestrian wait threshold |
| `reward` | Weights for the five reward components |
| `agent` | Network width, learning rate, PER parameters, ε-greedy schedule |
| `training` | Total episodes, eval/save frequency, checkpoint paths |

Override any value from the CLI via `merge_configs`, or pass a custom YAML file
to `--config`.

---

## Key Components

### `StateEncoder` (`src/environment/state_representation.py`)

Converts a `TrafficState` snapshot into a **flat, normalised float32 vector**
ready for neural-network input:

| Feature group | Size | Description |
|---------------|-----:|-------------|
| Queue lengths | 12 | Halting vehicles per lane (÷ 50) |
| Vehicle speeds | 12 | Average speed km/h per lane (÷ 60) |
| Waiting times | 12 | Average wait per lane in seconds (÷ 300) |
| Vehicle counts | 12 | Total vehicles in detection zone (÷ 50) |
| Phase one-hot | 8 | Current signal phase |
| Phase duration | 1 | Seconds in current phase (÷ 90) |
| Time of day | 1 | Normalised [0, 1] |
| Day of week | 7 | One-hot |
| Pedestrian features | 8 | Bool + wait time for 4 crosswalks |
| Neighbour embeddings | 64 | 4 neighbours × 16-float coordination vector |
| **Total** | **137** | |

### `DuelingDQNAgent` (`src/agents/dqn_agent.py`)

Implements **Dueling DQN** with **Prioritized Experience Replay (PER)**:

- **`DuelingDQNNetwork`** — shared feature encoder → separate value stream V(s)
  and advantage stream A(s,a); combined as Q = V + (A − mean(A)).
- **`SumTree`** — O(log n) binary tree for priority-based replay sampling.
- **`PrioritizedReplayBuffer`** — wraps SumTree; anneals importance-sampling
  weights β from `beta_start` → 1.0 over `beta_frames` steps.
- **`DuelingDQNAgent`** — ε-greedy action selection, Double DQN targets, Huber
  loss with IS weights, soft target-network updates.

### Baselines (`src/agents/baselines.py`)

Two reference controllers for benchmarking:

- **`FixedTimingBaseline`** — cycles through phases with Webster-style fixed
  durations; no observation needed.
- **`MaxPressureBaseline`** — at each decision point activates the phase with
  the highest lane-queue pressure (Varaiya 2013).

### `SumoEnvironment` (`src/environment/sumo_env.py`)

OpenAI Gymnasium-compatible environment backed by SUMO/TraCI:

- **Observation**: flat vector from `StateEncoder` (137-dim by default).
- **Action**: integer `phase_id × 4 + bucket_id` (8 phases × 4 duration
  buckets = 32 discrete actions).
- **Reward**: weighted sum of five components — pressure, queue reduction,
  throughput, excessive-wait penalty, pedestrian penalty.
- **Safety**: minimum green time enforced; violations counted in `info`.

### Utilities (`src/utils/`)

| Module | Key exports |
|--------|------------|
| `config.py` | `load_config(path)`, `merge_configs(base, override)`, `get_device_str()` |
| `metrics.py` | `EpisodeMetrics` (accumulate + summarise per-episode stats), `improvement_over_baseline()` |

---

## CI / Linting

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push:

1. **Lint** — `ruff check src/ tests/ scripts/` and `black --check` (line length 100).
2. **Tests** — `pytest tests/ --cov=src` on Python 3.10 and 3.11.

Run linting locally before pushing:

```bash
ruff check src/ tests/ scripts/
black --check src/ tests/ scripts/

# Auto-fix style issues
ruff check --fix src/ tests/ scripts/
black src/ tests/ scripts/
```

---

## What Is Not Yet Implemented

The following modules are planned in later phases but do **not** exist yet:

| Planned module | Roadmap phase |
|----------------|:---:|
| `src/federated/` — Flower-based federated training | Phase 3 |
| `src/symbolic/` — Safety rule engine + explainer | Phase 4 |
| `src/perception/` — YOLOv8-nano + ByteTrack pipeline | Phase 5 |
| `src/digital_twin/` — Bayesian sim-to-real calibration | Phase 6 |
| `src/dashboard/` — React + FastAPI monitoring dashboard | Phase 7 |
| Multi-agent environment wrapper | Phase 2 |
| Graph Attention Network agent | Phase 2 |

See [docs/roadmap.md](docs/roadmap.md) for the full 30-week development plan.

---

## Further Reading

| Document | Contents |
|----------|---------|
| [README.md](README.md) | One-page project summary, tech stack, expected results |
| [RESEARCH.md](RESEARCH.md) | Full SOTA review, system design, and implementation rationale |
| [docs/architecture.md](docs/architecture.md) | ASCII architecture diagrams: edge unit, data flow, communication |
| [docs/roadmap.md](docs/roadmap.md) | Phase-by-phase milestones and team/compute requirements |
