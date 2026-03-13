# AI-Based Smart Traffic Management System: Research & Plan

## A Federated Neuro-Symbolic Multi-Agent System with Digital Twin Calibration

---

## Table of Contents

1. [Problem Analysis](#1-problem-analysis)
2. [State-of-the-Art Literature Review](#2-state-of-the-art-literature-review)
3. [Gap Analysis & Motivation for Novel Approach](#3-gap-analysis--motivation-for-novel-approach)
4. [Proposed Approach: FedNS-Traffic](#4-proposed-approach-fedns-traffic)
5. [System Architecture](#5-system-architecture)
6. [Technical Design](#6-technical-design)
7. [Implementation Plan](#7-implementation-plan)
8. [Evaluation Framework](#8-evaluation-framework)
9. [Expected Outcomes](#9-expected-outcomes)
10. [References](#10-references)

---

## 1. Problem Analysis

### 1.1 Problem Statement

Urban traffic congestion at multi-directional intersections remains one of the most pressing challenges in intelligent transportation. Traditional fixed-timing traffic signal systems fail to adapt to dynamic traffic conditions, leading to:

- **Increased average wait times**: Up to 40% longer during peak hours compared to optimal timing
- **Fuel waste and emissions**: Idling vehicles at poorly timed signals contribute ~30% of urban CO₂ emissions
- **Cascading congestion**: Poor signal coordination at one intersection propagates delays across the network
- **Safety concerns**: Frustrated drivers take risks, increasing accident rates at congested intersections

### 1.2 Key Challenges

| Challenge | Description |
|-----------|-------------|
| **Multi-directional flow** | Intersections with 4+ approaches, each with variable turning movements |
| **Temporal variability** | Traffic patterns shift by time of day, day of week, and season |
| **Stochastic events** | Accidents, weather, construction, special events cause unpredictable surges |
| **Network effects** | Optimizing one intersection can worsen conditions at neighbors |
| **Scalability** | City-wide deployment requires managing thousands of intersections |
| **Real-time constraint** | Decisions must be made in sub-second timeframes |
| **Data privacy** | Camera/sensor data raises privacy concerns |
| **Explainability** | Traffic authorities need to understand why a decision was made |

### 1.3 Requirements for an Ideal Solution

1. **Real-time adaptation**: Sub-second response to changing conditions
2. **Multi-intersection coordination**: Network-level optimization, not just local
3. **Robustness**: Graceful handling of sensor failures, edge cases, and novel situations
4. **Scalability**: Must work for 10 intersections or 10,000
5. **Explainability**: Human-understandable reasoning for each signal decision
6. **Privacy-preserving**: Minimize raw data transmission and storage
7. **Transferability**: Models trained in one city should transfer to another
8. **Safety**: Hard constraints on minimum green times, pedestrian phases, emergency preemption

---

## 2. State-of-the-Art Literature Review

### 2.1 Traditional & Classical Approaches

| Method | Description | Limitations |
|--------|-------------|-------------|
| **Webster's Formula (1958)** | Optimal fixed cycle length from traffic volume | No real-time adaptation |
| **SCOOT (1981)** | Adaptive system using loop detector data | Centralized, slow adaptation |
| **SCATS (1982)** | Adaptive coordination from Sydney | Limited optimization scope |
| **Max-Pressure (Varaiya, 2013)** | Theoretical optimal: serve phase with highest "pressure" | Assumes perfect information, no coordination |

### 2.2 Deep Reinforcement Learning Approaches

#### 2.2.1 Single-Agent RL

- **IntelliLight (Wei et al., KDD 2018)**: Pioneering DRL approach using Deep Q-Network (DQN) with phase-aware state representation. Combined traffic images with intersection metadata. Demonstrated 20%+ improvement over traditional methods on real-world datasets.

- **PressLight (Wei et al., KDD 2019)**: Integrated the theoretical max-pressure concept into RL reward design. Used "pressure" (difference between upstream and downstream vehicle counts) as reward signal. Showed that this theoretically-grounded reward enables faster convergence and better performance.

- **AttendLight (Oroojlooy et al., NeurIPS 2020 Workshop)**: Applied attention mechanisms to traffic state encoding. Enabled generalization across different intersection topologies by learning which traffic movements to focus on.

- **GeneraLight (Zhang et al., 2020)**: Focused on generalization through meta-learning, allowing models trained on one intersection to transfer to unseen intersections with different configurations.

#### 2.2.2 Multi-Agent RL (MARL)

- **CoLight (Wei et al., CIKM 2019)**: Graph Attention Network (GAT) for multi-intersection coordination. Each intersection is a node; edges represent road connections. Agents attend to neighboring intersections' states to coordinate. Achieved state-of-the-art results on network-level metrics.

- **MPLight (Chen et al., NeurIPS 2020)**: FRAP (Phase competition) combined with parameter sharing across all agents. Demonstrated that sharing a single neural network with position-dependent inputs scales to 1000+ intersections.

- **MA2C (Chu et al., 2020)**: Multi-Agent Advantage Actor-Critic with spatial discount factor. Agents communicate with neighbors using a fingerprinting mechanism to stabilize training.

- **DuaLight (2024)**: Dual-agent architecture where a "planner" agent selects high-level strategies and a "controller" agent executes fine-grained phase timing. Achieves better long-horizon planning while maintaining reactive control.

- **Advanced-CoLight / Advanced-MPLight (Jiang et al., 2024)**: Enhanced versions incorporating transformer attention, hierarchical communication, and improved reward shaping. Set new benchmarks on CityFlow simulator.

#### 2.2.3 Emerging LLM-Based Approaches

- **LLMLight (2024)**: Uses Large Language Models as traffic signal controllers by converting traffic states to natural language descriptions. The LLM reasons about optimal phase selection. Shows surprising zero-shot transfer ability but high inference latency.

- **LA-Light (2024)**: LLM-Augmented approach where an LLM provides high-level strategic guidance while a lightweight RL agent handles real-time decisions. Combines the reasoning capability of LLMs with the speed of RL.

### 2.3 Computer Vision for Traffic Sensing

- **YOLO-based Vehicle Detection**: YOLOv8/v9 models achieve >95% mAP on vehicle detection at intersections, enabling camera-only traffic monitoring without loop detectors.
- **Vehicle Counting & Speed Estimation**: Transformer-based trackers (ByteTrack, BoT-SORT) enable accurate per-lane vehicle counting and speed estimation from standard CCTV cameras.
- **Pedestrian & Cyclist Detection**: Multi-class detection models can simultaneously detect vehicles, pedestrians, and cyclists for comprehensive intersection monitoring.

### 2.4 Digital Twin for Traffic

- **SUMO (Simulation of Urban MObility)**: Open-source microscopic traffic simulator widely used for RL training and evaluation. Supports realistic car-following and lane-changing models.
- **CityFlow**: High-performance traffic simulator designed for RL research. Supports large-scale networks with thousands of intersections.
- **Digital Twin Calibration**: Recent work on auto-calibrating simulation parameters (car-following models, route choice) from real-world sensor data to improve sim-to-real transfer.

### 2.5 Federated Learning for Traffic

- **FedSignal (2023)**: Federated learning for traffic signal control where each intersection trains locally and shares only model updates. Addresses privacy concerns and communication constraints.
- **Privacy-Preserving Traffic Optimization (2024)**: Differential privacy combined with federated learning to ensure individual vehicle trajectories cannot be reconstructed from shared models.

### 2.6 Summary Table: SOTA Comparison

| Approach | Real-time | Multi-agent | Explainable | Privacy | Transferable | Scalable |
|----------|:---------:|:-----------:|:-----------:|:-------:|:------------:|:--------:|
| SCOOT/SCATS | ✓ | ✗ | ✓ | ✓ | ✗ | ⚠️ |
| IntelliLight | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| PressLight | ✓ | ✗ | ⚠️ | ✗ | ⚠️ | ✗ |
| CoLight | ✓ | ✓ | ✗ | ✗ | ✗ | ⚠️ |
| MPLight | ✓ | ✓ | ✗ | ✗ | ⚠️ | ✓ |
| DuaLight | ✓ | ✓ | ⚠️ | ✗ | ⚠️ | ⚠️ |
| LLMLight | ⚠️ | ✗ | ✓ | ✗ | ✓ | ✗ |
| FedSignal | ✓ | ✓ | ✗ | ✓ | ✗ | ⚠️ |
| **FedNS-Traffic (Ours)** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

---

## 3. Gap Analysis & Motivation for Novel Approach

### 3.1 Identified Gaps

After thorough analysis of existing approaches, we identify the following critical gaps:

1. **Explainability Gap**: Pure neural RL approaches (CoLight, MPLight) are black boxes. Traffic authorities cannot audit or override decisions with confidence. LLM-based approaches offer explainability but sacrifice real-time performance.

2. **Privacy Gap**: Most MARL approaches require sharing raw traffic states (vehicle counts, speeds, camera feeds) between intersections. Federated approaches exist but don't incorporate coordination mechanisms.

3. **Sim-to-Real Gap**: RL agents trained in simulators often underperform in the real world due to distribution shift. No existing approach combines digital twin calibration with continuous online adaptation.

4. **Robustness Gap**: Current systems lack graceful degradation. When sensors fail or unprecedented events occur, agents often make poor decisions. No existing approach combines learned policies with symbolic safety rules.

5. **Scalability-Coordination Tradeoff**: Approaches that coordinate well (CoLight) don't scale to thousands of intersections. Approaches that scale (MPLight) sacrifice fine-grained coordination.

### 3.2 Our Unique Insight

**We propose combining neuro-symbolic AI with federated multi-agent RL and digital twin calibration.** This is novel because:

- **Neuro-symbolic**: A neural RL agent handles pattern recognition and optimization, while a symbolic reasoning layer enforces safety constraints, provides human-readable explanations, and handles edge cases with hand-crafted rules. This addresses the explainability and robustness gaps simultaneously.

- **Federated + Coordinated**: We design a novel federated training protocol where intersections share model gradients (not raw data) but also exchange compressed coordination signals during inference. This addresses the privacy gap without sacrificing coordination.

- **Digital Twin Calibration Loop**: A continuously calibrated digital twin serves as both a training environment and a "what-if" scenario tester, bridging the sim-to-real gap.

---

## 4. Proposed Approach: FedNS-Traffic

### 4.1 System Name

**FedNS-Traffic**: **Fed**erated **N**euro-**S**ymbolic **Traffic** Management System

### 4.2 Core Innovation

A three-layer architecture that uniquely combines:

```
┌─────────────────────────────────────────────────────────┐
│  Layer 3: Symbolic Reasoning & Explainability Engine     │
│  (Safety Rules, Explanation Generation, Override Logic)  │
├─────────────────────────────────────────────────────────┤
│  Layer 2: Federated Multi-Agent RL with Graph Attention  │
│  (Distributed Training, Compressed Coordination)         │
├─────────────────────────────────────────────────────────┤
│  Layer 1: Digital Twin + Real-time Perception Pipeline   │
│  (Sensor Fusion, Simulation Calibration, Edge Inference) │
└─────────────────────────────────────────────────────────┘
```

### 4.3 Key Components

#### 4.3.1 Layer 1: Digital Twin + Perception Pipeline

**Real-time Perception:**
- Edge-deployed YOLOv8-nano models for vehicle/pedestrian detection on CCTV feeds
- Lightweight ByteTrack for multi-object tracking and lane-level counting
- Sensor fusion combining camera data with loop detectors and radar (where available)
- All processing on edge devices (NVIDIA Jetson) — raw video never leaves the intersection

**Digital Twin Engine:**
- SUMO-based microscopic simulation of the managed road network
- Auto-calibration module that continuously adjusts simulation parameters to match real-world observations
- Used for: (a) RL pre-training, (b) what-if scenario testing, (c) counterfactual explanations

#### 4.3.2 Layer 2: Federated Multi-Agent RL

**Agent Architecture (per intersection):**
```
State: [queue_lengths, waiting_times, speeds, phase_info, time_features, neighbor_embeddings]
       ↓
Encoder: Shared Feature Extractor (MLP) 
       ↓
Graph Attention: Attend to compressed neighbor states (K-hop neighborhood)
       ↓
Dueling DQN: State-value + Phase-advantage streams
       ↓
Action: Next phase selection + Duration adjustment
```

**Novel Federated Protocol:**
- Each intersection trains its RL agent locally using real-time experience + digital twin rollouts
- Every N minutes, agents share encrypted gradient updates with a regional aggregation server
- The aggregation server performs FedAvg with intersection-topology-aware weighting
- Privacy guarantee: Differential privacy noise added to gradients; raw traffic data never transmitted

**Compressed Coordination Protocol:**
- During inference, each agent computes a low-dimensional coordination embedding (8-16 floats)
- Embeddings are broadcast to K-hop neighbors (K=2 by default)
- Graph attention layer incorporates neighbor embeddings into decision-making
- Total communication: ~128 bytes per intersection per decision cycle (sub-second latency)

#### 4.3.3 Layer 3: Symbolic Reasoning & Explainability

**Safety Rule Engine (hard constraints enforced before any action):**
```
RULE minimum_green: IF phase_active_time < MIN_GREEN[phase] THEN block_phase_change
RULE maximum_green: IF phase_active_time > MAX_GREEN[phase] THEN force_phase_change  
RULE pedestrian_safety: IF pedestrian_phase_requested AND wait_time > MAX_PED_WAIT THEN priority_insert
RULE emergency_preemption: IF emergency_vehicle_detected(direction) THEN preempt_to(direction)
RULE all_red_clearance: BETWEEN phase_changes INSERT all_red(CLEARANCE_TIME)
RULE conflict_check: IF proposed_phases_conflict(phase_a, phase_b) THEN reject_action
```

**Explainability Engine:**
- For each signal decision, generates a human-readable explanation:
  ```
  "Phase changed to NS-Green because:
   - North queue: 23 vehicles (high) vs East queue: 5 vehicles (low)
   - North approach has been waiting 45s (above threshold)
   - Coordination signal from upstream intersection indicates platoon arriving from North in ~15s
   - Safety check: All minimum green times satisfied, pedestrian phase not pending"
  ```
- Explanations logged for auditing and stored for 30 days
- Dashboard displays real-time reasoning for each intersection

**Edge Case Handler:**
- Symbolic rules for scenarios rarely seen in training data:
  - Sensor failure → Fall back to time-of-day plan
  - Major accident → Activate emergency routing coordination
  - Special event → Load pre-computed event traffic plan
  - Power outage recovery → All-way flash sequence for safety

### 4.4 What Makes This Approach Unique

| Innovation | Why It's New | Why It Matters |
|------------|-------------|----------------|
| Neuro-symbolic fusion | No prior traffic RL system combines neural RL with a symbolic safety/explainability layer | Addresses the critical trust gap that prevents real-world deployment |
| Federated + coordinated MARL | Existing federated traffic systems don't support real-time coordination; existing MARL systems aren't federated | Enables privacy-preserving city-wide deployment |
| Continuously calibrated digital twin | Prior work uses static simulators; we auto-calibrate to close sim-to-real gap | Dramatically improves real-world RL performance |
| Compressed coordination embeddings | Novel communication protocol balancing bandwidth and coordination quality | Enables real-time multi-agent coordination at scale |
| Edge-first architecture | Most systems require cloud inference; we run everything at the intersection | Sub-100ms decision latency, works during internet outages |

---

## 5. System Architecture

### 5.1 High-Level Architecture

```
                            ┌──────────────────────────┐
                            │     Central Dashboard    │
                            │  (Monitoring & Override)  │
                            └────────────┬─────────────┘
                                         │ HTTPS/WSS
                            ┌────────────▼─────────────┐
                            │   Regional Aggregation    │
                            │        Server(s)          │
                            │  ┌─────────────────────┐  │
                            │  │  Federated Averager  │  │
                            │  │  Digital Twin Calib. │  │
                            │  │  Global Analytics    │  │
                            │  └─────────────────────┘  │
                            └────────────┬─────────────┘
                       ┌─────────────────┼─────────────────┐
                       │                 │                  │
              ┌────────▼───────┐ ┌───────▼────────┐ ┌──────▼─────────┐
              │ Intersection 1 │ │ Intersection 2 │ │ Intersection N │
              │  Edge Unit     │ │  Edge Unit     │ │  Edge Unit     │
              │ ┌────────────┐ │ │ ┌────────────┐ │ │ ┌────────────┐ │
              │ │ Perception │ │ │ │ Perception │ │ │ │ Perception │ │
              │ │ RL Agent   │ │ │ │ RL Agent   │ │ │ │ RL Agent   │ │
              │ │ Sym. Rules │ │ │ │ Sym. Rules │ │ │ │ Sym. Rules │ │
              │ │ Dig. Twin  │ │ │ │ Dig. Twin  │ │ │ │ Dig. Twin  │ │
              │ └────────────┘ │ │ └────────────┘ │ │ └────────────┘ │
              └────────────────┘ └────────────────┘ └────────────────┘
                    ▲                   ▲                   ▲
              ┌─────┴─────┐      ┌─────┴─────┐      ┌─────┴─────┐
              │  Cameras  │      │  Cameras  │      │  Cameras  │
              │  Sensors  │      │  Sensors  │      │  Sensors  │
              │  Signal   │      │  Signal   │      │  Signal   │
              │  Controller│     │  Controller│     │  Controller│
              └───────────┘      └───────────┘      └───────────┘
```

### 5.2 Edge Unit Detail

```
┌─────────────────────────────────────────────────────────────────┐
│                    INTERSECTION EDGE UNIT                        │
│                   (NVIDIA Jetson Orin Nano)                      │
│                                                                  │
│  ┌──────────┐   ┌──────────────┐   ┌─────────────────────────┐ │
│  │ Camera   │──▶│ YOLOv8-nano  │──▶│ State Representation    │ │
│  │ Feed(s)  │   │ + ByteTrack  │   │ [counts, speeds, waits, │ │
│  └──────────┘   └──────────────┘   │  phase, time_features]  │ │
│  ┌──────────┐                      └────────────┬────────────┘ │
│  │ Loop Det.│──────────────────────────────────▶ │              │
│  │ Radar    │                                    │              │
│  └──────────┘                                    ▼              │
│                              ┌────────────────────────────────┐ │
│                              │      RL AGENT (Dueling DQN)    │ │
│  ┌────────────────┐          │  ┌──────────┐  ┌────────────┐ │ │
│  │ Neighbor Coord.│────────▶ │  │ Feature  │─▶│ Graph Attn │ │ │
│  │ Embeddings     │          │  │ Encoder  │  │ (Neighbors)│ │ │
│  └────────────────┘          │  └──────────┘  └──────┬─────┘ │ │
│                              │                       ▼       │ │
│                              │              ┌──────────────┐ │ │
│                              │              │ Dueling Head │ │ │
│                              │              │ V(s) + A(s,a)│ │ │
│                              │              └──────┬───────┘ │ │
│                              └─────────────────────┼─────────┘ │
│                                                    ▼           │
│                              ┌────────────────────────────────┐ │
│                              │   SYMBOLIC SAFETY ENGINE       │ │
│                              │   • Min/Max green check        │ │
│                              │   • Conflict detection         │ │
│                              │   • Emergency preemption       │ │
│                              │   • Pedestrian priority        │ │
│                              │   • Explanation generation     │ │
│                              └──────────────┬─────────────────┘ │
│                                             ▼                   │
│                              ┌────────────────────────────────┐ │
│                              │   SIGNAL CONTROLLER INTERFACE  │ │
│                              │   → Set phase + duration       │ │
│                              └────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  LOCAL DIGITAL TWIN (SUMO lite)                            │ │
│  │  • Pre-training environment  • What-if scenario tester     │ │
│  │  • Auto-calibrated from live data  • Counterfactual expl.  │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Data Flow

```
1. SENSE:    Cameras/Sensors → Edge Perception → Traffic State (every 1s)
2. SHARE:    Traffic State → Compressed Embedding → Broadcast to neighbors (every 1s)
3. DECIDE:   State + Neighbor Embeddings → RL Agent → Proposed Action (every 5-15s)
4. VALIDATE: Proposed Action → Symbolic Safety Engine → Safe Action + Explanation
5. ACT:      Safe Action → Signal Controller → Physical Traffic Lights
6. LEARN:    Experience Buffer → Local Training → Federated Gradient Sharing (every 5min)
7. CALIBRATE: Live Data vs Simulation → Digital Twin Calibration (every 15min)
```

### 5.4 Communication Architecture

```
┌─────────────────────────────────────────────┐
│          Communication Protocol             │
├─────────────────────────────────────────────┤
│                                             │
│  Real-time (every 1s):                      │
│  ┌──────────┐    MQTT/UDP     ┌──────────┐ │
│  │ Agent A  │ ──────────────▶ │ Agent B  │ │
│  │ Embedding│ ◀────────────── │ Embedding│ │
│  └──────────┘  128 bytes/msg  └──────────┘ │
│                                             │
│  Training (every 5 min):                    │
│  ┌──────────┐   gRPC/TLS     ┌──────────┐ │
│  │ Agent    │ ──────────────▶ │ Regional │ │
│  │ Gradient │                 │ Aggreg.  │ │
│  │ + DP noise│ ◀────────────  │ Server   │ │
│  └──────────┘  Avg'd model   └──────────┘ │
│                                             │
│  Monitoring (every 10s):                    │
│  ┌──────────┐  WebSocket     ┌──────────┐  │
│  │ Edge Unit│ ──────────────▶│Dashboard │  │
│  │ Metrics  │                │ (Web UI) │  │
│  └──────────┘                └──────────┘  │
│                                             │
└─────────────────────────────────────────────┘
```

---

## 6. Technical Design

### 6.1 State Representation

Each intersection agent observes:

```python
state = {
    # Per-lane features (for each incoming lane)
    "queue_lengths": [int] * num_lanes,        # vehicles waiting
    "vehicle_speeds": [float] * num_lanes,     # avg speed (km/h)
    "waiting_times": [float] * num_lanes,      # avg wait (seconds)
    "vehicle_counts": [int] * num_lanes,       # total vehicles in detection zone
    
    # Intersection features
    "current_phase": one_hot(num_phases),       # current signal phase
    "phase_duration": float,                    # seconds in current phase
    "time_of_day": float,                       # normalized [0, 1]
    "day_of_week": one_hot(7),                  # day encoding
    
    # Neighbor coordination (received embeddings)
    "neighbor_embeddings": [float] * (K * embed_dim),  # K neighbors × 16-dim
    
    # Pedestrian features
    "pedestrian_waiting": [bool] * num_crosswalks,
    "pedestrian_wait_time": [float] * num_crosswalks,
}
```

### 6.2 Action Space

```python
# Option 1: Phase selection (simpler, more common)
action = phase_id  # Select next phase from valid phases

# Option 2: Phase + Duration (our approach)
action = (phase_id, duration_bucket)
# duration_bucket ∈ {short: 10s, medium: 20s, long: 30s, extended: 45s}
# Final duration is adjusted by symbolic layer within [MIN_GREEN, MAX_GREEN]
```

### 6.3 Reward Design

Our reward function combines multiple objectives:

```python
def compute_reward(state, next_state, action):
    # Primary: Pressure-based (theoretically grounded)
    pressure = sum(upstream_counts) - sum(downstream_counts)
    r_pressure = -abs(pressure)  # Minimize pressure
    
    # Secondary: Queue reduction
    r_queue = -(sum(next_state.queue_lengths) - sum(state.queue_lengths))
    
    # Tertiary: Throughput
    r_throughput = vehicles_passed_in_interval
    
    # Penalty: Excessive waiting
    r_wait = -max(0, max(state.waiting_times) - WAIT_THRESHOLD)
    
    # Pedestrian factor
    r_ped = -sum(ped_wait > PED_THRESHOLD for ped_wait in state.pedestrian_wait_time)
    
    # Weighted combination
    reward = (0.4 * r_pressure + 
              0.25 * r_queue + 
              0.2 * r_throughput + 
              0.1 * r_wait + 
              0.05 * r_ped)
    
    return reward
```

### 6.4 Neural Network Architecture

```python
class FedNSTrafficAgent(nn.Module):
    """
    Dueling DQN with Graph Attention for multi-agent coordination.
    """
    def __init__(self, state_dim, num_phases, num_neighbors, embed_dim=16):
        super().__init__()
        
        # Feature Encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64)
        )
        
        # Graph Attention for neighbor coordination
        self.neighbor_attention = GraphAttentionLayer(
            in_features=embed_dim,
            out_features=32,
            num_heads=4
        )
        
        # Coordination embedding generator (shared with neighbors)
        self.embed_generator = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim)
        )
        
        # Dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(64 + 32, 64),  # encoded_state + attended_neighbors
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, num_phases * 4)  # phases × duration_buckets
        )
    
    def forward(self, state, neighbor_embeddings):
        # Encode local state
        encoded = self.encoder(state)
        
        # Generate coordination embedding to share
        coord_embedding = self.embed_generator(encoded)
        
        # Attend to neighbors
        attended = self.neighbor_attention(encoded, neighbor_embeddings)
        
        # Combine for decision
        combined = torch.cat([encoded, attended], dim=-1)
        
        # Dueling architecture
        value = self.value_stream(combined)
        advantage = self.advantage_stream(combined)
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        
        return q_values, coord_embedding
```

### 6.5 Federated Training Protocol

```python
class FederatedTrainer:
    """
    Topology-aware federated averaging with differential privacy.
    """
    def aggregate(self, local_updates, intersection_graph):
        # Weight by traffic volume (busier intersections contribute more)
        weights = compute_traffic_weights(local_updates)
        
        # Topology-aware grouping: cluster nearby intersections
        clusters = partition_by_topology(intersection_graph, num_clusters=K)
        
        # Aggregate within clusters first, then across clusters
        cluster_models = {}
        for cluster_id, members in clusters.items():
            cluster_models[cluster_id] = federated_average(
                models=[local_updates[m] for m in members],
                weights=[weights[m] for m in members]
            )
        
        # Global aggregation
        global_model = federated_average(
            models=list(cluster_models.values()),
            weights=[len(clusters[c]) for c in cluster_models]
        )
        
        return global_model
    
    def add_differential_privacy(self, gradients, epsilon=1.0, delta=1e-5):
        """Add calibrated Gaussian noise for (ε,δ)-differential privacy."""
        sensitivity = compute_gradient_sensitivity(gradients)
        sigma = sensitivity * sqrt(2 * log(1.25 / delta)) / epsilon
        noisy_gradients = gradients + torch.randn_like(gradients) * sigma
        return noisy_gradients
```

### 6.6 Symbolic Safety Engine

```python
class SymbolicSafetyEngine:
    """
    Hard constraint enforcement and explanation generation.
    Runs AFTER the RL agent proposes an action, BEFORE execution.
    """
    
    def __init__(self, config):
        self.min_green = config.min_green_times    # dict: phase -> seconds
        self.max_green = config.max_green_times    # dict: phase -> seconds
        self.clearance_time = config.all_red_time  # seconds
        self.max_ped_wait = config.max_pedestrian_wait
        self.rules = self._load_rules()
    
    def validate_and_explain(self, proposed_action, state):
        """
        Validate proposed action against safety rules.
        Returns (safe_action, explanation, overridden).
        """
        explanation_parts = []
        overridden = False
        action = proposed_action
        
        # Rule 1: Minimum green time
        if state.phase_duration < self.min_green[state.current_phase]:
            action = (state.current_phase, "maintain")
            explanation_parts.append(
                f"Maintained current phase: minimum green time "
                f"({self.min_green[state.current_phase]}s) not yet reached "
                f"(current: {state.phase_duration:.0f}s)"
            )
            overridden = True
        
        # Rule 2: Maximum green time  
        elif state.phase_duration > self.max_green[state.current_phase]:
            action = self._next_valid_phase(state)
            explanation_parts.append(
                f"Forced phase change: maximum green time "
                f"({self.max_green[state.current_phase]}s) exceeded"
            )
            overridden = True
        
        # Rule 3: Emergency vehicle preemption
        if state.emergency_vehicle_detected:
            direction = state.emergency_vehicle_direction
            action = self._emergency_phase(direction)
            explanation_parts.append(
                f"EMERGENCY: Preempting to {direction} phase for emergency vehicle"
            )
            overridden = True
        
        # Rule 4: Pedestrian priority
        if any(w > self.max_ped_wait for w in state.pedestrian_wait_times):
            if not self._phase_serves_pedestrians(action, state):
                action = self._insert_pedestrian_phase(state)
                explanation_parts.append(
                    f"Pedestrian priority: wait time exceeded threshold"
                )
                overridden = True
        
        # Generate full explanation
        if not overridden:
            explanation_parts.append(self._explain_rl_decision(action, state))
        
        explanation = "; ".join(explanation_parts)
        
        return action, explanation, overridden
    
    def _explain_rl_decision(self, action, state):
        """Generate human-readable explanation for RL decision."""
        phase, duration = action
        
        # Find dominant factors
        max_queue_lane = argmax(state.queue_lengths)
        max_wait_lane = argmax(state.waiting_times)
        
        explanation = (
            f"Selected phase {phase} ({duration}s) because: "
            f"Lane {max_queue_lane} has highest queue ({state.queue_lengths[max_queue_lane]} vehicles), "
            f"Lane {max_wait_lane} has longest wait ({state.waiting_times[max_wait_lane]:.0f}s)"
        )
        
        if state.neighbor_influence > 0.3:  # Threshold for notable coordination
            explanation += f"; Coordinating with upstream intersection (platoon arriving)"
        
        return explanation
```

### 6.7 Digital Twin Calibration

```python
class DigitalTwinCalibrator:
    """
    Continuously calibrates SUMO simulation to match real-world observations.
    Uses Bayesian Optimization to tune simulation parameters.
    """
    
    CALIBRATION_PARAMS = [
        "car_following.tau",        # Reaction time
        "car_following.accel",      # Max acceleration
        "car_following.decel",      # Max deceleration  
        "car_following.minGap",     # Minimum gap
        "lane_change.lcStrategic",  # Lane change eagerness
        "route_choice.beta",        # Route choice parameter
    ]
    
    def calibrate(self, real_world_data, sim_params):
        """
        Minimize discrepancy between real-world and simulated traffic metrics.
        """
        def objective(params):
            sim_data = self.run_simulation(params, duration=900)  # 15-min sim
            
            # Compare key metrics
            queue_error = mse(real_world_data.queues, sim_data.queues)
            speed_error = mse(real_world_data.speeds, sim_data.speeds)
            count_error = mse(real_world_data.counts, sim_data.counts)
            
            return 0.4 * queue_error + 0.3 * speed_error + 0.3 * count_error
        
        # Bayesian optimization
        optimizer = BayesianOptimization(
            f=objective,
            pbounds=self._get_param_bounds(),
            random_state=42
        )
        optimizer.maximize(init_points=5, n_iter=20)
        
        return optimizer.max['params']
```

---

## 7. Implementation Plan

### 7.1 Technology Stack

| Component | Technology | Justification |
|-----------|-----------|---------------|
| **RL Framework** | PyTorch + Stable-Baselines3 | Industry standard, good MARL support |
| **Traffic Simulator** | SUMO + CityFlow | SUMO for realism, CityFlow for speed |
| **CV Model** | YOLOv8-nano (Ultralytics) | Best speed/accuracy for edge deployment |
| **Object Tracker** | ByteTrack | Lightweight, SOTA tracking performance |
| **Edge Hardware** | NVIDIA Jetson Orin Nano | 40 TOPS AI performance, $199, low power |
| **Communication** | MQTT (coord.) + gRPC (training) | MQTT for low-latency pub/sub; gRPC for reliable model sync |
| **Backend** | Python + FastAPI | Async, high-performance API framework |
| **Dashboard** | React + D3.js + Leaflet | Interactive maps and real-time charts |
| **Database** | TimescaleDB (metrics) + Redis (state) | Time-series optimized + fast cache |
| **Federated Learning** | Flower (flwr) | Leading open-source FL framework |
| **Symbolic Engine** | Custom Python + Pyke/Drools | Lightweight rule engine |
| **Container** | Docker + K3s (edge Kubernetes) | Lightweight orchestration for edge |
| **CI/CD** | GitHub Actions | Automated testing and deployment |

### 7.2 Development Phases

#### Phase 1: Foundation (Weeks 1-4)

**Goal**: Set up simulation environment, basic RL agent, and project infrastructure.

| Week | Task | Deliverable |
|------|------|-------------|
| 1 | Project setup: repo structure, CI/CD, Docker configs | Working dev environment |
| 1 | SUMO network creation: Design 3×3 grid intersection network | SUMO `.net.xml` and `.rou.xml` files |
| 2 | State/action/reward implementation | `environment.py` with OpenAI Gym interface |
| 2 | Basic DQN agent for single intersection | `agent.py` with training loop |
| 3 | Train and evaluate single-agent DQN | Baseline performance metrics |
| 3 | Implement Dueling DQN + Prioritized Replay | Improved single-agent performance |
| 4 | CityFlow integration for faster training | Dual-simulator support |
| 4 | Unit tests and documentation | 80%+ code coverage |

**Phase 1 Milestone**: Single-intersection agent outperforms fixed-timing by 15%+

#### Phase 2: Multi-Agent & Coordination (Weeks 5-8)

| Week | Task | Deliverable |
|------|------|-------------|
| 5 | Multi-agent environment: independent agents on 3×3 grid | `multi_agent_env.py` |
| 5 | Coordination embedding design and neighbor communication | `coordination.py` |
| 6 | Graph Attention Network for neighbor information | `graph_attention.py` |
| 6 | Train multi-agent system with parameter sharing | Coordinated MARL agents |
| 7 | Evaluate: independent vs. coordinated multi-agent | Comparison report |
| 7 | Scale to 5×5 grid, test scalability | Scalability benchmarks |
| 8 | Hyperparameter optimization (Optuna) | Tuned model |
| 8 | Integration tests | Full pipeline working |

**Phase 2 Milestone**: Multi-agent system outperforms single-agent by 10%+ on network metrics

#### Phase 3: Federated Learning & Privacy (Weeks 9-11)

| Week | Task | Deliverable |
|------|------|-------------|
| 9 | Flower-based federated training setup | `federated_trainer.py` |
| 9 | Topology-aware aggregation strategy | `topology_aggregation.py` |
| 10 | Differential privacy implementation | Privacy-preserving training |
| 10 | Evaluate: centralized vs. federated training | Performance comparison |
| 11 | Federated training with compressed coordination | Full FedMARL system |
| 11 | Privacy audit and compliance documentation | Privacy report |

**Phase 3 Milestone**: Federated training achieves within 5% of centralized, with DP guarantee (ε≤1)

#### Phase 4: Symbolic Safety & Explainability (Weeks 12-14)

| Week | Task | Deliverable |
|------|------|-------------|
| 12 | Symbolic rule engine implementation | `safety_engine.py` |
| 12 | Safety rules: min/max green, conflict, pedestrian, emergency | Rule library |
| 13 | Explanation generation system | `explainer.py` |
| 13 | Integration: RL agent + symbolic engine | Neuro-symbolic agent |
| 14 | Edge case handler: sensor failure, accidents, special events | Robustness module |
| 14 | Evaluate explainability quality (human evaluation) | Explainability report |

**Phase 4 Milestone**: 100% safety rule compliance, explanations rated "understandable" by 90%+ evaluators

#### Phase 5: Computer Vision & Edge Deployment (Weeks 15-18)

| Week | Task | Deliverable |
|------|------|-------------|
| 15 | YOLOv8-nano fine-tuning on traffic datasets | Vehicle detection model |
| 15 | ByteTrack integration for lane-level counting | Tracking pipeline |
| 16 | Sensor fusion: camera + loop detector | Robust perception |
| 16 | Edge optimization: TensorRT, model quantization | Edge-ready models |
| 17 | Jetson Orin deployment and benchmarking | Edge inference pipeline |
| 17 | Latency profiling: end-to-end decision time | Performance benchmarks |
| 18 | Integration testing: perception → RL → signal control | Full edge pipeline |
| 18 | Resilience testing: sensor failures, network issues | Robustness report |

**Phase 5 Milestone**: End-to-end latency < 100ms on Jetson Orin; handles sensor failures gracefully

#### Phase 6: Digital Twin & Calibration (Weeks 19-21)

| Week | Task | Deliverable |
|------|------|-------------|
| 19 | Digital twin auto-calibration module | `digital_twin_calibrator.py` |
| 19 | Bayesian optimization for sim parameter tuning | Calibration pipeline |
| 20 | Sim-to-real transfer evaluation | Transfer learning report |
| 20 | What-if scenario testing interface | Scenario tester |
| 21 | Counterfactual explanation generation | "What would have happened if..." |
| 21 | Continuous calibration loop (real-time) | Auto-calibrating digital twin |

**Phase 6 Milestone**: Calibrated simulation matches real-world within 10% error on key metrics

#### Phase 7: Dashboard & Deployment (Weeks 22-25)

| Week | Task | Deliverable |
|------|------|-------------|
| 22 | Backend API: FastAPI for metrics, control, explanations | REST/WebSocket API |
| 22 | Real-time monitoring dashboard (React + D3.js) | Web dashboard v1 |
| 23 | Interactive map with per-intersection status (Leaflet) | Map visualization |
| 23 | Historical analytics and reporting | Analytics module |
| 24 | Manual override and emergency controls | Operator interface |
| 24 | Alert system: anomaly detection, failures | Alerting pipeline |
| 25 | Docker/K3s packaging for deployment | Deployment artifacts |
| 25 | Documentation: user guide, API docs, deployment guide | Full documentation |

**Phase 7 Milestone**: Production-ready system with monitoring dashboard

#### Phase 8: Evaluation & Paper (Weeks 26-30)

| Week | Task | Deliverable |
|------|------|-------------|
| 26-27 | Comprehensive evaluation on multiple networks | Evaluation results |
| 28 | Ablation study: contribution of each component | Ablation report |
| 29 | Comparison with SOTA baselines | Benchmark comparison |
| 30 | Research paper writing | Draft paper |

### 7.3 Project Structure

```
research/
├── README.md                          # Project overview
├── RESEARCH.md                        # This document
├── docs/
│   ├── architecture.md                # Detailed architecture docs
│   ├── api.md                         # API documentation
│   ├── deployment.md                  # Deployment guide
│   └── evaluation.md                  # Evaluation methodology
├── src/
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── sumo_env.py                # SUMO-based Gym environment
│   │   ├── cityflow_env.py            # CityFlow-based environment
│   │   ├── multi_agent_env.py         # Multi-agent wrapper
│   │   └── state_representation.py    # State encoding
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── dqn_agent.py              # Dueling DQN agent
│   │   ├── graph_attention.py         # GAT for coordination
│   │   ├── coordination.py            # Embedding exchange protocol
│   │   └── fedns_agent.py             # Full FedNS agent
│   ├── federated/
│   │   ├── __init__.py
│   │   ├── flower_client.py           # Flower FL client
│   │   ├── flower_server.py           # Flower FL server
│   │   ├── topology_aggregation.py    # Topology-aware FedAvg
│   │   └── differential_privacy.py    # DP mechanisms
│   ├── symbolic/
│   │   ├── __init__.py
│   │   ├── safety_engine.py           # Rule enforcement
│   │   ├── rules/                     # Rule definitions
│   │   │   ├── safety_rules.py
│   │   │   ├── pedestrian_rules.py
│   │   │   └── emergency_rules.py
│   │   └── explainer.py               # Explanation generation
│   ├── perception/
│   │   ├── __init__.py
│   │   ├── detector.py                # YOLOv8 vehicle detection
│   │   ├── tracker.py                 # ByteTrack integration
│   │   ├── counter.py                 # Lane-level counting
│   │   └── sensor_fusion.py           # Multi-sensor fusion
│   ├── digital_twin/
│   │   ├── __init__.py
│   │   ├── calibrator.py              # Auto-calibration
│   │   ├── scenario_tester.py         # What-if analysis
│   │   └── counterfactual.py          # Counterfactual explanations
│   ├── dashboard/
│   │   ├── backend/
│   │   │   ├── main.py                # FastAPI server
│   │   │   ├── routes/
│   │   │   └── websocket/
│   │   └── frontend/
│   │       ├── src/
│   │       └── package.json
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py                 # Evaluation metrics
│       ├── visualization.py           # Plotting utilities
│       └── config.py                  # Configuration management
├── configs/
│   ├── default.yaml                   # Default hyperparameters
│   ├── sumo_networks/                 # SUMO network files
│   └── training/                      # Training configurations
├── tests/
│   ├── test_environment.py
│   ├── test_agent.py
│   ├── test_safety_engine.py
│   ├── test_federated.py
│   └── test_perception.py
├── scripts/
│   ├── train.py                       # Training entry point
│   ├── evaluate.py                    # Evaluation entry point
│   ├── deploy.py                      # Deployment script
│   └── calibrate.py                   # Digital twin calibration
├── docker/
│   ├── Dockerfile.edge                # Edge device image
│   ├── Dockerfile.server              # Server image
│   └── docker-compose.yml             # Full system compose
├── requirements.txt
├── pyproject.toml
└── .github/
    └── workflows/
        ├── ci.yml                     # CI pipeline
        └── deploy.yml                 # CD pipeline
```

---

## 8. Evaluation Framework

### 8.1 Simulation Benchmarks

| Benchmark Network | Intersections | Description |
|-------------------|:------------:|-------------|
| **Single Intersection** | 1 | 4-way, 3 lanes each direction |
| **3×3 Grid** | 9 | Standard grid, uniform demand |
| **5×5 Grid** | 25 | Larger grid, heterogeneous demand |
| **Arterial Corridor** | 10 | Linear corridor with coordination needs |
| **Real-World: Hangzhou** | 16 | Real topology and traffic data |
| **Real-World: Jinan** | 12 | Real topology and traffic data |
| **Real-World: New York** | 48 | Manhattan grid with real demand |

### 8.2 Baseline Methods

| Baseline | Category | Description |
|----------|----------|-------------|
| **Fixed-Time** | Traditional | Webster's optimal fixed timing |
| **Max-Pressure** | Classical adaptive | Serve highest-pressure phase |
| **SOTL** | Rule-based adaptive | Self-Organizing Traffic Lights |
| **IntelliLight** | Single-agent RL | DQN with phase-aware state |
| **PressLight** | Single-agent RL | Pressure-based reward DQN |
| **CoLight** | Multi-agent RL | Graph attention MARL |
| **MPLight** | Multi-agent RL | Parameter sharing MARL |
| **DuaLight** | Multi-agent RL | Dual-agent architecture |
| **Advanced-CoLight** | Multi-agent RL | Current SOTA on CityFlow |

### 8.3 Key Performance Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Average Travel Time** | Σ(trip_end - trip_start) / N_trips | Primary metric: lower is better |
| **Average Waiting Time** | Σ(time_speed<0.1) / N_vehicles | Time vehicles spend nearly stopped |
| **Average Queue Length** | Σ(queue_length) / (N_intersections × T) | Vehicles waiting per intersection |
| **Throughput** | N_completed_trips / T | Vehicles completing trips per hour |
| **Max Queue Length** | max(queue_length) | Worst-case congestion indicator |
| **Delay Index** | actual_travel_time / free_flow_travel_time | Congestion severity ratio |
| **Fairness (Jain's Index)** | (Σxi)² / (n × Σxi²) | Equity of wait times across directions |
| **CO₂ Emissions** | Σ(emission_per_vehicle × idle_time) | Environmental impact |

### 8.4 System Performance Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| **Decision Latency** | < 100ms | Time from state observation to action |
| **Perception Latency** | < 50ms | Camera frame to traffic state |
| **Communication Latency** | < 10ms | Neighbor embedding exchange |
| **Training Convergence** | < 500 episodes | Episodes to 90% of final performance |
| **Privacy Budget** | ε ≤ 1.0 | Differential privacy guarantee |
| **Safety Compliance** | 100% | All hard constraints always satisfied |
| **Explanation Quality** | > 4/5 | Human rating of explanation clarity |
| **Uptime** | 99.9% | System availability |

### 8.5 Ablation Studies

To validate each component's contribution:

| Experiment | Components Removed | Purpose |
|------------|-------------------|---------|
| No coordination | Graph attention, embeddings | Value of multi-agent coordination |
| No federation | Local training only | Value of federated learning |
| No symbolic layer | Safety engine | Value of neuro-symbolic approach |
| No digital twin | Train directly on real data | Value of simulation pre-training |
| No privacy | Remove DP noise | Privacy-performance tradeoff |
| No edge processing | Cloud-only inference | Value of edge deployment |

---

## 9. Expected Outcomes

### 9.1 Performance Targets

Based on current SOTA results and our novel contributions:

| Metric | Fixed-Time | Current SOTA | Our Target | Improvement |
|--------|:----------:|:------------:|:----------:|:-----------:|
| Avg. Travel Time | 300s | 220s | 195s | ~11% over SOTA |
| Avg. Wait Time | 120s | 65s | 55s | ~15% over SOTA |
| Throughput | 800 veh/hr | 1100 veh/hr | 1200 veh/hr | ~9% over SOTA |
| Max Queue | 45 veh | 25 veh | 20 veh | ~20% over SOTA |

### 9.2 Key Contributions

1. **First neuro-symbolic multi-agent RL system for traffic**: Combines the optimization power of deep RL with the safety guarantees and explainability of symbolic reasoning.

2. **Privacy-preserving coordinated traffic management**: Novel federated protocol enabling city-wide coordination without sharing raw traffic data.

3. **Continuously calibrated digital twin**: Auto-calibration closes the sim-to-real gap, enabling more effective RL training and deployment.

4. **Edge-first architecture**: All inference happens at the intersection, achieving sub-100ms latency and resilience to network outages.

5. **Comprehensive explainability**: Every signal decision comes with a human-readable explanation, enabling trust and regulatory compliance.

### 9.3 Research Publications

Target venues for publication:

| Venue | Type | Focus |
|-------|------|-------|
| KDD | Conference | Full system paper |
| NeurIPS | Conference | Federated neuro-symbolic RL |
| IEEE ITSC | Conference | Traffic-specific results |
| Transportation Research Part C | Journal | Comprehensive evaluation |
| Nature Machine Intelligence | Journal | High-impact systems paper |

---

## 10. References

### Key Papers

1. Wei, H., et al. "IntelliLight: A Reinforcement Learning Approach for Intelligent Traffic Light Control." KDD 2018.
2. Wei, H., et al. "PressLight: Learning Max Pressure Control to Coordinate Traffic Signals in Arterial Network." KDD 2019.
3. Wei, H., et al. "CoLight: Learning Network-level Cooperation for Traffic Signal Control." CIKM 2019.
4. Chen, C., et al. "Toward A Thousand Lights: Decentralized Deep Reinforcement Learning for Large-Scale Traffic Signal Control." AAAI 2020.
5. Oroojlooy, A., et al. "AttendLight: Universal Attention-Based Reinforcement Learning Model for Traffic Signal Control." NeurIPS 2020 Workshop.
6. Zhang, H., et al. "GeneraLight: Improving Environment Generalization of Traffic Signal Control via Meta Reinforcement Learning." CIKM 2020.
7. Chu, T., et al. "Multi-Agent Deep Reinforcement Learning for Large-Scale Traffic Signal Control." IEEE TITS 2020.
8. Varaiya, P. "Max pressure control of a network of signalized intersections." Transportation Research Part C 2013.
9. Jiang, Z., et al. "DuaLight: Dual-Agent Reinforcement Learning for Traffic Signal Control." 2024.
10. Tang, L., et al. "LLMLight: Large Language Models as Traffic Signal Control Agents." 2024.
11. McMahan, B., et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS 2017 (FedAvg).
12. Dwork, C., et al. "The Algorithmic Foundations of Differential Privacy." Foundations and Trends 2014.
13. Garcez, A., et al. "Neurosymbolic AI: The 3rd Wave." Artificial Intelligence Review 2023.
14. Lopez, P.A., et al. "Microscopic Traffic Simulation using SUMO." IEEE ITSC 2018.
15. Zhang, Y., et al. "ByteTrack: Multi-Object Tracking by Associating Every Detection Box." ECCV 2022.
16. Jocher, G., et al. "Ultralytics YOLOv8." 2023.

### Open Source Tools & Frameworks

- SUMO: https://eclipse.dev/sumo/
- CityFlow: https://cityflow-project.github.io/
- Flower (Federated Learning): https://flower.ai/
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
- Ultralytics YOLOv8: https://docs.ultralytics.com/
- PyTorch: https://pytorch.org/
- FastAPI: https://fastapi.tiangolo.com/

---

*Document Version: 1.0*
*Last Updated: March 2026*
*Authors: Research Team*
