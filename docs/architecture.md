# System Architecture: FedNS-Traffic

## Overview

FedNS-Traffic is a three-layer federated neuro-symbolic traffic management system designed for real-time adaptive signal control at urban intersections with heavy multi-directional traffic.

---

## High-Level Architecture

```
                             ┌──────────────────────────┐
                             │    Central Dashboard      │
                             │  (React + D3.js + Leaflet)│
                             │  - Real-time monitoring   │
                             │  - Manual override        │
                             │  - Historical analytics   │
                             └────────────┬─────────────┘
                                          │ HTTPS / WebSocket
                             ┌────────────▼─────────────┐
                             │  Regional Aggregation     │
                             │       Server(s)           │
                             │  ┌─────────────────────┐  │
                             │  │ Federated Averager   │  │
                             │  │ (Topology-Aware)     │  │
                             │  │ Differential Privacy │  │
                             │  │ Global Analytics     │  │
                             │  │ Digital Twin Master  │  │
                             │  └─────────────────────┘  │
                             └────────────┬─────────────┘
                                          │ gRPC/TLS (every 5 min)
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                      │
           ┌────────▼───────┐   ┌────────▼───────┐   ┌─────────▼──────┐
           │ Intersection 1 │   │ Intersection 2 │   │ Intersection N │
           │   Edge Unit    │   │   Edge Unit    │   │   Edge Unit    │
           │  (Jetson Orin) │   │  (Jetson Orin) │   │  (Jetson Orin) │
           └────────────────┘   └────────────────┘   └────────────────┘
                 ▲    ◀── MQTT/UDP coordination (every 1s) ──▶    ▲
           ┌─────┴─────┐                                 ┌────┴──────┐
           │ Cameras    │                                 │ Cameras   │
           │ Sensors    │                                 │ Sensors   │
           │ Signal Ctrl│                                 │ Signal Ctrl│
           └───────────┘                                 └───────────┘
```

---

## Edge Unit Architecture (Per Intersection)

Each intersection is equipped with an autonomous edge computing unit that performs all real-time processing locally:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INTERSECTION EDGE UNIT                            │
│                   (NVIDIA Jetson Orin Nano)                          │
│                                                                      │
│  ════════════════════════ PERCEPTION LAYER ══════════════════════    │
│  ┌──────────────┐   ┌──────────────┐   ┌─────────────────────────┐ │
│  │  CCTV Camera  │──▶│  YOLOv8-nano │──▶│  State Representation   │ │
│  │  Feed (4x)    │   │  + ByteTrack │   │  [queue_len, speed,     │ │
│  └──────────────┘   └──────────────┘   │   wait_time, count]     │ │
│  ┌──────────────┐                      │  per lane (12-16 lanes) │ │
│  │ Loop Detector │─────────────────────▶│  + phase, time_of_day   │ │
│  │ Radar (opt.) │                      │  + pedestrian_state     │ │
│  └──────────────┘                      └────────────┬────────────┘ │
│                                                      │              │
│  ══════════════════════ DECISION LAYER ═════════════ │ ═══════════  │
│                                                      ▼              │
│  ┌────────────────┐          ┌────────────────────────────────────┐ │
│  │ Neighbor Coord. │────────▶│       RL AGENT (Dueling DQN)      │ │
│  │ Embeddings      │         │  ┌──────────┐  ┌───────────────┐  │ │
│  │ (from MQTT)     │         │  │ Feature   │─▶│ Graph Attention│  │ │
│  └────────────────┘         │  │ Encoder   │  │ (K-hop nbrs)  │  │ │
│                              │  └──────────┘  └───────┬───────┘  │ │
│  ┌────────────────┐         │                         ▼          │ │
│  │ Coord. Embedding│◀───────│              ┌────────────────┐    │ │
│  │ (to neighbors)  │        │              │ Dueling Head   │    │ │
│  │ (via MQTT)      │        │              │ V(s) + A(s,a)  │    │ │
│  └────────────────┘         │              └───────┬────────┘    │ │
│                              └─────────────────────┼─────────────┘ │
│                                                     ▼              │
│  ═══════════════════ SAFETY & EXPLAINABILITY ═══════════════════   │
│                              ┌────────────────────────────────────┐ │
│                              │     SYMBOLIC SAFETY ENGINE         │ │
│                              │                                    │ │
│                              │  ✓ Min/Max green time check        │ │
│                              │  ✓ Phase conflict detection        │ │
│                              │  ✓ Emergency vehicle preemption    │ │
│                              │  ✓ Pedestrian priority             │ │
│                              │  ✓ All-red clearance intervals     │ │
│                              │  ✓ Sensor failure fallback         │ │
│                              │                                    │ │
│                              │  Output: Safe Action + Explanation │ │
│                              │  "Phase → NS-Green (20s) because   │ │
│                              │   N queue: 23 veh, waited 45s..."  │ │
│                              └──────────────┬─────────────────────┘ │
│                                              ▼                      │
│                              ┌────────────────────────────────────┐ │
│                              │   SIGNAL CONTROLLER INTERFACE      │ │
│                              │   NTCIP / Custom GPIO → Lights     │ │
│                              └────────────────────────────────────┘ │
│                                                                      │
│  ═══════════════════════ TRAINING LAYER ════════════════════════    │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  LOCAL TRAINING + DIGITAL TWIN                                 │ │
│  │  • Experience replay buffer (last 24h)                        │ │
│  │  • Local DQN training with real + simulated experience         │ │
│  │  • SUMO-lite digital twin (auto-calibrated every 15min)        │ │
│  │  • Gradient computation → DP noise → Upload to aggregator     │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Pipeline

### Real-Time Decision Loop (every 1-5 seconds)

```
Step 1: SENSE
  Camera frames (30fps) ──▶ YOLOv8-nano (50ms) ──▶ ByteTrack ──▶ Per-lane counts & speeds
  Loop detectors ──▶ Vehicle presence & occupancy
  Sensor Fusion ──▶ Unified Traffic State (updated every 1s)

Step 2: SHARE  
  Local State ──▶ Embedding Generator (16 floats) ──▶ MQTT broadcast to K-hop neighbors
  Receive neighbor embeddings (128 bytes each, <10ms latency)

Step 3: DECIDE
  State + Neighbor Embeddings ──▶ Dueling DQN with Graph Attention ──▶ Proposed Phase + Duration
  (Every 5-15 seconds, aligned with minimum phase duration)

Step 4: VALIDATE
  Proposed Action ──▶ Symbolic Safety Engine ──▶ Safe Action + Human-Readable Explanation
  (Hard constraints enforced; RL action overridden if unsafe)

Step 5: ACT
  Safe Action ──▶ Signal Controller ──▶ Physical Traffic Lights Updated
  Explanation ──▶ Logged to database + sent to dashboard

Step 6: OBSERVE
  New Traffic State ──▶ Reward Computation ──▶ Experience stored in replay buffer
```

### Training Loop (every 5 minutes)

```
Step 1: LOCAL TRAINING
  Experience Buffer ──▶ Sample mini-batches ──▶ DQN gradient update
  Digital Twin ──▶ Generate additional simulated experience ──▶ Train

Step 2: FEDERATED SYNC
  Local Gradients ──▶ Add DP Noise (ε=1.0) ──▶ Encrypt ──▶ Upload to Regional Server
  Regional Server ──▶ Topology-Aware FedAvg ──▶ Aggregated Model ──▶ Download to edges

Step 3: DIGITAL TWIN CALIBRATION (every 15 minutes)
  Real-world metrics ──▶ Compare with simulation ──▶ Bayesian Optimization
  ──▶ Updated simulation parameters ──▶ Better training data
```

---

## Communication Protocol

### Inter-Intersection (Real-Time Coordination)

```
Protocol: MQTT v5 over local network (or UDP for ultra-low latency)
Frequency: Every 1 second
Payload: 128 bytes (16 × float32 coordination embedding)
Topology: K-hop neighborhood (K=2 default)

Message Format:
{
  "intersection_id": "INT_001",
  "timestamp": 1709312400,
  "embedding": [0.12, -0.34, 0.56, ...],    // 16 floats
  "phase": 3,                                  // current phase index
  "queue_pressure": 0.78                       // normalized pressure
}
```

### Edge-to-Server (Federated Training)

```
Protocol: gRPC with TLS 1.3
Frequency: Every 5 minutes
Payload: ~2MB (model gradients + DP noise)
Privacy: (ε=1.0, δ=1e-5)-differential privacy

Steps:
1. Edge computes local gradients
2. Clips gradient norm to sensitivity bound
3. Adds calibrated Gaussian noise
4. Encrypts and uploads via gRPC
5. Server performs topology-aware FedAvg
6. Aggregated model downloaded to all edges
```

### Edge-to-Dashboard (Monitoring)

```
Protocol: WebSocket (WSS)
Frequency: Every 10 seconds
Payload: ~1KB JSON

Message Format:
{
  "intersection_id": "INT_001",
  "timestamp": "2026-03-13T19:00:00Z",
  "current_phase": "NS_GREEN",
  "phase_duration": 15,
  "queue_lengths": {"N": 12, "S": 8, "E": 5, "W": 3},
  "avg_wait_times": {"N": 34.2, "S": 22.1, "E": 12.5, "W": 8.3},
  "throughput_last_5min": 142,
  "last_explanation": "Phase NS-Green (20s): North queue highest (12 veh, 34s wait)",
  "safety_overrides": 0,
  "system_health": "HEALTHY"
}
```

---

## Technology Decisions

### Why Dueling DQN (not PPO/SAC)?

- **Discrete action space**: Traffic phases are naturally discrete; DQN excels here
- **Dueling architecture**: Separates state value from action advantage — important for traffic where many states have similar values but different optimal actions
- **Sample efficiency**: Better than policy gradient methods for this action space
- **Stable training**: More stable in multi-agent settings than continuous-action methods

### Why Graph Attention (not GCN/Message Passing)?

- **Attention weights**: Learn which neighbors are most relevant (e.g., upstream intersection with approaching platoon)
- **Dynamic importance**: Attention weights change per timestep unlike fixed GCN weights
- **Interpretable**: Attention maps show which neighbor influenced the decision — aids explainability

### Why Federated (not Centralized)?

- **Privacy**: Raw traffic data (camera feeds, vehicle trajectories) never leaves the intersection
- **Scalability**: No central bottleneck; each edge trains independently
- **Resilience**: System continues operating if server goes down
- **Regulatory**: Compliant with data locality requirements (GDPR, etc.)

### Why SUMO (not CityFlow only)?

- **Realism**: More accurate car-following, lane-changing, and pedestrian models
- **Calibration**: Extensive calibration tools and real-world network import
- **Ecosystem**: Large community, traffic engineering tools, NEMA signal support
- **CityFlow for speed**: Use CityFlow for fast RL exploration, SUMO for final evaluation and digital twin

### Why Edge (not Cloud)?

- **Latency**: Sub-100ms end-to-end vs. 200-500ms with cloud round-trip
- **Reliability**: Works during internet outages
- **Privacy**: Video processed locally; only embeddings transmitted
- **Cost**: No cloud compute bills; one-time hardware investment
- **Bandwidth**: No video streaming needed

---

## Hardware Specifications

### Edge Unit (per Intersection)

| Component | Specification | Purpose |
|-----------|--------------|---------|
| **Compute** | NVIDIA Jetson Orin Nano (40 TOPS) | AI inference + local training |
| **Memory** | 8GB LPDDR5 | Model + experience buffer |
| **Storage** | 128GB NVMe | Logs, local twin, model checkpoints |
| **Network** | Gigabit Ethernet + WiFi 6 | MQTT coordination + gRPC sync |
| **Camera Interface** | 4× MIPI CSI-2 or USB 3.0 | Connect intersection cameras |
| **Signal Interface** | GPIO / RS-232 / NTCIP | Traffic signal controller |
| **Power** | 15W typical | Solar-compatible |
| **Enclosure** | IP67 weatherproof | Outdoor deployment |

### Regional Server

| Component | Specification | Purpose |
|-----------|--------------|---------|
| **Compute** | 32-core CPU + NVIDIA T4 GPU | Federated aggregation + analytics |
| **Memory** | 128GB RAM | Aggregate model + digital twin master |
| **Storage** | 2TB SSD | Historical data + model archive |
| **Network** | 10 Gbps | Handle 100+ intersection uploads |

### Estimated Cost

| Item | Unit Cost | Quantity (50 intersections) | Total |
|------|:---------:|:--------------------------:|:-----:|
| Jetson Orin Nano | $199 | 50 | $9,950 |
| Cameras (existing CCTV) | $0 | — | $0 |
| Weatherproof enclosure | $150 | 50 | $7,500 |
| Networking equipment | $100 | 50 | $5,000 |
| Regional server | $5,000 | 1 | $5,000 |
| Installation per unit | $200 | 50 | $10,000 |
| **Total** | | | **$37,450** |

*~$750 per intersection — significantly cheaper than traditional adaptive systems ($15,000-$50,000 per intersection)*

---

## Security Architecture

```
┌─────────────────────────────────────────────────┐
│              Security Layers                     │
├─────────────────────────────────────────────────┤
│                                                  │
│  Data Privacy:                                   │
│  • Video processed on-edge; never transmitted    │
│  • (ε,δ)-Differential Privacy on all gradients   │
│  • No individual vehicle tracking stored         │
│                                                  │
│  Communication Security:                         │
│  • TLS 1.3 for all gRPC connections             │
│  • MQTT with TLS + client certificates          │
│  • Encrypted at rest (AES-256)                  │
│                                                  │
│  Access Control:                                 │
│  • Role-based access (RBAC) for dashboard       │
│  • API authentication (JWT + API keys)           │
│  • Physical security for edge units             │
│                                                  │
│  Integrity:                                      │
│  • Signed firmware updates                       │
│  • Model integrity verification (checksums)      │
│  • Audit logs for all overrides and changes      │
│                                                  │
│  Resilience:                                     │
│  • Fallback to fixed-timing on any failure       │
│  • Watchdog process monitors agent health        │
│  • Automatic restart and recovery                │
│                                                  │
└─────────────────────────────────────────────────┘
```

---

*See [RESEARCH.md](../RESEARCH.md) for the full technical details and literature review.*
