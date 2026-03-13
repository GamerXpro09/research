# Implementation Roadmap: FedNS-Traffic

## Timeline Overview (30 Weeks)

```
Phase 1: Foundation          ████████░░░░░░░░░░░░░░░░░░░░░░  Weeks 1-4
Phase 2: Multi-Agent         ░░░░░░░░████████░░░░░░░░░░░░░░  Weeks 5-8
Phase 3: Federated Learning  ░░░░░░░░░░░░░░░░██████░░░░░░░░  Weeks 9-11
Phase 4: Neuro-Symbolic      ░░░░░░░░░░░░░░░░░░░░░░██████░░  Weeks 12-14
Phase 5: Edge & CV           ░░░░░░░░░░░░░░░░░░░░░░░░████████ Weeks 15-18
Phase 6: Digital Twin        ░░░░░░░░░░░░░░░░░░░░░░░░░░██████ Weeks 19-21
Phase 7: Dashboard & Deploy  ░░░░░░░░░░░░░░░░░░░░░░░░░░████████ Weeks 22-25
Phase 8: Evaluation & Paper  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░██████████ Weeks 26-30
```

---

## Phase 1: Foundation (Weeks 1-4)

**Goal**: Set up simulation environment, basic RL agent, and project infrastructure.

### Week 1: Project Setup
- [ ] Initialize Python project with `pyproject.toml`
- [ ] Set up development environment (Docker, pre-commit hooks, CI)
- [ ] Install SUMO and CityFlow simulators
- [ ] Create SUMO network: 4-way single intersection
- [ ] Create SUMO network: 3×3 grid intersection network
- [ ] Generate traffic demand files (low, medium, high, peak scenarios)

### Week 2: Environment Implementation
- [ ] Implement `SumoEnvironment` with OpenAI Gym interface
- [ ] Define state representation (per-lane features, phase, time)
- [ ] Define action space (phase selection + duration buckets)
- [ ] Implement reward function (pressure + queue + throughput + wait)
- [ ] Unit tests for environment

### Week 3: Single-Agent RL
- [ ] Implement Dueling DQN agent with prioritized experience replay
- [ ] Training loop with epsilon-greedy exploration
- [ ] Train on single intersection, evaluate vs. fixed-timing
- [ ] Hyperparameter tuning (learning rate, buffer size, batch size)

### Week 4: Baseline & Testing
- [ ] Implement fixed-timing and max-pressure baselines
- [ ] CityFlow environment wrapper for faster training
- [ ] Comprehensive unit tests (80%+ coverage)
- [ ] Documentation for Phase 1 components

**✅ Milestone**: Single-agent outperforms fixed-timing by 15%+ on average travel time

---

## Phase 2: Multi-Agent & Coordination (Weeks 5-8)

**Goal**: Scale to multiple intersections with coordinated agents.

### Week 5: Multi-Agent Environment
- [ ] Implement `MultiAgentEnvironment` wrapper
- [ ] Independent agents: each intersection has separate DQN
- [ ] Coordination embedding design (16-float compressed state)
- [ ] MQTT-simulated neighbor communication protocol

### Week 6: Graph Attention Network
- [ ] Implement Graph Attention Layer (multi-head)
- [ ] Integrate GAT into agent architecture
- [ ] Coordinate embedding exchange between neighbors
- [ ] Train coordinated multi-agent system on 3×3 grid

### Week 7: Evaluation & Scaling
- [ ] Compare: independent vs. coordinated agents
- [ ] Scale to 5×5 grid network
- [ ] Parameter sharing experiments (shared vs. independent weights)
- [ ] Network-level metrics evaluation

### Week 8: Optimization
- [ ] Hyperparameter optimization with Optuna
- [ ] Ablation study: attention heads, embedding dimension, K-hop range
- [ ] Integration tests for full multi-agent pipeline
- [ ] Performance profiling and optimization

**✅ Milestone**: Coordinated multi-agent outperforms independent agents by 10%+ on network delay

---

## Phase 3: Federated Learning & Privacy (Weeks 9-11)

**Goal**: Enable privacy-preserving distributed training.

### Week 9: Federated Infrastructure
- [ ] Set up Flower federated learning framework
- [ ] Implement `FlowerClient` for intersection agents
- [ ] Implement `FlowerServer` with FedAvg aggregation
- [ ] Test basic federated training loop

### Week 10: Advanced Federated Features
- [ ] Topology-aware aggregation (cluster nearby intersections)
- [ ] Traffic-volume-weighted averaging
- [ ] Differential privacy: gradient clipping + Gaussian noise
- [ ] Privacy budget tracking and reporting

### Week 11: Integration & Evaluation
- [ ] Combine federated training with real-time coordination
- [ ] Compare: centralized vs. federated training performance
- [ ] Privacy audit: verify DP guarantees (ε ≤ 1.0)
- [ ] Communication efficiency analysis

**✅ Milestone**: Federated training achieves within 5% of centralized performance with (ε=1.0)-DP

---

## Phase 4: Symbolic Safety & Explainability (Weeks 12-14)

**Goal**: Add safety guarantees and human-readable explanations.

### Week 12: Safety Rule Engine
- [ ] Implement `SymbolicSafetyEngine` class
- [ ] Safety rules: minimum/maximum green times
- [ ] Conflict detection and phase compatibility matrix
- [ ] Emergency vehicle preemption logic
- [ ] Pedestrian priority rules

### Week 13: Explainability Engine
- [ ] Implement explanation generation for each decision
- [ ] Template-based natural language explanations
- [ ] Factor attribution: which state features drove the decision
- [ ] Integration: RL agent → Safety Engine → Final Action + Explanation

### Week 14: Edge Cases & Robustness
- [ ] Sensor failure detection and fallback strategies
- [ ] Anomaly detection for unusual traffic patterns
- [ ] Special event handling (concerts, sports, construction)
- [ ] Human evaluation of explanation quality (survey)

**✅ Milestone**: 100% safety compliance; explanations rated ≥4/5 by traffic engineers

---

## Phase 5: Computer Vision & Edge Deployment (Weeks 15-18)

**Goal**: Real-world perception pipeline on edge hardware.

### Week 15: Vehicle Detection
- [ ] Fine-tune YOLOv8-nano on traffic intersection datasets
- [ ] Multi-class detection: cars, trucks, buses, motorcycles, pedestrians, cyclists
- [ ] Evaluate detection accuracy (target: >90% mAP50)

### Week 16: Tracking & Counting
- [ ] Integrate ByteTrack for multi-object tracking
- [ ] Implement lane-level vehicle counting
- [ ] Speed estimation from tracked trajectories
- [ ] Sensor fusion: camera + loop detector agreement

### Week 17: Edge Optimization
- [ ] TensorRT optimization for YOLOv8-nano on Jetson
- [ ] INT8 quantization for RL agent
- [ ] End-to-end latency benchmarking
- [ ] Memory and power profiling

### Week 18: Integration Testing
- [ ] Full pipeline: camera → detection → tracking → state → RL → signal
- [ ] Stress testing under various conditions
- [ ] Sensor failure simulation and recovery
- [ ] Field testing documentation

**✅ Milestone**: End-to-end latency < 100ms on Jetson Orin; perception accuracy > 90%

---

## Phase 6: Digital Twin & Calibration (Weeks 19-21)

**Goal**: Auto-calibrated simulation for improved training and scenario testing.

### Week 19: Auto-Calibration
- [ ] Implement Bayesian optimization for SUMO parameter tuning
- [ ] Calibration objective: minimize real vs. simulated metric discrepancy
- [ ] Parameters: car-following, lane-changing, route choice models

### Week 20: Sim-to-Real Transfer
- [ ] Evaluate RL agents trained in calibrated vs. uncalibrated simulation
- [ ] Domain randomization during training for robustness
- [ ] Online fine-tuning protocol from real-world data

### Week 21: Scenario Testing
- [ ] What-if scenario testing interface
- [ ] Counterfactual explanation generation
- [ ] Continuous calibration loop (runs every 15 minutes)

**✅ Milestone**: Calibrated sim matches real-world within 10% error on key metrics

---

## Phase 7: Dashboard & Deployment (Weeks 22-25)

**Goal**: Production-ready monitoring and deployment.

### Week 22: Backend API
- [ ] FastAPI server: metrics, control, explanations endpoints
- [ ] WebSocket server for real-time updates
- [ ] TimescaleDB for time-series storage
- [ ] Redis for real-time state cache

### Week 23: Frontend Dashboard
- [ ] React dashboard with D3.js visualizations
- [ ] Interactive Leaflet map with per-intersection status
- [ ] Real-time charts: queue lengths, wait times, throughput
- [ ] Explanation viewer for each intersection

### Week 24: Operations
- [ ] Manual override and emergency control panel
- [ ] Alert system: anomalies, failures, threshold breaches
- [ ] Historical analytics and reporting
- [ ] Role-based access control

### Week 25: Deployment
- [ ] Docker images for edge and server components
- [ ] K3s orchestration for edge fleet management
- [ ] Deployment automation scripts
- [ ] User guide and operator documentation

**✅ Milestone**: Production-ready system with full monitoring and control

---

## Phase 8: Evaluation & Paper (Weeks 26-30)

**Goal**: Comprehensive evaluation and research publication.

### Weeks 26-27: Full Evaluation
- [ ] Evaluate on all benchmark networks (single, grid, arterial, real-world)
- [ ] Compare against all baselines (Fixed, MaxPressure, SOTL, IntelliLight, PressLight, CoLight, MPLight, DuaLight)
- [ ] Statistical significance tests

### Week 28: Ablation Study
- [ ] No coordination ablation
- [ ] No federation ablation
- [ ] No symbolic layer ablation
- [ ] No digital twin ablation
- [ ] No privacy ablation

### Week 29: Analysis
- [ ] Scalability analysis (10 to 1000+ intersections)
- [ ] Robustness analysis (sensor failures, demand spikes)
- [ ] Privacy-utility tradeoff analysis
- [ ] Computational efficiency analysis

### Week 30: Paper
- [ ] Write research paper for KDD/NeurIPS/IEEE ITSC
- [ ] Prepare supplementary materials
- [ ] Open-source release preparation

**✅ Milestone**: Paper submitted; open-source code released

---

## Resource Requirements

### Team

| Role | Count | Responsibility |
|------|:-----:|---------------|
| ML/RL Engineer | 2 | Agent development, training, federated learning |
| CV Engineer | 1 | Perception pipeline, edge optimization |
| Backend Engineer | 1 | API, database, deployment |
| Frontend Engineer | 1 | Dashboard development |
| Traffic Engineer | 1 (consultant) | Domain expertise, rule design, evaluation |
| Research Lead | 1 | Architecture, paper writing, coordination |

### Compute

| Resource | Specification | Duration |
|----------|--------------|----------|
| Training GPU | 4× NVIDIA A100 (40GB) | Weeks 1-21 |
| CI/CD | GitHub Actions runners | Continuous |
| Edge dev kits | 4× Jetson Orin Nano | Weeks 15-25 |
| Regional server | 1× GPU server (T4) | Weeks 22-30 |

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|:----------:|:------:|------------|
| Sim-to-real gap too large | Medium | High | Digital twin calibration + domain randomization |
| Federated convergence issues | Medium | Medium | Topology-aware aggregation + careful hyperparameter tuning |
| Edge latency too high | Low | High | Model pruning, quantization, TensorRT optimization |
| Safety rule conflicts with RL | Low | Critical | Symbolic layer has absolute priority; extensive testing |
| Scalability bottleneck | Medium | Medium | Compressed embeddings + hierarchical communication |

---

*For full technical details, see [RESEARCH.md](../RESEARCH.md)*
*For architecture diagrams, see [architecture.md](architecture.md)*
