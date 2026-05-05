# AoT-PPO: Ahead-of-Time Predictive Scheduling for Multi-Tenant Distributed ML Training

This repository contains a research notebook that simulates **Ahead-of-Time (AoT) scheduling** for multi-tenant distributed machine learning workloads on a **k=4 fat-tree network**.  
The goal is to reduce network contention during AllReduce communication by predicting burst patterns, optimizing job offsets, and training a **PPO-based scheduler** that chooses between **ring** and **tree** collective communication strategies.

## What this project does

- Models a **fat-tree data center network** with congestion on core links.
- Simulates **5 concurrent training tenants** with different periods, bandwidth demands, and priorities.
- Compares **ring AllReduce** and **tree AllReduce** traffic patterns.
- Uses an **AoT profiler + EDF choreographer** to search for better job offsets.
- Trains a **PPO agent** to select scheduling actions dynamically.
- Benchmarks against static baselines such as:
  - All-Ring
  - All-Tree
  - Priority-Tree
  - Reactive PPO

## Key results

In the notebook’s final evaluation, **AoT-PPO** achieved the best overall reward among the tested policies:

| Policy | Reward | ρ | Jain | κ | λ |
|---|---:|---:|---:|---:|---:|
| All-Ring | 81.29 ± 0.00 | 0.119 | 0.566 | 0.932 | +0.0000 |
| All-Tree | 66.58 ± 0.00 | 0.101 | 0.391 | 0.926 | +1.0000 |
| Priority-Tree | 80.84 ± 0.00 | 0.134 | 0.491 | 0.926 | +0.6000 |
| Reactive PPO | 60.20 ± 1.11 | 0.080 | 0.394 | 0.932 | +0.6165 |
| **AoT-PPO (ours)** | **86.32 ± 1.63** | **0.142** | **0.540** | **0.931** | **+0.5204** |

AoT-PPO also improved reward over:
- **All-Ring** by **+5.04**
- **All-Tree** by **+19.74**
- **Priority-Tree** by **+5.48**
- **Reactive PPO** by **+26.13**

## Repository contents

- `AoT_PPO.ipynb` — main notebook with the full simulation, training, and evaluation pipeline

The notebook also generates figures such as:
- `aot_ppo_final_results.png`
- `aot_burst_rhythm_final.png`

## Method overview

### 1) Network model
A **k=4 fat-tree** topology is built with:
- core switches
- aggregation switches
- edge switches
- hosts

Cross-pod traffic is routed through the core, making congestion visible and measurable.

### 2) Workload model
The notebook simulates five heterogeneous DNN training jobs:
- Transformer XL
- ResNet-50
- BERT finetune
- GPT-2 medium
- VGG-19

Each job has its own:
- worker count
- compute / communication phases
- bandwidth demand
- priority
- offset / period

### 3) Scheduling strategies
The project compares:
- **Ring AllReduce**
- **Tree AllReduce**
- **AoT offset optimization**
- **PPO-based dynamic scheduling**

### 4) Reward design
The reward combines:
- delivered throughput
- fairness
- congestion penalty
- latency bonus

This makes the agent prefer schedules that reduce contention while keeping useful parallelism.

## Requirements

The notebook uses:

- `numpy`
- `networkx`
- `matplotlib`

Install them with:

```bash
pip install numpy networkx matplotlib
```

## How to run

1. Clone the repository.
2. Open `AoT_PPO.ipynb` in Jupyter Notebook or JupyterLab.
3. Run the cells from top to bottom.
4. The notebook will:
   - build the fat-tree topology,
   - simulate the workloads,
   - train the PPO agents,
   - evaluate the baselines,
   - and save the final plots.

## Notes

- This is a **simulation study**, not a production scheduler.
- The notebook is designed to demonstrate how **prediction + scheduling** can reduce congestion in distributed training.
- Results may vary slightly depending on random seed and runtime environment.

## Citation / use

If you reuse this work, please credit the project as:

**AoT-PPO: Ahead-of-Time Predictive Scheduling for Multi-Tenant Distributed ML Training**

---

If you want, I can also turn this into a more polished GitHub README with badges, a project structure section, and a “Future Work” section.
