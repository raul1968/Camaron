# ROCA Unified Kernel v2.0 — Build & Extension Guide

## Overview

This repository contains the **ROCA (Routed Orbital Capsule Architecture) Unified Kernel v2.0**, a deterministic cognitive engine built on:

* Capsule-based knowledge representation
* Routing-by-agreement (no gradients, no neural nets)
* Orbital hierarchy (0–3)
* Gravity-based relevance
* Temporal frame graph (timeline memory)
* Hardware-aware execution

This kernel is **self-contained and bootable**.

---

## Entry Point

Run the system:

```bash
python roca_kernel.py
```

This executes:

* Kernel bootstrap
* Seed capsule injection
* 3 cognition cycles
* Query test
* Persistence to disk

---

## Core Architecture

### 1. Capsule (Fundamental Unit)

Each capsule contains:

* Identity: `id`, `kind`, `name`
* State: `state`, `content`, `metadata`
* Cognition: `confidence`, `agreement_score`
* Physics: `gravity_score`, `orbit_level`, `lane`
* Memory: `created_at`, `last_used_at`
* Graph: `lineage`, `shadows`, `merged_into`

Required methods:

```python
evaluate_agreement(other) -> float
merge(other) -> Capsule
```

---

### 2. Routing Engine (Consensus System)

Implements **routing-by-agreement**:

* Pairwise comparison
* Threshold: `0.63`
* Rounds: `4`
* Greedy merging (highest agreement first)

No gradients. Fully deterministic.

---

### 3. Gravity + Orbit System

Capsules behave like mass in a cognitive space.

#### Gravity Rules:

* Increases on usage / agreement
* Decays over time
* Clamped [0, 1]

#### Orbit Levels:

| Level | Meaning          |
| ----- | ---------------- |
| 0     | Core identity    |
| 1     | Active reasoning |
| 2     | Working memory   |
| 3     | Raw ingestion    |

Transitions:

* Promote if gravity ≥ 0.8
* Demote if gravity < 0.2

---

### 4. Temporal Graph (Memory)

Each cycle produces a **FrameNode**:

* Active capsules
* Orbit distribution
* Merge edges
* Gravity stats

Stored as:

```
Json/roca_timeline.jsonl
```

This enables:

* Replay
* Debugging
* Future learning systems

---

### 5. Adaptive Resource Manager

Hardware-aware execution:

* Detects CPU + RAM
* Sets capsule budget
* Adjusts under load

Heuristic:

```
~20 capsules per CPU core
~64MB per capsule
```

---

### 6. Persistence

Capsules are saved to:

```
Json/roca_knowledge_base.json
```

Includes:

* Full capsule graph
* Routing stats
* Timeline stats

---

## Main Loop (Deterministic Cognition Cycle)

```python
while running:
    ingest_input()
    active_set = select_by_gravity()
    route_by_agreement()
    update_gravity()
    update_orbits()
    write_frame()
    persist_state()
```

---

## Input System

Text input is:

1. Chunked
2. Embedded deterministically (SHA-based)
3. Converted into **TOPIC capsules**
4. Injected at **orbit level 3**

---

## Query System

```python
kernel.query("text", top_k=5)
```

Returns:

* Semantic matches (cosine similarity)
* Sorted by relevance
* Includes orbit + gravity

---

## Directory Structure (Expected)

```
project_root/
│
├── roca/
│   └── roca_kernel.py
│
├── Json/
│   ├── roca_knowledge_base.json
│   └── roca_timeline.jsonl
│
└── docs/
    └── BUILD_GUIDE.md   <-- (this file)
```

---

## How to Extend the System

### 1. Add New Capsule Types

Extend:

```python
class CapsuleKind(Enum):
```

Examples:

* `POSE`
* `CAMERA`
* `STYLE`
* `TRANSITION`

---

### 2. Add New Capsule Behaviors

Modify:

```python
evaluate_agreement()
merge()
```

Examples:

* Graph-based agreement
* Temporal agreement
* Multi-capsule consensus

---

### 3. Add Skills (Execution Capsules)

Create `SKILL` capsules with:

```python
state = {
    "callable": function_reference
}
```

Then build a dispatcher that executes them when selected.

---

### 4. Build a ROCA Agent Layer

Create a wrapper:

```python
class ROCAAgent:
    def think()
    def act()
    def observe()
```

This sits **on top of the kernel**, not inside it.

---

### 5. UI / Visualization

Recommended:

* Orbit visualization (radial graph)
* Capsule nodes with gravity size
* Timeline playback

---

### 6. CapsuleTimeline (Future Integration)

This kernel already supports:

* FrameNodes
* Temporal edges

You can extend it into:

* Multi-track timeline (pose, camera, style)
* Interpolation between frames
* Segment regeneration

---

## Design Constraints (Important)

### MUST:

* Remain deterministic
* No gradient descent
* No hidden state outside capsules
* All cognition = routing + gravity

### MAY:

* Use controlled randomness in merges
* Add new capsule kinds
* Add execution layers

---

## What This System Is

This is:

* A cognitive kernel
* A symbolic + statistical hybrid
* A graph-based reasoning engine

---

## What This System Is NOT

This is NOT:

* A neural network
* A transformer
* A black-box model

---

## Next Steps (Recommended)

1. Build Agent Layer (decision + action)
2. Add Skill Execution System
3. Implement CapsuleTimeline
4. Create Visualization UI
5. Add persistent long-term learning policies

---

## Final Note

This kernel is already capable of:

* Self-organization
* Memory formation
* Concept merging
* Temporal tracking

Everything else (agents, animation, tools) is built **on top of this**, not inside it.
