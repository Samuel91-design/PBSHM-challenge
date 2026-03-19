# PBSHM Coding Challenge — Doctoral Candidate Assignment

**Structural Health Monitoring Group · ETH IBK**

---

## Overview

This repository contains the data and instructions for a take-home coding assignment on **Population-Based Structural Health Monitoring (PBSHM)**.

You will work with a simulated population of 50 structures with **variable geometry** and build models that exploit population-level information to improve damage detection — with a Graph Neural Network (GNN) as the required centrepiece.

| | |
|---|---|
| **Presentation** | 10-minute talk + Q&A |
| **Language** | Your choice (Python, MATLAB, R, …) |
| **Submission** | Email your GitHub repo link by the agreed deadline |

---

## Dataset

The dataset simulates a population of 50 shear-frame structures (e.g. wind turbine towers or offshore platforms) with a **variable number of storeys (4–8)**. Each structure is a small graph — nodes are storeys, edges are inter-storey connections. Structures vary in size and physical properties, so feature matrices are **not the same shape** across the population. Approximately 30% of structures are damaged (localised stiffness reduction in one storey).

This variability is intentional: handling geometrically heterogeneous members is a core challenge of PBSHM.

| File | Description |
|---|---|
| `structures.json` | One entry per structure: geometry, per-storey node features, edge list, damage label |
| `population_edges.csv` | Population-level similarity graph connecting structures (k-NN, k=5) |
| `population_edge_weights.csv` | Same with cosine similarity weights |
| `population_metadata.json` | Full description of every field, design notes — **read this first** |
| `generate_dataset.py` | Script used to generate the data (for reference/reproducibility) |

### Structure of `structures.json`

Each entry looks like this:

```json
{
  "structure_id": 7,
  "n_storeys": 5,
  "damaged": 1,
  "damage_storey": 2,
  "edges": [[0,1],[1,2],[2,3],[3,4]],
  "feature_names": ["mass_kg", "stiffness_Nm", "height_m", "nat_freq_Hz", "mac"],
  "node_features": [
    {"storey": 0, "mass_kg": 1821.4, "stiffness_Nm": 1103452.1, "height_m": 3.91, "nat_freq_Hz": 2.341, "mac": 0.9981},
    {"storey": 1, "mass_kg": 2104.7, "stiffness_Nm": 1244810.3, "height_m": 4.12, "nat_freq_Hz": 5.871, "mac": 0.9903}
  ]
}
```

### Node features

| Feature | Description |
|---|---|
| `mass_kg` | Storey mass [kg] |
| `stiffness_Nm` | Inter-storey stiffness [N/m] — reduced at the damaged storey |
| `height_m` | Storey height [m] |
| `nat_freq_Hz` | Natural frequency of the dominant mode for this storey [Hz] (with sensor noise) |
| `mac` | MAC value of that mode vs. the undamaged reference [-] |

### Loading the data

```python
# Python
import json, pandas as pd
structures = json.load(open("structures.json"))
pop_edges  = pd.read_csv("population_edges.csv")
```

```matlab
% MATLAB
structures = jsondecode(fileread("structures.json"));
pop_edges  = readtable("population_edges.csv");
```

```r
# R
library(jsonlite)
structures <- fromJSON("structures.json")
pop_edges  <- read.csv("population_edges.csv")
```

---

## Tasks

### Task 1 — Explore the population

Characterise the dataset. Visualise the population graph, the distribution of physical and modal features, and the variation in structure geometry across the population.

Comment on which features carry the most damage information and justify any preprocessing choices. Note that structures have different numbers of nodes — how does this affect how you represent or compare them?

> We are looking for physical intuition, not just plots.

---

### Task 2 — Single-structure surrogate (baseline)

Build a surrogate model that predicts damage for each structure using **only that structure's own features** — no population information. Choose your model and justify the choice, bearing in mind that structures have variable size.

Report appropriate classification metrics (accuracy, F1, ROC-AUC) using cross-validation. This is your **non-population baseline**.

> Architecture is your decision. What matters is the reasoning behind it and how you handle variable-length inputs.

---

### Task 3 — Population-level damage detection with a GNN ⬅ required

Implement a **Graph Neural Network** that exploits both the internal structure graph of each member and the population-level similarity graph to perform **graph-level binary classification** (damaged / healthy per structure). The provided population edges are a starting point — you may modify or replace them.

- Train and evaluate using the same protocol as Task 2 so results are directly comparable
- Visualise the learned graph-level embeddings and comment on what the GNN has encoded
- Discuss: what does message passing represent physically in the context of a heterogeneous PBSHM population?

> This is the only task where the model type is prescribed. Library, architecture, and pooling strategy are your choice.

---

### Task 4 — Transferability to unseen structures

Design and implement a scenario where **5 structures are withheld entirely during training** and must be classified at test time by connecting them to the population.

- How do you decide which population edges to assign to the new structures?
- Compare performance to your Task 2 baseline — does the population help?
- What are the failure modes? Under what conditions would the population graph mislead rather than help, particularly given geometric heterogeneity?

> We are particularly interested in the critical reflection, not just the accuracy number.

---

### Task 5 — Population graph construction *(extension, if time permits)*

The provided edges are one possible population graph, built from summary statistics. Propose and implement an alternative edge construction strategy. Analyse how the graph topology affects GNN performance and what this implies for practical PBSHM — particularly when the population contains very heterogeneous members.

> Quality of argument over quantity of experiments.

---

## Deliverables

1. **Your code** in whatever format suits your tool (notebook, `.m` scripts, `.py` files, etc.) — it must be runnable
2. **3–5 slide deck** (PDF or PowerPoint) for the 10-minute presentation
3. **A brief README** in your repo: how to run your code, key design decisions, known limitations

> Clarity of reasoning matters more than accuracy. Comment your code — you will be asked to explain every line in the presentation.

---

## Marking rubric

| Criterion | Points |
|---|---|
| Code correctness & reproducibility | 20 |
| Justification of surrogate choice and handling of variable geometry | 20 |
| Understanding of GNN mechanics and physical interpretation | 25 |
| Transfer scenario — design quality and critical analysis | 20 |
| Presentation clarity & depth in Q&A | 15 |
| **Total** | **100** |

---

## 10-minute presentation structure

| Time | Content |
|---|---|
| 0 – 2 min | Why PBSHM? Why a graph? Frame the problem in your own words |
| 2 – 4 min | Your surrogate choice and GNN design — what you chose and why |
| 4 – 7 min | Results — baseline vs. GNN vs. transfer, with physical interpretation |
| 7 – 10 min | Limitations, open questions, what you would explore with more time |

> Slides covering results and the transfer scenario are the most revealing. We want to see whether you connect numerical results to physical SHM intuition — not just report metrics.

---

## Rules

- Any open-source library or toolbox is permitted
- Internet access for documentation is permitted
- You may use AI coding assistants, but **be prepared to explain every line in your presentation**
- Collaboration with other candidates is not permitted
- Submit by emailing your GitHub repo link before the agreed deadline; repos must remain accessible for review

---

*Questions about the data or setup? Contact the group before starting.*
