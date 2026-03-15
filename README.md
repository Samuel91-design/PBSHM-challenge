# PBSHM Coding Challenge — Doctoral Candidate Assignment

**Structural Health Monitoring Group · ETH IBK**

---

## Overview

This repository contains the data and instructions for a take-home coding assignment on **Population-Based Structural Health Monitoring (PBSHM)**.

You will work with a simulated population of 50 structures and build models that exploit population-level information to improve damage detection — with a Graph Neural Network (GNN) as the required centrepiece.

| | |
|---|---|
| **Presentation** | 10-minute talk + Q&A |
| **Language** | Your choice (Python, MATLAB, R, …) |
| **Submission** | Email your GitHub repo link by the agreed deadline |

---

## Dataset

The dataset simulates a population of 50 three-DOF shear-frame structures (e.g. wind turbine towers) with varying mass, stiffness, damping, and geometric properties. Modal features have been extracted from vibration measurements. Approximately 30% of structures are damaged (localised stiffness reduction).

| File | Description |
|---|---|
| `node_features.csv` | 12 features per structure: structural parameters + natural frequencies + MAC values |
| `labels.csv` | Binary damage label per structure (0 = healthy, 1 = damaged) |
| `edges.csv` | Pre-computed population graph — k-NN (k=5) based on structural similarity |
| `edge_weights.csv` | Cosine similarity weight for each edge |
| `population_metadata.json` | Column descriptions, graph statistics, generation notes — **read this first** |
| `generate_dataset.py` | Script used to generate the data (for reference/reproducibility) |

### Loading the data

```python
# Python
import pandas as pd
features = pd.read_csv("node_features.csv")
labels   = pd.read_csv("labels.csv")
edges    = pd.read_csv("edges.csv")
```

```matlab
% MATLAB
features = readtable("node_features.csv");
labels   = readtable("labels.csv");
edges    = readtable("edges.csv");
```

```r
# R
features <- read.csv("node_features.csv")
labels   <- read.csv("labels.csv")
edges    <- read.csv("edges.csv")
```

---

## Tasks

### Task 1 — Explore the population

Characterise the dataset. Visualise the population graph (node colour = damage label), the distribution of features, and the separation between healthy and damaged structures.

Comment on which features seem most informative for damage detection and justify any preprocessing choices.

> We are looking for physical intuition, not just plots.

---

### Task 2 — Single-structure surrogate (baseline)

Build a surrogate model that predicts damage for each structure using **only that structure's own features** — no population information. Choose your model and justify the choice.

Report appropriate classification metrics (accuracy, F1, ROC-AUC) using cross-validation. This is your **non-population baseline**.

> Architecture is your decision. What matters is the reasoning behind it.

---

### Task 3 — Population-level damage detection with a GNN ⬅ required

Implement a **Graph Neural Network** that exploits the population graph for node-level damage classification. The provided edges are a starting point — you may modify or replace them.

- Train and evaluate using the same protocol as Task 2 so results are directly comparable
- Visualise the learned node embeddings and comment on what the GNN has encoded
- Discuss: what does the message-passing mechanism represent physically in a PBSHM context?

> This is the only task where the model type is prescribed. Library and architecture are your choice.

---

### Task 4 — Transferability to unseen structures

Design and implement a scenario where **5 structures are withheld entirely during training** and must be labelled at test time by connecting them to the population.

- How do you decide which edges to assign to the new nodes?
- Compare performance to your Task 2 baseline — does the population help?
- What are the failure modes? Under what conditions would the population graph mislead rather than help?

> We are particularly interested in the critical reflection, not just the accuracy number.

---

### Task 5 — Population graph construction *(extension, if time permits)*

The provided edges are one possible population graph. Propose and implement an alternative edge construction strategy. Analyse how graph topology affects GNN performance and what this implies for PBSHM practice — particularly for heterogeneous populations.

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
| Justification of surrogate choice and experimental design | 20 |
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

> Slides covering the results and transfer scenario are the most revealing. We want to see whether you connect numerical results to physical SHM intuition — not just report metrics.

---

## Rules

- Any open-source library or toolbox is permitted
- Internet access for documentation is permitted
- You may use AI coding assistants, but **be prepared to explain every line in your presentation**
- Collaboration with other candidates is not permitted
- Submit by emailing your GitHub repo link before the agreed deadline; repos must remain accessible for review

---

*Questions about the data or setup? Contact the group before starting your 6-hour window.*
