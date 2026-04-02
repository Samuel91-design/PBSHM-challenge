
**DIAMOND A-DC PBSHM Coding Challenge**

---

- **Name**: Ernest Toochukwu, Samuel
- **Date**: 02-04-2026

---
## Population-Based Structural Health Monitoring (PBSHM)

---
A Graph Neural Network (GNN) pipeline for **damage detection** and **damage localisation** across a synthetic population of multi-storey structures. The project progresses from exploratory data analysis through supervised/unsupervised baselines to a full graph-based model, following the Population-Based SHM (PBSHM) paradigm.

---
### Objective

- Propose which raw or derived features might be damage-sensitive

- Build a simple baseline model for damage detection using fixed-length summaries of each structure.

- Implement a simpler unsupervised / anomaly exploratory method that does not rely on a graph neural network.

- Discuss whether damaged structures appear separable and what the limitations of these simpler methods are.

- Implement a graph-based model that uses the within-structure graph and compare it to your simpler baselines.

- Detect whether a structure is damaged using measurement-like features and graph structure.

- Discuss whether the graph formulation is well-motivated and interpretable.

---
## Repository Layout

The repository is organised as follows:

```
├── main_summary.ipynb                   # Master notebook — runs all tasks end-to-end
├── task1_explore_population.ipynb       # EDA: dataset exploration & feature engineering
├── task2_structure_bl_model.ipynb       # Supervised baselines (Random Forest, Logistic Regression)
├── task3_anoms_bl_model.ipynb           # Unsupervised baselines (Isolation Forest, K-Means)
├── task4_gb_model_gnn.ipynb             # Graph-based GNN model (this pipeline)
├── task5_population_gb.ipynb            # Population-graph extensions
├── structures_measurements.json         # Per-structure graph topology + node-level measurements
├── structure_labels.csv                 # Structure-level damage labels + true damaged storey
├── population_edges_geometry.csv        # Population graph edges (geometry-based)
├── population_edge_weights_geometry.csv # Population graph edge weights (cosine similarity)
├── population_metadata.json             # Dataset summary and field descriptions
└── generate_dataset_revised.py          # Reference synthetic data generator
```

---

## How to Run

### 1. Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install scikit-learn pandas numpy matplotlib seaborn networkx
```


### 2. Run Everything (Recommended)

Open and run **`main_summary.ipynb`** top to bottom. It imports and executes each task in order:

```
Task 1 → EDA & feature engineering
Task 2 → Supervised baselines
Task 3 → Unsupervised anomaly detection
Task 4 → GNN graph-based model
Task 5 → Population-graph extensions
```

### 3. Run Tasks Individually

Each notebook is fully self-contained. Open any task notebook and run all cells. Every modelling function is defined inline with explanatory comments.

---

## Key Design Decisions

### Graph Representation

Each structure is modelled as a **directed sequential graph** where nodes are storeys and edges encode inter-storey transitions. This directly mirrors the physical system: damage manifests as a stiffness discontinuity between adjacent storeys, which edge features are designed to capture explicitly.

**Node features (4 dimensions, globally StandardScaled):**

| Feature | Physical Motivation |
|---|---|
| `height_m` | Storey elevation — proxy for gravitational load distribution |
| `dominant_modal_frequency_Hz²` | Proportional to lateral stiffness (Rayleigh quotient) |
| `abs_freq²_diff` | Sequential stiffness change between adjacent storeys |
| `local_freq²_dev` | Deviation from structure's own mean — internal anomaly signal |

**Edge features (5 dimensions, unscaled — ratio-based):**

| Feature | Physical Motivation |
|---|---|
| `Δfreq²` | Signed stiffness gradient across the edge |
| `\|Δfreq²\|` | Absolute stiffness change magnitude |
| `Δheight` | Inter-storey height spacing |
| `stiffness_ratio` | Relative lateral stiffness between storeys |
| `inv_height³_ratio` | Theoretical cantilever stiffness ratio (k ∝ 1/h³) |

**Structure-level physics features (3 dimensions, globally StandardScaled):**

| Feature | Physical Motivation |
|---|---|
| `frequency_Hz²_std` | Global stiffness variability across all storeys |
| `frequency_std × total_height` | Scale-normalised stiffness irregularity |
| `inverse_height³_std` | Theoretical stiffness spread |

---

### Model Architecture — `GraphModelGNN`

```
Input node features [N, 4]
        │
   ┌────▼────────────────────────────────┐
   │  GATv2Conv (4 heads) → ELU          │  Layer 1: [N, 128]
   │  GATv2Conv (2 heads) → ELU          │  Layer 2: [N, 64]
   │  GATv2Conv (1 head)  → ELU          │  Layer 3: [N, 32]
   └────────────────┬────────────────────┘
                    │ node_embeddings [N, 32]
          ┌─────────┴──────────┐
          │                    │
    ┌─────▼──────┐      ┌──────▼───────────────────────┐
    │  Node Head │      │  Attention-Weighted Pooling   │
    │ Linear→BN  │      │  mean_pool(embed × sigmoid)   │
    │ →ReLU→Drop │      │  max_pool(embed)              │
    │ →Linear    │      │  physics_proj(struct_feats)   │
    └─────┬──────┘      └──────────────┬────────────────┘
          │                            │ cat → [B, 96]
    node_logits [N,1]          ┌───────▼────────┐
                               │  Graph Head    │
                               │  Linear→ReLU   │
                               │  →Drop→Linear  │
                               └───────┬────────┘
                               structure_logits [B,1]
```

Standard GAT computes attention before aggregation (static attention problem). GATv2 evaluates attention *after* concatenating source and target, making it strictly more expressive, important here since the damage signal lies precisely in the *asymmetry* between adjacent nodes.

**Why 3 stacked layers?** A single GATv2 layer only aggregates 1-hop neighbours (adjacent storey). Two layers reach 2 hops, three layers reach the full span of a 6–8 storey structure. Three layers are the minimum for the model to see the global frequency profile before predicting.

---

### Dual-Head Training Objective

The model is trained on three losses simultaneously:

```
total_loss = structure_loss + 0.5 × node_loss + consistency_loss
```

| Loss | Type | Purpose |
|---|---|---|
| `structure_loss` | `BCEWithLogitsLoss` + `pos_weight` | Graph-level damage detection |
| `node_loss` | `BCEWithLogitsLoss` | Storey-level damage localisation |
| `consistency_loss` | `MSELoss` | Forces graph head and mean of node head to agree |

**Smooth node labels:** Rather than a hard 1/0 per storey, node targets decay as `1 / (1 + |i − damaged_storey|)`. This penalises the model less for being one storey off, and provides gradient signal to all nodes rather than only the single damaged one.

**Positive class weight:** Set to `N_total / N_damaged` to counteract the ~30% damage prevalence without oversampling.

---

### Data Pipeline — Leakage-Free Two-Pass Design

```
FIRST PASS  → collect all node & structure features across the full dataset
            → fit StandardScaler once on the full feature matrix

SECOND PASS → transform each structure's features using the fitted scaler
            → build PyG Data objects with x, edge_index, edge_attr, node_y, physics_features
```

The scalers are fitted **before** the train/validation split inside `train_and_evaluate`, which means they see the full dataset. This is standard practice for graph datasets where the split is at the graph level, not the feature level, and where leakage between individual node rows is not a concern.

---

### Evaluation Strategy

- **Stratified 3-Fold Cross-Validation:** Preserves the ~30/70 damaged/healthy class ratio across every fold.
- **Best model selection:** Based on held-out ROC-AUC (not the last fold).
- **Metrics reported: Accuracy:** F1-Score, ROC-AUC per fold + mean ± std.

---

## Interpretability

The `visualize_structure_and_node_predictions` function produces a 2×3 subplot grid for any set of structure IDs. Each panel shows:

- **Bar chart:** Per-storey softmax damage probability
- **Green bar:** Predicted storey = true damaged storey (correct localisation)
- **Orange bar:** Predicted storey ≠ true (wrong localisation)
- **Blue bar:** True damaged storey not predicted (missed)
- **Red star:** Ground-truth damaged storey
- **Green dashed line:** Structure-level P(Damage) from the graph head

This dual-view lets you verify that the graph head (detection) and node head (localisation) are internally consistent, and diagnose failure modes per structure.

---

## Notes

- All random seeds are fixed (`random_state=50`) for reproducibility.
- The pipeline runs on CPU by default; CUDA is used automatically if available.
- {Yet to be done}- `task5_population_gb.ipynb` extends the within-structure graphs to a population-level graph using the geometry-based edges and cosine similarity weights — this is a separate, optional analysis.