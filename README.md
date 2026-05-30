# GraCA: Gradient-guided Graph Connection Assessment

Official implementation of **GraCA**, a gradient-based framework for identifying and pruning task-harmful edges in graphs for GNN training.

## Overview

GraCA leverages hidden-layer gradient behavior during GNN training to compute edge-level risk scores. It identifies edges that are harmful to task optimization and prunes them before downstream training, improving graph quality for semi-supervised learning.

Key components:
- **Direction Consistency (D)**: Measures whether gradient updates from an edge align with the optimization direction
- **Relative Strength (M)**: Captures the relative gradient contribution of each edge
- **Uncertainty Weighting (ρ)**: Down-weights edges where pseudo-labels are unreliable
- **Per-node Adaptive Pruning**: Local threshold and budget per node, preserving minimum degree

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run smoke test (Cora, seed=0)
python scripts/smoke_test.py

# Run full clean experiment on Cora
python scripts/run_graca.py --config configs/graca_lite_cora.yaml --seed 0

# Run noisy-edge experiment
python scripts/run_noisy_edge_experiment.py \
    --config configs/graca_lite_cora.yaml --seed 0 \
    --noise_type low_feature_similarity --noise_ratio 0.10
```

## Project Structure

```
GraCA/
├── src/
│   ├── graca/              # Core GraCA module
│   │   ├── gradient_collector.py  # Gradient collection from hidden layers
│   │   ├── edge_scoring.py        # D, M, rho, H, R, P computation
│   │   ├── pruning.py             # Per-node adaptive pruning
│   │   ├── ema_teacher.py         # EMA teacher for stable pseudo labels
│   │   ├── pseudo_label.py        # Soft pseudo labels with confidence
│   │   └── consistency_loss.py    # Consistency regularization (Full GraCA)
│   ├── models/             # GNN architectures (GCN, GAT, GraphSAGE)
│   ├── training/           # Training loops, losses, evaluation
│   ├── baselines/          # Baseline methods
│   ├── eval/               # Evaluation metrics and noise injection
│   └── utils/              # Config, seed, device utilities
├── configs/                # YAML configs per dataset
├── scripts/                # Experiment runners
├── tests/                  # Unit tests
├── results_clean/          # New experiment results (unified schema)
├── paper_tables_clean/     # Auto-generated paper tables
└── sanitized_graphs_clean/ # Saved pruned graphs
```

## Running Experiments

### Core Experiment Matrix

```bash
# Run all clean experiments (6 datasets, 10 seeds, all methods)
python scripts/run_core_matrix.py --phase clean --seeds 0-9

# Run all noisy experiments (4 datasets, 4 noise types, 4 ratios, 10 seeds)
python scripts/run_core_matrix.py --phase noisy --seeds 0-9

# Run ablation experiments (Cora/CiteSeer/PubMed, noisy 10%)
python scripts/run_core_matrix.py --phase ablation --seeds 0-9

# Run oracle experiments
python scripts/run_core_matrix.py --phase oracle --seeds 0-9

# Dry run (show commands without executing)
python scripts/run_core_matrix.py --phase all --seeds 0-9 --dry_run
```

### Individual Experiments

```bash
# GraCA-lite on a specific dataset
python scripts/run_graca.py --config configs/graca_lite_cora.yaml --seed 0

# Baselines (Original, Random, Homophily)
python scripts/run_baselines.py --config configs/graca_lite_cora.yaml --seed 0

# Oracle GraCA (uses all labels - diagnostic only)
python scripts/run_oracle.py --config configs/oracle_cora.yaml --seed 0

# Noisy-edge experiment with specific noise type
python scripts/run_noisy_edge_experiment.py \
    --config configs/graca_lite_cora.yaml --seed 0 \
    --noise_type cross_class_oracle --noise_ratio 0.20

# Ablation experiment
python scripts/run_ablation_noisy.py \
    --config configs/graca_lite_cora.yaml --seed 0 \
    --variant direction_only

# Edge score diagnostics
python scripts/analyze_edge_scores.py \
    --config configs/graca_lite_cora.yaml --seed 0 \
    --noise_type low_feature_similarity --noise_ratio 0.10
```

### Noise Types

| Type | Labels Used | Description |
|------|-------------|-------------|
| `low_feature_similarity` | None | Connects nodes with lowest feature cosine similarity |
| `cross_class_train_safe` | Train + high-confidence pseudo | Cross-class edges using only safe labels |
| `random_inter_community` | None (feature clustering) | Cross-cluster edges via k-means on features |
| `cross_class_oracle` | ALL labels | Cross-class using full labels (oracle/diagnostic only) |

### Generating Paper Tables

```bash
# Validate results first
python scripts/validate_results.py --dir results_clean/

# Build all paper tables
python scripts/build_final_tables.py --results_dir results_clean/ --output_dir paper_tables_clean/
```

This generates:
- `table1_clean_accuracy.csv` - Clean graph accuracy with paired t-tests
- `table2_noisy_accuracy.csv` - Noisy graph accuracy
- `table3_bad_edge_detection.csv` - Bad-edge detection precision/recall/F1
- `table4_ablation_noisy.csv` - Ablation results on noisy graphs
- `table5_oracle_gap.csv` - Oracle vs practical gap analysis
- `table6_scalability.csv` - Runtime comparison
- `statistical_tests.csv` - All pairwise paired t-tests

## Result Schema

All results in `results_clean/` use a unified CSV schema with fields:
- `experiment_type`: clean | noisy_edge | oracle | ablation | scalability
- `method`: Original | DropEdge | Random-Matched | DegreeAwareRandom-Matched | Similarity-Pruning | Homophily-TrainOnly | GraCA-lite | GraCA-Oracle
- `actual_prune_ratio`: Real fraction of edges removed
- `edge_homophily_before/after`: Graph homophily before/after pruning
- `bad_edge_precision/recall/f1`: For noisy-edge experiments only

## Running Tests

```bash
python tests/test_pruning.py
python tests/test_scoring.py
python tests/test_result_schema.py
```

## Datasets

### Homophilic
Cora, CiteSeer, PubMed, AmazonComputers, AmazonPhoto, CoauthorCS, CoauthorPhysics

### Heterophilic
Actor, Texas, Cornell, Wisconsin, Roman-empire, Amazon-ratings, Minesweeper, Tolokers, Questions

## Citation

```bibtex
@article{graca2024,
  title={GraCA: Gradient-guided Graph Connection Assessment for Graph Neural Networks},
  author={},
  year={2024}
}
```
