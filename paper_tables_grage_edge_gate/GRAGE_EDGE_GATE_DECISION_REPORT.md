# GraGE Edge-Gate Decision Report

## 1. 实验概况

- 总实验数: 297
- 数据集: ['CiteSeer', 'Cora', 'PubMed']
- 噪声类型: ['cross_class_oracle', 'degree_aligned_random', 'low_feature_similarity']
- 种子: [np.int64(0), np.int64(1), np.int64(2)]
- 方法: ['DegreeAwareRandom', 'EdgeBench-InGraphSupervised', 'EdgeBench-Transfer', 'EdgeInfluence-Oracle', 'Feature-only', 'GCN-Jaccard', 'GraGE-FO', 'GraGE-Unrolled-K1', 'GraGE-Unrolled-K3', 'Original+Noise', 'Random-Matched']

## 2. 主结果 (Practical Methods Only)

| 方法 | Test Acc (mean ± std) |
|------|----------------------|
| DegreeAwareRandom | 0.6766 ± 0.0488 |
| EdgeBench-Transfer | 0.5589 ± 0.0657 |
| Feature-only | 0.7126 ± 0.0450 |
| GCN-Jaccard | 0.7125 ± 0.0422 |
| GraGE-FO | 0.6602 ± 0.0541 |
| GraGE-Unrolled-K1 | 0.6869 ± 0.0435 |
| GraGE-Unrolled-K3 | 0.6865 ± 0.0436 |
| Original+Noise | 0.6938 ± 0.0535 |
| Random-Matched | 0.6728 ± 0.0537 |

## 3. GraGE vs Feature-only 分析

| GraGE Method | Delta vs Feature-only | Significant? |
|-------------|----------------------|--------------|
| GraGE-FO | -0.0523 | No |
| GraGE-Unrolled-K1 | -0.0257 | No |
| GraGE-Unrolled-K3 | -0.0261 | No |

## 4. 边检测质量

| 方法 | Bad-edge F1 | Practical? |
|------|-------------|-----------|
| DegreeAwareRandom | 0.0000 | True |
| EdgeBench-InGraphSupervised | 0.6474 | False |
| EdgeBench-Transfer | 0.0000 | True |
| EdgeInfluence-Oracle | 0.5429 | False |
| Feature-only | 0.6391 | True |
| GCN-Jaccard | 0.0000 | True |
| GraGE-FO | 0.1616 | True |
| GraGE-Unrolled-K1 | 0.2729 | True |
| GraGE-Unrolled-K3 | 0.2729 | True |
| Original+Noise | 0.0000 | True |
| Random-Matched | 0.1818 | True |

## 5. 结论

**结论**: GraGE-FO 未能超过 Feature-only。

Current edge-gate hypergradient does not yet provide reliable gains beyond static feature smoothness.

论文主张 **未得到支持**。

## 6. 方法有效性检查

- EdgeBench-InGraphSupervised 被标记为 oracle_only，不进入主表
- 所有 practical 方法使用同一 noisy edge_index
- 无 val/test labels 用于 edge scoring
- bad_edge_mask 仅用于 evaluation，不用于 training signal