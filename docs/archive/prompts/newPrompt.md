# ClaudeCode Prompt: GraGE-Hybrid 动态校准实验验证

请继续维护 GitHub 仓库 `J-oMarch/GraCA`，不要新建仓库。当前最新方向不是让 pure edge-gate hypergradient 单独替代 Feature-only，而是验证：

```text
GraGE-Hybrid = static feature smoothness prior + training-dynamics edge-gate calibration
```

论文主 claim 调整为：

```text
Training dynamics calibrates static feature smoothness into task-aware graph sanitization.
```

核心判据：

```text
GraGE-Hybrid > Feature-only
```

尤其应在 `feature_similar_cross_class`、`degree_aligned_random`、`random_inter_community` 等 non-feature-biased 噪声上超过 Feature-only。`low_feature_similarity` 天然偏向 Feature-only，只能作为 sanity check，不能作为主卖点。

## 0. 当前必须承认的问题

当前 `results_clean/grage_edge_gate/diagnostic/results.csv` 显示：

```text
Feature-only        0.7126
GraGE-Unrolled-K1   0.6869
GraGE-Unrolled-K3   0.6865
GraGE-FO            0.6602
```

因此 pure GraGE 不支持论文主张。请不要继续包装 pure hypergradient 成功。下一阶段目标是验证 **dynamic calibration over Feature-only** 是否有效。

## 1. 修复真正的 unrolled hypergradient

当前 `src/grage/unrolled_hypergradient.py` 中使用：

```python
with torch.no_grad():
    p.data = p.data - inner_lr * g.data
```

这会切断 `theta_K(m)` 对 `edge_gate` 的计算图，不能作为真正 unrolled hypergradient。

请修改：

```text
src/grage/unrolled_hypergradient.py
```

要求：

1. 使用 `torch.func.functional_call` 或等价 functional parameter update。
2. 内层更新必须保留计算图：

   ```text
   theta_{k+1} = theta_k - alpha * grad_theta L_support(theta_k, m)
   ```

3. 外层计算：

   ```text
   d L_score(theta_K(m), m) / d m_e
   ```

4. 支持 K=1、K=3、K=5。
5. 增加测试，确认 K=1/K3 的结果不是和 first-order 完全相同，并且计算图没有被 `.data` 或 `no_grad` 切断。

## 2. 新增 GraGE-Hybrid scorer

新增：

```text
src/grage/hybrid_score.py
```

实现：

```python
def rank_normalize(score, higher_is_risk=True):
    ...

def compute_grage_hybrid_score(
    feature_risk,
    dynamic_grad,
    lambda_pos=0.25,
    lambda_neg=0.25,
    degree=None,
    degree_norm=False,
    mode="pos_neg",
):
    """
    feature_risk: Tensor[E], e.g. 1 - cosine(x_u, x_v)
    dynamic_grad: Tensor[E], d L_score / d m_e

    mode:
      "feature_only"
      "grad_only"
      "neg_grad"
      "abs_grad"
      "feature_plus_grad"
      "feature_plus_pos"
      "feature_pos_neg"
      "feature_pos_neg_degree"

    Return:
      hybrid_score: Tensor[E], higher means more harmful
      diagnostics: dict
    """
```

核心公式：

```text
R = rank(feature_risk)
R += lambda_pos * rank(relu(dynamic_grad))
R -= lambda_neg * rank(relu(-dynamic_grad))
```

解释：

- `relu(dynamic_grad)`：增加该边会提高 score loss，应提升剪除优先级；
- `relu(-dynamic_grad)`：增加该边会降低 score loss，应保护，降低剪除优先级；
- `rank` 用于消除尺度不一致；
- `degree_norm` 用于避免高阶节点边被系统性误删。

## 3. 新增 score variant sweep

新增脚本：

```text
scripts/run_grage_hybrid_sweep.py
```

在已实现的 edge-gate score 基础上，对每个 graph unit 计算以下方法：

```text
Feature-only
GraGE-FO-grad
GraGE-FO-neggrad
GraGE-FO-absgrad
GraGE-Hybrid-FO-pos
GraGE-Hybrid-FO-posneg
GraGE-Hybrid-FO-posneg-degree
GraGE-Hybrid-UnrolledK1-posneg
GraGE-Hybrid-UnrolledK3-posneg
GraGE-Hybrid-UnrolledK5-posneg
```

参数 sweep：

```text
lambda_pos: 0.05, 0.1, 0.25, 0.5, 1.0
lambda_neg: 0.0, 0.05, 0.1, 0.25, 0.5
score_ratio: 0.2, 0.3, 0.5
degree_norm: false, true
```

输出：

```text
results_clean/grage_hybrid_sweep/results.csv
```

每行字段必须包含：

```text
dataset, noise_type, seed, method, lambda_pos, lambda_neg,
score_ratio, degree_norm, test_acc, test_f1, val_acc,
bad_edge_precision, bad_edge_recall, bad_edge_f1,
edge_score_auc, actual_prune_ratio,
edge_homophily_before, edge_homophily_after,
num_edges_before, num_edges_after, runtime, notes
```

## 4. 新增 feature-similar cross-class noise

修改：

```text
src/eval/noise_injection.py
```

新增：

```text
feature_similar_cross_class
```

构造逻辑：

1. 候选边端点真实类别不同；
2. 端点 feature cosine 高于分位数阈值，例如 top 30%；
3. 不连接已有边；
4. 按 undirected pair 注入；
5. `bad_edge_mask` 只用于 evaluation，不用于 practical scoring。

这个噪声是关键实验，因为 Feature-only 很难删除“高特征相似但跨类冲突”的边。如果 GraGE-Hybrid 能在这里超过 Feature-only，论文主张才更有说服力。

## 5. 修复 baseline 的边检测指标

当前 Jaccard / DegreeAwareRandom 已经支持 noisy `edge_index_override`，但结果中 bad-edge F1 和 homophily_after 可能仍不完整。

请确保：

1. `run_jaccard_pruning` 返回 `prune_mask` 或可恢复的 removed edge set；
2. `run_degree_aware_random` 返回 `prune_mask`；
3. 所有方法都计算：

   ```text
   bad_edge_precision / recall / f1
   edge_homophily_after
   ```

4. `num_edges_before` 在同一 dataset/noise/seed 下对所有 practical 方法一致。

## 6. 实验矩阵

先跑 small validation，不要直接 full matrix。

### Stage A: signal diagnostic

```text
datasets: Cora, CiteSeer, PubMed
noise_types:
  cross_class_oracle
  feature_similar_cross_class
  degree_aligned_random
  random_inter_community
  low_feature_similarity
seeds: 0, 1, 2
downstream_model: GCN
prune_ratio: 0.2
```

### Stage B: confirmation

只在 Stage A 中最好的 2-3 个 GraGE-Hybrid variant 上跑：

```text
datasets: Cora, CiteSeer, PubMed, Actor, Texas, Wisconsin
noise_types:
  feature_similar_cross_class
  degree_aligned_random
  random_inter_community
  cross_class_oracle
  low_feature_similarity
seeds: 0, 1, 2, 3, 4
downstream_models: GCN, GAT, GraphSAGE
```

## 7. 表格与判定报告

新增：

```text
scripts/build_grage_hybrid_tables.py
paper_tables_grage_hybrid/
  main_practical_acc.csv
  hybrid_vs_feature_only.csv
  edge_detection_f1.csv
  noise_type_breakdown.csv
  ablation_pos_neg_degree.csv
  clean_graph_non_degradation.csv
  GRAGE_HYBRID_DECISION_REPORT.md
```

报告必须回答：

1. GraGE-Hybrid 是否超过 Feature-only？
2. 提升主要来自哪些噪声？
3. 在 `feature_similar_cross_class` 上是否显著超过 Feature-only？
4. 在 `low_feature_similarity` 上是否接近 Feature-only？
5. 正梯度增强和负梯度保护各自是否有效？
6. degree normalization 是否减少误删？
7. clean graph 是否不明显破坏 Original？

## 8. 成功标准

只有满足以下条件，才可以支持 AAAI 论文主张：

```text
GraGE-Hybrid > Feature-only
```

并且：

1. 在多数 non-feature-biased noisy settings 中平均提升 >= 0.5pp；
2. paired test 显著或至少方向稳定；
3. `feature_similar_cross_class` 上明显优于 Feature-only；
4. bad-edge F1 高于或接近 Feature-only；
5. clean graph 不明显低于 Original；
6. 所有 practical scoring 不使用 val/test labels；
7. 不使用 target graph 的 bad_edge_mask 训练 practical 方法。

如果失败，请诚实输出：

```text
Current training-dynamics calibration does not provide reliable gains beyond static feature smoothness.
```

不要用 EdgeBench-InGraphSupervised 或 EdgeInfluence-Oracle 的结果替代主 claim。

## 9. 提交要求

运行：

```bash
python -m pytest tests/test_edge_gate.py tests/test_pruning.py tests/test_result_schema.py
python scripts/run_grage_hybrid_sweep.py --stage signal
python scripts/build_grage_hybrid_tables.py --results results_clean/grage_hybrid_sweep/results.csv
```

最终提交并推送到 GitHub。建议 commit message：

```text
feat: evaluate GraGE hybrid dynamic calibration over feature smoothness
```
