# GraCA 65f997f 受控实验复核报告

复核对象：GitHub `J-oMarch/GraCA`  
最新 commit：`65f997f feat: 受控对比实验 - 3数据集×10种子`  
核心结果文件：`results_clean/controlled/controlled_results.csv`

## 1. 总体结论

这次结果比上一轮更完整，确实做了 3 个数据集、10 个 seed、Original / Random / Homophily / EdgeInfluence 的受控对比，并且 noisy graph 上的数值提升很大。

但是，当前结果仍然不能作为论文主 claim 的证据。原因是两个关键泄漏仍然存在：

1. `EdgeInfluence` 的核心分数 `delta_softmax` 仍使用全量真实标签 `y[src]`。
2. noisy graph 实验中 teacher 仍然用 clean graph `data.edge_index` 预测。

因此当前结果应被标注为：

> Oracle-label scoring + oracle-noise construction + clean-teacher diagnostic upper bound.

它可以说明“如果知道或间接使用真实类别信息，删除跨类/低相似边能显著提升 noisy graph 表现”，但不能说明 GraCA 是半监督、无泄漏、可实际使用的方法。

## 2. 结果是否复现了 ClaudeCode 总结

基本复现。

### Clean Graph

聚合 `controlled_results.csv` 后，clean graph 结果如下：

| Dataset | Model | Original | Random | Homophily | EdgeInfluence |
|---|---|---:|---:|---:|---:|
| Cora | GCN | 78.48 | 76.02 | 78.43 | 78.06 |
| CiteSeer | GCN | 66.41 | 64.74 | 66.62 | 64.19 |
| PubMed | GCN | 76.02 | 74.31 | 76.08 | 78.81 |
| PubMed | GAT | 77.81 | 76.89 | 77.81 | 80.21 |

补充完整观察：

- Cora：EdgeInfluence 不优于 Original。
- CiteSeer：GCN 和 GraphSAGE 明显变差。
- PubMed：GCN/GAT/GraphSAGE 均提升，值得继续研究。

### Noisy Graph

noisy graph 使用 `cross_class_oracle 20%`，结果确实很强：

| Dataset | Model | Original | Random | EdgeInfluence | EdgeInfluence - Original | Bad-edge F1 |
|---|---|---:|---:|---:|---:|---:|
| Cora | GCN | 71.19 | 68.38 | 76.09 | +4.90 | 0.511 |
| CiteSeer | GCN | 60.21 | 59.09 | 64.13 | +3.92 | 0.499 |
| PubMed | GCN | 67.56 | 66.31 | 75.85 | +8.29 | 0.461 |

但这些数字不能直接用于 practical method claim，因为 scoring 和 teacher 仍有泄漏。

## 3. 代码中的关键问题

### 3.1 全量标签泄漏仍然存在

`src/graca/edge_influence.py`：

```python
p_uc_full = probs_full[dst].gather(1, y[src].unsqueeze(1)).squeeze(1)
p_uc_ablated = probs_ablated.gather(1, y[src].unsqueeze(1)).squeeze(1)
delta_softmax = p_uc_full - p_uc_ablated
```

这里 `y[src]` 是所有节点的真实标签。对 Cora/CiteSeer/PubMed 的 public split 来说，绝大多数节点在半监督训练中不应该暴露标签。因此当前 EdgeInfluence score 是 oracle-label score。

### 3.2 noisy graph 使用 clean teacher

`scripts/run_controlled_comparison.py`：

```python
model, teacher, _, _ = train_proxy(config, data, num_features, num_classes, device, args.seed)
teacher_probs = teacher.predict(x, data.edge_index)
```

即使 `--noisy` 时已经构造了：

```python
edge_index = noise_result["noisy_edge_index"]
```

teacher 仍然在 clean `data.edge_index` 上预测。这会让 EdgeInfluence 的评分条件强于 Original+Noise / Random / Homophily。

### 3.3 CSV 中存在重复追加

`controlled_results.csv` 中 Cora clean 部分不是严格 10 seeds：

- Original 每个模型 12 行；
- EdgeInfluence/Random/Homophily 每个模型 11 行；
- seed 0 出现重复。

论文表生成前必须清空结果目录或按 `(dataset, experiment_type, method, model, seed, noise_type)` 去重。

### 3.4 Homophily baseline 不等价于 20% 裁剪

Homophily-TrainOnly 只删除两端都有训练标签且类别不同的边。在 public split 下，这类边极少：

- Cora noisy：实际裁剪约 `0.12%`
- CiteSeer noisy：实际裁剪约 `0.13%`
- PubMed noisy：约 `0%`

所以它不是 20% matched pruning baseline，而是一个“train-label-only conservative baseline”。可以保留，但不能说 EdgeInfluence 公平击败了一个同裁剪率 Homophily baseline。

## 4. 当前结果能支持什么

可以支持：

1. 在 oracle cross-class noise 下，删除跨类/低相似边能显著恢复 GNN 性能。
2. 当前 oracle-label EdgeInfluence score 能较好找到注入坏边，F1 约 `0.46-0.51`。
3. PubMed clean graph 上，20% 边裁剪可能有正则化/净化收益。

不能支持：

1. 不能证明半监督无泄漏 GraCA 有效。
2. 不能把 noisy graph 的 +3.9% 到 +8.3% 当作 practical method 结果。
3. 不能声称 `delta_softmax + feature_cosine` 是不使用真实标签的方法。
4. 不能声称已经完成了足以投稿的实验矩阵。

## 5. 对论文方向的判断

当前最合理的论文定位有两种。

### 方向 A：Oracle Diagnostic Paper

如果短期内 practical score 修复后效果下降明显，可以写成：

> harmful edge diagnostic and upper-bound study under semi-supervised graph learning.

核心贡献是分析跨类边、低相似边、oracle LOO、oracle-label score 的影响。

这类文章需要诚实标注 oracle setting，不能声称 practical deployment。

### 方向 B：Practical Graph Cleaning Method

如果要写成方法论文，必须先修复泄漏，然后证明：

- `combined_pseudo_label` 不使用 val/test labels；
- noisy graph teacher 不使用 clean graph；
- 在 `low_feature_similarity` 或 train-safe noise 下仍优于 baselines；
- 至少 3-6 个数据集、10 seeds；
- 有 feature-only / pseudo-only / LOO-only / random / degree-random / GDC 或 DropEdge 类 baseline。

目前还没达到方向 B。

## 6. 下一步给 ClaudeCode 的 Prompt

```text
请继续修复 GraCA 当前 65f997f 版本。注意：当前受控实验结果很强，但仍然存在两个核心泄漏，不能作为论文主表。你的任务不是继续扩大实验，而是先把实验改成无泄漏 practical setting，并保留 oracle setting 作为 upper bound。

任务 1：拆分 oracle-label score 和 practical pseudo-label score

修改 `src/graca/edge_influence.py`。

当前代码：

```python
p_uc_full = probs_full[dst].gather(1, y[src].unsqueeze(1)).squeeze(1)
p_uc_ablated = probs_ablated.gather(1, y[src].unsqueeze(1)).squeeze(1)
delta_softmax = p_uc_full - p_uc_ablated
```

这是 oracle-label score，必须重命名为：

```python
delta_softmax_oracle_label
```

新增 practical score：

```python
node_label_for_score = teacher_probs.argmax(dim=1)
node_label_for_score[train_mask] = y[train_mask]

p_uc_full_pseudo = probs_full[dst].gather(1, node_label_for_score[src].unsqueeze(1)).squeeze(1)
p_uc_ablated_pseudo = probs_ablated.gather(1, node_label_for_score[src].unsqueeze(1)).squeeze(1)
delta_softmax_pseudo_label = p_uc_full_pseudo - p_uc_ablated_pseudo
```

要求：

- practical score 不能读取非 train 节点真实标签；
- return dict 同时输出 oracle 和 pseudo 两套 score；
- 所有 downstream practical 实验默认用 pseudo；
- oracle score 只能用于 diagnostic/upper-bound。

任务 2：修复 noisy graph teacher 泄漏

修改 `train_proxy` 或新增参数：

```python
train_proxy(..., edge_index_override=None)
```

要求：

- 如果 `edge_index_override` 非空，proxy/teacher 训练必须使用 override graph；
- teacher.predict 也必须使用当前实验图；
- noisy practical 实验中禁止使用 `data.edge_index` clean graph；
- 只有显式 `--oracle_clean_teacher` 时才允许 clean teacher，并写入结果字段。

修改 `scripts/run_controlled_comparison.py`：

- noisy=False：使用 clean graph；
- noisy=True practical：teacher 在 noisy graph 上训练和预测；
- noisy=True oracle diagnostic：可以额外跑 clean teacher，但 method 名必须带 `OracleCleanTeacher`。

任务 3：新增结果字段

在 `controlled_results.csv` 中新增字段：

```text
score_type
oracle_label_score
oracle_noise
clean_teacher_used
score_component
```

每个结果必须明确标注。

任务 4：重新运行最小无泄漏矩阵

输出到新文件，不能追加旧结果：

```text
results_clean/controlled_v2/controlled_v2_results.csv
```

实验矩阵：

- datasets: Cora, CiteSeer, PubMed
- seeds: 0-9
- downstream: GCN, GAT, GraphSAGE
- clean graph:
  - Original
  - Random-Matched
  - feature_only
  - combined_pseudo_label
  - combined_oracle_label  # diagnostic only
- noisy graph:
  - noise_type:
    - cross_class_oracle
    - low_feature_similarity
    - train_safe_oracle_v2
  - methods:
    - Original+Noise
    - Random-Matched
    - feature_only
    - combined_pseudo_label
    - combined_oracle_label  # diagnostic only
    - OracleCleanTeacherCombined  # diagnostic only, optional

任务 5：生成报告

新增：

```text
CONTROLLED_V2_VALIDITY_REPORT.md
```

报告必须分开三类结果：

1. Practical：无 oracle label，无 clean teacher；
2. Oracle label diagnostic；
3. Oracle clean teacher diagnostic。

报告必须回答：

- practical `combined_pseudo_label` 是否仍优于 Original+Noise？
- practical `combined_pseudo_label` 是否优于 feature_only？
- oracle-label score 比 practical score 高多少？
- clean teacher 比 noisy teacher 高多少？
- PubMed clean graph 的提升在 practical score 下是否仍存在？
- 如果 practical score 不显著，请明确写“不支持论文主 claim”。

任务 6：结果文件卫生

- 每次运行前清空 `results_clean/controlled_v2/`；
- CSV 中 `(dataset, experiment_type, noise_type, method, downstream_model, seed)` 必须唯一；
- 写一个校验脚本检查重复行和缺失行。

完成后运行 pytest，提交并推送 GitHub。
```

## 7. 预计修复后结果

我预计修复后会出现明显分层：

1. `combined_oracle_label + clean_teacher` 仍然很强，接近当前结果。
2. `combined_oracle_label + noisy_teacher` 会略降。
3. `combined_pseudo_label + noisy_teacher` 会明显下降。
4. 如果 `feature_only` 接近 `combined_pseudo_label`，说明当前有效性主要来自特征相似度，而不是 EdgeInfluence。

只有当第 3 类 practical 结果仍然稳定优于 `Original+Noise`、`Random-Matched`、`feature_only`，才可以继续按方法论文推进。

