# GraCA b0268b4 实验复核报告

复核时间：2026-05-31  
复核对象：GitHub `J-oMarch/GraCA`，最新 commit `b0268b4 feat: EdgeInfluence 实验 - 可行性验证通过 + Cora 10-seed 结果`

## 1. 总体结论

当前结果还不能支撑论文主 claim，也不能证明 EdgeInfluence/GraCA 已经是一个有效的实用方法。

ClaudeCode 总结中的 “可行性验证通过” 主要来自：

```text
combined = zscore(delta_softmax) + zscore(-feature_cosine)
```

但当前 `delta_softmax` 使用了所有节点真实标签 `y[src]`，这在半监督节点分类设定下是标签泄漏。与此同时，noisy graph 实验中的 teacher 仍然在 clean graph 上训练/预测，这相当于评分阶段使用了实验中不应可见的干净图。

因此，当前结果可以作为“oracle/diagnostic 上限”参考，但不能作为论文里 practical method 的核心证据。

## 2. 关键证据

### 2.1 combined score 存在标签泄漏

`src/graca/edge_influence.py` 中：

```python
p_uc_full = probs_full[dst].gather(1, y[src].unsqueeze(1)).squeeze(1)
p_uc_ablated = probs_ablated.gather(1, y[src].unsqueeze(1)).squeeze(1)
delta_softmax = p_uc_full - p_uc_ablated
```

这里的 `y[src]` 是所有源节点真实标签，包括 val/test/unlabeled 节点。半监督设定中，模型不能使用这些标签做边评分。

这会显著抬高 AUC：

| score | AUC |
|---|---:|
| L_weighted | 0.588 |
| L_raw | 0.593 |
| L_oracle | 0.586 |
| delta_softmax | 0.701 |
| feature_cosine | 0.713 |
| combined_delta_cos | 0.789 |

可以看到，真正的 LOO 类影响分数仍然只有约 `0.59`；“通过”的 `0.789` 来自 `delta_softmax + feature_cosine`，其中 `delta_softmax` 使用了全量真实标签。

### 2.2 noisy graph 实验使用了 clean graph teacher

`scripts/run_edge_influence.py` 中 noisy graph 分支虽然构造了 `edge_index = noisy_edge_index`，但 teacher 训练和预测仍使用 clean graph：

```python
model, teacher, train_log, _ = train_proxy(config, data, ...)
teacher_probs = teacher.predict(x, data.edge_index)  # use clean graph for teacher
```

这不符合真实 noisy graph sanitization。合理做法应是：

1. 如果实验设定是 noisy graph，则 proxy/teacher 必须在 noisy_edge_index 上训练或至少预测；
2. 不能在评分时使用 clean `data.edge_index`；
3. clean graph 只能用于构造 oracle diagnostic 或计算上限，不能进入 practical method。

### 2.3 没有真正实现 edge_gate autograd influence

仓库仍未生成：

```text
results_v2/diagnostic/edge_gate_diagnostic_results.csv
V2_EDGE_GATE_DIAGNOSTIC_REPORT.md
```

模型 `GCN/GAT/GraphSAGE` 的 `forward` 也没有 `edge_gate` 参数。因此上轮要求的：

```text
S_e = d L_score / d m_e
```

仍然没有被真正验证。

### 2.4 下游 clean graph 结果不支持方法优越性

从 `results_clean/main/results.csv` 和 `results/main/baseline_results.csv` 聚合：

| Model | Original | EdgeInfluence | Homophily | DropEdge |
|---|---:|---:|---:|---:|
| GCN | 78.45 | 78.39 | 78.53 | 78.30 |
| GAT | 81.96 | 81.65 | 82.18 | 82.25 |
| GraphSAGE | 76.66 | 76.85 | 76.76 | 76.11 |

EdgeInfluence 在 clean graph 上没有稳定提升：

- GCN：低于 Original `0.06pp`
- GAT：低于 Original `0.31pp`
- GraphSAGE：高于 Original `0.19pp`

这只能说明它在 clean graph 上“基本不破坏”，不能说明方法有效。

### 2.5 noisy graph 结果缺少公平 baseline 矩阵

当前 `results_clean/main/results.csv` 有 EdgeInfluence 的 noisy `cross_class_oracle 20%` 结果：

| Model | EdgeInfluence noisy test acc | bad-edge F1 |
|---|---:|---:|
| GCN | 73.66 | 0.423 |
| GAT | 77.00 | 0.423 |
| GraphSAGE | 73.01 | 0.423 |

但新实验结果文件中没有同一设置下的：

- Original+Noise
- Random-Matched
- DegreeAwareRandom-Matched
- Similarity
- Homophily-TrainOnly
- feature_cosine-only
- delta_softmax without true labels

所以不能判断 EdgeInfluence 是否真正优于 baseline。尤其是当前 F1 来自 `cross_class_oracle` 噪声和泄漏评分，不能作为 practical claim。

## 3. 对 ClaudeCode 总结的逐条判断

### Stage 1: AUC=0.789

不应视为实用方法通过。该 AUC 来自 `combined_delta_cos`，其中 `delta_softmax` 使用 `y[src]` 全量真实标签。它更接近 oracle diagnostic。

### Stage 2: AUC=0.785 oracle LOO

这是 oracle 上限，不是 practical method。它说明问题可能有信号，但不能证明当前方法可用。

### Stage 3: top-20% 边中跨类边比例 0.79x

这是负面信号。`0.79x` 表示 top-20% 里跨类边比例低于随机基准，不是有效性证据。JSON 中也显示：

```text
top_20_pct_cross_ratio = 0.254
random_cross_ratio = 0.325
effectiveness_ratio = 0.782
bad_edge_f1 = 0.055
```

这里的 Stage 3 实际上不支持方法有效。

### Stage 4: 所有 5 种噪声 AUC > 0.70

需要降级为 oracle/diagnostic 结果。因为 combined score 使用了真实标签，并且部分噪声构造也使用 oracle 标签。

## 4. 当前能写进论文的内容

可以写：

1. 图中 injected harmful/cross-class edges 对 GNN 有明显伤害。
2. Oracle LOO 或使用全量标签的 diagnostic score 能识别一部分 harmful edges。
3. 简单特征相似度和预测变化组合在 Cora 上有一定诊断信号。

不能写：

1. 不能声称当前 EdgeInfluence 是无标签/半监督可用方法。
2. 不能声称 GraCA 在 noisy graph 上已经显著优于 baseline。
3. 不能用 `combined_delta_cos` 的 AUC=0.789 作为 practical method 的主结果。
4. 不能把 clean-graph teacher 用于 noisy graph practical evaluation。

## 5. 下一步必须修复

优先级从高到低：

1. 移除 scoring 中的全量真实标签泄漏。
2. noisy graph 下 teacher/proxy 必须在 noisy graph 上训练和预测。
3. 增加同一 noisy setting 下的完整 baseline matrix。
4. 将 `combined_delta_cos` 拆成：
   - `combined_oracle_label`
   - `combined_pseudo_label`
   - `feature_only`
   - `delta_softmax_pseudo`
5. 补充真正的 edge_gate autograd influence，或明确放弃这条线。
6. 所有表格必须区分 practical、oracle-noise、oracle-scoring。

## 6. 可直接发给 ClaudeCode 的修复 Prompt

```text
请继续修复 GraCA 当前 b0268b4 版本。注意：这不是扩展实验，而是修复实验有效性。当前 `combined_delta_cos` 使用了 `y[src]` 全量真实标签，noisy graph 实验还使用 clean graph teacher，这会导致标签泄漏和 clean graph 泄漏。请按下面要求修复。

任务 1：拆分 oracle scoring 和 practical scoring

1. 在 `src/graca/edge_influence.py` 中保留当前使用 `y[src]` 的 `delta_softmax`，但重命名为：

   ```python
   delta_softmax_oracle_label
   combined_oracle_label
   ```

2. 新增 practical 版本：

   ```python
   delta_softmax_pseudo_label
   combined_pseudo_label
   ```

   要求：
   - 对 train_mask 节点，可以使用真实训练标签；
   - 对非 train_mask 节点，只能使用 teacher_probs.argmax 或 soft teacher distribution；
   - 不能读取 val/test/unlabeled 的真实 `y`；
   - 输出中必须同时保存 oracle 和 practical 两套 score，但论文主表只能用 practical。

3. 新增 `feature_only = -feature_cosine` baseline。

任务 2：修复 noisy graph teacher 泄漏

1. 修改 `scripts/run_edge_influence.py`：
   - 如果 `--noisy`，proxy/teacher 必须在 `noisy_edge_index` 上训练或至少用 noisy edge_index 做预测；
   - 不能在 practical scoring 中调用 `teacher.predict(x, data.edge_index)`；
   - clean graph teacher 只能作为 `oracle_clean_teacher` diagnostic，不能进入 practical method。

2. 如果现有 `train_proxy` 不支持外部 edge_index，请修改接口：

   ```python
   train_proxy(..., edge_index_override=None)
   ```

   并确保训练 loss、teacher update、teacher.predict 都基于 override graph。

任务 3：补齐公平 noisy baseline matrix

在同一个脚本或新脚本中运行：

- Dataset: Cora
- Seeds: 0-9
- Noise: cross_class_oracle 20%, low_feature_similarity 20%, train_safe_oracle_v2 20%
- Models: GCN, GAT, GraphSAGE
- Methods:
  - Original+Noise
  - Random-Matched
  - DegreeAwareRandom-Matched
  - Homophily-TrainOnly
  - feature_only
  - combined_pseudo_label
  - combined_oracle_label  # 只作为 oracle diagnostic

输出到：

```text
results_clean/noisy_v2/noisy_v2_results.csv
```

每行必须包含：

- method
- oracle_scoring: true/false
- oracle_noise: true/false
- clean_teacher_used: true/false
- seed
- dataset
- downstream_model
- noise_type
- noise_ratio
- test_acc
- val_acc
- bad_edge_precision
- bad_edge_recall
- bad_edge_f1
- actual_prune_ratio
- edge_homophily_before
- edge_homophily_after

任务 4：重新生成报告

新增：

```text
VALIDITY_FIXED_REPORT.md
```

报告必须分三块：

1. Practical results：不能使用 oracle labels，不能使用 clean teacher。
2. Oracle scoring diagnostic：允许使用全标签，但必须明确标注。
3. Oracle noise diagnostic：如果噪声构造使用 full labels，也必须明确标注。

报告中必须回答：

- practical `combined_pseudo_label` 是否优于 `feature_only`？
- practical `combined_pseudo_label` 是否优于 `Original+Noise`？
- practical `combined_pseudo_label` 是否优于 `Random-Matched` 和 `Homophily-TrainOnly`？
- oracle score 与 practical score 的差距是多少？
- 如果 practical 结果不显著，请明确写“不支持论文主 claim”。

验收标准：

1. 不允许论文主表使用 `combined_oracle_label`。
2. 不允许 practical noisy graph 实验使用 clean teacher。
3. `results_clean/noisy_v2/noisy_v2_results.csv` 必须存在且包含所有方法。
4. `VALIDITY_FIXED_REPORT.md` 必须给出 mean ± std。
5. pytest 通过。

完成后提交并推送 GitHub。
```

## 7. 预计修复后的可能结果

我预计修复泄漏后，AUC 和 bad-edge F1 会明显下降。可能出现三种情况：

1. `combined_pseudo_label` 仍显著优于 feature-only 和 Random-Matched：方法有希望，但主线应改成“pseudo-label guided feature/prediction disagreement pruning”，而不是当前的 LOO EdgeInfluence。
2. `combined_pseudo_label` 接近 feature-only：说明主要有效信号来自特征相似度，论文应降级 GraCA 的梯度贡献。
3. `combined_pseudo_label` 不如 Homophily/Random：应停止当前主 claim，改成 negative result 或重新设计方法。

