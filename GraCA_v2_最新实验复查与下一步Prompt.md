# GraCA v2 最新实验复查与下一步 Prompt

复查对象：GitHub 仓库 `J-oMarch/GraCA` 最新 HEAD `64d3d6c`，提交信息为 `feat: 实现 EdgeInfluence 并完成可行性验证`。

## 1. 总体结论

当前最新实验不能支撑论文主 claim，也不能证明 GraCA / EdgeInfluence 已经有效。

仓库这次新增的结果是一个诚实的可行性验证，而不是成功的完整 v2 实验。核心结果显示：

- `feasible: false`
- 实用可用的最好 AUC：`0.688`
- 预设阈值：`0.75`
- Oracle LOO AUC：`0.786`
- Oracle 移除跨类边准确率提升：`+8.5%`

这说明“移除错误/跨类边确实可能提升 GNN 表现”这个问题本身有价值，但当前可实现的评分方法还不能可靠找出这些边。因此目前更适合写成失败诊断或方法探索报告，不足以支撑一篇声称方法有效的论文。

## 2. 最新实验结果解读

`results_clean/diagnostics/verify_idea_Cora_42.json` 中的关键指标如下：

| 指标 | 数值 | 解释 |
|---|---:|---|
| best_auc_practical | 0.688 | 最好的实用信号仍低于 0.75 |
| best_auc_oracle | 0.786 | 使用 oracle 信息时可以达到可用水平 |
| oracle_loo_auc | 0.786 | 全前向 LOO + 真实标签有效，但不可作为实用方法 |
| vectorized_loo_auc_oracle_labels | 0.551 | 向量化近似即使使用真实标签也接近随机 |
| vectorized_loo_auc_pseudo_labels | 0.431 | 使用伪标签后信号反而更差 |
| kl_divergence_auc | 0.688 | 当前最好的实用信号，但仍不足 |
| pseudo_label_agreement_auc | 0.658 | 有弱信号 |
| hidden_cosine_auc | 0.623 | 旧隐藏相似度思想较弱 |
| feature_cosine_auc | 0.597 | 特征相似度也不足 |
| oracle_accuracy_gain | 0.085 | oracle 删除跨类边可提升 8.5pp |

可以支持的结论：

1. 图中跨类/有害边确实可能伤害 GNN。
2. 如果有 oracle 标签或昂贵的全边 leave-one-out，删除这些边可以提升性能。
3. 当前近似 EdgeInfluence、伪标签评分、隐藏余弦评分都不能稳定识别这些边。

不能支持的结论：

1. 不能声称 GraCA 已经优于 baseline。
2. 不能声称 EdgeInfluence 可以实用地识别 harmful edges。
3. 不能声称当前实验足以支持论文发表。

## 3. 代码实现和上轮 Prompt 的偏差

上轮要求的核心实验是 edge-gate autograd influence：

```text
S_e = d L_score / d m_e
```

其中 `m_e` 是每条边的可微门控变量。

但最新仓库没有真正实现这个实验。

### 3.1 模型没有支持 edge_gate

`src/models/gcn.py`、`src/models/gat.py`、`src/models/sage.py` 的 `forward` 仍然只有：

```python
forward(x, edge_index, return_hidden=False, retain_hidden_grad=False)
```

没有：

```python
edge_gate: Optional[torch.Tensor] = None
```

因此当前代码没有办法计算 `edge_gate.grad`，也没有真正测试一阶边影响函数。

### 3.2 当前 EdgeInfluence 是向量化 LOO 近似，不是梯度影响函数

`src/graca/edge_influence.py` 中使用的是：

```python
h_ablated = ReLU((d_v * h_full[v] - h_full[u]) / (d_v - 1))
```

这个公式对当前 PyG GCNConv + self-loop + normalized adjacency + BatchNorm + dropout 的模型并不严格成立。它把已经聚合后的 `h_full[u]` 当成从 `u` 到 `v` 的原始 message，这在数学上不是精确 leave-one-out。

实验结果也证明了这一点：

- Oracle full LOO AUC = 0.786
- Vectorized LOO oracle-label AUC = 0.551

也就是说，问题不是只有伪标签噪声；近似本身已经破坏了主要信号。

### 3.3 没有完整 v2 实验矩阵

仓库没有产出以下上轮要求的文件：

```text
results_v2/diagnostic/v2_diagnostic_results.csv
V2_DIAGNOSTIC_REPORT.md
docs/PAPER_REVISION_PLAN_V2.md
paper_tables_v2/
```

也没有系统比较：

- Original+Noise
- Random-Matched
- Similarity
- Homophily-TrainOnly
- node_cosine_legacy
- edge_influence_first_order
- edge_influence_unrolled
- low_helpfulness

因此这轮实验只能算作“可行性预检”，不能算作完整论文实验。

## 4. 对方法本身的判断

当前原始方法，即“节点隐藏梯度/相似性推断边是否有害”，大概率不是一个足够强的主方法。

原因是：

1. 节点级梯度相似性不是边级因果贡献。
2. harmful edge 的定义依赖模型、标签目标和训练动态，不等价于两端节点隐藏表示是否相似。
3. 在 semi-supervised setting 中，大量节点没有真实标签，伪标签噪声会淹没单边影响。
4. 单条边对高阶 GNN 表示的影响很小，尤其在节点度较大时，信噪比很低。

但“图结构中存在可移除的有害边”这个问题不是无效的。Oracle 结果说明这个方向有研究价值。真正需要替换的是 scoring 机制。

## 5. 下一步建议

下一步不要继续扩大数据集，也不要急着写论文。应该先做一个严格的最小闭环实验，回答一个问题：

> 不使用 oracle 标签，只使用训练标签内部划分得到的 score loss，可微 edge-gate influence 是否能超过 0.75 AUC，并在下游准确率上超过 matched baseline？

如果不能，应该停止把它包装成有效方法，转向 negative result / diagnostic paper。

如果能，再扩展到 Cora、CiteSeer、PubMed、Actor、Texas、Wisconsin，加入多 seed、多噪声类型和完整消融。

## 6. 可直接发送给 ClaudeCode 的 Prompt

```text
你现在需要继续完善 GraCA 项目，但不要再扩大无效实验。请严格完成下面的最小验证任务，目标是判断“真正的 edge-gate autograd influence”是否有效。

当前仓库已经有一个向量化 LOO 近似实验，但它不是我要的最终方法。请不要把现有 `src/graca/edge_influence.py` 当成最终 EdgeInfluence。它只能保留为 `vectorized_loo_legacy` baseline。

任务 1：实现真正的 edge_gate 支持

1. 修改 `src/models/gcn.py`，让 GCN forward 支持：

   ```python
   forward(
       x,
       edge_index,
       edge_gate=None,
       return_hidden=False,
       retain_hidden_grad=False,
   )
   ```

2. 对 GCNConv，优先通过 `edge_weight=edge_gate` 实现边门控。如果原本存在 edge_weight，则使用 `edge_weight * edge_gate`。
3. `edge_gate is None` 时，模型输出必须和旧版本完全一致。
4. 先只要求 GCN 支持 edge_gate，不要急着支持 GAT / GraphSAGE。
5. 增加单元测试：
   - edge_gate 全 1 时输出和不传 edge_gate 一致；
   - edge_gate.requires_grad=True 时，反向传播后 `edge_gate.grad is not None`；
   - edge_gate 全 0 时邻居传播被明显改变。

任务 2：实现 first-order edge influence

新增文件：

```text
src/graca/edge_gate_influence.py
```

实现函数：

```python
def compute_edge_gate_influence(
    model,
    x,
    edge_index,
    y,
    score_mask,
    use_abs=False,
):
    """
    Return:
      score: Tensor[E], higher means more harmful candidate to prune
      raw_grad: Tensor[E]
    """
```

逻辑：

1. 设置 `model.eval()`，关闭 dropout。
2. 构造：

   ```python
   edge_gate = torch.ones(edge_index.size(1), device=edge_index.device, requires_grad=True)
   ```

3. 前向：

   ```python
   logits = model(x, edge_index, edge_gate=edge_gate)
   ```

4. 只在 `score_mask` 上计算 supervised CE：

   ```python
   L_score = F.cross_entropy(logits[score_mask], y[score_mask])
   ```

5. 反向得到：

   ```python
   grad = torch.autograd.grad(L_score, edge_gate)[0]
   ```

6. 评分定义：
   - `harmful_score = grad`，因为若 `grad > 0`，减小 edge gate 会一阶降低 loss；
   - 同时输出 `helpfulness = -grad`，用于比较 low-helpfulness pruning。

任务 3：实现 train 内部 support/score split

新增工具函数：

```text
src/utils/mask_split.py
```

实现：

```python
def split_train_support_score(train_mask, score_ratio=0.3, seed=0):
    ...
```

要求：

1. 只从 train_mask 内部划分。
2. support_mask 用于训练 proxy。
3. score_mask 用于计算 edge influence。
4. score_mask 不允许使用 test_mask。

任务 4：实现最小诊断脚本

新增脚本：

```text
scripts/run_edge_gate_diagnostic.py
```

实验设置：

- dataset: Cora
- model: GCN
- seeds: 0, 1, 2
- noise_type:
  - cross_class_oracle
  - low_feature_similarity
- noise_ratio: 0.1, 0.2
- prune_ratio: 0.05, 0.1, 0.2

比较方法：

1. Original+Noise
2. Random-Matched
3. Similarity
4. vectorized_loo_legacy
5. edge_gate_harmful_score
6. edge_gate_low_helpfulness

每个设置需要输出：

- dataset
- seed
- model
- noise_type
- noise_ratio
- prune_ratio
- method
- test_acc
- val_acc
- bad_edge_precision
- bad_edge_recall
- bad_edge_f1
- cross_class_auc
- actual_prune_ratio
- num_edges_before
- num_edges_after

输出到：

```text
results_v2/diagnostic/edge_gate_diagnostic_results.csv
```

任务 5：生成报告

新增：

```text
V2_EDGE_GATE_DIAGNOSTIC_REPORT.md
```

报告必须回答：

1. edge_gate_harmful_score 的 cross_class_auc 是否稳定超过 0.75？
2. edge_gate_harmful_score 是否稳定超过 Random-Matched 和 Similarity？
3. edge_gate_low_helpfulness 是否比 harmful_score 更有效？
4. test_acc 是否相对 Original+Noise 有稳定提升？
5. 如果无效，明确写出“不支持论文主 claim”，不要美化结果。

验收标准：

1. `pytest` 通过。
2. `results_v2/diagnostic/edge_gate_diagnostic_results.csv` 存在且非空。
3. `V2_EDGE_GATE_DIAGNOSTIC_REPORT.md` 存在。
4. 报告中必须有均值 ± 标准差表格。
5. 如果 edge_gate 方法没有明显优于 baseline，请不要修改论文正文去声称方法有效。

完成后提交并推送到 GitHub。
```

## 7. 预计结果

我预计会出现三种可能：

### 情况 A：edge-gate influence 明显有效

如果 `edge_gate_harmful_score` 在 Cora 上跨 seed 平均 AUC 达到 `0.75+`，且下游 test accuracy 稳定高于 Original+Noise 和 Random-Matched，那么说明原来的节点梯度思想需要被替换，论文主方法应改写成：

> 基于可微边门控的一阶结构影响函数。

这时再扩展数据集和写论文。

### 情况 B：AUC 有弱信号但下游不提升

如果 AUC 在 `0.65-0.75`，但 test accuracy 没有稳定提升，说明 scoring 有一定相关性但不够构成有效净化方法。论文不能按方法有效来写，只能作为诊断型工作继续打磨。

### 情况 C：AUC 和下游都不稳定

如果 AUC 低于 `0.65` 或不超过 Similarity / Random-Matched，则应停止当前 GraCA 主线。最合理的处理是把已有结果写成 negative result：

> 真实有害边存在，但在低标签半监督设定下，仅依赖梯度/伪标签的单边评分信噪比不足。

这比继续堆实验更符合科学性。

