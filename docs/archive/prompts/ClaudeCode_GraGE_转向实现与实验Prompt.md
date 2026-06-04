# ClaudeCode Prompt: 将当前项目转向 GraGE 可微边门控图演化

## 结论先行

请不要新建仓库。请在当前 GraCA 项目中继续修改。

原因：

1. 当前项目已有数据加载、噪声注入、baseline、结果 schema、controlled_v2 实验和历史消融。
2. Feature-only 已经被验证为强 baseline，后续 GraGE 必须直接击败它。
3. 新建仓库会浪费已有实验基础，也不利于复用 controlled_v2 的无泄漏检查。

建议做法：

- 保留旧 GraCA / EdgeInfluence-Pseudo 代码作为 ablation；
- 新增 GraGE 模块；
- 新增独立结果目录 `results_clean/grage_diagnostic/`；
- 不覆盖旧结果。

## 项目新方向

论文方向正式调整为：

```text
Training-Dynamics-Guided Graph Evolution via Differentiable Edge Gates
```

中文理解：

```text
基于训练动态的可微边门控图结构演化
```

核心观点：

> 图结构不是静态输入，而是由模型训练过程中的边级贡献信号动态演化。一条边是否保留，不由特征相似度直接决定，而由它对模型训练目标、验证目标、预测稳定性和泛化行为的边际贡献决定。

注意：

1. Feature-only 不是主方法，只是强 baseline 和可选 smooth prior。
2. 旧 `EdgeInfluence-Pseudo` 不是主方法，只作为 ablation。
3. 新方法必须实现真正的 differentiable edge gate 和 support/score training dynamics。

## 任务 1：实现 GCN edge_gate

修改：

```text
src/models/gcn.py
```

要求 forward 支持：

```python
def forward(
    self,
    x,
    edge_index,
    edge_gate=None,
    return_hidden=False,
    retain_hidden_grad=False,
):
    ...
```

要求：

1. `edge_gate is None` 时行为与旧版本一致。
2. `edge_gate.shape == [edge_index.size(1)]`。
3. 对 PyG `GCNConv`，通过 `edge_weight=edge_gate` 实现边门控。
4. `edge_gate.requires_grad=True` 时，反向传播后 `edge_gate.grad` 必须非空。
5. 第一阶段只支持 GCN，GAT/GraphSAGE 暂不实现 edge gate。

新增测试：

```text
tests/test_edge_gate_gcn.py
```

测试：

1. 全 1 gate 与不传 gate 输出接近；
2. gate 可获得梯度；
3. 全 0 gate 与全 1 gate 输出有显著差异。

## 任务 2：实现 train 内部 support/score split

新增：

```text
src/utils/mask_split.py
```

实现：

```python
def split_train_support_score(train_mask, score_ratio=0.3, seed=0):
    """
    Split train_mask into support_mask and score_mask.
    Only train labels can be used.
    """
```

要求：

1. 只从 train_mask 内部划分；
2. support_mask 用于训练 proxy；
3. score_mask 用于计算边门控超梯度；
4. 不允许使用 val/test labels。

## 任务 3：实现 GraGE first-order edge gate influence

新增：

```text
src/grage/edge_gate_influence.py
```

如果项目没有 `src/grage/`，请新建该目录并添加 `__init__.py`。

实现：

```python
def compute_edge_gate_first_order(
    model,
    x,
    edge_index,
    y,
    score_mask,
    smooth_prior=None,
    smooth_lambda=0.0,
):
    """
    Return:
      harmful_score: Tensor[E], larger means more harmful
      raw_grad: Tensor[E]
    """
```

逻辑：

1. `model.eval()`。
2. 构造：

   ```python
   edge_gate = torch.ones(edge_index.size(1), device=edge_index.device, requires_grad=True)
   ```

3. 前向：

   ```python
   logits = model(x, edge_index, edge_gate=edge_gate)
   ```

4. 只在 `score_mask` 上计算 CE：

   ```python
   L_score = F.cross_entropy(logits[score_mask], y[score_mask])
   ```

5. 如提供 `smooth_prior`：

   ```python
   L_score = L_score + smooth_lambda * (edge_gate * smooth_prior).mean()
   ```

6. 计算：

   ```python
   grad = torch.autograd.grad(L_score, edge_gate)[0]
   harmful_score = grad
   ```

7. 不允许使用全量真实标签。
8. 不允许使用 clean graph teacher。
9. 不允许把 feature cosine 直接加到 score 中，除非作为 smooth prior 正则。

## 任务 4：实现 GraGE unrolled hypergradient

同文件中新增：

```python
def compute_edge_gate_unrolled(
    model,
    x,
    edge_index,
    y,
    support_mask,
    score_mask,
    inner_lr=0.01,
    unroll_steps=1,
    smooth_prior=None,
    smooth_lambda=0.0,
):
    ...
```

目标：

近似：

```text
S_e = ∂ L_score(θ_K(m), m) / ∂ m_e
```

其中：

```text
θ_{k+1} = θ_k - α ∇_θ L_support(θ_k, m)
```

实现建议：

1. 如果 functional GCN 实现复杂，先实现 2-layer GCN 的 functional forward。
2. 如果完整 unrolled 太慢，先支持 Cora/CiteSeer。
3. 输出 raw_grad 和 harmful_score。
4. 报告中必须说明 first-order 与 unrolled 的差异。

## 任务 5：实现图演化裁剪

新增：

```text
src/grage/graph_evolution.py
```

实现：

```python
def evolve_graph_by_score(
    edge_index,
    harmful_score,
    num_nodes,
    prune_ratio=0.2,
    min_degree=1,
    undirected=True,
):
    ...
```

要求：

1. 高 harmful_score 优先删除；
2. 无向图成对删除；
3. 保持 `min_degree`；
4. 实际裁剪率尽量接近目标；
5. 返回 `pruned_edge_index, prune_mask, graph_stats`。

可以复用现有 `prune_graph`，但必须确认分数方向正确。

## 任务 6：实现 GraGE 诊断实验脚本

新增：

```text
scripts/run_grage_diagnostic.py
```

实验矩阵：

```text
datasets:
  Cora
  CiteSeer
  PubMed

seeds:
  0-9

noise_type:
  cross_class_oracle
  low_feature_similarity
  random_inter_community
  degree_aligned_random

noise_ratio:
  0.2

downstream:
  GCN
  GAT
  GraphSAGE
```

方法：

```text
Original+Noise
Random-Matched
DegreeAwareRandom-Matched
Feature-only
EdgeInfluence-Pseudo-old
GraGE-first-order
GraGE-unrolled
GraGE-first-order+smooth-prior
GraGE-unrolled+smooth-prior
Oracle-label-score  # diagnostic only
```

输出：

```text
results_clean/grage_diagnostic/grage_results.csv
```

必须包含字段：

```text
dataset
seed
experiment_type
noise_type
noise_ratio
method
score_type
oracle_label_score
clean_teacher_used
uses_feature_prior
downstream_model
test_acc
val_acc
test_f1
bad_edge_precision
bad_edge_recall
bad_edge_f1
actual_prune_ratio
edge_homophily_before
edge_homophily_after
num_edges_before
num_edges_after
runtime
```

## 任务 7：生成理论文档

新增：

```text
docs/GRAGE_THEORY.md
```

内容必须包括：

1. Differentiable edge gates；
2. Support/score bilevel objective；
3. First-order edge influence；
4. Unrolled hypergradient；
5. Feature smoothness as prior, not main score；
6. 与 Feature-only、GCN-Jaccard、DropEdge、Pro-GNN 的区别。

## 任务 8：生成实验决策报告

新增：

```text
GRAGE_DECISION_REPORT.md
```

报告必须回答：

1. GraGE-first-order 是否超过 Feature-only？
2. GraGE-unrolled 是否超过 Feature-only？
3. smooth prior 是否有帮助？
4. 训练动态信号是否提供了超越静态特征相似性的增益？
5. 哪些数据集/噪声类型失败？
6. 当前结果是否足以支撑论文主 idea？

判定标准：

### 成立

如果 GraGE 在多数 noisy setting 中超过 Feature-only，平均提升至少 1pp，并且 clean graph 不明显破坏 Original，则可以支持论文主线：

```text
Training dynamics provides task-aware edge evolution signals beyond static feature smoothness.
```

### 不成立

如果 GraGE 仍弱于 Feature-only，或只在 oracle-label 下有效，必须明确写：

```text
Current training-dynamics edge gate does not outperform static feature smoothness.
The GraGE hypothesis is not supported by current experiments.
```

## 验收标准

1. `pytest` 通过。
2. `tests/test_edge_gate_gcn.py` 通过。
3. `results_clean/grage_diagnostic/grage_results.csv` 存在且非空。
4. `docs/GRAGE_THEORY.md` 存在。
5. `GRAGE_DECISION_REPORT.md` 存在。
6. 报告必须明确比较 GraGE vs Feature-only。
7. 不允许把 Feature-only 写成主方法。
8. 不允许把旧 EdgeInfluence-Pseudo 写成主方法。

完成后提交并推送 GitHub。

