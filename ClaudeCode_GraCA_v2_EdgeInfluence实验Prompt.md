# ClaudeCode 执行 Prompt：GraCA-v2 Edge Influence 最小验证实验

下面内容可直接复制给 ClaudeCode 执行。目标是先验证 `edge-gate influence` 是否真正有效，再决定是否修改论文内容。

```text
你现在继续修改 J-oMarch/GraCA。当前目标是验证一个新的 GraCA-v2 方法是否有效，而不是扩展大规模实验。

背景：
当前 GraCA 的 node-gradient cosine score 无法识别 injected bad edges：
- P_avg AUC 接近 0.5
- noisy-edge 下 GraCA-lite 低于 Original+Noise
因此需要改为 edge-level influence score。

核心新假设：
对每条边 e 引入连续 gate m_e，message passing 使用 m_e 控制该边信息传递。若

    S_e = ∂ L_score / ∂ m_e | m_e = 1

则：
- S_e > 0 表示增大该边权会增大 scoring loss，删除该边可能降低 loss，边是 harmful；
- S_e < 0 表示该边降低 scoring loss，边是 helpful。

一阶泰勒近似：

    L(m_e = 0) - L(m_e = 1) ≈ - ∂L / ∂m_e

所以 harmful_score = ReLU(S_e)，helpful_score = ReLU(-S_e)。

请严格按下面阶段执行。

阶段 0：建立 v2 分支式实现，不破坏旧方法
1. 保留旧 node-gradient cosine scoring，命名为 `node_cosine_legacy`。
2. 新增 scoring variant：
   - `edge_influence`
   - `edge_influence_unrolled`
   - `low_helpfulness`
3. 配置项新增：
   ```yaml
   scoring:
     variant: edge_influence
     gate_scope: edge
     use_unrolled: false
     support_score_split: true
     score_split_ratio: 0.5
   ```
4. 所有新实验写入：
   - `results_v2/`
   - `paper_tables_v2/`
   - `sanitized_graphs_v2/`
5. 不要继续 append 到旧 `results/` 或 `results_clean/`。

阶段 1：支持 edge_gate message passing
请修改 GCN/GAT/GraphSAGE，使 forward 支持可选参数：

```python
edge_gate: Optional[torch.Tensor] = None
```

要求：
1. `edge_gate.shape == [E]`。
2. `edge_gate` 不传时，模型行为与原来完全一致。
3. 对 GCN：
   - 优先用 PyG `GCNConv` 的 `edge_weight` 参数实现；
   - 如果原本已有 edge_weight，则乘以 edge_gate。
4. 对 GraphSAGE：
   - 如果 PyG `SAGEConv` 不直接支持 edge_weight，请新增一个简单的 gated SAGE message passing 实现，或用 MessagePassing subclass 包装；
   - 不要只忽略 edge_gate。
5. 对 GAT：
   - 先允许 edge_gate 乘到 message 或 attention output；
   - 如果实现复杂，可以第一阶段只在 GCN 上验证 v2，但代码必须明确报错或标注 GAT/SAGE 暂不支持，不要静默忽略。
6. 所有模型 forward 仍支持：
   - return_hidden
   - retain_hidden_grad

验收测试：
1. edge_gate 全 1 时，输出与不传 edge_gate 基本一致。
2. edge_gate 全 0 时，邻居信息明显被屏蔽。
3. edge_gate.requires_grad=True 时，反向传播后 edge_gate.grad 非 None。

阶段 2：实现 edge-gate influence scorer
新增文件：

```text
src/graca/edge_influence.py
```

实现函数：

```python
def collect_edge_influence_scores(
    model,
    x,
    edge_index,
    y,
    train_mask,
    score_mask,
    teacher_probs=None,
    rho_train=None,
    lambda_s=1.0,
    deterministic=True,
    create_graph=False,
):
    ...
```

逻辑：
1. 冻结模型参数：
   ```python
   for p in model.parameters(): p.requires_grad_(False)
   ```
2. 新建：
   ```python
   edge_gate = torch.ones(E, device=device, requires_grad=True)
   ```
3. 用 `model(..., edge_gate=edge_gate)` 前向。
4. 用 `score_mask` 计算 `L_score`：
   - 如果 teacher_probs/rho_train 存在，可加入 soft pseudo loss；
   - 第一版建议只用 `score_mask` 上的 supervised CE，避免伪标签噪声。
5. 反向得到：
   ```python
   S = edge_gate.grad
   ```
6. 返回：
   ```python
   {
     "S": S,
     "harmful": relu(S),
     "helpful": relu(-S),
     "score": relu(S)
   }
   ```
7. 恢复模型参数 requires_grad 状态。

注意：
- 不要用 test_mask。
- score_mask 必须来自 train 内部划分，或者 val_mask 只在明确 transductive validation setting 下使用。优先使用 train 内部分割。

阶段 3：support/score split
新增工具函数：

```text
src/data/splits.py
```

或扩展已有文件，实现：

```python
def split_train_support_score(train_mask, ratio=0.5, seed=0):
    ...
```

要求：
1. 只在 train_mask 内部分割。
2. support_mask 用于训练 proxy。
3. score_mask 用于 edge influence scoring。
4. 两者不重叠。
5. test_mask 不参与任何评分。

训练流程：
1. ProxyGNN 只用 support_mask 训练。
2. Edge scoring 只用 score_mask 计算 `L_score`。
3. Downstream 仍用完整 train_mask 在净化图上重新训练。

阶段 4：实现 unrolled edge influence
新增可选 variant：

```text
edge_influence_unrolled
```

一阶 unrolled 逻辑：
1. 当前参数为 θ。
2. 用 support_mask 计算：
   ```python
   L_support(θ, m)
   ```
3. 做一步虚拟更新：
   ```python
   θ' = θ - lr_inner * ∇_θ L_support
   ```
4. 用 θ' 和 score_mask 计算：
   ```python
   L_score(θ', m)
   ```
5. 计算：
   ```python
   d L_score(θ', m) / d m_e
   ```
6. 如果完整 functional_call 实现复杂，先实现 GCN-only 版本，并明确限制。

阶段 5：把 pruning 接到新 score
修改或新增统一入口：

```python
compute_pruning_scores(..., variant="edge_influence")
```

支持：
1. `edge_influence`: score = harmful = ReLU(S)
2. `low_helpfulness`: score = -helpful 或 score = -ReLU(-S)，实际裁剪 helpful 最低的边
3. `node_cosine_legacy`: 保留旧 P
4. `edge_influence_unrolled`: 使用 unrolled S

对 undirected 图：

```python
score_uv = mean(score_u_to_v, score_v_to_u)
```

然后复用现有 per-node adaptive pruning。

阶段 6：最小诊断实验，先不要跑大矩阵
新增脚本：

```text
scripts/run_v2_diagnostic.py
```

只跑：

```text
Dataset: Cora
Noise types:
- low_feature_similarity
- cross_class_oracle

Noise ratio:
- 0.10

Seeds:
- 0, 1, 2

Model:
- GCN only

Methods:
- Original+Noise
- Random-Matched
- Similarity-Pruning
- GraCA-node-cosine-legacy
- GraCA-edge-influence
- GraCA-low-helpfulness
- GraCA-edge-influence-unrolled，如果已实现
```

每个方法输出：
1. test_acc
2. actual_prune_ratio
3. bad_edge_precision
4. bad_edge_recall
5. bad_edge_f1
6. AUC
7. AP
8. edge_homophily_before/after

关键要求：
- Random-Matched 必须使用 GraCA-edge-influence 的 actual_prune_ratio，不能用 beta。
- Similarity-Pruning 也要 matched ratio。
- 所有 pruning method 必须返回真实 prune_mask，用于 bad-edge detection。
- AUC/AP 必须基于 method 的边级 score 对 bad_edge_mask 计算。
- 输出到：
  ```text
  results_v2/diagnostic/v2_diagnostic_results.csv
  ```

阶段 7：判断是否有效
运行完诊断后，生成：

```text
V2_DIAGNOSTIC_REPORT.md
```

报告必须回答：

1. edge_influence 的 AUC 是否 > 0.65？
2. edge_influence 的 bad_edge_f1 是否显著高于 Random-Matched？
3. edge_influence 的 noisy test_acc 是否高于 Original+Noise？
4. low_helpfulness 是否比 harmful_score 更有效？
5. unrolled 是否比 first-order 更有效？
6. 如果所有 gradient-based variants 仍接近随机，明确说失败，不要粉饰。
7. 如果某个 variant 有效，给出下一步扩大实验矩阵的命令。

阶段 8：如果 v2 有效，再准备论文内容修改建议
只有当满足以下条件时，才修改论文内容：

```text
AUC >= 0.65
bad_edge_f1 > Random-Matched
noisy test_acc >= Original+Noise
```

如果满足，请新增：

```text
docs/PAPER_REVISION_PLAN_V2.md
```

内容包括：
1. 将方法从 node-gradient cosine 改为 edge-gate influence。
2. 新数学定义：
   - m_e edge gate
   - S_e = ∂L_score / ∂m_e
   - Taylor approximation
   - harmful/helpful definition
3. 新算法伪代码。
4. 新实验表格设计。
5. 删除或降级旧 node_cosine 内容为 ablation。
6. 明确 semi-supervised legality：support/score split only inside train_mask。

如果不满足，不要改论文正文，只写失败分析和下一步方法建议。

非常重要：
- 不要用 test labels 做 scoring。
- 不要手写结果。
- 不要只看 accuracy，必须看 bad-edge AUC/F1。
- 不要扩展到更多数据集，直到 Cora v2 diagnostic 成立。
- 当前任务的成败标准是：edge-level influence 是否比 node-cosine 更能识别 injected harmful edges。
```

## 执行后的判断标准

ClaudeCode 完成后，请重点上传或保留以下文件：

```text
results_v2/diagnostic/v2_diagnostic_results.csv
V2_DIAGNOSTIC_REPORT.md
docs/PAPER_REVISION_PLAN_V2.md  # 仅当 v2 有效时生成
```

判断逻辑：

- 如果 `edge_influence` 有效，文章主线可以改成 **edge-gate influence graph sanitization**。
- 如果 `low_helpfulness` 有效而 `harmful_score` 无效，文章主线应改成 **low optimization support edge pruning**。
- 如果二者都无效，说明“监督梯度做边清洗”这条路线至少在当前设定下不成立，应转向 feature/structure + gradient 混合方法。

