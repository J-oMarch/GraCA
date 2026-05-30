# ClaudeCode_实现与实验指南

本文档用于指导 Claude Code 从零实现 Practical GraCA / GraCA-lite 的模型代码、baseline、ablation、实验流水线、结果聚合与自动调优。实现时应优先保证实验合法性、可复现性和模块可替换性，再逐步扩展复杂功能。

## 1. 实现目标

本项目目标是实现一套完整的图结构净化实验框架，用于验证 Practical GraCA 是否能在不使用测试标签的标准半监督节点分类设定下，基于任务梯度行为净化图结构，并提升下游 GNN 的分类性能。

第一阶段实现 `GraCA-lite`，核心流程包括：

1. 加载节点分类数据集。
2. 训练 ProxyGNN。
3. 使用 EMA Teacher 生成 soft pseudo label。
4. 基于训练标签与 soft pseudo label 构造 `L_proxy`。
5. 基于 deterministic scoring loss 构造 `L_score`。
6. 采集隐藏表示梯度。
7. 计算边级 `D_vu`、`M_vu`、`rho_vu`、`H_vu`、`R_vu`、`P_vu`。
8. 执行 per-node adaptive top-budget pruning。
9. 保存 sanitized graph。
10. 在 sanitized graph 上从零训练 downstream GNN。
11. 实现 baseline、ablation、oracle upper-bound、结果聚合与多 seed 汇总。

第二阶段扩展为 `Full Practical GraCA`，加入 consistency regularization、weak/strong graph augmentation、多层梯度、多 checkpoint temporal stability、bridge protection 和大图实验。

必须严格区分：

- `GraCA-lite / Practical GraCA`：合法半监督主方法，不使用测试标签。
- `Oracle GraCA`：仅用于 upper-bound / diagnostic study，单独隔离，不进入主实验表。

## 2. Claude Code 执行协议

Claude Code 应按 milestone 推进，不要一次性生成全部复杂功能。每个 milestone 必须满足验收标准后再进入下一步。

执行原则：

- 先实现最小可运行闭环：数据加载 -> Original GNN -> ProxyGNN -> 梯度采集 -> 边打分 -> 裁剪 -> 下游重训。
- 每新增一个模块，添加最小测试或 sanity check。
- 所有随机实验必须固定 seed，并将 seed 写入结果 CSV。
- 所有结果必须可复现、可聚合、可区分 method 和 oracle。
- 禁止在 practical 模式下读取 `test_mask` 对应标签参与任何 loss、pseudo-label 构造、edge scoring 或 pruning 决策。
- 如果某个高级 baseline 实现成本过高，先保留接口和配置项，完成核心 GraCA 与基础 baseline。

推荐开发顺序：

1. `M1` 数据加载与泄漏检查。
2. `M2` GCN/GAT/GraphSAGE 与 Original baseline。
3. `M3` EMA Teacher 与 soft pseudo label。
4. `M4` ProxyGNN 训练。
5. `M5` hidden gradient collection。
6. `M6` edge scoring。
7. `M7` graph pruning 与 graph stats。
8. `M8` downstream retraining。
9. `M9` baselines、ablation、aggregation。
10. `M10` 自动调参与完整实验矩阵。

## 3. 项目技术栈

建议技术栈：

```text
Python >= 3.9
PyTorch >= 2.0
PyTorch Geometric >= 2.4
NumPy
scikit-learn
PyYAML
pandas
tqdm
networkx
matplotlib
```

可选：

```text
ogb
tensorboard
wandb
torch-scatter
```

第一阶段不强制使用 wandb，优先保证 CSV 结果完整、可复现、可聚合。

## 4. 推荐项目目录结构

```text
GraCA/
├── configs/
│   ├── graca_lite_cora.yaml
│   ├── graca_lite_citeseer.yaml
│   ├── graca_lite_pubmed.yaml
│   ├── graca_lite_actor.yaml
│   ├── oracle_cora.yaml
│   ├── downstream_gcn.yaml
│   ├── downstream_gat.yaml
│   └── baselines.yaml
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py
│   │   ├── splits.py
│   │   └── leakage_check.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── gcn.py
│   │   ├── gat.py
│   │   ├── sage.py
│   │   └── model_factory.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_proxy.py
│   │   ├── train_downstream.py
│   │   ├── losses.py
│   │   ├── evaluator.py
│   │   └── early_stopping.py
│   ├── graca/
│   │   ├── __init__.py
│   │   ├── ema_teacher.py
│   │   ├── pseudo_label.py
│   │   ├── scoring_loss.py
│   │   ├── gradient_collector.py
│   │   ├── edge_scoring.py
│   │   ├── pruning.py
│   │   ├── save_graph.py
│   │   └── oracle.py
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── original.py
│   │   ├── dropedge.py
│   │   ├── random_pruning.py
│   │   ├── homophily_pruning.py
│   │   ├── gdc.py
│   │   ├── gnnguard.py
│   │   └── prognn.py
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── graph_stats.py
│   │   ├── result_writer.py
│   │   └── aggregate.py
│   └── utils/
│       ├── __init__.py
│       ├── seed.py
│       ├── config.py
│       ├── logger.py
│       ├── device.py
│       └── io.py
├── scripts/
│   ├── run_graca.py
│   ├── run_downstream.py
│   ├── run_baselines.py
│   ├── run_oracle.py
│   ├── run_ablation.py
│   ├── run_sweep.py
│   └── aggregate_results.py
├── results/
│   ├── main/
│   ├── oracle/
│   ├── ablation/
│   ├── baselines/
│   ├── sweeps/
│   └── aggregated/
├── checkpoints/
│   ├── proxy/
│   ├── teacher/
│   └── downstream/
├── sanitized_graphs/
│   ├── graca_lite/
│   ├── oracle/
│   ├── random/
│   └── homophily/
├── logs/
├── README.md
└── requirements.txt
```

## 5. 配置文件设计

所有实验使用 YAML 配置。示例：`configs/graca_lite_cora.yaml`

```yaml
dataset:
  name: Cora
  root: data/
  split: public
  normalize_features: true
  add_self_loops: true
  undirected: true

proxy_model:
  name: GCN
  hidden_dim: 64
  num_layers: 2
  dropout: 0.5
  activation: relu

downstream_model:
  names: [GCN, GAT, GraphSAGE]
  hidden_dim: 64
  num_layers: 2
  dropout: 0.5

training:
  epochs: 300
  lr: 0.01
  weight_decay: 0.0005
  patience: 100
  use_val_for_early_stopping: true
  device: cuda

teacher:
  use_ema: true
  ema_decay: 0.99
  update_after_epoch: 1

pseudo:
  tau: 0.8
  alpha: 1.0
  epsilon_rho: 0.05
  lambda_p: 1.0
  use_soft_label: true
  hard_pseudo: false

scoring:
  lambda_s: 1.0
  deterministic: true
  collect_layer: last
  checkpoints: [100, 150, 200, 250, 300]
  use_multi_checkpoint: true
  eta: 1.0
  eps: 1.0e-12

pruning:
  beta: 0.2
  min_degree: 1
  lambda_theta: 0.0
  use_local_threshold: true
  use_top_budget: true
  protect_self_loops: true
  undirected_score_reduce: mean

experiment:
  method: graca_lite
  oracle_only: false
  run_id: graca_lite_cora_seed42
  seeds: [42]

logging:
  save_checkpoints: true
  save_sanitized_graph: true
  save_edge_scores: true
  result_dir: results/main/
  graph_dir: sanitized_graphs/graca_lite/
```

Oracle 配置必须显式包含：

```yaml
experiment:
  method: oracle
  oracle_only: true
```

否则禁止访问 full labels。

## 6. 数据集模块

需要支持：

```text
Cora, CiteSeer, PubMed, Actor, Texas, Cornell, Wisconsin
```

可选：

```text
ogbn-arxiv
```

### 6.1 `load_dataset(config)`

文件：`src/data/load_data.py`

输入：

```python
config: dict
```

输出：

```python
data: torch_geometric.data.Data
num_features: int
num_classes: int
```

`data` 至少包含：

```python
data.x
data.edge_index
data.y
data.train_mask
data.val_mask
data.test_mask
```

对于没有标准 split 的数据集，需要在 `src/data/splits.py` 中生成固定随机划分，并保存到磁盘，保证可复现。

### 6.2 `get_split_masks(data)`

输出：

```python
train_mask: torch.BoolTensor
val_mask: torch.BoolTensor
test_mask: torch.BoolTensor
unlabeled_mask: torch.BoolTensor
```

定义：

```python
unlabeled_mask = ~train_mask
```

注意：在 transductive node classification 中，未标注节点可以包含 validation/test 节点的特征与结构，但不能使用其真实标签。

### 6.3 `ensure_no_test_label_leakage(config, masks, mode)`

文件：`src/data/leakage_check.py`

必须检查：

1. 若 `mode != "oracle"`，任何 loss index 不得包含 `test_mask`。
2. 若 `config["experiment"]["oracle_only"] is False`，禁止使用 full labels。
3. 若 `method == "graca_lite"`，`L_proxy` 与 `L_score` 只能访问 `train_mask` 标签。
4. 若输出文件路径包含 `oracle`，必须 `oracle_only=true`。
5. 主结果聚合必须排除 `oracle_only=true` 的结果。

## 7. 模型模块

需要实现：

- GCN
- GAT
- GraphSAGE

文件：

```text
src/models/gcn.py
src/models/gat.py
src/models/sage.py
```

统一接口：

```python
class BaseGNN(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_hidden: bool = False,
        retain_hidden_grad: bool = False,
    ):
        ...
```

若 `return_hidden=False`：

```python
return logits
```

若 `return_hidden=True`：

```python
return logits, hidden_list
```

其中：

```python
hidden_list: List[Tensor]  # each tensor shape: [num_nodes, hidden_dim]
```

如果 `retain_hidden_grad=True`，对需要采集梯度的 hidden tensor 调用：

```python
hidden.retain_grad()
```

第一阶段只采集最后一层 hidden representation。

### 7.1 Model Factory

文件：`src/models/model_factory.py`

```python
def build_model(
    name: str,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_layers: int,
    dropout: float,
    **kwargs,
):
    ...
```

支持：

```python
name in ["GCN", "GAT", "GraphSAGE"]
```

## 8. EMA Teacher 模块

文件：`src/graca/ema_teacher.py`

### 8.1 `EMATeacher`

```python
class EMATeacher:
    def __init__(self, student_model: torch.nn.Module, decay: float):
        ...

    def update(self, student_model: torch.nn.Module):
        ...

    @torch.no_grad()
    def predict(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        ...

    def eval(self):
        ...
```

初始化：

```python
teacher = deepcopy(student_model)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)
```

EMA 更新：

```python
theta_bar = decay * theta_bar + (1 - decay) * theta
```

Teacher 预测必须：

```python
with torch.no_grad():
    logits = teacher(x, edge_index)
    q = torch.softmax(logits, dim=-1)
```

禁止 teacher 反向传播。

## 9. ProxyGNN 训练模块

文件：`src/training/train_proxy.py`

第一阶段实现 GraCA-lite：

$$
\mathcal{L}_{proxy}
=
\mathcal{L}_{sup}
+
\lambda_p\mathcal{L}_{soft}.
$$

暂不实现 consistency。

### 9.1 Supervised Loss

```python
loss_sup = F.cross_entropy(logits[train_mask], y[train_mask])
```

只能访问 `train_mask` 标签。

### 9.2 Soft Pseudo Label

文件：`src/graca/pseudo_label.py`

```python
def compute_soft_pseudo_labels(
    teacher_probs: torch.Tensor,
    train_mask: torch.Tensor,
    unlabeled_mask: torch.Tensor,
    tau: float,
    alpha: float,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        q: teacher_probs, shape [N, C]
        confidence: shape [N]
        entropy: shape [N]
        rho_train: shape [N]
    """
```

计算：

$$
q_v=p_{\bar{\theta}}(v),
$$

$$
c_v=\max_k q_{v,k},
$$

$$
H(q_v)=-\sum_k q_{v,k}\log(q_{v,k}+\epsilon).
$$

训练可靠性：

$$
\rho_v^{train}
=
\begin{cases}
1, & v\in\mathcal{V}_L,\\
c_v^\alpha\left(1-\frac{H(q_v)}{\log C}\right), & v\in\mathcal{V}_U,\ c_v\ge\tau,\\
0, & v\in\mathcal{V}_U,\ c_v<\tau.
\end{cases}
$$

实现时训练节点的 \(\rho_v^{train}\) 可单独处理，伪标签 loss 仅对 unlabeled 节点计算。

### 9.3 Soft Pseudo Loss

```python
log_probs = F.log_softmax(student_logits, dim=-1)
loss_soft_node = F.kl_div(
    log_probs,
    q.detach(),
    reduction="none",
).sum(dim=-1)

weights = rho_train[unlabeled_mask]
loss_soft = (weights * loss_soft_node[unlabeled_mask]).sum()
loss_soft = loss_soft / (weights.sum() + eps)
loss_proxy = loss_sup + lambda_p * loss_soft
```

训练日志至少记录：

```text
loss_sup
loss_soft
pseudo_coverage
mean_confidence
val_acc
```

## 10. Scoring Loss 模块

文件：`src/graca/scoring_loss.py`

GraCA-lite scoring loss：

$$
\mathcal{L}_{score}
=
\mathcal{L}_{sup}^{det}
+
\lambda_s\mathcal{L}_{soft}^{det}.
$$

要求：

1. 使用原始 deterministic `edge_index`。
2. 不使用 DropEdge。
3. 不使用 feature masking。
4. 不使用 consistency。
5. 不访问测试标签。
6. teacher soft label 必须 `detach()` 固定。

函数：

```python
def compute_scoring_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    unlabeled_mask: torch.Tensor,
    teacher_probs: torch.Tensor,
    rho_train: torch.Tensor,
    lambda_s: float,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ...
```

返回：

```python
loss_score
loss_sup_det
loss_soft_det
```

## 11. Hidden Gradient Collection

文件：`src/graca/gradient_collector.py`

第一阶段只采集最后一层 hidden gradient。

```python
def collect_hidden_gradients(
    model: torch.nn.Module,
    data,
    teacher_probs: torch.Tensor,
    rho_train: torch.Tensor,
    train_mask: torch.Tensor,
    unlabeled_mask: torch.Tensor,
    lambda_s: float,
    collect_layer: str = "last",
) -> dict:
    """
    Returns:
        {
            "hidden": Tensor [N, H],
            "grad": Tensor [N, H],
            "logits": Tensor [N, C],
            "loss_score": float
        }
    """
```

流程：

```python
model.train()
model.zero_grad(set_to_none=True)

logits, hidden_list = model(
    data.x,
    data.edge_index,
    return_hidden=True,
    retain_hidden_grad=True,
)

hidden = hidden_list[-1]
loss_score, _, _ = compute_scoring_loss(...)
loss_score.backward()

grad = hidden.grad.detach().clone()
hidden = hidden.detach().clone()
model.zero_grad(set_to_none=True)
```

注意：

- hidden 必须参与计算图。
- 必须在 backward 前调用 `retain_grad()`。
- backward 后读取 `hidden.grad`。
- 采集后清空梯度，避免污染后续训练。

Sanity checks：

```python
assert grad.shape == hidden.shape
assert torch.isfinite(grad).all()
assert grad.abs().sum() > 0
```

若使用多 checkpoint，保存：

```python
grad_list: list[Tensor]
```

## 12. Edge Scoring 模块

文件：`src/graca/edge_scoring.py`

输入：

```python
edge_index: LongTensor [2, E]
grad: Tensor [N, H]
rho_score: Tensor [N]
eta: float
epsilon_rho: float
eps: float
```

### 12.1 方向约定

PyG `edge_index` 约定：

```python
src = edge_index[0]  # neighbor u
dst = edge_index[1]  # target v
```

实现中统一解释为：

```text
u -> v 表示 neighbor u 对 target v 的贡献
```

### 12.2 计算 \(D_{vu}\)

$$
D_{vu}=\cos(g_v,g_u).
$$

实现：

```python
g_u = grad[src]
g_v = grad[dst]
D = F.cosine_similarity(g_v, g_u, dim=-1, eps=eps)
```

输出形状：

```python
D: Tensor [E]
```

### 12.3 计算 \(M_{vu}\)

$$
M_{vu}
=
\frac{\|g_u\|_2}
{\operatorname{mean}_{j\in\mathcal{N}(v)}\|g_j\|_2+\epsilon}.
$$

实现：

```python
grad_norm = torch.norm(grad, p=2, dim=-1)  # [N]
src_norm = grad_norm[src]                  # [E]
mean_norm_per_dst = scatter_mean(src_norm, dst, dim=0, dim_size=num_nodes)
M = src_norm / (mean_norm_per_dst[dst] + eps)
```

如果环境中没有 `torch_scatter`，用 `scatter_add_` 自行实现 mean：

```python
sum_norm = torch.zeros(num_nodes, device=grad.device).scatter_add_(0, dst, src_norm)
deg = torch.zeros(num_nodes, device=grad.device).scatter_add_(0, dst, torch.ones_like(src_norm))
mean_norm_per_dst = sum_norm / deg.clamp_min(1)
```

### 12.4 计算 \(\rho_v^{score}\) 与 \(\rho_{vu}\)

```python
def compute_rho_score(
    teacher_probs: torch.Tensor,
    train_mask: torch.Tensor,
    unlabeled_mask: torch.Tensor,
    tau: float,
    alpha: float,
    epsilon_rho: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    ...
```

评分可靠性：

$$
\rho_v^{score}
=
\begin{cases}
1, & v\in\mathcal{V}_L,\\
c_v^\alpha\left(1-\frac{H(q_v)}{\log C}\right), & v\in\mathcal{V}_U,\ c_v\ge\tau,\\
\epsilon_\rho, & v\in\mathcal{V}_U,\ c_v<\tau.
\end{cases}
$$

边可靠性：

$$
\rho_{vu}
=
\rho_v^{score}
\cdot
\operatorname{clip}(\rho_u^{score},\epsilon_\rho,1).
$$

实现：

```python
rho_v = rho_score[dst]
rho_u = rho_score[src].clamp(min=epsilon_rho, max=1.0)
rho_vu = rho_v * rho_u
```

### 12.5 计算 \(H_{vu}\)、\(R_{vu}\)、\(P_{vu}\)

单 checkpoint：

$$
H_{vu}=\rho_{vu}\max(D_{vu},0)M_{vu},
$$

$$
R_{vu}=\rho_{vu}\max(-D_{vu},0)M_{vu},
$$

$$
P_{vu}=R_{vu}-\eta H_{vu}.
$$

实现：

```python
H = rho_vu * torch.clamp(D, min=0.0) * M
R = rho_vu * torch.clamp(-D, min=0.0) * M
P = R - eta * H
```

多 checkpoint：

```python
H = torch.stack(H_all, dim=0).mean(dim=0)
R = torch.stack(R_all, dim=0).mean(dim=0)
P = R - eta * H
```

### 12.6 无向图处理

PyG 通常用双向 `edge_index` 表示无向图。如果存在 `u -> v` 和 `v -> u`，建议逐向计算分数，删除时同步删除双向边。

推荐第一阶段策略：

1. 保留 directed edge scoring。
2. 构造无向 key：`tuple(sorted((u, v)))`。
3. 对同一无向边的两个方向风险分数取平均。
4. prune 时删除两个方向。
5. self-loop 默认保护，不参与裁剪。

## 13. Graph Pruning 模块

文件：`src/graca/pruning.py`

```python
def prune_graph(
    edge_index: torch.Tensor,
    risk_score: torch.Tensor,
    num_nodes: int,
    beta: float,
    min_degree: int,
    lambda_theta: float = 0.0,
    undirected: bool = True,
    protect_self_loops: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Returns:
        pruned_edge_index: LongTensor [2, E']
        prune_mask: BoolTensor [E]  # True means edge removed
        graph_stats: dict
    """
```

### 13.1 Per-node Local Threshold

对每个 target node \(v\)：

$$
\mathcal{P}_v=\{P_{vu}:u\in\mathcal{N}(v)\}.
$$

局部阈值：

$$
\theta_v
=
\operatorname{mean}(\mathcal{P}_v)
+
\lambda_\theta \operatorname{std}(\mathcal{P}_v).
$$

第一阶段可设置：

```yaml
lambda_theta: 0.0
```

即只裁剪高于局部均值的高风险边。

### 13.2 Top-Budget

$$
b_v
=
\min
\left(
\lfloor \beta|\mathcal{N}(v)|\rfloor,
|\mathcal{N}(v)|-d_{min}
\right).
$$

若 \(b_v\le0\)，节点 \(v\) 不裁剪。

### 13.3 Minimum Degree Protection

删除边前检查：

```python
degree[v] > min_degree
degree[u] > min_degree
```

对无向图，需要同时保护两端。禁止删除 self-loop：

```python
if u == v and protect_self_loops:
    keep_edge()
```

### 13.4 输出统计

`graph_stats` 至少包含：

```python
{
    "num_edges_before": int,
    "num_edges_after": int,
    "prune_ratio": float,
    "isolated_nodes": int,
    "min_degree": int,
    "mean_degree": float,
    "largest_connected_component_ratio": float,
}
```

## 14. Downstream Retraining 模块

文件：`src/training/train_downstream.py`

目标：在 sanitized graph 上从零训练 GCN / GAT / GraphSAGE。

```python
def train_downstream(
    model_name: str,
    data,
    pruned_edge_index: torch.Tensor,
    config: dict,
    seed: int,
) -> dict:
    ...
```

要求：

1. 重新初始化模型。
2. 不加载 ProxyGNN 权重。
3. 只使用 `train_mask` 标签计算 loss。
4. 使用 `val_mask` early stopping。
5. 最终报告 `test_mask` accuracy。
6. 不允许使用测试标签调参。

训练 loss：

```python
loss = F.cross_entropy(logits[train_mask], y[train_mask])
```

输出：

```python
{
    "val_acc": float,
    "test_acc": float,
    "best_epoch": int,
    "runtime": float,
}
```

## 15. Oracle GraCA 模块

文件：`src/graca/oracle.py`

Oracle 仅用于 upper-bound / diagnostic study。

配置必须：

```yaml
experiment:
  method: oracle
  oracle_only: true
```

Oracle 允许构造：

$$
\mathcal{L}_{oracle}
=
\sum_{v\in\mathcal{V}}
CE(y_v,p_\theta(v)).
$$

但必须满足：

1. 输出目录为 `results/oracle/`。
2. sanitized graph 保存到 `sanitized_graphs/oracle/`。
3. 结果 CSV 字段 `oracle_only=True`。
4. `aggregate_results.py` 默认排除 oracle。
5. 主表禁止读取 oracle 文件。

```python
def run_oracle_graca(config: dict):
    assert config["experiment"]["oracle_only"] is True
    ...
```

代码中必须明确写注释：

```python
# ORACLE ONLY: uses full labels for diagnostic upper-bound.
# Do not include in main semi-supervised results.
```

## 16. Baseline 模块

第一阶段至少实现：

### 16.1 Original

不裁剪原图，直接训练 downstream GNN。

文件：`src/baselines/original.py`

### 16.2 DropEdge

训练时随机 drop edge。

配置：

```yaml
dropedge_rate: 0.2
```

DropEdge 不保存固定 sanitized graph，它是训练正则策略。

### 16.3 Random Pruning

按照 GraCA 相同裁剪比例随机删边。目的：证明提升不是因为简单稀疏化。

### 16.4 Homophily Pruning

第一阶段实现合法弱版本：

- 只使用训练标签节点之间的同/异类信息；
- 或只使用 high-confidence pseudo label；
- 禁止使用测试标签。

第二阶段预留：

- GDC
- GNNGuard
- ProGNN
- Jaccard-GCN

如果高级 baseline 实现困难，先保留接口，并在 README 标注待补充。

## 17. Ablation 实验

必须支持以下 ablation：

| Ablation | 设置 | 目的 |
|---|---|---|
| w/o EMA | 直接使用 student prediction | 验证 EMA teacher 稳定性 |
| hard pseudo | 使用 `argmax(q)` 构造 CE | 验证 soft label 必要性 |
| w/o reliability | 设置所有 \(\rho_v=1\) | 验证 uncertainty reliability |
| harmful-only | \(P_{vu}=R_{vu}\) | 验证 helpful 抵消项 |
| helpful-only | 裁剪低 \(H_{vu}\) | 验证 harmful 分数必要性 |
| global threshold | 使用全局阈值裁剪 | 验证 local adaptive pruning |
| train-only | \(L_{score}=L_{sup}\) | 验证 soft pseudo 覆盖全图的贡献 |
| oracle | full-label loss | upper-bound 分析 |

## 18. 实验矩阵

### 18.1 主实验

数据集：

```text
Cora, CiteSeer, PubMed, Actor, Texas, Cornell, Wisconsin
```

方法：

```text
Original, DropEdge, Random Pruning, Homophily Pruning, GraCA-lite
```

下游模型：

```text
GCN, GAT, GraphSAGE
```

证明点：GraCA-lite 是否稳定超过 Original 和随机稀疏化。

### 18.2 Oracle 上界实验

方法：

```text
Oracle GraCA
```

证明点：梯度边信号是否存在上界收益。

### 18.3 消融实验

优先数据集：

```text
Cora, PubMed, Actor, Texas
```

证明点：每个模块是否有独立贡献。

### 18.4 鲁棒性实验

人工加边噪声：

```text
noise_ratio = [0.05, 0.10, 0.20, 0.30]
```

噪声构造：

- 随机添加跨类边：只允许使用训练标签或 synthetic oracle 标注的实验设置。
- 完全随机重连：不使用标签。
- 对抗扰动：第二阶段可选。

证明点：GraCA 是否能清除 task-harmful noisy edges。

### 18.5 迁移实验

ProxyGNN 固定为：

```text
GCN
```

Downstream：

```text
GCN, GAT, GraphSAGE
```

证明点：sanitized graph 是否 downstream transferable。

### 18.6 可扩展性实验

指标：

```text
runtime, memory, num_edges, pruning time
```

可选数据：

```text
ogbn-arxiv
```

## 19. 自动调优与失败诊断

Claude Code 应实现轻量 sweep，不需要一开始使用复杂 HPO。

### 19.1 第一阶段调参范围

```yaml
pseudo.tau: [0.7, 0.8, 0.9]
pseudo.lambda_p: [0.5, 1.0, 2.0]
scoring.eta: [0.5, 1.0, 2.0]
pruning.beta: [0.05, 0.10, 0.20, 0.30]
pruning.lambda_theta: [0.0, 0.5, 1.0]
pruning.min_degree: [1, 2]
```

调参规则：

- 只能使用 validation accuracy 选择超参数。
- 不能使用 test accuracy 选择超参数。
- 最终 test accuracy 只在固定配置后报告。
- 每组 sweep 记录完整配置。

### 19.2 失败诊断

如果 GraCA-lite 明显低于 Original，依次检查：

1. `edge_index` 方向是否反了。
2. `hidden.grad` 是否非零。
3. `D_vu` 是否同时存在正负值。
4. `M_vu` 是否出现 NaN 或极端值。
5. `rho_train` 是否几乎全 0。
6. `tau` 是否过高导致 pseudo coverage 太低。
7. `beta` 是否过大导致过度裁剪。
8. 是否误删 self-loop。
9. minimum degree protection 是否生效。
10. practical 模式是否意外使用 test labels。

建议保存诊断统计：

```text
pseudo_coverage
mean_confidence
mean_D
frac_negative_D
mean_M
mean_H
mean_R
mean_P
prune_ratio
isolated_nodes
```

## 20. 结果保存格式

所有实验结果保存为 CSV。

路径：

```text
results/main/results.csv
results/oracle/oracle_results.csv
results/ablation/ablation_results.csv
results/baselines/baseline_results.csv
results/sweeps/sweep_results.csv
```

字段：

```text
run_id
seed
dataset
method
oracle_only
proxy_model
downstream_model
prune_ratio
num_edges_before
num_edges_after
isolated_nodes
min_degree
mean_degree
largest_connected_component_ratio
val_acc
test_acc
best_epoch
runtime
config_path
graph_path
checkpoint_path
```

示例：

```csv
run_id,seed,dataset,method,oracle_only,proxy_model,downstream_model,prune_ratio,num_edges_before,num_edges_after,isolated_nodes,val_acc,test_acc,runtime
graca_lite_cora_seed42,42,Cora,GraCA-lite,False,GCN,GCN,0.18,10556,8654,0,0.812,0.806,34.2
```

## 21. 运行命令设计

### 21.1 运行 GraCA-lite

```bash
python scripts/run_graca.py --config configs/graca_lite_cora.yaml
```

### 21.2 在 sanitized graph 上训练 downstream

```bash
python scripts/run_downstream.py \
  --config configs/graca_lite_cora.yaml \
  --graph sanitized_graphs/graca_lite/Cora_seed42.pt \
  --model GCN
```

### 21.3 运行 baseline

```bash
python scripts/run_baselines.py --config configs/baselines.yaml
```

### 21.4 运行 oracle

```bash
python scripts/run_oracle.py --config configs/oracle_cora.yaml
```

### 21.5 运行 ablation

```bash
python scripts/run_ablation.py \
  --config configs/graca_lite_cora.yaml \
  --ablation no_reliability
```

### 21.6 运行 sweep

```bash
python scripts/run_sweep.py \
  --config configs/graca_lite_cora.yaml \
  --sweep configs/sweeps/graca_lite_small.yaml
```

### 21.7 聚合结果

```bash
python scripts/aggregate_results.py \
  --input results/main/ \
  --output results/aggregated/main_summary.csv \
  --exclude_oracle true
```

## 22. 防止标签泄漏硬规则

必须在代码层面实现以下检查。

### 22.1 Practical 训练阶段

禁止：

```python
y[test_mask]
```

进入任何 loss。只允许：

```python
y[train_mask]
```

用于 supervised loss。

### 22.2 Practical Scoring 阶段

禁止 `test_mask` 参与 `L_score`。

### 22.3 Pseudo Label

pseudo label 只能来自 teacher prediction：

```python
q = teacher(x, edge_index)
```

不能来自真实测试标签。

### 22.4 Oracle 隔离

只有以下两个条件同时满足，才能使用 full labels：

```yaml
oracle_only: true
method: oracle
```

### 22.5 聚合排除

`aggregate_results.py` 默认：

```python
df = df[df["oracle_only"] == False]
```

### 22.6 文件路径隔离

Oracle 输出必须进入：

```text
results/oracle/
sanitized_graphs/oracle/
```

### 22.7 断言检查

Practical 模式必须通过：

```python
assert not torch.any(test_mask & loss_mask)
```

## 23. 第一阶段 GraCA-lite 开发计划

### M1 数据加载

完成：

- Cora
- CiteSeer
- PubMed
- Actor
- Texas
- Cornell
- Wisconsin

验收：

```python
data.x
data.edge_index
data.y
data.train_mask
data.val_mask
data.test_mask
```

均存在且 shape 正确。

### M2 模型训练

完成：

- GCN
- GAT
- GraphSAGE
- Original baseline

验收：Original 精度接近常见范围。

### M3 EMA Teacher

完成：

- 初始化
- update
- predict

验收：teacher 参数不参与梯度更新。

### M4 ProxyGNN + Soft Pseudo Loss

完成：

$$
\mathcal{L}_{proxy}
=
\mathcal{L}_{sup}
+
\lambda_p\mathcal{L}_{soft}.
$$

验收：可记录 pseudo coverage。

### M5 Gradient Collection

完成：

- hidden `retain_grad`
- backward
- collect grad

验收：

```python
grad.shape == hidden.shape
grad.abs().sum() > 0
```

### M6 Edge Scoring

完成：

- \(D_{vu}\)
- \(M_{vu}\)
- \(\rho_{vu}\)
- \(H_{vu}\)
- \(R_{vu}\)
- \(P_{vu}\)

验收：输出每条边分数，无 NaN/Inf。

### M7 Pruning

完成：

- per-node local pruning
- top budget
- minimum degree protection

验收：裁剪比例合理，孤立节点不增加或被限制。

### M8 Downstream Retraining

完成：

- GCN
- GAT
- GraphSAGE

验收：输出 val/test acc。

### M9 Baseline + Result Aggregation

完成：

- Original
- DropEdge
- Random Pruning
- Homophily Pruning
- CSV aggregation

验收：生成主实验表。

## 24. 第二阶段 Full Practical GraCA 扩展

第二阶段在 GraCA-lite 验证有效后再实现。

### 24.1 Consistency Loss

加入：

$$
\mathcal{L}_{proxy}
=
\mathcal{L}_{sup}
+
\lambda_p\mathcal{L}_{soft}
+
\lambda_c\mathcal{L}_{cons}.
$$

### 24.2 Weak / Strong Graph Augmentation

Weak view：

- 原图
- 轻微 feature dropout

Strong view：

- edge dropout
- feature masking

注意：scoring 阶段仍必须 deterministic。

### 24.3 Multi-layer Gradient

从：

```python
hidden_list[-1]
```

扩展为：

```python
hidden_list[0], hidden_list[1], ...
```

最终分数对 layer 平均。

### 24.4 Multi-checkpoint Temporal Stability

保存多个 checkpoint：

```text
epoch 100, 150, 200, 250, 300
```

计算多 checkpoint 平均分数。

### 24.5 Bridge Protection

使用 networkx 检测桥边，禁止删除 bridge edge。

### 24.6 OGB 大图

支持：

```text
ogbn-arxiv
```

需要 mini-batch 或 neighbor sampling。

## 25. 验收标准

### 25.1 Oracle 有效

Oracle GraCA 应满足：

```text
Oracle GraCA > Original
Oracle GraCA > Random Pruning
```

若 Oracle 无效，说明梯度打分实现或原始假设存在问题。

### 25.2 Practical 有效

GraCA-lite 至少应在主要数据集上满足：

```text
GraCA-lite >= Original
GraCA-lite >= DropEdge
GraCA-lite >= Random Pruning
```

理想结果：

```text
平均提升 1%~3%
```

若 Practical 明显低于 Original，检查 pseudo label 质量、pruning ratio、gradient collection、edge direction、测试标签泄漏等问题。

### 25.3 可迁移性

同一 sanitized graph 应对多个 downstream model 有收益：

```text
GCN Proxy -> GCN/GAT/GraphSAGE Downstream
```

若只对 Proxy 同构模型有效，则 downstream transferability 不足。

### 25.4 无测试标签泄漏

必须满足：

```text
Practical loss masks do not overlap with test_mask.
Oracle results are excluded from main tables.
```

### 25.5 图统计正确

检查：

```text
num_edges_after < num_edges_before
prune_ratio within beta
minimum degree protected
isolated nodes not increased or controlled
```

### 25.6 多 seed 可复现

至少运行：

```text
seeds = [0, 1, 2, 3, 4]
```

报告：

```text
mean ± std
```

### 25.7 继续推进标准

如果出现以下情况，说明 idea 需要调整：

1. Oracle 明显无效。
2. Practical 全数据集低于 Original。
3. Random Pruning 与 GraCA-lite 无差异。
4. 只在 Cora 有效。
5. Pruning 后大量节点孤立。
6. GraCA-lite 只对 ProxyGNN 有效，不能迁移到其他 downstream models。

如果出现以下情况，说明方法具备继续推进价值：

1. Oracle 明显优于 Original。
2. GraCA-lite 达到 Oracle 收益的 40%~70%。
3. GraCA-lite 平均优于 DropEdge 和 Random Pruning。
4. Sanitized graph 对 GCN/GAT/GraphSAGE 均有收益。
5. 在加噪图上优势更明显。
6. 多 seed 方差可控。
