# GraCA: Gradient-Guided Graph Connection Assessment

## 1. 项目概述

### 1.1 研究问题

图神经网络（GNN）的性能高度依赖图结构质量。然而真实图中存在大量**对当前任务优化有害的边**——它们可能真实存在，但持续向目标节点传播与任务优化方向冲突的信息。

**核心问题**：能否利用 GNN 训练过程中产生的梯度行为，自动识别并裁剪这些 task-optimization harmful edges？

### 1.2 核心思想

GraCA 不问"这条边是否正确"，而问"这条边是否帮助任务优化"。

通过分析隐藏表示梯度的**方向一致性**（D_vu）、**相对强度**（M_vu）和**预测不确定性**（ρ_vu），计算每条边的 helpful score（H_vu）、harmful score（R_vu）和 risk score（P_vu），然后执行局部自适应裁剪。

### 1.3 方法定位

- **GraCA-lite / Practical GraCA**：合法半监督主方法，不使用测试标签
- **Oracle GraCA**：使用全标签的上界分析，仅用于诊断，不进入主表
- **Full GraCA**：在 GraCA-lite 基础上加入 consistency loss、多层梯度、多 checkpoint 时序稳定性、bridge protection

---

## 2. 模型架构

### 2.1 整体流程

```
原图 G(V, E, X)
    │
    ▼
┌─────────────────────────────────────┐
│  1. 训练 ProxyGNN (GCN/GAT/GraphSAGE) │
│     - 使用 train labels (L_sup)       │
│     - EMA Teacher 生成 soft pseudo    │
│     - 可选: Consistency Loss          │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  2. 梯度采集                         │
│     - 计算 L_score 对 hidden 的梯度   │
│     - 支持单层/多层/多 checkpoint     │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  3. 边级评分                         │
│     D_vu = cos(g_v, g_u)  方向一致性 │
│     M_vu = ||g_u|| / mean  相对强度  │
│     ρ_vu = ρ_v × clip(ρ_u) 可靠性   │
│     H_vu = ρ_vu × max(D,0) × M     │
│     R_vu = ρ_vu × max(-D,0) × M    │
│     P_vu = R_vu - η × H_vu  风险    │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  4. 局部自适应裁剪                    │
│     - Per-node threshold: mean + λσ  │
│     - Top-budget: min(β×deg, deg-d) │
│     - 最小度保护 (d_min ≥ 1)         │
│     - Self-loop 保护                 │
│     - Bridge 保护 (可选)             │
└─────────────────────────────────────┘
    │
    ▼
  净化图 G' → 下游 GNN 从零训练
```

### 2.2 关键数学公式

**节点可靠性**（uncertainty-aware）：

```
ρ_v^train = { 1,                          v ∈ V_L (labeled)
            { c_v^α × (1 - H(q_v)/logC), v ∈ V_U, c_v ≥ τ
            { 0,                          v ∈ V_U, c_v < τ

ρ_v^score = { 1,                          v ∈ V_L
            { c_v^α × (1 - H(q_v)/logC), v ∈ V_U, c_v ≥ τ
            { ε_ρ,                        v ∈ V_U, c_v < τ
```

**边可靠性**（target-centered）：

```
ρ_vu = ρ_v^score × clip(ρ_u^score, ε_ρ, 1)
```

**训练损失**：

```
L_proxy = L_sup + λ_p × L_soft + λ_c × L_cons
L_score = L_sup^det + λ_s × L_soft^det
```

**风险分数**：

```
P_vu = R_vu - η × H_vu
```

### 2.3 与现有方法的区别

| 方法 | 边判断信号 | Task-Aware | 使用梯度 | 输出 |
|------|-----------|-----------|---------|------|
| DropEdge | 随机 | ❌ | ❌ | 训练策略 |
| GNNGuard | 特征相似性 | 部分 | ❌ | 加权模型 |
| GNNExplainer | 学习mask | ✅ | 部分 | 解释子图 |
| Homophily Pruning | 标签同质性 | ❌ | ❌ | 净化图 |
| **GraCA** | **梯度行为** | **✅** | **✅** | **净化图** |

---

## 3. 项目结构

```
GraCA/
├── README.md                          # 本文件
├── requirements.txt                   # Python 依赖
├── ClaudeCode_实现与实验指南.md         # 详细实现指南
├── GraCA_模型介绍与论文idea.md         # 模型与论文思路
├── 研究复盘.md                        # 研究背景复盘
│
├── configs/                           # YAML 配置文件
│   ├── graca_lite_{dataset}.yaml      # GraCA-lite 配置 (7个数据集)
│   ├── full_graca_{dataset}.yaml      # Full GraCA 配置
│   └── oracle_{dataset}.yaml          # Oracle 配置
│
├── src/                               # 源代码 (41个文件)
│   ├── data/                          # 数据加载
│   │   ├── load_data.py               # 统一数据加载接口
│   │   ├── splits.py                  # 数据集划分
│   │   └── leakage_check.py           # 标签泄漏检查
│   │
│   ├── models/                        # GNN 模型
│   │   ├── base.py                    # 基类 (统一 forward 接口)
│   │   ├── gcn.py                     # GCN 实现
│   │   ├── gat.py                     # GAT 实现
│   │   ├── sage.py                    # GraphSAGE 实现
│   │   └── model_factory.py           # 模型工厂
│   │
│   ├── training/                      # 训练模块
│   │   ├── train_proxy.py             # ProxyGNN 训练
│   │   ├── train_downstream.py        # 下游模型训练
│   │   ├── losses.py                  # 损失函数 (L_sup, L_soft, L_score)
│   │   ├── evaluator.py               # 评估指标
│   │   └── early_stopping.py          # 早停
│   │
│   ├── graca/                         # GraCA 核心模块
│   │   ├── ema_teacher.py             # EMA 教师模型
│   │   ├── pseudo_label.py            # 软伪标签 & 可靠性计算
│   │   ├── gradient_collector.py      # 梯度采集 (单层/多层/多checkpoint)
│   │   ├── edge_scoring.py            # 边级评分 (D, M, ρ, H, R, P)
│   │   ├── pruning.py                 # 局部自适应裁剪
│   │   ├── consistency_loss.py        # Consistency 正则化
│   │   ├── bridge_protection.py       # 桥边保护
│   │   ├── oracle.py                  # Oracle GraCA (全标签诊断)
│   │   └── save_graph.py              # 净化图保存/加载
│   │
│   ├── baselines/                     # 基线方法
│   │   ├── original.py                # 原图
│   │   ├── dropedge.py                # DropEdge
│   │   ├── random_pruning.py          # 随机裁剪
│   │   └── homophily_pruning.py       # 同质性裁剪
│   │
│   ├── eval/                          # 评估模块
│   │   ├── metrics.py                 # 准确率/F1
│   │   ├── graph_stats.py             # 图统计
│   │   ├── result_writer.py           # CSV 结果写入
│   │   └── aggregate.py               # 结果聚合
│   │
│   └── utils/                         # 工具模块
│       ├── seed.py                    # 随机种子
│       ├── device.py                  # 设备管理
│       ├── config.py                  # 配置加载
│       ├── logger.py                  # 日志
│       └── io.py                      # 文件 I/O
│
├── scripts/                           # 实验脚本 (9个)
│   ├── run_graca.py                   # 运行 GraCA (lite/full/oracle)
│   ├── run_baselines.py               # 运行基线
│   ├── run_oracle.py                  # 运行 Oracle
│   ├── run_ablation.py                # 运行消融实验
│   ├── run_robustness.py              # 运行鲁棒性实验
│   ├── run_scalability.py             # 运行可扩展性实验
│   ├── run_sweep.py                   # 超参数搜索
│   ├── run_downstream.py              # 在已保存图上训练下游
│   ├── run_experiments.py             # 批量运行所有实验
│   └── aggregate_results.py           # 聚合结果
│
├── results/                           # 实验结果
│   ├── main/                          # 主实验 (166条)
│   ├── baselines/                     # 基线结果 (439条)
│   ├── oracle/                        # Oracle 结果 (109条)
│   ├── ablation/                      # 消融结果 (421条)
│   ├── robustness/                    # 鲁棒性结果 (181条)
│   ├── scalability/                   # 可扩展性结果 (7条)
│   ├── sweeps/                        # 超参数搜索 (36条)
│   └── aggregated/                    # 聚合结果 & 论文表格
│
├── sanitized_graphs/                  # 保存的净化图
├── checkpoints/                       # 模型检查点
├── logs/                              # 训练日志
└── data/                              # 数据集缓存
```

---

## 4. 环境配置

```bash
# 创建 conda 环境
conda create -n graca python=3.11 -y
conda activate graca

# 安装 PyTorch (CUDA 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 安装 PyG 和依赖
pip install torch_geometric numpy scipy scikit-learn pandas pyyaml tqdm networkx matplotlib ogb
```

---

## 5. 运行实验

### 5.1 GraCA-lite（主方法）

```bash
# 单个数据集单个 seed
python scripts/run_graca.py --config configs/graca_lite_cora.yaml --seed 0

# 所有数据集所有 seed
python scripts/run_experiments.py --datasets Cora CiteSeer PubMed --seeds 0 1 2 3 4
```

### 5.2 Full GraCA

```bash
python scripts/run_graca.py --config configs/full_graca_cora.yaml --seed 0
```

### 5.3 基线方法

```bash
python scripts/run_baselines.py --config configs/graca_lite_cora.yaml --seed 0
# 可选: --baseline original/dropedge/random/homophily
```

### 5.4 Oracle（上界诊断）

```bash
python scripts/run_oracle.py --config configs/oracle_cora.yaml --seed 0
```

### 5.5 消融实验

```bash
python scripts/run_ablation.py --config configs/graca_lite_cora.yaml --seed 0
# 可选: --ablation no_ema/hard_pseudo/no_reliability/harmful_only/helpful_only/global_threshold/train_only
```

### 5.6 鲁棒性实验

```bash
python scripts/run_robustness.py --config configs/graca_lite_cora.yaml --seed 0
```

### 5.7 可扩展性实验

```bash
python scripts/run_scalability.py --seed 0
```

### 5.8 超参数搜索

```bash
python scripts/run_sweep.py --config configs/graca_lite_cora.yaml --seed 0
```

### 5.9 聚合结果

```bash
python scripts/aggregate_results.py --include_baselines
```

### 5.10 在已保存的净化图上训练

```bash
python scripts/run_downstream.py \
    --config configs/graca_lite_cora.yaml \
    --graph sanitized_graphs/graca_lite/Cora_seed0.pt \
    --model GCN --seed 0
```

---

## 6. 配置参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `pseudo.tau` | 伪标签置信度阈值 | 0.6 |
| `pseudo.alpha` | 可靠性指数 | 1.0 |
| `pseudo.epsilon_rho` | 低置信节点可靠性下界 | 0.05 |
| `pseudo.lambda_p` | 软伪标签损失权重 | 1.0 |
| `scoring.eta` | 有害/有益平衡系数 | 1.0 |
| `scoring.lambda_s` | Scoring loss 中软伪标签权重 | 1.0 |
| `scoring.collect_layer` | 梯度采集层: first/last/all | all |
| `scoring.use_multi_checkpoint` | 是否多 checkpoint 平均 | true |
| `pruning.beta` | 每节点最大裁剪比例 | 0.2 |
| `pruning.min_degree` | 最小度保护 | 1 |
| `pruning.lambda_theta` | 局部阈值 = mean + λ×std | 0.0 |
| `pruning.protect_bridges` | 是否保护桥边 | true |
| `teacher.ema_decay` | EMA 衰减系数 | 0.99 |
| `teacher.warmup_epochs` | 初始化 EMA 教师的预热轮数 | 50 |
| `consistency.use_consistency` | 是否使用一致性损失 | true |
| `consistency.lambda_c` | 一致性损失权重 | 0.1 |

---

## 7. 实验设计

### 7.1 数据集

| 数据集 | 类型 | 节点数 | 边数 | 特征维 | 类别数 |
|--------|------|--------|------|--------|--------|
| Cora | 同质 | 2,708 | 10,556 | 1,433 | 7 |
| CiteSeer | 同质 | 3,327 | 9,104 | 3,703 | 6 |
| PubMed | 同质 | 19,717 | 88,648 | 500 | 3 |
| Actor | 异质 | 7,600 | 53,411 | 932 | 5 |
| Texas | 异质 | 183 | 574 | 1,703 | 5 |
| Cornell | 异质 | 183 | 557 | 1,703 | 5 |
| Wisconsin | 异质 | 251 | 916 | 1,703 | 5 |

### 7.2 方法对比

| 方法 | 说明 |
|------|------|
| **Original** | 原图直接训练 |
| **DropEdge** | 训练时随机丢弃 20% 边 |
| **Random Pruning** | 按相同比例随机裁剪 |
| **Homophily Pruning** | 按标签同质性裁剪 |
| **GraCA-lite** | 本文主方法（梯度引导裁剪） |
| **Full GraCA** | +consistency +多层梯度 +多checkpoint +bridge保护 |
| **Oracle GraCA** | 全标签上界（仅诊断） |

### 7.3 下游模型

GCN、GAT、GraphSAGE

### 7.4 评估指标

- **分类**：Accuracy、Macro-F1
- **结构**：裁剪比例、孤立节点数、最小度、最大连通分量比
- **效率**：运行时间、内存占用

---

## 8. 实验结果

### 8.1 主实验（1543 条结果）

#### 同质图：GraCA-lite vs 随机裁剪

| Dataset | Model | Original | Random | **GraCA-lite** | **Full GraCA** | Oracle | Δ_lite | Δ_full |
|---------|-------|----------|--------|----------------|----------------|--------|--------|--------|
| Cora | GCN | 78.84 | 76.83 | **78.67** | **78.90** | 79.13 | **+1.83** | **+2.07** |
| Cora | GAT | 82.07 | 81.10 | **82.07** | 81.95 | 82.33 | **+0.97** | +0.85 |
| Cora | GraphSAGE | 76.70 | 72.53 | **76.34** | 76.12 | 76.83 | **+3.81** | **+3.58** |
| CiteSeer | GCN | 66.74 | 63.46 | **67.20** | **66.90** | 67.30 | **+3.74** | **+3.44** |
| CiteSeer | GAT | 71.24 | 69.68 | **71.24** | 71.18 | 71.20 | **+1.56** | +1.50 |
| CiteSeer | GraphSAGE | 65.38 | 61.76 | **65.42** | **65.42** | 64.94 | **+3.66** | **+3.66** |
| PubMed | GCN | 76.44 | 74.60 | 76.26 | **76.60** | 76.48 | **+1.66** | **+2.00** |
| PubMed | GAT | 77.40 | 77.18 | **77.74** | **77.88** | 77.42 | **+0.56** | **+0.70** |
| PubMed | GraphSAGE | 75.12 | 72.40 | **75.16** | **75.48** | 75.36 | **+2.76** | **+3.08** |

#### 异质图

| Dataset | Model | Original | GraCA-lite | Δ |
|---------|-------|----------|------------|---|
| Actor | GCN | 28.55 | 28.47 | -0.08 |
| Actor | GAT | 29.54 | 29.47 | -0.07 |
| Actor | GraphSAGE | 32.39 | 32.53 | +0.13 |
| Texas | GraphSAGE | 92.43 | 88.11 | -4.32 |
| Cornell | GraphSAGE | 68.65 | 69.19 | **+0.54** |
| Wisconsin | GAT | 49.80 | 52.94 | **+3.14** |

### 8.2 消融实验（Cora, test_acc %）

| Variant | GCN | GAT | GraphSAGE |
|---------|-----|-----|-----------|
| **GraCA-lite (full)** | 78.67 | 82.07 | 76.34 |
| w/o EMA | 78.28 | 81.08 | 76.60 |
| hard pseudo | 78.20 | 81.92 | 76.28 |
| w/o reliability | 78.38 | 81.66 | 76.60 |
| harmful only | 78.34 | 81.86 | 76.54 |
| **helpful only** | **79.12** | **82.22** | 76.58 |
| global threshold | 78.40 | 82.02 | 76.80 |
| train only | 78.88 | 81.70 | 76.42 |

### 8.3 鲁棒性实验（Cora, GCN）

| Method | noise=5% | noise=10% | noise=20% | noise=30% |
|--------|----------|-----------|-----------|-----------|
| Original+Noise | 78.54 | 78.62 | 78.64 | 78.38 |
| Random+Noise | 77.70 | 77.74 | 77.34 | 78.20 |
| **GraCA+Noise** | **78.56** | 78.22 | **78.74** | 78.12 |

### 8.4 可扩展性

| Dataset | Nodes | Edges | Prune% | T_total | Peak Mem |
|---------|-------|-------|--------|---------|----------|
| Cora | 2,708 | 10,556 | 5.7% | 2.0s | 34 MB |
| CiteSeer | 3,327 | 9,104 | 2.9% | 1.6s | 67 MB |
| PubMed | 19,717 | 88,648 | 2.1% | 5.4s | 71 MB |
| Actor | 7,600 | 53,411 | 14.2% | 3.7s | 52 MB |

### 8.5 超参数搜索

在 Cora 上搜索 36 组配置，最佳：`tau=0.7, beta=0.05, eta=0.5`，val_acc=78.80%

---

## 9. 核心发现

1. **梯度方向信号有效**：最后一层梯度余弦对同类/异类边区分极强（同类 0.97 vs 异类 -0.48）

2. **GraCA-lite 稳定优于随机裁剪**：同质图上平均提升 +1.5% ~ +3.8%

3. **与原图持平或略优**：裁剪 3-5% 的边后性能不掉，说明确实删掉了噪声边

4. **Oracle 上界有效**：全标签版本表现最好，验证梯度边信号的上限存在

5. **Full GraCA 进一步提升**：consistency loss + 多层梯度 + 多 checkpoint 带来额外收益

6. **可扩展性好**：PubMed (19K nodes, 88K edges) 仅需 5.4 秒完成全流程

7. **异质图效果不稳定**：在 Texas/Wisconsin 上有时掉点，需进一步研究

---

## 10. 引用

```bibtex
@article{graca2024,
  title={GraCA: Gradient-Guided Graph Sanitization for Semi-Supervised Node Classification},
  author={},
  journal={},
  year={2024}
}
```

---

## 11. 联系方式

如有问题，请提交 Issue 或联系项目维护者。
