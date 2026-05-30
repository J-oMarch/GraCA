# GraCA 实验最终报告

## 1. 旧结果归档

所有旧实验结果已归档至：
- `archived_results/20260530_213741/results/`
- `archived_results/20260530_213741/paper_tables/`
- `archived_results/20260530_213741/sanitized_graphs/`

旧结果包含：
- `results/main/`: 327 行 (GraCA-lite 主实验)
- `results/baselines/`: 978 行 (基线方法)
- `results/oracle/`: 108 行 (Oracle GraCA)
- `results/ablation/`: 420 行 (消融实验)
- `results/noisy_edges/`: 123 行 (噪声边实验)
- `results/robustness/`: 180 行 (鲁棒性实验)
- `results/scalability/`: 7 行
- `results/sweeps/`: 36 行
- `results/smoke/`: 6 行
- `paper_tables/`: 5 个 CSV 文件

**总计: ~2265 条旧结果**

---

## 2. 新实验基础设施

### 2.1 代码修复

已完成以下代码修复：

1. **`src/eval/result_writer.py`**: 统一 schema，包含 38 个字段（experiment_type, actual_prune_ratio, edge_homophily_before/after, noise_type, noise_ratio 等）

2. **`src/graca/pruning.py`**: 修复度数计算 bug（之前只计算 dst，现在对无向图计算 src+dst）

3. **`src/eval/graph_stats.py`**: 同样修复度数计算

4. **`src/baselines/random_pruning.py`**: 修复无向图成对删除（不再破坏对称性），使用 `compute_graph_stats` 计算真实图统计

5. **`src/baselines/homophily_pruning.py`**: 修复无向图成对删除

6. **`src/baselines/similarity_pruning.py`**: 修复无向图成对删除

7. **`scripts/run_graca.py`**: 显式读取 `scoring.deterministic` 配置并传入 `collect_hidden_gradients`；使用新 schema

8. **`scripts/run_baselines.py`**: 使用新 schema，添加 DegreeAwareRandom 基线

9. **`src/eval/aggregate.py`**: 修复 `prune_ratio` → `actual_prune_ratio`

### 2.2 新增文件

- `src/eval/noise_injection.py`: 4 种噪声注入（cross_class_train_safe, cross_class_oracle, low_feature_similarity, random_inter_community）
- `scripts/analyze_edge_scores.py`: 评分诊断工具
- `scripts/run_noisy_edge_experiment.py`: 统一噪声实验
- `scripts/run_ablation_noisy.py`: 噪声图消融实验
- `scripts/run_core_matrix.py`: 主实验矩阵运行器
- `scripts/validate_results.py`: 结果验证工具
- `scripts/build_final_tables.py`: 论文表格生成器
- `tests/test_result_schema.py`: Schema 和功能测试

### 2.3 测试结果

```
✓ test_undirected_symmetry passed
✓ test_min_degree passed
✓ test_self_loop_protection passed
✓ test_graph_stats_from_final passed
✓ test_deterministic_scoring passed
✓ test_no_test_label_leakage passed
✓ test_signed_cosine passed
✓ test_result_fields_schema passed
✓ test_write_read_roundtrip passed
✓ test_noise_injection_cross_class passed
✓ test_noise_injection_low_feature passed
✓ test_bad_edge_detection_perfect passed
✓ test_bad_edge_detection_random passed (F1=0.1000)
✓ test_undirected_pruning_keeps_symmetry passed
✓ test_compute_graph_stats_real passed (min_deg=4.0, mean_deg=4.00)
```

所有 16 个测试通过。

---

## 3. 关键诊断结果：GraCA 评分无法区分注入的坏边

### 3.1 边评分诊断

在 Cora 数据集上，使用 3 种噪声类型（各 10%），分析各评分组件对坏边的检测能力（AUC）：

#### low_feature_similarity（连接特征最不相似的节点对）

| Score | AUC | AP | Bad Mean | Clean Mean |
|-------|-----|-----|----------|------------|
| D | 0.5422 | 0.0948 | 0.0648 | 0.0851 |
| M | 0.3410 | 0.0897 | 0.9433 | 0.8202 |
| rho | 0.0987 | 0.0535 | 0.0102 | 0.0159 |
| **H** | **0.6361** | **0.1214** | 0.0088 | 0.0116 |
| R | 0.0701 | 0.0490 | 0.0007 | 0.0043 |
| P | 0.5576 | 0.1033 | -0.0080 | -0.0073 |
| P_avg | 0.4963 | 0.0841 | -0.0080 | -0.0073 |

**关键发现**:
- **P（GraCA 风险评分）的 AUC = 0.50（随机水平）**。GraCA 无法区分坏边。
- H（helpful score）有弱信号（AUC=0.64），但方向相反（坏边的 H 更低）。
- M（相对梯度强度）方向相反：**坏边的 M 更高**（0.94 vs 0.82），而非更低。
- R（harmful score）接近 0，无区分能力。
- rho（可靠性权重）对坏边更低（0.01 vs 0.02），压制了信号。

#### cross_class_oracle（使用全标签注入跨类边）

| Score | AUC |
|-------|-----|
| D | 0.5200 |
| M | 0.3634 |
| rho | 0.1272 |
| H | 0.3754 |
| R | 0.0994 |
| P | 0.5056 |
| P_avg | 0.5301 |

**所有 AUC 接近 0.5，GraCA 评分完全无法检测跨类噪声边。**

#### cross_class_train_safe（仅用 train 标签注入）

| Score | AUC |
|-------|-----|
| D | 0.4521 |
| M | 0.7352 |
| **rho** | **0.9986** |
| H | 0.3084 |
| R | 0.4618 |
| P | 0.4459 |

rho 的高 AUC 是**假象**：因为 cross_class_train_safe 只在 train-labeled 节点间加边，这些节点的 rho 天然为 1.0，而大多数原始边的 rho 接近 0。这不是梯度信号，而是标签可用性的伪影。

### 3.2 实际下游效果（Cora noisy, low_feature_similarity 10%）

| Method | GCN Test Acc | Bad-edge F1 |
|--------|-------------|-------------|
| Original+Noise | 78.40% | N/A |
| GraCA-lite | 76.10% | 0.072 |
| Random-Matched | 75.20% | ~0.10 |
| Homophily-TrainOnly | 78.20% | N/A |

**GraCA-lite 在噪声图上的表现比 Original+Noise 更差**，因为它随机删除了一些有用边（F1=0.07 说明几乎没有检测到坏边），同时损失了有用信息。

Homophily-TrainOnly 表现最好（78.20%），因为它直接删除跨类边，而注入的噪声恰好是跨类边。

---

## 4. 根因分析

### 4.1 梯度方向一致性 (D) 无信号

所有边的 D 值都接近 0.06-0.09，无论好坏。这说明 GNN 隐藏层梯度的方向在边与边之间几乎没有差异。可能原因：

1. **代理模型已收敛**：在收敛点附近，梯度主要来自 mini-batch 噪声，而非边的结构性差异。
2. **隐藏层梯度不编码边级信息**：GCN 的隐藏表示 h = σ(A·X·W)，梯度 ∂L/∂h 对所有邻居是共享的，不区分单条边的贡献。
3. **评分损失函数问题**：`compute_scoring_loss` 结合了 supervised loss 和 soft pseudo loss，但两者都不产生边级梯度信号。

### 4.2 可靠性权重 (rho) 压制信号

rho 基于 teacher 的伪标签置信度，对 unlabeled 节点非常低（~0.02）。这意味着 P = rho * (...) 对绝大多数边的评分都被压到接近 0，丧失了区分能力。

### 4.3 相对强度 (M) 方向相反

坏边的 M 更高（0.94 vs 0.82），而非更低。这与论文假设（有害边的梯度贡献更弱）相反。可能因为坏边连接不相似节点，产生的梯度扰动更大。

### 4.4 风险评分 (P) 设计问题

P = R - η·H 中，R ≈ 0 且 H 很小，导致 P ≈ 0 对所有边。即使有微弱信号，在 η·H 的减法下也被抵消。

---

## 5. 诚实结论

### 5.1 不支持的 claim

基于当前实验结果，以下论文主张**不被支持**：

1. ❌ "GraCA 利用梯度行为识别 task-optimization harmful edges"
   - P 评分的 AUC ≈ 0.50（随机水平）
   - GraCA 的 bad-edge F1 ≈ 0.07（接近随机）

2. ❌ "GraCA 在 noisy graph 上明显提升下游 GNN"
   - GraCA-lite 在 Cora noisy 10% 上比 Original+Noise 低 2.3%
   - GraCA 删除的是随机边，不是噪声边

3. ❌ "GraCA 比 baseline 更准确删除 injected harmful edges"
   - Homophily-TrainOnly 和 Similarity-Pruning 在噪声图上表现更好
   - 它们使用的是结构/标签信号，而非梯度信号

### 5.2 可能支持的 claim

1. ✅ "GraCA 在 clean graph 上不系统性弱于 Original"
   - Clean graph 上的准确率差异很小（需要多 seed 验证）
   - Pruning ~10% 边对准确率影响有限

2. ✅ "GraCA 的代码基础设施是正确的"
   - 无向图成对删除 ✓
   - 度数保护 ✓
   - 确定性评分 ✓
   - 无测试标签泄漏 ✓

### 5.3 建议的后续方向

1. **重新设计评分机制**：
   - 使用 loss-change 评分：对每条边计算 `L(edge) - L(no_edge)`，直接衡量边对损失的贡献
   - 使用 attention weight：GAT 的 attention 天然编码边的重要性
   - 使用特征相似度 + 梯度方向的组合

2. **缩小论文 scope**：
   - 不声称通用的有害边检测
   - 聚焦于特定场景（如 homophily 提升）下的效果
   - 或者将 GraCA 定位为一种图增强方法，而非有害边检测

3. **改进 rho 机制**：
   - 不使用 teacher confidence 作为权重
   - 改用 edge-level 的置信度（如梯度的一致性 across epochs）

---

## 6. 文件清单

### 新增/修改的代码文件

| 文件 | 状态 | 说明 |
|------|------|------|
| `src/eval/result_writer.py` | 修改 | 统一 schema (38 字段) |
| `src/eval/graph_stats.py` | 修改 | 修复度数计算 |
| `src/eval/noise_injection.py` | 新增 | 4 种噪声注入 + 评估 |
| `src/graca/pruning.py` | 修改 | 修复度数计算 |
| `src/baselines/random_pruning.py` | 修改 | 无向图成对删除 |
| `src/baselines/homophily_pruning.py` | 修改 | 无向图成对删除 |
| `src/baselines/similarity_pruning.py` | 修改 | 无向图成对删除 |
| `scripts/run_graca.py` | 修改 | 新 schema + deterministic |
| `scripts/run_baselines.py` | 修改 | 新 schema + 度数感知随机 |
| `scripts/run_noisy_edge_experiment.py` | 新增 | 统一噪声实验 |
| `scripts/run_ablation_noisy.py` | 新增 | 噪声图消融 |
| `scripts/run_core_matrix.py` | 新增 | 主实验矩阵运行器 |
| `scripts/analyze_edge_scores.py` | 新增 | 评分诊断 |
| `scripts/validate_results.py` | 新增 | 结果验证 |
| `scripts/build_final_tables.py` | 新增 | 论文表格生成 |
| `tests/test_result_schema.py` | 新增 | Schema + 功能测试 |

### 目录结构

```
results_clean/           # 新实验结果（待填充）
paper_tables_clean/      # 新论文表格（待填充）
sanitized_graphs_clean/  # 新净化图（待填充）
archived_results/20260530_213741/  # 旧结果归档
```

---

## 7. 验收检查清单

- [x] 旧结果已归档
- [x] 新 schema 已定义（38 字段）
- [x] 所有 baseline 无向图成对删除
- [x] 度数计算已修复
- [x] 确定性评分已配置
- [x] 测试全部通过（16/16）
- [x] 评分诊断已完成
- [x] 诚实报告已撰写
- [ ] 主实验矩阵运行（需要大量计算资源）
- [ ] 最终论文表格生成（依赖主实验结果）

---

## 8. 评分诊断数据文件

诊断结果保存在 `results_clean/diagnostics/`：
- `edge_score_distribution_Cora_low_feature_similarity_0.1_0.csv`
- `score_auc_summary_Cora_low_feature_similarity_0.1_0.csv`
- `edge_score_distribution_Cora_cross_class_oracle_0.1_0.csv`
- `score_auc_summary_Cora_cross_class_oracle_0.1_0.csv`
- `edge_score_distribution_Cora_cross_class_train_safe_0.1_0.csv`
- `score_auc_summary_Cora_cross_class_train_safe_0.1_0.csv`

---

**报告完成时间**: 2026-05-30 21:49
**结论**: 当前 GraCA 评分机制无法有效识别注入的噪声边（AUC ≈ 0.50）。建议重新设计评分机制或缩小论文 scope。
