# GraCA / EdgeInfluence 实验最终报告

## 1. 旧结果归档

所有旧实验结果已归档至：
- `archived_results/20260530_213741/`

## 2. 代码基础设施修复

### 已完成的修复

1. **度数计算 bug**: `pruning.py`, `graph_stats.py` 修复无向图度数计算
2. **无向图成对删除**: `random_pruning.py`, `homophily_pruning.py`, `similarity_pruning.py` 修复
3. **统一 result schema**: 38 字段，包含 experiment_type, actual_prune_ratio, edge_homophily 等
4. **确定性评分**: `run_graca.py` 显式读取 `scoring.deterministic`
5. **4 种噪声注入**: cross_class_train_safe, cross_class_oracle, low_feature_similarity, random_inter_community
6. **16/16 测试全部通过**

## 3. 可行性验证结果（核心发现）

### 3.1 验证方法

在 Cora 数据集上，seed=42，使用训练好的 teacher GCN，测试多种评分方法识别跨类边的能力：

| 方法 | AUC | 是否 ≥ 0.75 |
|------|-----|-------------|
| **Oracle LOO（全前向传播，真实标签）** | **0.786** | ✓ |
| 向量化 LOO（近似，真实标签） | 0.551 | ✗ |
| 向量化 LOO（近似，伪标签） | 0.431 | ✗ |
| 向量化 LOO（混合目标） | 0.561 | ✗ |
| KL 散度 | 0.688 | ✗ |
| 伪标签一致性 | 0.658 | ✗ |
| 隐藏层余弦相似度 | 0.623 | ✗ |
| 特征余弦相似度 | 0.597 | ✗ |
| rho 可靠性权重 | 0.512 | ✗ |
| P(true class) oracle | 0.548 | ✗ |
| P(u_class@v) 变化 | 0.527 | ✗ |
| 熵变化 | 0.576 | ✗ |
| KL + 隐藏 + 特征 组合 | 0.652 | ✗ |

### 3.2 Oracle 实验：问题本身是否可解？

```
Original accuracy:           79.20%
Oracle cross-class removal:  87.70% (+8.5%)
Inverse (remove same-class): 52.10% (-27.1%)
```

**结论：问题本身是可解的。移除跨类边可以显著提升准确率（+8.5%）。但当前方法无法可靠识别跨类边。**

### 3.3 根因分析

**为什么 Oracle LOO 有效（AUC=0.786）而向量化近似无效（AUC=0.55）？**

1. **近似误差**：向量化方法使用 `h_ablated = ReLU((d·h - h_u)/(d-1))` 近似 leave-one-out 隐藏表示。但 GCN 有 BatchNorm 层，BN 的 running mean/variance 是基于全图统计的。当移除一条边时，BN 的归一化效果不同，导致近似误差。

2. **信号微弱**：对于度数 ~10 的节点，移除一条边只改变隐藏表示约 10%。这个微小变化经过第二层传播后进一步衰减。

3. **伪标签噪声**：使用 teacher 伪标签作为 loss 目标引入额外噪声。teacher 在 Cora 上的准确率只有 ~78%，伪标签错误率 ~22%，这淹没了微弱的 leave-one-out 信号。

4. **特征空间重叠**：Cora 的跨类边和同类边在特征空间中高度重叠（特征余弦相似度 AUC 仅 0.60），使得基于特征/隐藏表示的方法难以区分。

### 3.4 各数据集特征相似度检测能力

| 数据集 | 边同质性 | 跨类边比例 | 特征余弦 AUC |
|--------|---------|-----------|-------------|
| Cora | 0.810 | 19.0% | 0.597 |
| CiteSeer | 0.736 | 26.4% | 0.595 |
| PubMed | 0.802 | 19.8% | 0.539 |
| AmazonComputers | 0.777 | 22.3% | 0.440 |

**所有数据集的特征相似度 AUC < 0.75。AmazonComputers 甚至低于 0.5（信号反转）。**

## 4. 诚实结论

### 4.1 不支持的 claim

基于当前实验结果：

1. ❌ **"EdgeInfluence 可以可靠识别跨类边"**
   - 最佳实用方法 AUC = 0.69（KL 散度），低于 0.75 阈值
   - Oracle LOO AUC = 0.786，但需要 O(E) 次全前向传播，不可扩展

2. ❌ **"EdgeInfluence 在 noisy graph 上提升下游 GNN"**
   - 无法可靠识别噪声边，因此无法有效裁剪

3. ❌ **"EdgeInfluence 比 baseline 更准确删除 harmful edges"**
   - 特征相似度（AUC=0.60）与 EdgeInfluence（AUC=0.55-0.69）效果相当
   - 两者都低于实用阈值

### 4.2 Oracle 实验支持的发现

1. ✅ **"跨类边确实对 GNN 有害"**
   - Oracle 移除跨类边：+8.5% 准确率提升
   - 反向移除同类边：-27.1% 准确率下降

2. ✅ **"全前向传播 LOO 可以检测跨类边"**
   - Oracle LOO AUC = 0.786（使用真实标签）
   - 但计算复杂度 O(E × forward_pass)，不可扩展

### 4.3 失败原因

1. **近似方法精度不足**：GCN 的 BatchNorm 和非线性使得线性 leave-one-out 近似误差过大
2. **伪标签质量不足**：teacher 准确率 ~78%，伪标签错误率淹没了 leave-one-out 信号
3. **信号噪声比太低**：移除一条边对节点预测的影响（~10%）小于模型和标签的噪声
4. **特征空间不分离**：跨类边和同类边在特征/隐藏空间中高度重叠

### 4.4 可能的改进方向

1. **更高精度的近似**：
   - 使用二阶泰勒展开（考虑 Hessian 项）
   - 使用 BN 的 per-node 统计而非 running statistics
   - 但会增加计算成本

2. **更好的 teacher**：
   - 使用更高准确率的 teacher（如 GAT、GraphSAGE ensemble）
   - 使用多 teacher 投票减少伪标签噪声

3. **不同的评分目标**：
   - 不使用 loss change，而是使用 prediction change 的方向
   - 使用 margin-based 目标而非 probability-based

4. **组合方法**：
   - 将 EdgeInfluence 与特征相似度、结构特征（度数、聚类系数）组合
   - 训练一个轻量分类器来组合多个信号

5. **缩小论文 scope**：
   - 将 EdgeInfluence 定位为 "edge influence estimation framework"
   - 不声称通用的有害边检测
   - 聚焦于 oracle 诊断分析场景

## 5. 文件清单

### 新增/修改的代码文件

| 文件 | 说明 |
|------|------|
| `src/graca/edge_influence.py` | EdgeInfluence 评分模块 |
| `src/eval/noise_injection.py` | 4 种噪声注入 |
| `src/eval/result_writer.py` | 统一 schema (38 字段) |
| `src/graca/pruning.py` | 修复度数计算 |
| `src/baselines/random_pruning.py` | 无向图成对删除 |
| `src/baselines/homophily_pruning.py` | 无向图成对删除 |
| `src/baselines/similarity_pruning.py` | 无向图成对删除 |
| `scripts/verify_idea.py` | 可行性验证 |
| `scripts/analyze_edge_scores.py` | 评分诊断 |
| `scripts/run_noisy_edge_experiment.py` | 统一噪声实验 |
| `scripts/run_ablation_noisy.py` | 噪声图消融 |
| `scripts/run_core_matrix.py` | 主实验矩阵运行器 |
| `scripts/validate_results.py` | 结果验证 |
| `scripts/build_final_tables.py` | 论文表格生成 |
| `tests/test_result_schema.py` | Schema + 功能测试 |

### 诊断数据

`results_clean/diagnostics/`:
- `verify_idea_Cora_42.json` - 可行性验证结果
- `edge_score_distribution_*.csv` - 评分分布
- `score_auc_summary_*.csv` - AUC 汇总

## 6. 验收检查清单

- [x] 旧结果已归档
- [x] 代码基础设施已修复
- [x] 16/16 测试通过
- [x] 可行性验证已完成
- [x] 诚实报告已撰写
- [x] Oracle 实验确认问题可解
- [x] 失败原因已分析
- [x] 改进方向已提出

---

**报告完成时间**: 2026-05-31
**结论**: EdgeInfluence 的向量化近似方法在 Cora 上无法达到 AUC ≥ 0.75 的阈值。Oracle LOO（全前向传播）达到 AUC=0.786，但计算成本过高。问题本身是可解的（oracle 移除跨类边 +8.5%），但当前近似方法的精度和伪标签质量不足以可靠检测跨类边。
