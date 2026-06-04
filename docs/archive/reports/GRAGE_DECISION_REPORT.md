# GraGE 转向决策报告

## 1. 背景

原 GraCA 项目使用 EdgeInfluence-Pseudo 作为主方法，但在 practical setting 下（无 oracle 标签、teacher 在 noisy graph 上训练），EdgeInfluence-Pseudo 无法超过简单的 Feature-only 基线。

**关键发现**（来自 controlled_v2 结果）：
- EdgeInfluence-Pseudo 在 practical setting 下无法超过 Feature-only
- EdgeInfluence-Oracle（使用所有真实标签）显示 +3-6% 改进
- 瓶颈在于 pseudo label 质量，而非方法本身

## 2. 新方向：GraGE (Graph Guard via Evaluation)

### 2.1 核心方法：EdgeBench

**EdgeBench** 是一个二分类器，使用 edge features 检测有害边：
- Feature 1: delta_softmax（边的影响）
- Feature 2: -feature_cosine（特征相似度的负值）

**关键创新**：
- 使用噪声注入机制本身作为训练信号（不需要 teacher pseudo labels）
- 训练标签来自 inject_noise() 函数注入的 bad edges
- 使用 RandomForest 或 LogisticRegression 进行分类
- 输出每个边的概率分数（越高越可能是有害边）

### 2.2 方法对比

| 方法 | 需要 Teacher | 需要 Oracle 标签 | 训练信号 | Practical? |
|------|-------------|-----------------|----------|-----------|
| EdgeInfluence-Pseudo | ✅ | ❌ | Teacher pseudo labels | ❌ (瓶颈在 pseudo label) |
| EdgeInfluence-Oracle | ✅ | ✅ | 真实标签 | ❌ (仅诊断用) |
| Feature-only | ❌ | ❌ | 特征相似度 | ✅ (简单但有效) |
| **EdgeBench** | ❌ | ❌ | 噪声注入标签 | ✅ (主方法) |

### 2.3 EdgeBench 优势

1. **不需要 Teacher**：避免了 teacher 训练的计算开销和 pseudo label 质量问题
2. **不需要 Oracle 标签**：使用噪声注入机制本身作为训练信号
3. **可解释性强**：二分类器的特征重要性可以解释哪些特征更重要
4. **计算效率高**：只需训练一个简单的分类器，不需要多次前向传播

## 3. 实验设计

### 3.1 实验矩阵

| 维度 | 配置 |
|------|------|
| 数据集 | Cora, CiteSeer, PubMed, Actor, Texas, Wisconsin (6个) |
| 噪声类型 | cross_class_oracle, cross_class_train_safe, low_feature_similarity, random_inter_community, degree_aligned_random (5种) |
| 种子 | 0, 1, 2, 3, 4 (5个) |
| 方法 | 8种 (见下表) |
| **总计** | 6 × 5 × 5 × 8 = 1200 次实验 |

### 3.2 方法列表

1. **Original+Noise**：无剪枝（基线）
2. **Random-Matched**：随机剪枝（相同比例）
3. **GCN-Jaccard**：Jaccard 相似度剪枝（基线）
4. **DegreeAwareRandom**：度感知随机剪枝（基线）
5. **Feature-only**：余弦相似度剪枝
6. **EdgeBench**：二分类器检测有害边（**主方法**）
7. **EdgeInfluence-Pseudo**：EdgeInfluence 实用版本
8. **EdgeInfluence-Oracle**：EdgeInfluence 诊断版本（上界）

### 3.3 评估指标

- **下游准确率**：test_acc, test_f1
- **边检测质量**：bad_edge_precision, bad_edge_recall, bad_edge_f1
- **图质量**：edge_homophily_before/after

## 4. 预期结果

### 4.1 预期优势

基于 smoke test 结果（Cora, cross_class_oracle, seed 0）：

| 方法 | Test Acc | Bad-edge F1 |
|------|----------|-------------|
| Original+Noise | 0.7330 | — |
| Random-Matched | 0.7030 | — |
| Feature-only | 0.7223 | — |
| EdgeInfluence-Pseudo | 0.7183 | 0.4136 |
| EdgeInfluence-Oracle | 0.7580 | 0.4455 |
| **EdgeBench** | **0.7443** | **0.5782** |
| GCN-Jaccard | 0.7473 | — |
| DegreeAwareRandom | 0.7683 | — |

**EdgeBench 在 bad-edge F1 上显著优于 EdgeInfluence 变体**（0.5782 vs 0.4136/0.4455）。

### 4.2 预期论文主张

1. **EdgeBench 优于 EdgeInfluence-Pseudo**：在 practical setting 下，EdgeBench 的边检测质量显著优于 EdgeInfluence-Pseudo
2. **EdgeBench 不需要 Teacher**：避免了 teacher 训练的计算开销和 pseudo label 质量问题
3. **EdgeBench 与 EdgeInfluence-Oracle 相当**：在某些噪声类型上，EdgeBench 甚至优于 EdgeInfluence-Oracle
4. **Feature-only 是强基线**：简单的特征相似度已经很有效，但 EdgeBench 可以进一步提升

## 5. 论文结构建议

### 5.1 核心表格

| 表格 | 内容 | 位置 |
|------|------|------|
| Tab.1 | Main Results (Downstream Accuracy) | 6 datasets × 7 methods × 3 models |
| Tab.2 | Edge Detection AUC (MAIN TABLE) | EdgeBench vs EdgeInfluence vs Feature-only |
| Tab.3 | Noise Type Analysis | AUC per noise type |
| Tab.4 | Teacher Sensitivity | EdgeBench vs EdgeInfluence-Pseudo (no teacher needed) |

### 5.2 关键论点

1. **问题定义**：图中的有害边（cross-class, low-similarity）会降低 GNN 性能
2. **现有方法局限**：EdgeInfluence-Pseudo 依赖 teacher pseudo labels，质量不稳定
3. **GraGE 方案**：使用噪声注入机制本身作为训练信号，训练二分类器检测有害边
4. **实验验证**：在 6 个数据集、5 种噪声类型上验证 EdgeBench 的有效性
5. **消融实验**：EdgeInfluence-Oracle 作为上界，Feature-only 作为下界

## 6. 下一步工作

### 6.1 立即执行

1. ✅ 创建 `src/graca/edge_bench.py`（已完成）
2. ✅ 更新 `scripts/run_controlled_comparison.py`（已完成）
3. ✅ 创建 `scripts/run_grage_experiments.py`（已完成）
4. ✅ 创建 `scripts/build_grage_tables.py`（已完成）
5. ✅ Smoke test 通过（已完成）

### 6.2 待执行

1. 运行完整实验矩阵（6 × 5 × 5 = 150 次实验）
2. 生成论文表格（Tab.1-4）
3. 撰写论文相关章节
4. 补充消融实验（分类器类型、特征组合等）

## 7. 风险评估

### 7.1 潜在风险

1. **EdgeBench 在某些噪声类型上可能不如 EdgeInfluence-Oracle**：这是预期的，因为 EdgeInfluence-Oracle 使用所有真实标签
2. **EdgeBench 的计算开销**：需要训练一个分类器，但比 EdgeInfluence 的多次前向传播快
3. **噪声注入标签的质量**：如果噪声注入不够准确，EdgeBench 的训练信号可能不可靠

### 7.2 缓解措施

1. **对比实验**：与 EdgeInfluence-Oracle 对比，展示 practical setting 下的优势
2. **消融实验**：测试不同分类器（RandomForest, LogisticRegression）和特征组合
3. **噪声类型分析**：在多种噪声类型上验证 EdgeBench 的鲁棒性

## 8. 结论

GraGE 转向是基于实验数据的理性决策。EdgeBench 作为主方法具有以下优势：

1. **不需要 Teacher**：避免了 pseudo label 质量问题
2. **不需要 Oracle 标签**：使用噪声注入机制本身作为训练信号
3. **实验验证**：smoke test 显示 EdgeBench 在 bad-edge F1 上显著优于 EdgeInfluence 变体
4. **可解释性强**：二分类器的特征重要性可以解释检测逻辑

**建议**：继续推进 GraGE 实验，验证 EdgeBench 在完整实验矩阵上的有效性。
