# ClaudeCode 执行 Prompt：完善 GraCA 代码与实验

下面内容可直接复制给 ClaudeCode 执行。

```text
你现在接手 GitHub 仓库 J-oMarch/GraCA。请先完整阅读 README、configs、scripts、src 目录和现有研究文档，然后按下面要求修复和完善项目。

目标：把当前 GraCA prototype 整理成一个可复现、无测试标签泄漏、baseline 公平、数据集更充分、实验结果可自动聚合的研究代码库。不要伪造任何实验结果；README 中所有结果必须来自真实 CSV。

重要说明：
GitHub 远端仓库中可能缺失 src/data/load_data.py、src/data/leakage_check.py、src/data/splits.py 等文件，但这些文件可能只是本地存在、上传时没有被 git 追踪。请先验证本地工作区是否存在这些文件：
1. 如果本地存在，请检查它们是否被 .gitignore 排除或未 git add，并将其纳入版本控制。
2. 如果本地不存在，再按要求补齐实现。
3. 不要在未验证本地文件前重复造轮子或覆盖已有实现。

阶段 1：修复仓库可运行性
1. 验证并补齐 src/data/ 模块：
   - src/data/__init__.py
   - src/data/load_data.py
   - src/data/splits.py
   - src/data/leakage_check.py
2. load_data.py 至少支持：
   - Cora
   - CiteSeer
   - PubMed
   - Actor
   - Texas
   - Cornell
   - Wisconsin
3. 数据加载统一返回：
   - data
   - num_features
   - num_classes
4. practical 模式严禁 test_mask 标签进入：
   - proxy training loss
   - pseudo-label construction
   - gradient scoring loss
   - pruning decision
   - baseline pruning decision
5. val_mask 只能用于 early stopping 和超参选择，不能用于构造 edge score，除非明确是 oracle/diagnostic 模式。
6. 新增 scripts/smoke_test.py，能在 Cora seed=0 上跑通：
   - Original
   - GraCA-lite
   - Random Pruning
   并输出 smoke_results.csv。
7. 修正 README 中不存在的目录和文件描述。README 不能声称存在 results/、sanitized_graphs/、checkpoints/ 中的实验产物，除非它们真实存在或由脚本生成。

阶段 2：修复 GraCA 核心算法实现
1. gradient collection 必须支持 deterministic scoring：
   - scoring.deterministic=true 时，评分阶段使用 model.eval()，关闭 dropout。
   - 不要让随机 dropout 影响同一 checkpoint 的 edge score。
2. collect_layer=last 的定义要统一：
   - Practical GraCA 和 Oracle GraCA 都使用输出层前的最后一个 hidden representation。
   - 不要 practical 用 hidden、oracle 用 logits。
3. Multi-checkpoint gradient averaging 要保证每个 checkpoint 对应自己的 forward、teacher_probs 或明确说明 teacher_probs 固定策略。
4. 对 undirected=True 的图，pruning 必须成对删除 (u,v) 和 (v,u)，不能只删单向边。
5. 修复 pruning.py 里的 degree 统计和 degree 更新：
   - min_degree、isolated_nodes、mean_degree、largest_connected_component_ratio 必须基于裁剪后的真实 edge_index 重新计算。
   - 不要用中间状态 degree 直接当最终统计。
6. self-loop 必须可配置保护。
7. bridge protection 如果开启，要在无向图上正确工作；如果成本较高，先保证默认关闭且不会影响主实验。
8. 添加最小单元测试：
   - 无向图裁剪后仍保持对称。
   - min_degree 生效。
   - self-loop 不被删除。
   - scoring.deterministic=true 时，同一模型同一输入重复评分结果一致。

阶段 3：修复 baseline 公平性
1. Random Pruning baseline 必须支持两种模式：
   - config beta 模式
   - match_graca_ratio 模式
2. 主表必须使用 match_graca_ratio：Random 删除比例必须等于 GraCA 的 actual_prune_ratio，而不是 pruning.beta。
3. 增加 degree-aware random pruning baseline，尽量保持每个节点的删除预算和 GraCA 接近。
4. 增加 feature-similarity pruning / Jaccard-GCN 风格 baseline：
   - 对 bag-of-words 特征可用 Jaccard。
   - 对连续特征可用 cosine similarity。
5. Homophily pruning 拆成两个版本：
   - legal_train_only：只使用 train_mask 两端均有标签的边。
   - oracle_label：使用全标签，只能放 oracle/diagnostic 表，不能进入 practical 主表。
6. 所有 baseline 输出字段统一：
   - run_id
   - seed
   - dataset
   - method
   - oracle_only
   - downstream_model
   - actual_prune_ratio
   - num_edges_before
   - num_edges_after
   - isolated_nodes
   - min_degree
   - mean_degree
   - val_acc
   - test_acc
   - test_f1
   - best_epoch
   - runtime
   - config_path

阶段 4：扩展更合适的数据集
请扩展数据集模块和 configs，新增以下数据集。能直接用 PyG/OGB 加载的就实现；不能稳定加载的先建立清晰接口和 TODO，不要伪造结果。

同质图：
1. Amazon Computers
2. Amazon Photo
3. Coauthor CS
4. Coauthor Physics
5. WikiCS

大图：
1. ogbn-arxiv
   - 使用官方 split
   - 使用 OGB evaluator
   - 如果全量开销太大，额外支持 ogbn-arxiv-subset 配置，但必须标注为 subset，不得冒充 full ogbn-arxiv。

异质图：
1. Actor 保留
2. Roman-empire
3. Amazon-ratings
4. Minesweeper
5. Tolokers
6. Questions

每个数据集加载后记录并输出：
- num_nodes
- num_edges
- num_features
- num_classes
- train/val/test size
- edge homophily
- split type
- 是否 public split / random split / official split

阶段 5：新增 noisy-edge robustness 实验
新增 scripts/run_noisy_edges.py 和相关模块，实现：
1. 对 clean graph 注入 task-harmful edges：
   - cross-class random edges
   - low feature similarity edges
   - optional hub noise edges
2. 噪声比例：
   - 5%
   - 10%
   - 20%
   - 30%
3. 注入时必须保存 bad_edge_mask 或 bad_edge_set，用于评估 GraCA 是否真的删掉了噪声边。
4. 指标包括：
   - downstream test_acc
   - bad-edge removal precision
   - bad-edge removal recall
   - bad-edge removal F1
   - clean-edge mistakenly removed ratio
5. 对比方法：
   - Original+Noise
   - Random Pruning matched ratio
   - Degree-aware Random
   - Jaccard/Cosine similarity pruning
   - Homophily legal_train_only
   - GraCA-lite
   - Oracle GraCA
6. 至少先在 Cora、CiteSeer、PubMed、Amazon Computers 上跑 smoke 版本；大规模数据集只创建配置，不强制立即跑完。

阶段 6：完善结果聚合和论文表格
1. 修改 scripts/aggregate_results.py，使其可以聚合：
   - main clean results
   - baseline results
   - noisy-edge robustness results
   - oracle results
   - ablation results
2. 所有正式实验至少支持 10 seeds。
3. 输出 mean ± std。
4. 对 GraCA vs Original、GraCA vs best practical baseline 做 paired t-test。
5. 自动生成：
   - paper_tables/main_homophily.csv
   - paper_tables/main_heterophily.csv
   - paper_tables/noisy_edge_robustness.csv
   - paper_tables/ablation.csv
   - paper_tables/oracle_gap.csv
6. README 中的实验结果表必须由真实 CSV 生成。请新增 scripts/update_readme_tables.py 或在 README 中只说明如何生成表格，不要手写无法追溯的结果。

阶段 7：重新设计消融实验
请补充或修正 ablation：
1. no_ema
2. hard_pseudo
3. no_reliability
4. harmful_only
5. helpful_only
6. no_relative_strength
7. no_uncertainty
8. global_threshold
9. local_top_budget
10. first_layer
11. last_layer
12. all_layers
13. deterministic_off

每个消融必须输出 actual_prune_ratio，否则不同裁剪比例下的 accuracy 不能直接比较。

阶段 8：最终验收
完成后请运行：
1. python scripts/smoke_test.py
2. 针对 pruning/scoring/data leakage 的单元测试
3. 至少一个 Cora seed=0 的 GraCA-lite + matched Random + Original 实验
4. 至少一个 Cora noise=10% 的 noisy-edge 实验

最后请给出：
1. 修改文件列表
2. 新增文件列表
3. 已运行命令
4. 测试结果
5. 生成的 CSV 路径
6. 当前还不能跑或需要更多算力的数据集
7. 下一步建议的完整实验命令

重要约束：
- 不要伪造实验结果。
- 不要把 oracle 结果混入 practical 主表。
- 不要使用 test labels 参与 practical 的任何训练、打分、裁剪。
- 不要大范围重构无关模块。
- 先保证可复现、合法、公平，再考虑 Full GraCA 的复杂增强。
```

## 预计结果

完成后，仓库应从 prototype 变成可复现实验框架：

- `scripts/run_graca.py`、`scripts/run_baselines.py`、`scripts/smoke_test.py` 可以正常运行。
- 本地存在但未追踪的关键文件会被识别并纳入 git；不存在的文件会被补齐。
- Practical GraCA 与 Oracle GraCA 被严格隔离。
- Random baseline 使用 matched prune ratio，不再因删边比例过大而被不公平压低。
- 无向图裁剪不会破坏边对称性。
- README 不再出现代码或结果不存在但文档声称存在的问题。
- 能输出 clean graph 与 noisy-edge 两类实验表。

比较现实的实验预期：

- 在 clean Cora/CiteSeer/PubMed 上，GraCA 相比 Original 可能只是持平或小幅提升，约 `0% ~ +1%`。
- 公平 matched Random 后，GraCA 相对 Random 的优势可能小于当前 README 表格。
- 在注入跨类噪声边后，如果 idea 成立，GraCA 应该明显优于 Random/Jaccard，尤其在 `10% ~ 30%` 噪声下更容易体现。
- 如果 bad-edge removal precision/recall 明显高于 baseline，这会比单纯 accuracy 更能支撑论文核心主张。

达到发文章门槛的关键标志：

1. 无测试标签泄漏。
2. 多数据集、多 seed 稳定。
3. matched baseline 公平。
4. noisy-edge 检测指标明显更好。
5. 消融能证明 gradient direction、uncertainty、local pruning 各自有贡献。
