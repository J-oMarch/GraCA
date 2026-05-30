# GraCA_模型介绍与论文idea

## 1. Executive Summary

图神经网络（Graph Neural Networks, GNNs）通过图结构进行消息传递，因此模型性能不仅取决于节点特征质量，还高度依赖图结构本身的质量。然而在真实场景中，大量边虽然客观存在，却未必对当前任务有益。例如，在引文网络中，跨领域引用可能是真实存在的，但对于论文主题分类任务而言可能引入噪声；在社交网络中，弱关系连接也可能降低节点表示的判别能力。因此，图结构中真正需要解决的问题并非“哪些边是错误的”，而是“哪些边对当前任务优化持续产生负贡献”。

Practical GraCA（Practical Gradient-guided Graph Connection Assessment）是一种面向半监督节点分类任务的图结构净化框架。与传统图结构学习方法主要依赖特征相似性、结构统计量或图重构目标不同，GraCA 从模型自身学习行为出发，通过分析 GNN 训练过程中隐藏表示的梯度方向、梯度相对强度以及预测不确定性，评估每条边在当前任务中的优化贡献。

GraCA 的核心思想是：如果邻居节点的信息长期推动目标节点沿着与任务优化一致的方向更新，则该边应被保留；反之，如果邻居节点长期向目标节点传播与任务优化相冲突的信息，则该边应被视为 task-optimization harmful edge，并在图结构净化过程中被优先裁剪。

最初的 GraCA 版本依赖全图标签计算监督梯度，可以验证梯度边信号的有效性，但存在测试标签泄漏问题，因此只能作为 Oracle 上界分析。Practical GraCA 在此基础上进行了系统修订，采用 train-supervised learning、EMA teacher、soft pseudo label、uncertainty-aware gradient scoring 等机制，在不使用测试标签的前提下构建合法半监督框架。

最终输出并非新的 GNN 模型，而是一张经过净化的图结构：

$$
\mathcal{G} \rightarrow \mathcal{G}'.
$$

该净化图可用于重新训练 GCN、GAT、GraphSAGE 等不同下游模型，因此 GraCA 属于 data-centric graph sanitization 方法，而非传统意义上的模型结构创新。

## 2. 研究背景与动机

### 2.1 GNN 为什么依赖图结构

现代 GNN 的核心机制是 message passing。对于节点 \(v\)，第 \(l+1\) 层表示通常表示为：

$$
h_v^{(l+1)}
=
\phi
\left(
h_v^{(l)},
\operatorname{AGG}
\left(
\{h_u^{(l)}:u\in\mathcal{N}(v)\}
\right)
\right),
$$

其中 \(\mathcal{N}(v)\) 表示节点 \(v\) 的邻域。该过程意味着节点表示本质上由自身特征和邻居信息共同决定。因此，好邻居会提供有益信息，坏邻居会传播噪声信息，无关邻居会稀释有效信号。GNN 的性能上限很大程度上取决于图结构质量。

### 2.2 边可能真实存在，但对任务有害

传统观点常隐含“边存在即有价值”的假设。实际上，在节点分类任务中，一个真实存在的邻居可能属于不同语义群体、位于类别边界、属于异常节点，或者与当前分类目标无关。即使边 \((u,v)\in\mathcal{E}\) 是真实存在的，也不意味着它一定有助于降低分类损失。

GraCA 不尝试判断 \((u,v)\) 是否真实，而尝试判断 \((u,v)\) 是否有利于当前任务优化。

### 2.3 为什么 Data-Centric Graph Learning 有价值

Data-centric AI 的核心思想是改善数据质量，而不是只设计更复杂模型。在图学习场景下，对应的问题是 graph sanitization：给定原图 \(\mathcal{G}\)，得到净化图 \(\mathcal{G}'\)，使其能服务多个下游模型。

GraCA 的最终产物不是新的 GNN，而是新的图。这一点使它更接近数据清洗、图结构净化和鲁棒图学习，而不是普通的模型结构改造。

### 2.4 为什么按同质性删边不够

很多图净化方法隐含同质性假设：若 \(y_u=y_v\)，边应保留；若 \(y_u\neq y_v\)，边应删除。这种思路存在三个问题：

- 异类邻居并不一定无用；
- 边界节点天然需要跨类连接；
- 仅按标签同质性删除边容易破坏连通性和结构角色信息。

GraCA 不关注 \(y_u=y_v\) 是否成立，而关注邻居 \(u\) 是否帮助目标节点 \(v\) 优化任务损失。

## 3. 核心问题定义

给定图：

$$
\mathcal{G}=(\mathcal{V},\mathcal{E},X),
$$

其中 \(\mathcal{V}\) 为节点集合，\(\mathcal{E}\) 为边集合，\(X\in\mathbb{R}^{N\times d}\) 为节点特征矩阵。训练节点、验证节点、测试节点分别为：

$$
\mathcal{V}_{train},\quad \mathcal{V}_{val},\quad \mathcal{V}_{test}.
$$

在标准半监督节点分类设定下，主方法只能使用训练标签，不能使用测试标签。记有标签训练节点集合为：

$$
\mathcal{V}_{L}=\mathcal{V}_{train},
$$

未标注或不可直接监督节点集合为：

$$
\mathcal{V}_{U}=\mathcal{V}\setminus\mathcal{V}_{L}.
$$

GraCA 的目标不是恢复真实图 \(\mathcal{E}^{*}\)，因为真实结构本身可能并不利于当前任务。GraCA 的目标是学习：

$$
\mathcal{E}\rightarrow\mathcal{E}',
$$

使 \(\mathcal{E}'\) 更适合当前任务优化。

若边 \((v,u)\) 长期传播与任务优化相冲突的信息，则称其为 task-optimization harmful edge。GraCA 只试图识别这种任务优化意义上的有害边，不声称恢复真实图结构或检测真实错误边。

## 4. 原始 Idea 与问题

### 4.1 Full-label Oracle GraCA

最初版本构建 ProxyGNN \(f_\theta\)，使用全图标签计算：

$$
\mathcal{L}_{full}
=
\sum_{v\in\mathcal{V}}
CE(y_v,\hat{y}_v).
$$

然后计算隐藏表示梯度：

$$
g_v
=
\frac{\partial \mathcal{L}_{full}}{\partial h_v}.
$$

对于边 \((v,u)\)，分析 \(g_v\) 与 \(g_u\) 的方向关系：

$$
\cos(g_v,g_u).
$$

若二者长期反向，则认为该边在当前任务优化中可能有害。

### 4.2 为什么 Oracle 有价值

Oracle GraCA 证明了一件重要事情：梯度行为确实包含边贡献信息。已有实验观察表明，即使只删除部分边，下游 GNN 也可能显著提升。这说明“梯度方向和强度可作为边级任务信号”这一假设值得继续推敲。

### 4.3 为什么 Oracle 不能作为主方法

Oracle 使用 \(y_v,\forall v\in\mathcal{V}\)，即包含测试标签。这违反标准半监督设定。更严重的是，图结构已经通过测试标签被改写，即使后续下游模型只用训练标签，也已经间接使用了测试信息。因此 Oracle 只能作为 upper-bound / diagnostic study，不能作为正式方法或 SOTA 对比结果。

### 4.4 为什么硬伪标签替代不稳

直接使用：

$$
\hat{y}_v=\arg\max p_v
$$

构造硬伪标签交叉熵，会把错误预测当作真实标签，放大 teacher bias，并将错误梯度写入边评分。这是伪标签版 GraCA 的核心瓶颈。

### 4.5 为什么固定阈值不合理

统一阈值 \(s_{vu}<\gamma\) 忽略度数差异、图密度差异和邻域结构差异，可能导致稀疏节点被过度裁剪、稠密节点裁剪不足。因此 Practical GraCA 采用 per-node local adaptive pruning。

### 4.6 为什么绝对梯度范数不合理

随着训练收敛，绝对梯度范数通常会下降：

$$
\|g_v\|_2 \rightarrow 0.
$$

因此，绝对梯度不可跨 epoch、跨节点直接比较。Practical GraCA 使用邻域内相对梯度强度，而不是绝对范数。

## 5. Practical GraCA 总体方案

Practical GraCA 包含七个阶段：

1. 训练 ProxyGNN \(f_\theta\)，得到基础表示与分类预测。
2. 使用 EMA teacher \(f_{\bar{\theta}}\) 生成更稳定的 soft pseudo label。
3. 根据 teacher confidence 和 entropy 计算节点可靠性。
4. 在 deterministic graph view 上计算 scoring loss，避免随机增强干扰边评分。
5. 采集隐藏表示梯度 \(g_v^{(l,t)}\)。
6. 计算 helpful score、harmful score 和 risk score。
7. 对每个节点执行局部自适应裁剪，得到净化图 \(\mathcal{G}'\)，再让下游 GNN 从零训练。

## 6. 完整数学公式

### 6.1 图、模型与 Teacher

给定图：

$$
\mathcal{G}=(\mathcal{V},\mathcal{E},X),
$$

标签空间：

$$
\mathcal{Y}=\{1,\dots,C\}.
$$

ProxyGNN 为 \(f_\theta\)，输出类别概率：

$$
p_\theta(v)=\operatorname{softmax}(z_v).
$$

EMA teacher 为 \(f_{\bar{\theta}}\)，参数更新为：

$$
\bar{\theta}_t
=
\mu \bar{\theta}_{t-1}
+
(1-\mu)\theta_t,
$$

其中 \(\mu\in(0,1)\)，常用范围为 \(0.99\sim0.999\)。Teacher soft pseudo label 为：

$$
q_v=p_{\bar{\theta}}(v).
$$

### 6.2 Confidence 与 Entropy

定义 teacher confidence：

$$
c_v=\max_k q_{v,k}.
$$

定义预测熵：

$$
H(q_v)
=
-
\sum_{k=1}^{C}
q_{v,k}\log q_{v,k}.
$$

置信度越高、熵越低，节点预测越可靠。

### 6.3 节点可靠性

训练阶段可靠性为：

$$
\rho_v^{train}
=
\begin{cases}
1, & v\in\mathcal{V}_{L},\\
c_v^\alpha\left(1-\frac{H(q_v)}{\log C}\right), & v\in\mathcal{V}_{U},\ c_v\ge\tau,\\
0, & v\in\mathcal{V}_{U},\ c_v<\tau.
\end{cases}
$$

低置信节点不参与伪标签监督，以避免 teacher bias 直接污染分类边界。

边评分阶段可靠性为：

$$
\rho_v^{score}
=
\begin{cases}
1, & v\in\mathcal{V}_{L},\\
c_v^\alpha\left(1-\frac{H(q_v)}{\log C}\right), & v\in\mathcal{V}_{U},\ c_v\ge\tau,\\
\epsilon_\rho, & v\in\mathcal{V}_{U},\ c_v<\tau.
\end{cases}
$$

其中 \(\epsilon_\rho\ll 1\)，如 \(0.05\)。低置信节点不应主导评分，但仍保留少量结构上下文影响。

### 6.4 边可靠性

GraCA 的评分对象是“邻居 \(u\) 对目标节点 \(v\) 的贡献”，因此采用 target-centered 设计：

$$
\rho_{vu}
=
\rho_v^{score}
\cdot
\operatorname{clip}
\left(
\rho_u^{score},
\epsilon_\rho,
1
\right).
$$

目标节点可靠性占主导，邻居可靠性作为温和修正。

### 6.5 Proxy Training Loss

监督损失为：

$$
\mathcal{L}_{sup}
=
\sum_{v\in\mathcal{V}_{L}}
CE
\left(
y_v,
p_\theta(v)
\right).
$$

Soft pseudo loss 为：

$$
\mathcal{L}_{soft}
=
\sum_{v\in\mathcal{V}_{U}}
\rho_v^{train}
KL
\left(
q_v
\Vert
p_\theta(v)
\right).
$$

Consistency loss 为可选增强：

$$
\mathcal{L}_{cons}
=
\sum_{v\in\mathcal{V}}
\rho_v^{train}
KL
\left(
p_{\bar{\theta}}(v|\mathcal{G}^{weak})
\Vert
p_\theta(v|\mathcal{G}^{strong})
\right).
$$

训练阶段总损失为：

$$
\mathcal{L}_{proxy}
=
\mathcal{L}_{sup}
+
\lambda_p\mathcal{L}_{soft}
+
\lambda_c\mathcal{L}_{cons}.
$$

第一阶段 GraCA-lite 可暂不实现 \(\mathcal{L}_{cons}\)，即令 \(\lambda_c=0\)。

### 6.6 Scoring Loss

边评分阶段关闭随机增强，使用 deterministic graph：

$$
\mathcal{L}_{score}
=
\mathcal{L}_{sup}^{det}
+
\lambda_s\mathcal{L}_{soft}^{det}.
$$

\(\mathcal{L}_{score}\) 不包含 consistency loss，因为 consistency gradient 反映的是扰动不变性，不完全等价于分类任务贡献。

### 6.7 隐藏表示梯度

对于第 \(l\) 层、checkpoint \(t\) 的节点表示 \(h_v^{(l,t)}\)，定义：

$$
g_v^{(l,t)}
=
\frac
{\partial\mathcal{L}_{score}}
{\partial h_v^{(l,t)}}.
$$

未标注节点即使没有真实标签，也可能通过 soft pseudo loss 获得直接梯度，或通过 message passing 对有监督节点的损失产生间接梯度。

### 6.8 梯度方向一致性

$$
D_{vu}^{(l,t)}
=
\cos
\left(
g_v^{(l,t)},
g_u^{(l,t)}
\right).
$$

若 \(D_{vu}^{(l,t)}>0\)，说明方向一致；若 \(D_{vu}^{(l,t)}<0\)，说明方向冲突。

### 6.9 相对梯度强度

$$
M_{vu}^{(l,t)}
=
\frac
{
\left\|
g_u^{(l,t)}
\right\|_2
}
{
\operatorname{mean}_{j\in\mathcal{N}(v)}
\left\|
g_j^{(l,t)}
\right\|_2
+
\epsilon
}.
$$

该项表示邻居 \(u\) 相对于目标节点 \(v\) 局部邻域的影响强度。

### 6.10 Helpful、Harmful 与 Risk Score

Helpful score：

$$
H_{vu}
=
\rho_{vu}
\cdot
\frac{1}{|\mathcal{T}||\mathcal{L}|}
\sum_{t,l}
\max
\left(
D_{vu}^{(l,t)},
0
\right)
M_{vu}^{(l,t)}.
$$

Harmful score：

$$
R_{vu}
=
\rho_{vu}
\cdot
\frac{1}{|\mathcal{T}||\mathcal{L}|}
\sum_{t,l}
\max
\left(
-D_{vu}^{(l,t)},
0
\right)
M_{vu}^{(l,t)}.
$$

Risk score：

$$
P_{vu}=R_{vu}-\eta H_{vu},
$$

其中 \(\eta>0\) 为平衡系数。\(P_{vu}\) 越大，边 \((v,u)\) 越可能是 task-optimization harmful edge。

### 6.11 局部自适应裁剪

对每个节点 \(v\)，统计邻域风险集合：

$$
\mathcal{P}_v
=
\{
P_{vu}:u\in\mathcal{N}(v)
\}.
$$

局部阈值为：

$$
\theta_v
=
\operatorname{mean}(\mathcal{P}_v)
+
\lambda_\theta
\operatorname{std}(\mathcal{P}_v).
$$

节点 \(v\) 的最大裁剪预算为：

$$
b_v
=
\min
\left(
\lfloor
\beta |\mathcal{N}(v)|
\rfloor,
|\mathcal{N}(v)|-d_{min}
\right).
$$

候选裁剪集合为：

$$
\mathcal{C}_v
=
\operatorname{Top}_{b_v}
\left(
\{
u\in\mathcal{N}(v):
P_{vu}>\theta_v
\}
\right).
$$

最终净化边集为：

$$
\mathcal{E}'
=
\mathcal{E}
\setminus
\{
(v,u):
u\in\mathcal{C}_v
\}.
$$

得到净化图：

$$
\mathcal{G}'=(\mathcal{V},\mathcal{E}',X).
$$

对于无向图，可将双向风险平均：

$$
P_{\{v,u\}}
=
\frac{P_{vu}+P_{uv}}{2}.
$$

## 7. 每个模块的必要性

### 7.1 EMA Teacher

如果直接使用 student 预测，训练初期预测波动大，边评分会不稳定。EMA teacher 提供平滑预测 \(q_v\)，降低伪标签抖动。若没有 teacher，risk score 方差可能显著增大。

### 7.2 Soft Pseudo Label

Hard pseudo label 会把错误预测当作真值。Soft pseudo label 保留类别分布的不确定性，更适合用于结构净化，因为边评分需要稳定的任务趋势，而不是过早强化某个类别边界。

### 7.3 Confidence / Entropy Reliability

并非所有伪标签都可信。Reliability 抑制低置信节点主导 loss 和 edge score，降低 teacher bias 对图结构裁剪的污染。

### 7.4 \( \mathcal{L}_{proxy} \) 与 \( \mathcal{L}_{score} \) 分离

训练阶段可以使用 consistency regularization 提升表示稳定性；但边评分阶段希望梯度反映分类任务贡献，因此使用 deterministic scoring loss，避免随机增强和一致性约束带来的非任务梯度干扰。

### 7.5 Signed Cosine

若 \(\cos=-1\)，代表强烈方向冲突。若使用 \(|\cos|\)，则 \(|-1|=1\)，会把强有害边误判为高价值边。因此必须保留方向符号。

### 7.6 Relative Magnitude

绝对梯度范数不可跨 epoch 比较。相对梯度强度 \(M_{vu}\) 衡量邻居在目标节点局部邻域中的相对影响，更适合边级比较。

### 7.7 Helpful / Harmful 分离

很多边既不是强正向，也不是强负向。将 helpful 和 harmful 分开，可以避免把中性边误删。最终风险使用 \(P_{vu}=R_{vu}-\eta H_{vu}\)，使高有害、低有益的边优先被裁剪。

### 7.8 Local Adaptive Pruning

不同节点度数和邻域质量差异很大。全局阈值难以适配所有节点。局部排序与局部阈值更符合图结构异质性。

### 7.9 Minimum Degree Protection

若不限制最小度数，稀疏节点容易被裁成孤立点，导致 message passing 失效。因此裁剪预算必须保证 \(deg(v)\ge d_{min}\)。

## 8. Oracle GraCA 与 Practical GraCA

| 维度 | Oracle GraCA | Practical GraCA |
|---|---|---|
| 标签来源 | 全图真实标签 | 训练标签 + soft pseudo label |
| 是否使用测试标签 | 是 | 否 |
| 是否符合半监督设定 | 否 | 是 |
| 梯度来源 | full-label loss | semi-supervised scoring loss |
| 论文角色 | upper bound / diagnostic study | main method |
| 结果位置 | 单独上界表或分析表 | 主实验表 |
| 允许 claim | 梯度边信号存在、上界潜力 | 合法半监督图净化 |
| 禁止 claim | 实际部署有效、标准设置 SOTA | 恢复真实图结构、识别真实错误边 |

## 9. 与相关工作的系统对比

| 方法 | 方法目标 | 使用的边判断信号 | 是否 task-aware | 是否使用梯度行为 | 输出形式 | 是否 downstream transferable | 与 GraCA 的关键差异 |
|---|---|---|---|---|---|---|---|
| DropEdge | 正则化、缓解过拟合/过平滑 | 随机删边 | 否 | 否 | 训练策略 | 否 | 不判断边质量，随机扰动结构 |
| GNNGuard | 对抗鲁棒防御 | 特征相似性、边权重 | 部分 | 否 | 加权消息传递模型 | 部分 | 依赖静态相似性，不分析隐藏梯度方向 |
| GNNExplainer | 解释单个预测 | 学习解释 mask | 是 | 部分 | explanation subgraph | 否 | 目标是解释预测，不输出可复用净化图 |
| GraphCleaner | 图数据标签纠错 | 预测结果、邻域标签依赖 | 是 | 否 | 修正/检测标签 | 否 | 主要处理 mislabeled nodes，而非边级结构净化 |
| ProGNN | 鲁棒图结构学习 | 低秩、稀疏、平滑等结构先验 | 部分 | 否 | 学习后的图和模型 | 部分 | 依赖结构先验，不使用任务梯度行为评分边 |
| GDC | 图扩散预处理 | 扩散矩阵、图统计 | 否 | 否 | 预处理图 | 是 | 是结构扩散，不区分任务有害边 |
| Meta-gradient Sparsification | 边稀疏化 | validation/meta gradient | 是 | 是 | 稀疏图 | 部分 | 通常为双层优化，计算更重；GraCA 用隐藏表示梯度做离线边评估 |
| Self-training | 提升分类器 | pseudo label | 是 | 否 | 更强模型 | 否 | 目标是训练模型；GraCA 的输出是净化图 |
| Practical GraCA | 图结构净化 | hidden gradient + reliability | 是 | 是 | sanitized graph | 是 | 直接从任务优化梯度行为评估边贡献 |

GraCA 的核心区别不是问“边是否相似”，而是问“边是否帮助任务优化”。

## 10. 创新点与投稿价值

### 10.1 Contribution-style 创新点

1. 提出 task-gradient behavior 视角下的 graph sanitization 框架，将边质量从静态结构属性重新定义为任务优化贡献。
2. 提出 uncertainty-aware gradient edge assessment，在不使用测试标签的半监督设定下，通过 EMA teacher、soft pseudo label 和可靠性权重估计边级任务信号。
3. 提出 helpful-harmful risk decomposition，用 signed gradient direction 和 relative gradient magnitude 区分有益边、中性边和任务有害边。
4. 设计 downstream model transferable 的离线图净化流程，净化图可供 GCN、GAT、GraphSAGE 等下游模型从零训练。

### 10.2 投稿潜力评估

Practical GraCA 有 A 会/A 刊潜力，但前提是实验必须证明它不是 self-training、不是随机删边、不是简单同质性剪枝，也不是只在 Cora 上有效。

- AAAI / WWW：若多数据集稳定提升、消融完整、与结构学习和鲁棒 GNN baseline 对比充分，具备投稿潜力。
- KDD：需要突出 data-centric graph learning，并加入更大规模图或真实边噪声场景。
- TKDE / Information Sciences：如果实验系统、分析完整、方法稳健，期刊路线较合适。
- NeurIPS / ICML：目前理论深度和方法抽象性可能不足，除非补充更强理论或广泛任务泛化。

需要支撑投稿价值的关键结果包括：Practical GraCA 显著优于 Original、DropEdge、random pruning、homophily pruning、GNNGuard、GDC/ProGNN；Oracle 与 Practical 之间存在可解释 gap；净化图能跨下游模型迁移；在人工加噪边场景下鲁棒性提升；连通性和最小度保护有效。

## 11. 审稿风险与防御

| 质疑 | 风险原因 | 防御说法 | 需要实验支撑 |
|---|---|---|---|
| 是否标签泄漏 | 原始 Oracle 使用测试标签 | 主方法不使用测试标签；Oracle 单独作为 upper bound | 主表不含 Oracle，单独 upper-bound 表 |
| 是否只是 self-training | 使用了 soft pseudo label | 输出是净化图，不是 teacher/student 分类器；下游模型从零训练 | downstream transferability |
| Teacher bias 是否污染图 | 伪标签错误会影响梯度 | soft label、confidence/entropy reliability、deterministic scoring 降低污染 | w/o reliability、hard vs soft pseudo |
| 是否依赖 ProxyGNN | 边评分来自代理模型 | 不声称完全 model-agnostic，只声称 downstream transferable | 不同 proxy / downstream 组合 |
| 低同质图是否失效 | 异类边可能有用 | GraCA 不按标签删边，保留 helpful 异类边 | Actor、Texas、Cornell、Wisconsin |
| 是否破坏连通性 | 裁剪可能导致孤立节点 | local budget + minimum degree protection | isolated nodes、LCC ratio |
| 复杂度是否过高 | 需要梯度采集和边评分 | 离线一次执行，评分近似边线性复杂度 | runtime、memory、scalability |
| 与 GNNGuard 是否重合 | 都处理有害边 | GNNGuard 用特征相似性；GraCA 用任务梯度行为 | 与 GNNGuard 对比 |
| 与 GNNExplainer 是否重合 | 都用模型信号解释边 | GNNExplainer 输出解释子图；GraCA 输出全图净化结构 | explanation vs sanitization 分析 |
| 是否只在 Cora 有效 | 小型同质图容易过拟合 | 多数据集、多模型、多 seed | Cora/CiteSeer/PubMed/OGB/heterophily |
| 理论是否不足 | 梯度方向依据可能被质疑 | 用一阶近似解释 task-gradient conflict；Oracle 分析作为证据 | 理论附录 + oracle diagnostic |

## 12. 实验设计概览

### 12.1 数据集

同质图：Cora、CiteSeer、PubMed。

异质/低同质图：Actor、Texas、Cornell、Wisconsin，可选 Chameleon、Squirrel。

大图：ogbn-arxiv 或 ogbn-products 子集，视算力决定。

人工加噪图：随机跨类加边、随机重连、结构扰动攻击边。

### 12.2 Baselines

基础对比：Original graph、DropEdge、Random pruning with same ratio、Homophily pruning。

结构学习/预处理：GDC、ProGNN。

鲁棒图学习：GNNGuard。

可选高级对比：meta-gradient sparsification、Jaccard-GCN。

### 12.3 Downstream Models

GCN、GAT、GraphSAGE 为必选；APPNP 或 GCNII 可作为补充。

### 12.4 Metrics

分类指标：Accuracy、Macro-F1。

结构指标：Edge pruning ratio、homophily change、isolated node count、largest connected component ratio。

效率指标：training time、inference time、edge count reduction、memory。

统计指标：mean ± std over 5/10 seeds，必要时报告显著性检验。

### 12.5 实验表设计

主实验表：Practical GraCA vs baselines，证明合法半监督设置下有效。

Oracle 上界表：Oracle GraCA 单独展示，证明梯度边信号上界存在，不参与主对比。

消融表：w/o EMA teacher、hard pseudo、w/o reliability、harmful-only、helpful-only、global threshold、train-only。

鲁棒性表：不同噪声比例下的性能和删边质量。

迁移表：同一个 ProxyGNN 产生净化图，不同 downstream GNN 从零训练。

可扩展性表：节点数、边数、梯度采集时间、裁剪时间、总开销。

## 13. 论文包装

### 13.1 推荐标题

1. Practical GraCA: Gradient-Guided Graph Sanitization for Semi-Supervised Node Classification
2. Learning to Sanitize Graphs from Task Optimization Behaviors
3. Beyond Homophily: Gradient-Based Edge Assessment for Graph Refinement
4. Task-Aware Graph Sanitization via Hidden Gradient Analysis
5. Gradient-Guided Structure Refinement for Graph Neural Networks
6. Understanding Graph Edges Through Optimization Behaviors
7. Graph Sanitization with Uncertainty-Aware Gradient Signals
8. Learning Task-Optimization Harmful Edges for Graph Refinement

### 13.2 One-Sentence Pitch

Rather than asking whether an edge is correct, GraCA asks whether the edge helps optimize the task.

### 13.3 Abstract Draft

Graph neural networks heavily rely on graph structures, yet many existing graph refinement methods primarily focus on feature similarity, structural heuristics, or graph reconstruction objectives. We argue that a more fundamental question is whether a graph edge contributes positively to task optimization. Motivated by this observation, we propose Practical GraCA, a gradient-guided graph sanitization framework for semi-supervised node classification. Instead of identifying structurally incorrect edges, Practical GraCA detects task-optimization harmful edges by analyzing hidden representation gradients generated during model training. To avoid test-label leakage, Practical GraCA employs a ProxyGNN equipped with an EMA teacher and confidence-weighted soft pseudo labels to produce reliable optimization signals under the standard semi-supervised setting. We further introduce uncertainty-aware edge assessment based on signed gradient direction consistency, relative gradient influence, and prediction reliability. The resulting edge risk scores are used to perform adaptive graph sanitization through local neighborhood pruning. Unlike model-specific approaches, Practical GraCA outputs a sanitized graph that can be reused by multiple downstream GNN architectures. Extensive experiments on homophilic and heterophilic benchmarks demonstrate that Practical GraCA consistently improves node classification performance while preserving graph connectivity and robustness. These results suggest that task-driven gradient behaviors provide a practical signal for graph structure refinement.

### 13.4 Introduction 六段逻辑

第一段：GNN 依赖图结构，边决定信息传播路径。

第二段：真实图中存在大量任务无关或任务有害边。

第三段：现有方法多依赖结构启发、特征相似性或随机扰动，缺少任务优化行为视角。

第四段：梯度天然编码模型为了降低损失希望表示如何变化，可用于定义边的任务贡献。

第五段：直接使用 full-label gradient 会标签泄漏，因此需要合法半监督近似。

第六段：提出 Practical GraCA，并概述其 uncertainty-aware gradient edge assessment 与 downstream transferability。

### 13.5 可以说的 Claim

- GraCA performs graph sanitization from task-gradient behaviors.
- GraCA identifies task-optimization harmful edges, not necessarily incorrect edges.
- Sanitized graphs are downstream model transferable.
- Oracle GraCA provides diagnostic upper-bound evidence.

### 13.6 不能说的 Claim

- 恢复真实图结构。
- 检测所有错误边。
- 完全 model-agnostic。
- 理论最优。
- 在所有图 benchmark 上都是 SOTA。
