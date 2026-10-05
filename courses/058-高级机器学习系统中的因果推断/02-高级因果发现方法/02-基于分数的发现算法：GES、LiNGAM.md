# 基于分数的发现算法：GES、LiNGAM

来源：[原文](https://apxml.com/zh/courses/causal-inference-ml-systems/chapter-2-advanced-causal-discovery/score-based-discovery-algorithms)

[返回章节目录](README.md) · [返回课程目录](../README.md)

当基于约束的算法利用条件独立性测试来削减可能因果结构的空间时，基于分数的发现算法则采用不同路径。它们将因果发现构建为一个优化问题：定义一个评分函数，用于衡量候选图结构与观测数据的拟合程度，然后查找可能图形空间中能使此分数最大化（或最小化）的那个。

在基于分数的因果发现算法中，一个主要挑战在于搜索空间的庞大性。可能的有向无环图（DAG）数量随变量数量呈超指数增长，使得除了最小问题之外，穷举搜索在计算上不可行。因此，基于分数的方法依赖于启发式或贪婪搜索策略。我们将讨论两种主要的办法：贪婪等价搜索（GES）和线性非高斯无环模型（LiNGAM）。

### 贪婪等价搜索（GES）

贪婪等价搜索（GES）算法没有在单个DAG的空间中进行查找，而是巧妙地在*马尔可夫等价类*（MEC）空间中操作。MEC代表一组DAG，它们编码了相同的条件独立关系集合，因此在标准假设下，仅凭观测数据无法区分它们。在MEC上进行查找显著降低了搜索空间的复杂性。

GES通常分两阶段进行：

1. **前向等价搜索（FES）：** 算法从空图（无边）开始，迭代添加能使评分函数增加最多的单条边（$X \rightarrow Y$ 或 $Y \rightarrow X$），前提是此添加将当前图移动到不同的、得分更高的MEC。此过程持续到没有单边添加能提高分数为止。
2. **后向等价搜索（BES）：** 算法从FES阶段结束时获得的图开始，迭代移除能使评分函数增加最多（或减少最少）的单条边，同时确保结果图属于有效的MEC。此阶段持续到没有单边移除能提高分数为止。

GES的最终输出通常表示为完整部分有向无环图（CPDAG），它以图形方式表征所辨别的MEC。

#### 评分函数

评分函数的选择对GES来说是基本的。常用评分包括：

- **贝叶斯信息准则（BIC）/施瓦茨信息准则（SIC）：**
  
  $$
  Score_{BIC}(G; D) = \log L(\hat{\theta}_G | D) - \frac{\log n}{2} |G|
  $$
  
  其中，$L(\hat{\theta}_G | D)$ 是给定图 $G$ 和估计参数 (parameter) $\hat{\theta}_G$ 时数据 $D$ 的最大似然，$n$ 是样本量，而 $|G|$ 是模型中独立参数的数量（与图的复杂性相关，通常是边的数量加上每个节点条件分布的参数）。BIC比AIC对模型复杂性的惩罚更重，通常偏好稀疏图，特别是在大型数据集上。
- **赤池信息准则（AIC）：**
  
  $$
  Score_{AIC}(G; D) = \log L(\hat{\theta}_G | D) - |G|
  $$
  
  AIC的惩罚项比BIC小。
- **贝叶斯狄利克雷等价均匀（BDeu）评分：** 专门为离散变量设计，此评分在参数独立性和狄利克雷先验假设下计算边际似然 $P(D|G)$。它具有理想的*分数等价性*属性，意味着同一MEC中的所有DAG获得相同分数。

#### 假设与局限性

GES依赖于几个假设：

- **无环性：** 真实因果结构是DAG。
- **因果充分性：** 观测变量没有未观测到的共同原因（潜在混杂因素）。
- **忠实性：** 数据中存在的所有条件独立性都对应于真实因果图中的d-分离。
- **分布假设：** 特定分数通常暗示分布假设（例如，BIC/AIC通常假设连续数据为高斯分布或离散数据为多项式分布进行计算）。

在这些假设下，GES在统计上是一致的，这意味着它随样本量增加而收敛到真实的MEC。然而，它执行贪婪搜索，这意味着它可能陷入局部最优。其性能对评分函数的选择和基本假设的潜在违反敏感，尤其是因果充分性。

### 线性非高斯无环模型（LiNGAM）

LiNGAM提供了一种独特的方法，在特定假设下，它可以辨别*精确*的DAG结构，而不仅仅是MEC。它通过借助数据分布*形态*中包含的信息来实现这一点，特别要求非高斯性。

#### LiNGAM模型

LiNGAM的核心假设是数据生成过程是线性且无环的，带有非高斯误差项（新息）：


$$
X_i = \sum_{X_j \in PA_i} \beta_{ij} X_j + \epsilon_i
$$


其中，$X_i$ 是第 $i$ 个变量，$PA_i$ 代表其在DAG中的直接因果父节点集合，$\beta_{ij}$ 是表示从 $X_j$ 到 $X_i$ 直接效应强度的线性因果系数，而 $\epsilon_i$ 是独立的、非高斯误差项，具有零均值和非零方差。无环性意味着，如果根据因果结构对变量进行排序（例如，$X_{k_1}, X_{k_2}, ..., X_{k_d}$，其中 $k_1, ..., k_d$ 是 $1, ..., d$ 的一个置换），则通过应用此置换，可以将邻接矩阵 $B$（其中如果 $X_j \in PA_i$ 则 $B_{ij} = \beta_{ij}$，否则为0）变为严格下三角。

此模型可以写成矩阵形式：


$$
X = BX + \epsilon
$$


这意味着：


$$
X = (I - B)^{-1} \epsilon
$$


其中 $I$ 是单位矩阵。

#### 通过非高斯性和ICA辨别

LiNGAM背后的观点源于独立成分分析（ICA）。ICA是一种用于将多元信号分离成相加、独立、非高斯源信号的技术。在LiNGAM的背景下，观测变量 $X$ 是独立的非高斯误差项 $\epsilon$ 的线性混合。矩阵 $A = (I-B)^{-1}$ 作为混合矩阵。

ICA算法旨在找到一个*解混*矩阵 $W$，使得 $W X$ 能恢复原始独立源（在置换和缩放的范围内）。也就是说，$W \approx A^{-1} = (I-B)$。

LiNGAM利用了这样一个事实：如果找到正确的解混矩阵 $W$，它与因果结构矩阵 $B$ 存在特定关系：$W = P D (I - B)$，其中 $P$ 是置换矩阵，$D$ 是对角缩放矩阵。核心地，可以找到 $W$ 行的一个置换，使得置换后的矩阵 $W'$ 的零元素与 $(I-B)$ 中的零条目相对应。由于 $B$ 在正确的因果顺序中是下三角矩阵，所以 $(I-B)$ 也是下三角矩阵（对角线上为1）。因此，$W'$ 可以通过置换变为下三角。

LiNGAM算法（其基本形式）通常包含以下步骤：

1. **数据中心化：** 从每个变量中减去均值。
2. **执行ICA：** 将ICA算法（如FastICA）应用于中心化数据 $X$，以获得估计的解混矩阵 $W$。
3. **置换查找：** 查找置换矩阵 $P$，使得 $W' = PW$ 尽可能接近下三角（例如，在行归一化 (normalization)后，最小化对角线上方元素的平方和）。
4. **估计B：** 根据置换 $P$ 重新排序变量。从置换后的、缩放过的解混矩阵中估计连接强度 $B$。 $B$ 可以使用 $B = I - (W')^{-1}$ 从 $W'$ 推导（适当缩放后）。或者，根据辨别的因果顺序通过回归估计连接。
5. **剪枝（可选）：** 执行统计测试或使用自助法，以剪除估计 $B$ 中的弱连接或不显著的边。

#### 假设与优势

LiNGAM的主要优势在于它能够在其假设下辨别完整DAG（包括边方向）并估计因果强度：

- **线性：** 变量之间的关系是线性的。
- **无环性：** 没有反馈回路。
- **非高斯性：** 所有独立的误差项 $\epsilon_i$ 必须是非高斯的。最多只有一个误差项可以是高斯。
- **因果充分性：** 没有潜在混杂因素。

如果这些假设成立，LiNGAM提供了一个强大的发现工具。存在各种扩展来放宽一些假设，例如处理时间序列数据（VAR-LiNGAM）或提高计算效率（DirectLiNGAM）。

#### 局限性

严格的线性和非高斯性要求是LiNGAM的主要局限性。如果基础误差项是高斯的，它将从根本上失败，因为在这种情况下ICA无法唯一确定旋转。性能还取决于ICA估计和置换查找的质量。与GES一样，它假设因果充分性。

> LiNGAM数据生成过程的图示。独立的非高斯源（$\epsilon_i$）通过由系数（$\beta_{ij}$）定义的线性无环结构影响观测变量（$X_i$）。LiNGAM旨在从观测到的 $X_i$ 中恢复此结构。

### 基于分数方法的选择

- **GES** 在函数形式方面更普遍（取决于评分），但对于通过线性模型计算的BIC/AIC等标准评分，通常需要高斯假设。它辨别MEC，不一定是完整DAG，并要求因果充分性。
- **LiNGAM** 辨别完整DAG，但施加更严格的线性和非高斯性假设。它也要求因果充分性。

如果您的数据被强烈认为是由线性过程生成并表现出非高斯特性，LiNGAM是潜在恢复完整DAG结构的一个有吸引力的选择。如果线性有问题或数据看起来是高斯，GES（带有适当评分）可能更适用，同时承认它只会辨别MEC。

### 实现注意事项

`causal-learn` 等库提供了GES及相关算法的实现，通常允许指定不同的评分函数（例如，高斯数据的BIC）。`lingam` Python包提供了LiNGAM及其扩展（DirectLiNGAM、VAR-LiNGAM等）的各种实现。

应用这些方法时，请记住它们对假设的敏感性。在选择LiNGAM之前，评估数据分布（例如，使用正态性检验）。对于GES，考虑所选评分函数的影响以及对局部最优的潜在敏感性。使用领域知识或干预数据（如果可用）验证发现的结构总是建议的。基于分数的方法，尽管优化全局标准，但仍可能受数据质量和采用的特定启发式搜索策略的影响。

## 参考资料

- [A polynomial-time algorithm for learning Bayesian networks with local structure](https://link.springer.com/article/10.1023/A:1015696117135) — David M. Chickering (2002)
  Journal: Machine Learning; Publisher: Springer US; Volume: 48; Pages: 35-42; DOI: [10.1023/A:1015696117135](https://doi.org/10.1023/A%3A1015696117135)
  介绍了贪婪等价搜索（GES）算法，这是一种在马尔可夫等价类中进行因果发现的方法。
- [A Causal Discovery Algorithm for Linear Non-Gaussian Acyclic Models](https://www.jmlr.org/papers/volume7/shimizu06a/shimizu06a.pdf) — Shohei Shimizu, Patrik O. Hoyer, Antti Hyvärinen, Aapo Tankasalo (2006)
  Journal: Journal of Machine Learning Research; Volume: 7; Pages: 2003-2030
  本文提出了原始的LiNGAM框架，利用误差项的非高斯性来识别完整的因果DAGs。
- [Elements of Causal Inference: Foundations and Learning Algorithms](https://mitpress.mit.edu/books/elements-causal-inference) — Jonas Peters, Dominik Janzing, Bernhard Schölkopf (2017)
  Publisher: MIT Press; DOI: [10.7551/mitpress/9780262037380.001.0001](https://doi.org/10.7551/mitpress/9780262037380.001.0001)
  一本因果推断基础概念和算法的教科书，包括与基于得分的方法以及LiNGAM所使用的识别原理相关的讨论。
- [DirectLiNGAM: A direct method for learning a linear non-Gaussian acyclic model](https://www.jmlr.org/papers/volume12/shimizu11a/shimizu11a.pdf) — Shohei Shimizu, Takuya Inazumi, Yasuhiro Sogawa, Akira Hyvarinen, Yoshinobu Kawahara, Takashi Washio, Patrik O. Hoyer, Kenji Bollen (2011)
  Journal: Journal of Machine Learning Research; Publisher: Journal of Machine Learning Research; Volume: 12; Pages: 1225-1248; DOI: [10.5555/2012000.2012061](https://doi.org/10.5555/2012000.2012061)
  描述了DirectLiNGAM，这是对原始LiNGAM算法的改进，它直接估计因果顺序和系数，常用于实际应用。

---

[上一节](01-%E5%9F%BA%E4%BA%8E%E7%BA%A6%E6%9D%9F%E7%9A%84%E5%8F%91%E7%8E%B0%E7%AE%97%E6%B3%95%EF%BC%9APC%E3%80%81FCI%20%E7%AE%97%E6%B3%95%E7%9A%84%E5%BB%B6%E4%BC%B8.md) · [下一节](03-%E5%9B%A0%E6%9E%9C%E7%BB%93%E6%9E%84%E5%8F%91%E7%8E%B0%EF%BC%9A%E4%BD%BF%E7%94%A8%E5%B9%B2%E9%A2%84%E6%95%B0%E6%8D%AE.md)
