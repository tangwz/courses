# 进阶谱GNN (ChebNet, CayleyNets)

来源：[原文](https://apxml.com/zh/courses/graph-neural-networks-gnns/chapter-2-advanced-gnn-architectures/advanced-spectral-gnns)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管图卷积网络（GCN）在谱图理论与图上实际深度学习 (deep learning)之间建立了强连接，但其基础谱滤波器相对简单，通常是图拉普拉斯矩阵的一阶多项式。此限制使其难以获取复杂的谱模式并学习高度局部化的滤波器。为此，已发展出更精密的谱GNN架构，主要侧重于设计更具表现力与灵活性的图滤波器$g_\theta(\Lambda)$。两个主要实例是使用切比雪夫多项式近似的ChebNet，以及采用有理开莱滤波器的CayleyNets。

### ChebNet: 用于局部化的多项式滤波器

ChebNet [Defferrard et al., 2016] 的核心思路是使用正交多项式基，具体来说是第一类切比雪夫多项式$T_k(x)$，来近似所需的谱滤波器$g_\theta(\Lambda)$。这些多项式由递推关系定义：


$$
T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)
$$


基本情况为$T_0(x) = 1$ 和 $T_1(x) = x$。

切比雪夫多项式特别适用，因为它们定义在区间$[-1, 1]$上，并且归一化 (normalization)图拉普拉斯矩阵的特征值可以缩放到该范围。令$L$为图拉普拉斯矩阵（例如，$L = I - D^{-1/2}AD^{-1/2}$）。其特征值$\lambda_i$位于$[0, \lambda_{max}]$内，其中$\lambda_{max}$是最大特征值（对于归一化拉普拉斯矩阵，$\lambda_{max}$通常$\le 2$）。我们定义缩放后的拉普拉斯矩阵$\tilde{L} = \frac{2}{\lambda_{max}} L - I$，其特征值$\tilde{\lambda}_i$落在$[-1, 1]$内。

作用于特征值对角矩阵$\Lambda$的谱滤波器$g_\theta(\Lambda)$然后由一个$K$阶多项式近似：


$$
g_\theta(\Lambda) \approx \sum_{k=0}^{K} \theta_k T_k(\tilde{\Lambda})
$$


这里，$\theta = (\theta_0, ..., \theta_K)$是可学习的滤波器参数 (parameter)（多项式基的系数），而$T_k(\tilde{\Lambda})$将第$k$个切比雪夫多项式函数应用于$\tilde{\Lambda}$对角线上的每个特征值。

将此滤波器应用于图信号$x$得到：


$$
y = g_\theta(L) x = U g_\theta(\Lambda) U^T x \approx U \left( \sum_{k=0}^{K} \theta_k T_k(\tilde{\Lambda}) \right) U^T x = \sum_{k=0}^{K} \theta_k \left( U T_k(\tilde{\Lambda}) U^T \right) x
$$


重要地，$U T_k(\tilde{\Lambda}) U^T = T_k(U \tilde{\Lambda} U^T) = T_k(\tilde{L})$。这意味着我们无需显式特征分解即可计算滤波后的信号：


$$
y \approx \sum_{k=0}^{K} \theta_k T_k(\tilde{L}) x
$$


项$T_k(\tilde{L})x$可以通过直接在缩放后的拉普拉斯矩阵$\tilde{L}$和信号$x$上使用切比雪夫递推关系来高效地计算。一个基本特性是，应用$T_k(\tilde{L})$对应于严格从每个节点的$k$跳邻域聚合信息。这使得滤波器严格局部化在$K$跳范围内。

一个ChebNet层，其输入特征$H^{(l)} \in \mathbb{R}^{N \times F_{in}}$和输出特征$H^{(l+1)} \in \mathbb{R}^{N \times F_{out}}$定义为：


$$
H^{(l+1)}_{:, j} = \sigma \left( \sum_{i=1}^{F_{in}} \sum_{k=0}^{K} \Theta^{(l)}_{k, i, j} T_k(\tilde{L}) H^{(l)}_{:, i} \right) \quad \text{for } j=1, ..., F_{out}
$$


或者更紧凑地，将$\Theta_k^{(l)}$视为大小为$F_{in} \times F_{out}$的参数矩阵时：


$$
H^{(l+1)} = \sigma\left( \sum_{k=0}^{K} T_k(\tilde{L}) H^{(l)} \Theta_k^{(l)} \right)
$$


其中$\sigma$是非线性激活函数 (activation function)。

**与GCN的关联：** GCN可以看作是$K=1$时ChebNet的简化。设$K=1$则得到$y \approx \theta_0 T_0(\tilde{L})x + \theta_1 T_1(\tilde{L})x = \theta_0 x + \theta_1 \tilde{L}x$。通过进一步假设$\lambda_{max} \approx 2$，从而$\tilde{L} \approx L - I$，并使用单个参数$\theta = \theta_0 = -\theta_1$，我们恢复了类似于GCN传播规则的操作（经过重归一化技巧后）。ChebNet，在$K>1$时，能够通过控制多项式阶数$K$和系数$\theta_k$来学习更复杂的局部化滤波器。

### CayleyNets: 用于谱域放大的有理滤波器

CayleyNets [Levie et al., 2018] 提出使用基于开莱变换的有理谱滤波器，旨在谱域中实现更好的局部化，特别是用于选择特定频段（谱域放大）。

开莱变换定义为$c(x, h) = (x - ih) / (x + ih)$，其中$h > 0$是实参数 (parameter)，$i$是虚数单位。开莱多项式建立在此变换之上。一个作用于拉普拉斯矩阵$L$的基本开莱滤波器定义为：


$$
g_\theta(L) = \theta_0 I + 2 \text{Re} \sum_{j=1}^{r} \theta_j (hL - iI)^j (hL + iI)^{-j}
$$


这里，$r$控制着滤波器的复杂度（类似于ChebNet中的$K$），并且$\theta = (\theta_0, ..., \theta_r)$是可学习的系数（可能为复数，但通常受限）。参数$h$作为谱域放大参数。小的$h$将滤波器的能量集中在特征值$\lambda = 0$附近（低频），而大的$h$则更广泛地分布能量。

应用此滤波器需要计算诸如$(hL + iI)^{-j}x$的项。这涉及求解形如$(hL + iI)y = z$的复数线性系统。由于$hL+iI$与拉普拉斯矩阵相关，这些系统通常是稀疏的，并且可以使用共轭梯度等迭代方法高效求解。

**优点：** 开莱滤波器作为有理函数，能够比同阶多项式更有效地近似带通滤波器。这使得CayleyNets有可能从图信号中分离出特定频率范围，这对于某些需要区分低频（平滑）和高频（变异）分量的任务可能很有益。

### 比较ChebNet和CayleyNets

| 特性 | ChebNet | CayleyNets |
| --- | --- | --- |
| **滤波器类型** | 多项式 (切比雪夫) | 有理 (开莱变换) |
| **局部化** | 空间：严格$K$跳邻域 | 谱域：更好的频段选择 |
| **参数 (parameter)** | 多项式阶数$K$，系数$\theta_k$ | 滤波器阶数$r$，放大参数$h$，系数$\theta_j$ |
| **计算方式** | 递归多项式应用 ($T_k(L)x$) | 求解复数线性系统 ($(hL+iI)y=z$) |
| **灵活性** | 良好的通用滤波器近似 | 更佳的谱域放大能力 |
| **复杂度** | 每特征每层\$O(K | \mathcal{E} |

ChebNet和CayleyNets都代表着对基本GCN滤波器的显著进展，通过实现更复杂和定制化的谱滤波。ChebNet提供空间局部化滤波器，其范围（$K$）受到明确控制。CayleyNets在频域中提供可能更锐利的滤波器。两者间的选择取决于具体的应用要求：如果需要明确控制空间邻域大小，ChebNet是自然的选择。如果目标是精确的频率选择，CayleyNets可能更适合。

这些进阶谱方法展示了从图信号处理视角设计GNN的丰富性。尽管空间方法（将在后续讨论）因其归纳能力和更简单的公式而变得非常流行，但了解ChebNet和CayleyNets等谱方法能提供关于图卷积表现力及理论基础的宝贵见解。`ChebConv`等层的实现可在标准GNN库中找到，从而方便对这些强大架构进行实际试用。

## 参考资料

- [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://proceedings.neurips.cc/paper_files/paper/2016/file/04025958562c2f627f1ce62d2966a1e1-Paper.pdf) — Michaël Defferrard, Xavier Bresson, Pierre Vandergheynst (2016)
  Journal: Advances in Neural Information Processing Systems (NIPS); Publisher: NeurIPS; Volume: 29; DOI: [10.48550/arXiv.1606.09375](https://doi.org/10.48550/arXiv.1606.09375)
  提出ChebNet的原始论文，它使用切比雪夫多项式近似实现局部谱图卷积。
- [Graph Neural Networks: Foundations, Frontiers, and Applications](https://link.springer.com/book/10.1007/978-981-15-5120-7) — Zonghan Wu, Shirui Pan, Guodong Long, Jing Jiang, Xiaojun Chang, Chengqi Zhang (2020)
  Publisher: Springer; DOI: [10.1007/978-981-15-5120-7](https://doi.org/10.1007/978-981-15-5120-7)
  一本提供GNN架构广阔背景和理论基础的书籍，其中涵盖了谱方法。
- [Spectral Graph Theory](https://www.ams.org/books/cbms/092/) — Fan Chung (1997)
  Publisher: American Mathematical Society; Volume: 92
  一本经典的教科书，提供了谱图理论的数学基础，与理解谱GNN相关。

---

[上一节](04-%E5%9B%BETransformer.md) · [下一节](06-%E8%BF%9B%E9%98%B6%E7%A9%BA%E9%97%B4GNNs%20%28GraphSAGE%E5%8F%98%E4%BD%93%2C%20PNA%29.md)
