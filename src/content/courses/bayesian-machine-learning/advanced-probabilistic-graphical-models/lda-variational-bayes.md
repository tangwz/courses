---
course: "bayesian-machine-learning"
chapter: "advanced-probabilistic-graphical-models"
lesson: "lda-variational-bayes"
sourceId: 3621
sourceUrl: "https://apxml.com/zh/courses/bayesian-machine-learning/chapter-5-advanced-probabilistic-graphical-models/lda-variational-bayes"
title: "LDA的推断：变分贝叶斯"
description: "为LDA的可扩展主题建模实现变分贝叶斯推断。"
order: 7
plots: []
sourceHash: "6dc5f74a2ef0c74953467a16569ba0e3824ef4c4f551960be845e9686232cca9"
sourceCorrections: []
---

折叠吉布斯采样（Collapsed Gibbs Sampling）是一种用于潜在狄利克雷分配（LDA）后验推断的方法。然而，其迭代性质，即一次处理一个词元 (token)分配，对于大型语料库而言计算量可能很大。变分贝叶斯（VB）提供了一种确定性替代方法，通常可以更有效地扩展。VB将推断问题转化为一个优化问题，而非采样。

### LDA的变分贝叶斯方法

LDA推断中的主要挑战是计算后验分布 $p(\theta, z, \beta | w)$，其中 $\theta$ 代表文档-主题分布，$z$ 代表每个词的主题分配，$\beta$ 代表主题-词分布，$w$ 代表观测到的词。由于变量之间复杂的依赖关系，这个后验分布是难以处理的。

变分贝叶斯通过引入一个更简单的近似分布 $q(\theta, z, \beta)$ 来解决这个问题，该分布选自分布族 $\mathcal{Q}$。目标是找到分布族 $\mathcal{Q}$ 中最接近真实后验 $p$ 的分布 $q$。接近程度通常使用Kullback-Leibler（KL）散度 $KL(q || p)$ 来衡量。最小化这个KL散度等价于最大化数据对数边缘似然的下界，这个下界被称为证据下界（ELBO）：


$$
\mathcal{L}(q) = \mathbb{E}_q[\log p(\theta, z, \beta, w)] - \mathbb{E}_q[\log q(\theta, z, \beta)] \le \log p(w)
$$


对于LDA，变分族 $\mathcal{Q}$ 的常见选择是 *平均场* 族，该族假设近似后验中的潜在变量是相互独立的：


$$
q(\theta, z, \beta | \gamma, \phi, \lambda) = \prod_{d=1}^M q(\theta_d | \gamma_d) \prod_{k=1}^K q(\beta_k | \lambda_k) \prod_{n=1}^{N_d} q(z_{dn} | \phi_{dn})
$$


这里：

- $q(\theta_d | \gamma_d)$ 是一个狄利克雷分布，近似文档 $d$ 的主题混合后验，其参数 (parameter)是变分狄利克雷参数向量 (vector) $\gamma_d = (\gamma_{d1}, ..., \gamma_{dK})$。
- $q(\beta_k | \lambda_k)$ 是一个狄利克雷分布，近似主题 $k$ 的词分布后验，其参数是变分狄利克雷参数向量 $\lambda_k = (\lambda_{k1}, ..., \lambda_{kV})$，其中 $V$ 是词汇表 (vocabulary)大小。
- $q(z_{dn} | \phi_{dn})$ 是一个多项式分布，近似文档 $d$ 中第 $n$ 个词的主题分配后验，其参数是变分多项式参数向量 $\phi_{dn} = (\phi_{dn1}, ..., \phi_{dnK})$，其中 $\sum_k \phi_{dnk} = 1$。

参数 $\gamma$、$\phi$ 和 $\lambda$ 是我们为了最大化ELBO需要优化的 *变分参数*。

> 平均场近似假设变分分布 $q$ 中的潜在变量（$\theta$、$\beta$、$z$）之间相互独立，从而简化了真实后验 $p$ 中存在的难以处理的依赖关系。

### LDA的坐标上升变分推断（CAVI）

ELBO可以通过迭代坐标上升方法最大化。我们循环更新变分参数 (parameter)（$\phi$、$\gamma$、$\lambda$），每次固定其他参数。更新方程是通过对ELBO的每个参数求导，将其设为零并求解而得出的。结果更新通常具有直观的形式，类似于期望最大化（EM）算法中的更新。

1. **更新 $\phi_{dnk}$**：此更新估计词 $w_{dn}$ 属于主题 $k$ 的概率。它依赖于文档 $d$ 的主题比例的期望对数概率的当前估计（与 $\gamma_d$ 相关）以及词 $w_{dn}$ 在主题 $k$ 下的期望对数概率（与 $\lambda_k$ 相关）。

   
   $$
   \phi_{dnk} \propto \exp(\mathbb{E}_{q(\theta_d|\gamma_d)}[\log \theta_{dk}] + \mathbb{E}_{q(\beta_k|\lambda_k)}[\log \beta_{k, w_{dn}}])
   $$
   

   期望涉及双伽马函数 $\Psi$，因为 $\mathbb{E}[\log \theta_{dk}] = \Psi(\gamma_{dk}) - \Psi(\sum_{j=1}^K \gamma_{dj})$。在计算所有 $k$ 的右侧后，$\phi_{dn}$ 进行归一化 (normalization)，使得 $\sum_k \phi_{dnk} = 1$。
2. **更新 $\gamma_{dk}$**：此更新聚合了文档 $d$ 中所有词分配给主题 $k$ 的责任（$\phi_{dnk}$），并加上原始的狄利克雷先验参数 $\alpha_k$。它表示文档 $d$ 中分配给主题 $k$ 的预期词数。

   
   $$
   \gamma_{dk} = \alpha_k + \sum_{n=1}^{N_d} \phi_{dnk}
   $$
   
3. **更新 $\lambda_{kv}$**：此更新聚合了在所有文档中词 $v$ 的所有实例（其中 $w_{dn}=v$）分配给主题 $k$ 的责任（$\phi_{dnk}$），并加上原始的狄利克雷先验参数 $\eta_v$。它表示词 $v$ 由主题 $k$ 生成的预期次数。

   
   $$
   \lambda_{kv} = \eta_v + \sum_{d=1}^M \sum_{n: w_{dn}=v} \phi_{dnk}
   $$
   

这些更新会迭代进行，直到ELBO收敛，这意味着变分参数趋于稳定。

### 结果的解读

一旦收敛，优化的变分参数 (parameter)会提供后验分布的近似值：

- $\gamma^*$：参数化了文档 $d$ 的主题混合的近似后验狄利克雷分布 $q(\theta_d | \gamma^*_d)$。预期的主题比例为 $\mathbb{E}[\theta_{dk}] = \gamma^*_{dk} / \sum_{j=1}^K \gamma^*_{dj}$。
- $\lambda^*$：参数化了主题 $k$ 的词分布的近似后验狄利克雷分布 $q(\beta_k | \lambda^*_k)$。预期的词概率为 $\mathbb{E}[\beta_{kv}] = \lambda^*_{kv} / \sum_{v'=1}^V \lambda^*_{kv'}$。
- $\phi^*$：表示词 $w_{dn}$ 属于主题 $k$ 的优化后验概率 $q(z_{dn}=k | \phi^*_{dn})$。

### VB与折叠吉布斯采样的比较

| 特点 | 变分贝叶斯 (VB) | 折叠吉布斯采样 |
| --- | --- | --- |
| **方法** | 优化（最大化ELBO） | 采样 |
| **性质** | 确定性 | 随机性 |
| **速度** | 每轮迭代通常更快 | 每轮迭代可能较慢 |
| **可扩展性** | 更适合大型数据集 | 处理大数据可能存在困难 |
| **收敛性** | 收敛到ELBO的局部最优 | （在极限情况下）收敛到真实后验 |
| **准确度** | 提供近似值（有偏） | 渐近精确 |
| **实现** | 更新方程可能复杂 | 采样方法更简单 |
| **并行性** | 更易于某些并行化 | 可以并行化（例如，按文档） |

VB通常比吉布斯采样收敛快得多，特别是在大型数据集上。然而，由于它基于近似分布族（平均场假设）来优化下界，它可能收敛到一个对真实后验近似较差的解，相比于运行良好的吉布斯采样器。平均场假设倾向于低估后验分布的方差。

### 进一步扩展：随机变分推断

对于真正大规模的数据集，即使是标准的批量VB也可能太慢，因为它在$\lambda$的每次更新步骤中都需要处理整个语料库。随机变分推断（SVI）通过对ELBO使用随机梯度上升来解决这个问题。它在每一步处理小批量的文档，从而实现在线学习和显著加速，使得VB适用于网络规模的数据集。

总之，变分贝叶斯为LDA提供了一个强大且可扩展的推断框架。通过将推断视为优化问题，并使用因子化近似，它使我们能够从大型文本集合中有效地估计主题模型，补充了像折叠吉布斯采样这样的基于采样的方法。

## 参考资料

- [Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — David M. Blei, Andrew Y. Ng, Michael I. Jordan (2003)
  Journal: Journal of Machine Learning Research; Publisher: MIT Press; Volume: 3; Pages: 993-1022; DOI: [10.1162/jmlr.2003.3.blei03a](https://doi.org/10.1162/jmlr.2003.3.blei03a)
  介绍了潜在狄利克雷分配（LDA）模型及其基础的变分贝叶斯推断算法，包括平均场近似和坐标上升更新。
- [Variational Inference: A Review for Statisticians](https://www.tandfonline.com/doi/full/10.1080/01621459.2017.1285773) — David M. Blei, Alp Kucukelbir, Jon D. McAuliffe (2017)
  Journal: Journal of the American Statistical Association; Publisher: Taylor & Francis on behalf of the American Statistical Association; Volume: 112; Pages: 859-877; DOI: [10.1080/01621459.2017.1285773](https://doi.org/10.1080/01621459.2017.1285773)
  对变分推断进行了全面且现代的综述，详细阐述了理论基础、平均场近似、ELBO最大化及其与包括LDA在内的各种模型的关联。
- [Stochastic Variational Inference](http://jmlr.org/papers/v14/hoffman13a.html) — Matthew D. Hoffman, David M. Blei, Chong Wang, John Paisley (2013)
  Journal: Journal of Machine Learning Research; Volume: 14; Pages: 1303-1347; DOI: [10.5555/2503926.2503947](https://doi.org/10.5555/2503926.2503947)
  介绍了随机变分推断（SVI）算法，该算法通过使用小批量数据将变分贝叶斯扩展到大规模数据集，与本节中关于扩展性的讨论相关。
- [Pattern Recognition and Machine Learning](https://link.springer.com/book/9780387310732) — Christopher M. Bishop (2006)
  Publisher: Springer; Pages: 461-536
  第10章对变分推断进行了清晰且基础的介绍，包括其理论基础、平均场近似及其在各种概率模型中的应用。
