---
course: "bayesian-machine-learning"
chapter: "advanced-probabilistic-graphical-models"
lesson: "approximate-inference-large-pgms"
sourceId: 3616
sourceUrl: "https://apxml.com/zh/courses/bayesian-machine-learning/chapter-5-advanced-probabilistic-graphical-models/approximate-inference-large-pgms"
title: "大型概率图模型中的近似推断"
description: "将MCMC和变分推断技术应用于大型或难以精确推断的概率图模型。"
order: 4
plots: []
sourceHash: "bef3e1dcee6195c42f5adad4c241eeb67fdc34e5d646bbdbbddaf1c7ff3da30b"
sourceCorrections: []
---

精确推断算法，比如连接树算法，能为概率图模型中的概率查询提供准确答案，但它们常遇到计算瓶颈。对于树宽较高的图（连接密集的结构）或包含许多没有方便共轭关系的连续变量的模型，精确推断的计算成本会变得过高。计算边际概率或条件概率，例如 $P(X_i | E)$ 或 $P(X_i, X_j | E)$，可能需要对指数级大的状态空间进行求和或积分。现代机器学习 (machine learning)应用中常见的那些大型复杂图模型尤其如此。

当精确推断不可行时，我们转向近似推断方法。这些方法牺牲理论上的精确性以换取计算上的可行性，目标是为所需后验概率或期望提供足够准确的估计。你在前面章节中已经接触到的两种主要的近似推断技术是马尔可夫链蒙特卡罗（MCMC）方法和变分推断（VI）。我们来看看它们是如何应用于大型概率图模型的。

### MCMC方法在概率图模型中的应用

MCMC方法通过构建一个马尔可夫链来近似目标后验分布 $P(X | E)$（其中 $X$ 代表所有未观测变量的集合，$E$ 是观测到的证据），该马尔可夫链的平稳分布*就是*目标后验分布。经过足够的“预热”期后，从链中抽取的样本可以被视为来自 $P(X | E)$ 的（相关）样本，从而使我们能够估计后验边际、期望或其他感兴趣的量。

**概率图模型中的吉布斯抽样**

吉布斯抽样特别适合概率图模型，因为它利用了图结构中编码的条件独立关系。回顾一下，吉布斯抽样迭代地从给定所有其他变量当前状态的条件分布 $P(X_i | X_{-i}, E)$ 中抽取每个变量 $X_i$ 的样本。

概率图模型结构带来了一个显著的简化：给定其马尔可夫毯 $MB(X_i)$，变量 $X_i$ 条件独立于所有其他变量。马尔可夫毯包含它的父节点、子节点以及其子节点的其他父节点（有时称为“配偶”）。因此，抽样步骤简化为从以下分布中抽取样本：


$$
P(X_i | X_{-i}, E) = P(X_i | MB(X_i), E)
$$


如果从 $P(X_i | MB(X_i), E)$ 中抽样容易，这种局部计算会使吉布斯抽样高效。对于离散概率图模型或具有共轭先验的模型，情况通常如此。

> 节点 $X_i$（红色）的马尔可夫毯包含其父节点（$P_1, P_2$）、其子节点（$C_1, C_2$）以及其子节点的其他父节点（$S_1, S_2$），所有这些都以黄色突出显示。对于吉布斯抽样，我们只需要这些节点的值即可为 $X_i$ 抽取一个新状态。

**MCMC在概率图模型中的实际考量：**

- **混合性：** 采样器需要有效地在状态空间中移动。混合性差（样本之间自相关性高）会减少有效样本量。分块（将变量分组一起采样）等技术有时会有所帮助。
- **收敛性：** 评估链何时达到其平稳分布十分重要（使用第二章中讨论的迹图和 $\hat{R}$ 等诊断方法）。
- **计算成本：** 每次吉布斯扫描都需要对每个变量采样一次。成本取决于马尔可夫毯的大小以及从局部条件分布中采样的复杂性。对于非常大的图或混合缓慢的链，MCMC仍然可能需要大量计算。
- **非共轭性：** 如果条件分布 $P(X_i | MB(X_i), E)$ 不是我们可以轻松采样的标准分布（例如，在具有非共轭先验或复杂似然的模型中），我们可能需要在吉布斯采样器内嵌入 (embedding)其他采样方法，如 Metropolis-Hastings（Metropolis-within-Gibbs）。

### 变分推断在概率图模型中的应用

变分推断将推断重新定义为一个优化问题。VI 不通过采样，而是寻求在预定义族 $\mathcal{Q}$ 中找到一个分布 $Q(X)$，它能最好地近似真实的后验分布 $P(X | E)$。“最好”通常通过最小化 Kullback-Leibler (KL) 散度 $KL(Q(X) || P(X | E))$ 来衡量，这等同于最大化证据下界 (ELBO)，如第三章所述。

**平均场变分推断**

近似族 $\mathcal{Q}$ 最常见的选择是平均场族，其中分布 $Q(X)$ 在各个变量（或有时是变量组）上完全分解：


$$
Q(X) = \prod_{i=1}^{N} Q_i(X_i)
$$


这种分解假设了近似分布中变量之间的独立性，即使它们在真实后验中可能存在依赖关系。尽管这是一个强假设并且是近似误差的来源，但它带来了可处理的优化算法，例如坐标上升变分推断（CAVI）。

在平均场假设下，单个因子 $Q_j^*(X_j)$ 的最优更新由下式给出：


$$
\log Q_j^*(X_j) \propto \mathbb{E}_{Q_{-j}} [\log P(X, E)]
$$


这里，$\mathbb{E}_{Q_{-j}}[\cdot]$ 表示对所有其他因子 $Q_i(X_i)$ （其中 $i \neq j$）求期望。项 $\log P(X, E)$ 是所有变量（隐藏变量 $X$ 和观测变量 $E$）的联合概率的对数，可以从概率图模型的定义（条件概率分布或因子的乘积）中方便地得到。

重要的是，由于概率图模型结构的存在，期望 $\mathbb{E}_{Q_{-j}} [\log P(X, E)]$ 通常会简化。$\log P(X, E)$ 中涉及 $X_j$ 的项通常只依赖于 $X_j$ 及其在图中的邻居。期望只需要根据与这些邻近变量对应的因子 $Q_k(X_k)$ 来计算。CAVI 利用此规则迭代更新每个 $Q_j^*(X_j)$ 直到 ELBO 收敛。

**VI在概率图模型中的实际考量：**

- **速度：** VI 通常比 MCMC 快得多，特别是对于大型数据集，因为它用迭代优化步骤取代了采样。随机变分推断（SVI）可以通过在 ELBO 上使用随机梯度上升进一步将 VI 扩展到海量数据集。
- **偏差：** 平均场假设（或对族 $\mathcal{Q}$ 的任何限制）会引入偏差。VI 近似，特别是平均场，已知会低估后验分布的方差，并且可能无法捕捉多峰性。
- **实现复杂性：** 推导平均场更新方程需要计算相关期望，这在数学上可能比较复杂，尽管库通常会为标准模型自动化此过程。黑箱变分推断（BBVI）可以通过使用自动微分和梯度蒙特卡罗估计来缓解这个问题。
- **局部最优：** ELBO 的优化过程可能收敛到局部最优解，而不一定是全局最优解。

### 大型概率图模型中MCMC和VI的选择

MCMC和VI之间的选择很大程度上取决于具体问题、概率图模型的结构、数据规模以及对精度和速度的要求。

- **在以下情况使用MCMC：**
  - 精度至关重要，并且你需要渐近精确性的保证。
  - 计算预算允许较长的运行时间。
  - 你需要捕捉复杂的后验形状，包括多峰性和准确的方差估计。
  - 吉布斯抽样适用（条件抽样容易）。
- **在以下情况使用VI：**
  - 速度和可扩展性是主要考量，特别是对于非常大的数据集。
  - 后验的近似结果可以接受。
  - 你主要需要点估计（如后验均值）或对不确定性的大致描述。
  - 对于模型而言，推导或实现MCMC采样器过于复杂。

在许多涉及大型概率图模型的实际情况中，VI（尤其是SVI或BBVI）是一种有用的工具，可以在可接受的时间范围内获得相当好的推断结果，即使它无法达到MCMC的理论保证。理解这些高效近似推断方法之间的权衡，可以帮助你为具体的建模任务选择最合适的方法。

## 参考资料

- [Probabilistic Graphical Models: Principles and Techniques](https://mitpress.mit.edu/books/probabilistic-graphical-models) — Daphne Koller and Nir Friedman (2009)
  Publisher: MIT Press
  这本综合性教材是概率图模型的权威参考书，详细阐述了精确推断、近似方法的必要性，并深入涵盖了MCMC（包括PGM中的吉布斯采样）和变分推断。
- [Pattern Recognition and Machine Learning](https://www.springer.com/gp/book/9780387310732) — Christopher M. Bishop (2006)
  Publisher: Springer; DOI: [10.1007/978-0-387-45528-0](https://doi.org/10.1007/978-0-387-45528-0)
  一本广泛使用的教材，清晰而全面地介绍了马尔可夫链蒙特卡洛（MCMC）和变分推断的理论基础，包括均值场近似和ELBO。
- [Variational Inference: A Review for Statisticians](https://www.tandfonline.com/doi/full/10.1080/01621459.2017.1285773) — David M. Blei, Alp Kucukelbir, and Jon D. McAuliffe (2017)
  Journal: Journal of the American Statistical Association; Publisher: American Statistical Association; Volume: 112; Pages: 859-877; DOI: [10.1080/01621459.2017.1285773](https://doi.org/10.1080/01621459.2017.1285773)
  这篇极具影响力的综述文章全面概述了变分推断，涵盖了其理论基础、均值场近似以及随机变分推断等现代进展。
- [Bayesian Data Analysis](https://www.taylorfrancis.com/books/9780429113079) — Andrew Gelman, John B. Carlin, Hal S. Stern, David B. Dunson, Aki Vehtari, and Donald B. Rubin (2013)
  Publisher: CRC Press; Pages: 675; DOI: [10.1201/b16018](https://doi.org/10.1201/b16018)
  这本贝叶斯统计学的基础著作详细介绍了马尔可夫链蒙特卡洛方法，包括吉布斯采样、实际实现考量和收敛诊断，对于可靠的近似推断至关重要。
