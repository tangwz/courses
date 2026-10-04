---
course: "bayesian-machine-learning"
chapter: "variational-inference-techniques"
lesson: "comparing-mcmc-vi"
sourceId: 3578
sourceUrl: "https://apxml.com/zh/courses/bayesian-machine-learning/chapter-3-variational-inference-techniques/comparing-mcmc-vi"
title: "MCMC与VI优势比较"
description: "分析MCMC与VI在准确性、速度和可扩展性方面的权衡。"
order: 8
plots: []
sourceHash: "02dc699584364e91a2ea6053f60613ae7a6d01048c5f32df77ac0eebc9df3621"
sourceCorrections: []
---

变分推断（VI）将近似难以处理的后验分布视为一个优化问题，通常旨在最大化证据下界（ELBO）。马尔可夫链蒙特卡洛（MCMC）方法是近似这些分布的另一种方法。对这两种主要的后验分布近似技术进行比较，可以展现它们各自的优势和不足。VI和MCMC都旨在近似通常难以处理的后验分布 $p(\mathbf{z}|\mathbf{x})$，但它们实现此目的的方式截然不同。MCMC和VI的选择在很大程度上取决于具体的问题、模型复杂度、数据集大小以及后验近似所需的准确程度。

### 近似方法的性质

- **MCMC:** 这些方法是随机模拟技术。诸如Metropolis-Hastings、Gibbs采样、HMC和NUTS等算法会生成一系列样本 $(\mathbf{z}^{(1)}, \mathbf{z}^{(2)}, ..., \mathbf{z}^{(T)})$，其分布渐进地收敛到真实的后验 $p(\mathbf{z}|\mathbf{x})$。理论上，在无限计算时间下，MCMC能提供后验的精确表示。实际操作中，我们使用有限数量的样本，这会引入近似误差，并且必须仔细诊断收敛情况。
- **VI:** 这是一种确定性优化技术（尽管SVI在优化过程中引入了随机性，而非目标本身）。VI在预定义的族 $\mathcal{Q}$（例如，平均场高斯）中寻找一个最优分布 $q^*(\mathbf{z})$，使其最小化KL散度 $KL(q(\mathbf{z}) || p(\mathbf{z}|\mathbf{x}))$。这等同于最大化ELBO。近似质量本质上受到族 $\mathcal{Q}$ 表达能力的限制。如果真实的后验不能被任何 $q \in \mathcal{Q}$ 很好地表示，那么即使计算无限，VI也会提供一个有偏的近似。

### 准确性和不确定性量化 (quantization)

- **MCMC:** 优势在于其高准确性的潜力。在足够长的运行时间和成功收敛的情况下，样本能提供真实后验的丰富经验表示，捕捉复杂的形状、多峰性以及参数 (parameter)间的相关性。使用样本统计数据计算后验均值、方差、可信区间或任何其他期望通常是直接的。
- **VI:** 准确性受到所选变分族的限制。常见的平均场近似假设潜在变量（或变量组）之间存在后验独立性，$q(\mathbf{z}) = \prod_i q_i(z_i)$。这个假设常常无法捕捉真实后验中存在的关联。因此，VI，特别是平均场VI，倾向于低估后验分布的方差，并可能遗漏多峰性。尽管更复杂的族（例如，归一化 (normalization)流）可以提高准确性，但它们会增加计算和实现的复杂性。

### 计算成本与可扩展性

- **MCMC:** 计算量可能很大。每一步通常需要评估似然和先验，而HMC等方法还需要梯度计算。生成大量*独立*样本可能需要很长时间，因为MCMC样本之间存在相关性。扩展到非常大的数据集（$N \gg 1$）具有挑战性，尽管存在随机梯度MCMC（SG-MCMC）等方法来解决这个问题。并行化通常仅限于运行多个独立的链。
- **VI:** 通常比MCMC快得多，特别是对于大型模型或数据集。优化算法，特别是SVI中使用的随机梯度方法，可以利用小批量处理，非常适合大N的情况。与传统MCMC相比，其计算成本通常随数据量增长而表现更佳。优化程序有时可以比复杂的MCMC采样器更容易地从硬件加速（GPU）中获益。

### 实现和调优的便捷性

- **MCMC:** 如果条件分布简单，实现Metropolis-Hastings或Gibbs等基本采样器可能很简单。然而，HMC和NUTS等高效采样器更为复杂，需要仔细调整参数 (parameter)（步长、步数、质量矩阵），尽管PyMC或Stan等现代概率编程语言（PPL）已将大部分工作自动化。诊断收敛是一个必要且有时不简单的步骤。
- **VI:** 分析推导CAVI的更新方程可能繁琐且与模型紧密相关。SVI和BBVI，利用TensorFlow Probability或Pyro等框架中提供的自动微分工具，减轻了手动推导的需求。然而，VI带来了与优化相关的挑战：选择合适的学习率、优化器，并可能处理局部最优问题。收敛通常通过ELBO进行监控，ELBO表明了优化进展，但并不能直接保证后验近似相对于真实后验的质量。

### 诊断

- **MCMC:** 受益于一套相对成熟的收敛诊断方法。通过可视化检查迹线图、自相关图以及诸如潜在尺度缩减因子（$\hat{R}$）和有效样本量（ESS）等定量测量，有助于评估采样器是否收敛以及它对后验的遍历效率。
- **VI:** 诊断方法标准化程度较低。监控ELBO可确保优化过程已收敛，但收敛的ELBO仅表示在所选族 $\mathcal{Q}$ *内*找到了最佳近似。它不直接衡量 $q^*(\mathbf{z})$ 与真实后验 $p(\mathbf{z}|\mathbf{x})$ 的距离。评估VI近似的质量通常需要后验预测检查，或者在可行的情况下，将其结果与MCMC在较小数据子集上的结果进行比较。

### 总结表

| 特点 | 马尔可夫链蒙特卡洛 (MCMC) | 变分推断 (VI) |
| --- | --- | --- |
| **方法** | 采样 (随机模拟) | 优化 (确定性/随机梯度) |
| **目标** | 从真实后验 \$p(\mathbf{z} | \mathbf{x})\$ 渐进采样 |
| **准确性** | 潜力高 (渐进精确) | 受限于变分族 $\mathcal{Q}$；常有偏差 |
| **不确定性** | 通常能很好地捕捉方差和相关性 | 常低估方差；平均场在处理相关性方面表现不佳 |
| **速度** | 可能较慢，特别是对于复杂模型/大数据 | 通常快得多，特别是SVI |
| **可扩展性 (数据)** | 对于超大N有挑战 (SG-MCMC变体除外) | 通过小批量处理（SVI）能很好地扩展到大N |
| **实现** | 复杂采样器 (HMC/NUTS) 从头实现较难 | 手动推导 (CAVI) 可能较难；SVI/BBVI在AD帮助下更容易 |
| **调优** | 采样器参数 (parameter) (步长等)，收敛检查 | 优化参数 (学习率等)，ELBO监控 |
| **诊断** | 成熟的收敛诊断 ($\hat{R}$，ESS，迹线图) | 标准化程度较低；ELBO收敛，预测检查 |
| **并行性** | 主要通过多个独立链 | 优化通常可以并行化 |

### 何时选择哪种方法？

- **选择MCMC，当：**
  - 后验近似需要高度准确时。
  - 你需要准确捕捉后验中复杂的依赖关系或多峰性时。
  - 数据集大小和模型复杂程度允许合理的计算时间时。
  - 严谨的收敛评估是重要考量时。
- **选择VI，当：**
  - 计算速度和对大型数据集的可扩展性是主要考虑因素时。
  - 对后验有一个合理近似就足够，即使可能存在偏差（例如，低估方差）。
  - 模型符合支持SVI或BBVI等方法自动微分的框架时。
  - 它作为更大系统中的一个组成部分，且采样不切实际时（例如，某些强化学习 (reinforcement learning)或生成模型设置）。

实际上，选择并非总是严格二元的。VI有时可用于初始化MCMC采样器。此外，在较小的数据子集上比较这两种方法的结果，可以提供对后验性质和所涉及权衡的宝贵认识。理解此处概述的核心差异，使您能够根据具体的分析目标和计算限制做出明智的决定。

## 参考资料

- [Variational Inference: A Review for Statisticians](https://doi.org/10.1080/01621459.2017.1285773) — David M. Blei, Alp Kucukelbir, Jon D. McAuliffe (2017)
  Journal: Journal of the American Statistical Association; Publisher: Taylor & Francis; Volume: 112; Pages: 859-877; DOI: [10.1080/01621459.2017.1285773](https://doi.org/10.1080/01621459.2017.1285773)
  全面概述了变分推断，包括其理论基础、各种算法（如平均场、随机变分推断）和局限性，有助于理解变分推断的优点和缺点。
- [The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo](http://www.jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf) — Matthew D. Hoffman and Andrew Gelman (2014)
  Journal: Journal of Machine Learning Research; Volume: 15; Pages: 1593-1623
  介绍了No-U-Turn Sampler (NUTS)，这是一种高效且广泛使用的MCMC算法，它自适应地调整哈密顿蒙特卡洛的参数，解决了MCMC效率方面的问题。
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Diederik P Kingma, Max Welling (2013)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  引入了重参数化技巧，这是随机变分推断（SVI）等现代变分推断方法的关键技术，通过自动微分实现了有效的梯度优化。
- [Pattern Recognition and Machine Learning](https://www.springer.com/gp/book/9780387310732) — Christopher M. Bishop (2006)
  Publisher: Springer; DOI: [10.1007/978-0-387-44933-2](https://doi.org/10.1007/978-0-387-44933-2)
  一本经典教材，全面涵盖了马尔可夫链蒙特卡洛方法（第11章）和变分推断（第10章），为比较这两种方法提供了基础知识。
