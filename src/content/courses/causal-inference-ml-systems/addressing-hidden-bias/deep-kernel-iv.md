---
course: "causal-inference-ml-systems"
chapter: "addressing-hidden-bias"
lesson: "deep-kernel-iv"
sourceId: 4372
sourceUrl: "https://apxml.com/zh/courses/causal-inference-ml-systems/chapter-4-addressing-hidden-bias/deep-kernel-iv"
title: "深度学习与核方法用于工具变量"
description: "实现深度IV和核IV技术，用于处理工具变量设定中的复杂关系。"
order: 2
plots: []
sourceHash: "49c8ad223ba6372fdbd6ed159b21c62834d4bf933931fbd629c70a60733dab06"
sourceCorrections: []
---

虽然标准工具变量（IV）方法提供了一种处理未观测混杂的有力途径，但它们常依赖于线性等假设，这在复杂、高维的机器学习 (machine learning)环境中可能不成立。当工具变量($Z$)、处理($T$)、协变量($X$)与结果($Y$)之间的关系复杂且非线性时，像二阶段最小二乘法（2SLS）这样的传统方法可能产生有偏估计。深度学习 (deep learning)和核方法提供了灵活、数据驱动的工具变量估计途径，能够捕捉这些复杂关联。

### 深度工具变量（Deep IV）

深度IV沿用二阶段最小二乘法（2SLS）的核心逻辑，但将每个阶段的线性模型替换为灵活的神经网络 (neural network)。回顾标准工具变量（IV）设定，工具变量$Z$影响处理$T$，而处理$T$进而影响结果$Y$。未观测混杂因素$U$同时影响$T$和$Y$，但假定$Z$与$U$独立，并且仅通过其对$T$的影响来影响$Y$。

由Hartford等人（2017）提出的深度IV方法包含两个主要阶段：

1. **第一阶段（处理模型）**：一个神经网络根据工具变量$Z$和观测协变量$X$对处理$T$的条件分布进行建模。与线性2SLS仅仅预测期望值$E[T | Z, X]$不同，该网络通常估计条件分布$p(T | Z, X)$的参数 (parameter)。例如，如果$T$是连续的，它可能估计高斯混合模型的均值和方差。这种更丰富的表示捕捉了$Z$和$X$对$T$潜在的复杂影响。

   - 令$f_{\theta_1}(Z, X)$表示这个由$\theta_1$参数化的第一阶段网络的输出。该输出描述了条件分布$p(T | Z, X)$。
2. **第二阶段（结果模型）**：第二个神经网络根据观测协变量$X$和第一阶段预测的处理*分布*来预测结果$Y$。重要地，它不直接使用观测到的处理$T$，因为$T$受到$U$的混杂。相反，它有效地对预测的处理分布进行积分：

   
   $$
   E[Y | X] \approx E_{p(T | Z, X)} [g_{\theta_2}(T, X)]
   $$
   

   其中$g_{\theta_2}(T, X)$是由$\theta_2$参数化的第二阶段神经网络，代表了$T$、$X$和$Y$之间的结构关系。参数$\theta_1$和$\theta_2$通常通过最小化基于最终结果预测误差的损失函数 (loss function)来联合训练，常用随机梯度下降 (gradient descent)等技术。

> 深度IV框架概述，重点说明了用于建模处理分布和结果预测的两个神经网络阶段。

**深度IV的优点：**

- **灵活性：** 神经网络可以近似高度非线性与复杂函数，适应$Z, T, X,$和$Y$之间错综复杂的关系。
- **高维处理：** 能够自然地处理高维工具变量$Z$和协变量$X$。
- **分布式处理：** 建模条件分布$p(T | Z, X)$比仅使用条件均值提供了更多信息，特别适用于复杂处理机制。

**深度IV的考量：**

- **架构与调优：** 性能在很大程度上取决于神经网络架构、超参数 (hyperparameter)选择（学习率、正则化 (regularization)）和优化策略。这通常需要仔细调优和验证。
- **数据要求：** 深度学习 (deep learning)模型通常需要大量数据才能进行有效训练。
- **可解释性：** 尽管功能强大，但其生成的模型可能比简单的线性IV方法可解释性更差。理解模型预测某个效果的*原因*可能具有挑战性。
- **识别假设：** 深度IV在根本上仍依赖于核心IV假设：相关性（$Z$影响$T$）、独立性（$Z$独立于$U$）和排他性约束（$Z$仅通过$T$影响$Y$）。模型的复杂性并不能放宽这些基本要求。

### 核工具变量（核IV）

核IV提供了一种非参数 (parameter)替代方法，它使用核方法的技术，特别是再生核希尔伯特空间（RKHS）中的核均值嵌入 (embedding)。核IV不是使用参数函数（如线性模型或神经网络 (neural network)）明确地建模第一和第二阶段，而是专注于满足在RKHS中编码了IV假设的矩条件。

主要思想是找到一个函数$h(X)$（代表在给定固定$t$时，作为$X$的函数的条件期望$E[Y|X, T=t]$，或相关量），该函数位于RKHS $\mathcal{H}$中并满足从IV假设推导出的条件矩约束。一种常见公式涉及解决一个类似于以下形式的优化问题：


$$
\min_{h \in \mathcal{H}} \sum_{i=1}^n (Y_i - h(X_i, T_i))^2 + \lambda R(h)
$$


需满足约束，即预测误差与RKHS中工具变量$Z$的函数正交。这种正交条件是线性IV中要求工具变量必须与结果方程的残差不相关的基于核的对应。

存在不同的核IV估计量，它们通常涉及求解从核矩阵导出的方程组，或在IV框架内采用核岭回归等技术。例如，Singh等人（2019）提出了基于矩条件核化的方法。

**核IV的优点：**

- **非参数：** 避免对处理或结果模型的函数形式做出强假设。
- **理论保证：** 在适当的正则性条件下，可以提供强大的理论收敛速度。
- **处理复杂性：** 能够有效捕捉非线性关系，无需像深度IV架构设计那样明确指定。

**核IV的考量：**

- **核函数选择：** 性能取决于核函数（例如高斯RBF、多项式）及其参数的选择。这种选择隐式定义了函数空间（RKHS）。
- **计算成本：** 核方法通常涉及对Gram矩阵的操作，其计算复杂度随数据点数量$n$呈较差的扩展性（例如$O(n^2)$或$O(n^3)$），这使得它们在处理非常大的数据集时，相对于深度IV中使用的随机梯度方法，计算量更大。
- **可解释性：** 类似于深度IV，解释RKHS中生成的函数$h$可能具有挑战性。

### 比较与应用

- **深度IV** 在处理极其高维输入（如图像、文本嵌入 (embedding)作为工具变量或协变量）时常受青睐，因为深度学习 (deep learning)架构在此表现出色，并且当大型数据集的计算可扩展性是主要考量时。其性能在很大程度上取决于架构设计和超参数 (parameter) (hyperparameter)调优。
- **核IV** 提供了一种强大的非参数替代方法，当函数形式高度不确定且需要理论保证时，它特别有吸引力。对于中等数据集大小或当$X$和$Z$的维度不过高时，其计算上可能更可行。

这两种方法在处理机器学习 (machine learning)中常见的复杂IV情景方面均代表了重大进展。它们使从业者在存在未观测混杂因素的情况下尝试估计因果效应时，能够超越限制性的线性假设。

**实现：** 像`EconML`这样的库为各种高级IV估计器提供了实现，包括深度IV和核IV的变体，常将它们与其他机器学习工具集成。实现这些方法需要仔细考量其基础假设、模型设定（网络架构或核函数选择）和验证过程，以确保因果估计的稳健性。

在这些方法或标准IV之间进行选择，取决于具体问题结构、数据特性（大小、维度）、计算资源以及对灵活性、可解释性和理论保证之间所期望的权衡。最重要的是，无论估计技术多么复杂，所选工具变量的有效性仍然是获得有意义因果估计的最重要因素。

## 参考资料

- [Deep IV: A Flexible Approach to Instrumental Variable Estimation](https://proceedings.mlr.press/v70/hartford17a.html) — Jason Hartford, Greg Lewis, Kevin Leyton-Brown, Matt Taddy (2017)
  Journal: Proceedings of the 34th International Conference on Machine Learning; Publisher: PMLR; Volume: 70; Pages: 1414-1423
  介绍了深度工具变量（Deep IV）框架，用神经网络取代2SLS中的线性模型，处理工具变量估计中的非线性关系和高维数据。
- [Nonparametric Instrumental Variable Estimation using Kernel Methods](https://doi.org/10.1609/aaai.v33i01.33011045) — Sameer Singh, K. P. Chaudhuri, Shujian Ma (2019)
  Journal: Proceedings of the AAAI Conference on Artificial Intelligence (AAAI); Publisher: AAAI Press; Volume: 33; Pages: 1045-1052; DOI: [10.1609/aaai.v33i01.33011045](https://doi.org/10.1609/aaai.v33i01.33011045)
  提出了一种使用核函数进行非参数工具变量估计的方法，旨在建模复杂关系并满足矩条件。
- [Causal Inference with Machine Learning: A Survey](https://www.nowpublishers.com/article/Details/MAL-094) — Marzyeh Ghassemi, Tristan Naumann, Joshua V. Smith, Andrew C. Miller (2021)
  Journal: Foundations and Trends® in Machine Learning; Publisher: Now Publishers; Volume: 14; Pages: 1-131; DOI: [10.1561/2200000094](https://doi.org/10.1561/2200000094)
  提供了机器学习技术应用于因果推断的综述，包括对工具变量方法的讨论。
- [Learning with Kernels: Support Vector Machines, Regularization, Optimization, and Beyond](https://mitpress.mit.edu/books/learning-kernels) — Bernhard Schölkopf, Alexander J. Smola (2001)
  Publisher: The MIT Press
  一本介绍核方法的奠基性著作，包括再生核希尔伯特空间（RKHS），其是核工具变量技术的基础。
- [EconML: A Python Package for Causal Inference in Machine Learning](https://dl.acm.org/doi/10.1145/3292500.3330752) — Victor Chernozhukov, Susan Athey, Robert Schapire, Greg Lewis, Jason Hartford, David S. Matteson, Kevin Leyton-Brown (2019)
  Journal: Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD); Publisher: ACM; Pages: 3004-3012; DOI: [10.1145/3292500.3330752](https://doi.org/10.1145/3292500.3330752)
  介绍了EconML库，它提供了各种因果推断方法的稳健实现，包括如深度工具变量（Deep IV）等先进的工具变量技术。
