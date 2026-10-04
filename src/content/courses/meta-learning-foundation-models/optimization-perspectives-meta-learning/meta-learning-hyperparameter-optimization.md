---
course: "meta-learning-foundation-models"
chapter: "optimization-perspectives-meta-learning"
lesson: "meta-learning-hyperparameter-optimization"
sourceId: 4345
sourceUrl: "https://apxml.com/zh/courses/meta-learning-foundation-models/chapter-4-optimization-perspectives-meta-learning/meta-learning-hyperparameter-optimization"
title: "超参数优化间的关联"
description: "理解元学习与自动化超参数调优之间的关系。"
order: 3
plots: []
sourceHash: "b71df0389cc89c50e663cbd74e1052713085caec0a6f16e767552fca168a1cee"
sourceCorrections: []
---

从优化角度看元学习，它与超参数 (parameter) (hyperparameter)优化 (HPO) 展现出密切相似性。这两个方面都旨在优化控制学习过程本身的参数，而非直接优化单个任务的模型参数。当我们考虑许多元学习方法固有的双层优化结构时，这种联系变得尤为清晰。

### 双层优化类比

回顾之前介绍的双层优化一般形式：


$$
\min_{\lambda} F(\lambda, \theta^*(\lambda)) \quad \text{约束条件是} \quad \theta^*(\lambda) = \arg\min_{\theta} f(\lambda, \theta)
$$


这里，$\lambda$ 代表外层变量，$\theta$ 代表内层变量。$F$ 是外层目标函数，$f$ 是内层目标函数。

在标准的\*\*超参数 (parameter) (hyperparameter)优化 (HPO)\*\*中，目标通常是找到超参数 $\lambda$ (例如，学习率、正则化 (regularization)强度、架构选择)，这些超参数在模型 $\theta^*(\lambda)$ 使用它们在训练数据集上训练后，能最小化验证损失 $F$。内循环最小化训练损失 $f$：

- **外层变量 $\lambda$：** 超参数（学习率 $\alpha$、正则化 $\beta$ 等）。
- **内层变量 $\theta$：** 模型参数。
- **外层目标函数 $F$：** 在验证集上的表现（例如，验证损失）。
- **内层目标函数 $f$：** 在训练集上的表现（例如，训练损失）。


$$
\min_{\lambda} \mathcal{L}_{\text{val}}(\theta^*(\lambda); \mathcal{D}_{\text{val}}) \quad \text{约束条件是} \quad \theta^*(\lambda) = \arg\min_{\theta} \mathcal{L}_{\text{train}}(\theta; \mathcal{D}_{\text{train}}, \lambda)
$$


在**元学习**中，特别是像 MAML 这样的基于梯度的方法，其结构是类似的。目标是找到元参数 $\theta$（通常是初始参数），这些参数在模型已在对应的支持集上经过 $k$ 步适应之后，能最小化不同任务查询集上的平均损失 $F$。内循环执行这种特定于任务的适应，最小化任务损失 $f_i$：

- **外层变量 $\lambda$ (或元参数 $\theta_{\text{meta}}$)：** 通常是初始模型参数 $\theta$，但也可能包含适应学习率或其他元学习组件。
- **内层变量 $\phi$ (或任务参数 $\theta'_{\text{task}}$)：** 针对特定任务 $i$ 适应后的模型参数。
- **外层目标函数 $F$：** 在查询集 $\mathcal{Q}_i$ 上跨任务的平均表现。
- **内层目标函数 $f_i$：** 在任务 $i$ 的支持集 $\mathcal{S}_i$ 上的表现。

对于 MAML，这看起来像：


$$
\min_{\theta} \mathbb{E}_{T_i \sim p(T)} [\mathcal{L}_{\mathcal{Q}_i}(\theta'_i)] \quad \text{约束条件是} \quad \theta'_i = \text{Adapt}(\theta, \mathcal{S}_i)
$$


其中 $\text{Adapt}(\theta, \mathcal{S}_i)$ 通常涉及从 $\theta$ 开始，在 $\mathcal{L}_{\mathcal{S}_i}$ 上执行一个或多个梯度下降 (gradient descent)步骤。

> 双层结构突出显示了相似性：外循环优化控制内部学习过程的参数，并根据该内部过程的结果进行评估。

### 算法重叠

共同的双层结构意味着为一个方面开发的算法常常在另一个方面得到应用或存在对应的算法。

- **基于梯度的方法：** 通过内部优化过程计算梯度的技术是基于梯度的 HPO 和基于梯度的元学习的核心。在 HPO 中，这涉及对超参数 (parameter) (hyperparameter)的验证损失进行微分，通常需要通过训练过程进行微分（例如，使用隐式微分或通过展开的优化步骤进行反向传播 (backpropagation)）。这类似于 MAML 中元梯度的计算，后者要求通过在支持集上执行的适应步骤，对初始参数 $\theta$ 的查询集损失进行微分。像隐式 MAML (iMAML) 这样的算法直接运用与隐式微分相关的技术，这些技术在 HPO 中也得到使用。
- **黑盒 (black box)/无导数方法：** HPO 通常处理那些梯度不可用或计算不切实际的超参数（例如，离散的架构选择）。诸如贝叶斯优化、演化策略或强化学习 (reinforcement learning)等技术很常见。尽管在寻找初始参数 $\theta$ 的主流基于梯度的元学习中不那么常见，但这些优化技术可能与优化元学习的*超参数*（如元学习率、适应步骤 $k$、元学习器*内部*使用的网络架构选择）相关，或者在特定的元学习环境中，如元强化学习或元学习中的架构搜索。

### 元学习中“超参数 (parameter) (hyperparameter)”的构成？

从 HPO 的角度看，元学习外循环中正在优化的“超参数”是元参数本身。这些通常包括：

1. **模型初始参数 ($\theta$)：** MAML 和 Reptile 等算法的主要优化目标。目标是找到一个能够实现快速适应的初始参数。
2. **适应学习率 ($\alpha$)：** 一些元学习方法明确学习任务特定或每个参数的适应学习率，将其作为元参数的一部分。
3. **学习到的预处理/嵌入 (embedding)函数：** 在基于度量的元学习中，嵌入网络在外循环中学习。其参数就像控制内循环距离计算的超参数。
4. **元优化器参数：** 如果使用学习优化器（本身是参数化的“优化器函数”）进行适应，其参数在外循环中优化。

### 区别与考量

尽管有相似之处，但仍存在重要区别：

- **维度：** 元学习通常优化非常高维的元参数 (parameter)（例如，基础模型的全部初始权重 (weight) $\theta$），而 HPO 通常优化一组较小的标量或低维超参数 (hyperparameter)。
- **内循环结构：** 元学习中的内循环由支持集上的少样本适应过程明确定义，通常只涉及几个梯度步骤。HPO 中的内循环通常是在更大数据集上的完整模型训练过程。
- **目标差异：** 优化方面可能存在实质性差异。元学习方面受任务分布以及初始参数和适应动态之间相互影响。

理解与 HPO 的联系提供了一个有价值的框架。它显示元学习根本上是关于优化能够实现有效学习的*条件*或*参数*，很像 HPO 优化有效训练的条件（超参数）。来自 HPO 成熟研究的技术和见解可以启发新方法或提供分析工具，用于理解优化元学习系统的行为和难题，特别是当我们将其扩展到大型基础模型时。反过来，为处理元学习中高维外循环而开发的技术可能为特定的 HPO 问题提供见解。

## 参考资料

- [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://proceedings.mlr.press/v70/finn17a.html) — Chelsea Finn, Pieter Abbeel, Sergey Levine (2017)
  Journal: Proceedings of the 34th International Conference on Machine Learning; Publisher: PMLR; Volume: 70; Pages: 1126-1135; DOI: [10.5555/3305890.3306016](https://doi.org/10.5555/3305890.3306016)
  介绍模型无关元学习 (MAML) 的基础论文，这是一种具有双层优化结构的梯度元学习算法。
- [Gradient-Based Hyperparameter Optimization Through Reversible Learning](https://proceedings.mlr.press/v37/maclaurin15.pdf) — Dougal Maclaurin, David Duvenaud, and Ryan P. Adams (2015)
  Journal: Proceedings of the 32nd International Conference on Machine Learning (ICML); Volume: 37; Pages: 2115-2124; DOI: [10.1137/1.9781611974542.48](https://doi.org/10.1137/1.9781611974542.48)
  介绍了一种早期有效计算验证性能相对于超参数精确梯度的方法，这是梯度超参数优化的核心概念。
- [Automated Machine Learning: Methods, Systems, Challenges](https://link.springer.com/book/10.1007/978-3-030-05318-5) — Frank Hutter, Lars Kotthoff, Joaquin Vanschoren (2019)
  Publisher: Springer; DOI: [10.1007/978-3-030-05318-5](https://doi.org/10.1007/978-3-030-05318-5)
  一本关于自动化机器学习 (AutoML) 各个方面的著作，其中包含关于超参数优化 (HPO) 方法的专门章节。
- [Optimizing Millions of Hyperparameters by Implicit Differentiation](https://proceedings.neurips.cc/paper_files/paper/2016/file/332168923a19e5e7b57d4ad2287955dd-Paper.pdf) — Fabian Pedregosa (2016)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 29; Pages: 1195-1203; DOI: [10.5555/3045390.3045470](https://doi.org/10.5555/3045390.3045470)
  探讨了如何应用隐式微分来优化大量超参数，并与元学习中的高维元参数进行了比较。
