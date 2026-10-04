---
course: "meta-learning-foundation-models"
chapter: "meta-learning-foundations-revisited"
lesson: "meta-learning-problem-formulation"
sourceId: 4294
sourceUrl: "https://apxml.com/zh/courses/meta-learning-foundation-models/chapter-1-meta-learning-foundations-revisited/meta-learning-problem-formulation"
title: "元学习问题定义"
description: "详细定义了元训练、元测试、任务、支持集和查询集，以便更好地理解。"
order: 1
plots: []
sourceHash: "484075edfd6526189b0f56af61c65c7ff45df02c97662f941ba3c500879f2bce"
sourceCorrections: []
---

元学习从根本上改变了学习目标，从掌握单一任务转变为迅速高效地学习*新*任务的能力。与传统监督学习 (supervised learning)中模型为一项特定工作在一个大型数据集上训练不同，元学习在任务分布上运行。其目标是获取可迁移的知识或学习策略，以加速面对新任务时的适应，特别是当该新任务的数据稀缺时。这通常被描述为“学会学习”。

### 元学习设置：任务、支持集和查询集

元学习问题的核心是**任务**。一个任务 $T$ 代表一个具体的学习问题，例如分类一组新图像、在一种新的语言对之间进行翻译，或使语言模型适应独特的写作风格。元学习中的任务是从一个潜在的概率分布 $p(T)$ 中抽取的。此分布定义了元学习算法应处理的问题范围。

元学习过程通常包含两个阶段：

1. **元训练：** 算法接触到一系列任务，$D_{meta-train} = \{T_1, T_2, ..., T_N\}$，这些任务从 $p(T)$ 中采样。此阶段的目标不是完美掌握任何单个任务 $T_i$，而是学习一个通用模型初始化、一个学习过程或一个度量空间，以促进在从相同分布中抽取的*未来*任务上的快速学习。
2. **元测试：** 所学模型或过程在另一组*新*任务上进行评估，$D_{meta-test} = \{T'_1, T'_2, ..., T'_M\}$，这些任务也从 $p(T)$ 中采样，且在元训练期间未见过。在这些任务上的表现衡量了算法泛化其所学学习策略的能力。

### 元学习任务的构成

重要的一点是，元训练或元测试集中的每个独立任务 $T_i$ 本身都构成一个小型学习问题。它包含两个不同的数据子集：

- **支持集 ($S_i$)：** 这是一小组特定于任务 $T_i$ 的标记 (token)样本。其目的是提供模型在元学习过程中（通常在“内循环”中）*适应*或*学习*此特定任务细节所需的数据。对于 K-shot, N-way 分类任务，支持集通常包含 $N$ 个类别中的每个类别的 $K$ 个样本。形式上，$S_i = \{(x_{i,k}, y_{i,k})\}_{k=1}^{|S_i|}$。 $S_i$ 的大小有意保持较小，以体现少样本学习 (few-shot learning)的情形。
- **查询集 ($Q_i$)：** 这是来自*同一*任务 $T_i$ 的另一组标记样本，与支持集不相交 ($S_i \cap Q_i = \emptyset$)。其目的是在模型使用支持集 $S_i$ 适应后，评估其表现。在查询集上计算的损失通常驱动元参数 (parameter)的优化（“外循环”更新）。形式上，$Q_i = \{(x'_{i,j}, y'_{i,j})\}_{j=1}^{|Q_i|}$。

"每个任务内部划分为支持集和查询集非常重要。它在元训练期间模拟了少样本情形：模型必须*从*支持集学习，以便在该特定任务上*在*查询集上表现良好。"

> 元训练阶段单一步骤内的数据流。采样一个任务 $T_i$，将其分为支持集 ($S_i$) 和查询集 ($Q_i$)。使用 $S_i$ 适应元模型参数 ($\theta$) 以生成任务特定参数 ($\phi_i$)。随后在 $Q_i$ 上评估表现，所得损失指导元参数 $\theta$ 的更新。

### 元目标

设 $\theta$ 表示元学习模型的参数 (parameter)或定义我们学习过程的参数（例如，神经网络 (neural network)的初始权重 (weight)、优化器的参数）。使用支持集 $S_i$ 将这些通用参数 $\theta$ 适应到任务特定参数 $\phi_i$ 的过程可以用函数或算法 $\text{适应}$ 表示。因此，$\phi_i = \text{适应}(\theta, S_i)$。

元训练的最终目标是找到最优的元参数 $\theta^*$，使得在使用相应支持集 $S_i$ 适应后，在任务分布 $p(T)$ 上的查询集 $Q_i$ 的预期损失最小。如果 $L(Q_i, \phi_i)$ 表示适应模型 $\phi_i$ 在查询集 $Q_i$ 上的损失（例如，交叉熵、均方误差），元目标可以正式表述为：


$$
\theta^* = \arg\min_{\theta} \mathbb{E}_{T_i \sim p(T)} [ L(Q_i, \phi_i) ]
$$


$$
\theta^* = \arg\min_{\theta} \mathbb{E}_{T_i \sim p(T)} [ L(Q_i, \text{适应}(\theta, S_i)) ]
$$


在实践中，这个期望通过对每个元训练迭代中采样的一批任务的查询集损失求平均来近似。

### 与基础模型的关系

这种表述直接适用于使用大型基础模型的情况。在这里，$\theta$ 代表基础模型可能极其庞大的参数 (parameter)集（例如，Transformer）。 `适应` 函数可以是：

- 在 $S_i$ 上进行微调 (fine-tuning)（全部或部分 $\theta$）。
- 使用 $S_i$ 学习参数高效模块（如适配器或 LoRA 矩阵），同时保持基础 $\theta$ 固定。
- 使用 $S_i$ 的非参数方法（如对从 $\theta$ 派生的嵌入 (embedding)进行 k-最近邻）。
- 一种基于梯度的元学习算法（如 MAML）在 $S_i$ 上执行模拟适应步骤。

无论具体的 `适应` 机制是什么，元学习目标保持一致：找到初始参数 $\theta$（或生成它们的方法），使得模型在仅看到小支持集 $S_i$ 后，能在查询集 $Q_i$ 上表现良好。挑战在于，鉴于基础模型中 $\theta$ 的巨大规模，如何高效地执行元优化（找到 $\theta^*$）和适应（$\text{适应}(\theta, S_i)$），这是本课程中持续关注的一个主要议题。在查看旨在解决此问题的特定算法之前，理解这种核心问题结构非常重要。

## 参考资料

- [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://proceedings.mlr.press/v70/finn17a.html) — Chelsea Finn, Pieter Abbeel, Sergey Levine (2017)
  Journal: Proceedings of the 34th International Conference on Machine Learning; Publisher: PMLR; Volume: 70; Pages: 1126-1135; DOI: [10.48550/arXiv.1703.03400](https://doi.org/10.48550/arXiv.1703.03400)
  提出了一种通用的元学习算法，通过学习一组良好的初始参数，模型只需少量梯度步即可快速适应新任务。它阐明了内循环和外循环优化过程。
- [Prototypical Networks for Few-Shot Learning](https://neurips.cc/paper/2017/hash/2d9b2ed4f2c1cfbd37dd24d0ddb239f6-Abstract.html) — Jake Snell, Kevin Swersky, Richard Zemel (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: Neural Information Processing Systems Foundation, Inc. (NeurIPS); Volume: 30; DOI: [10.48550/arXiv.1703.05175](https://doi.org/10.48550/arXiv.1703.05175)
  介绍了一种基于度量的元学习方法，通过计算到每个类别原型表示的距离进行分类，这些原型从支持集示例中导出。这展示了支持集和查询集的使用。
- [Learning to learn by gradient descent by gradient descent](https://papers.nips.cc/paper/2016/hash/c5ff2543b53f4cc0ad3819a5b0aedee5-Abstract.html) — Marcin Andrychowicz, Misha Denil, Sergio Gómez, Matthew W Hoffman, David Pfau, Tom Schaul, Brendan Shillingford, Nando de Freitas (2016)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; Volume: 29; Pages: 3981-3989; DOI: [10.48550/arXiv.1606.04474](https://doi.org/10.48550/arXiv.1606.04474)
  提出使用神经网络学习优化器（元学习器），在深度学习背景下展示了“学习如何学习”的理念。它将元学习问题设定为优化一个优化过程。
