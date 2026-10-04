---
course: "meta-learning-foundation-models"
chapter: "meta-learning-foundations-revisited"
lesson: "meta-learning-taxonomy-advanced"
sourceId: 4298
sourceUrl: "https://apxml.com/zh/courses/meta-learning-foundation-models/chapter-1-meta-learning-foundations-revisited/meta-learning-taxonomy-advanced"
title: "元学习方法分类"
description: "元学习算法分类：基于梯度、基于度量和基于优化的观点。"
order: 2
plots: []
sourceHash: "088e329003be530d09b40a504d3330f5c635346f1f043e060aa6c4088373b390"
sourceCorrections: []
---

元学习问题的特点是其结构，涉及任务、支持集 ($S_i$) 和查询集 ($Q_i$)。为了管理为解决此问题而设计的各种算法，有必要进行系统性分类。元学习算法通常根据其实现快速适应的核心机制分为三大类。这些类别提供了一个有用的结构，尽管有些算法可能包含不止一个类别的元素。主要观点有：

1. **基于梯度的元学习：** 这些方法学习模型参数 (parameter)（通常是初始化），以便新任务能高效学习，只需使用任务特定的支持集进行少量梯度步骤。
2. **基于度量的元学习：** 这些方法学习一个嵌入 (embedding)函数或一个度量空间，其中同类别的示例会聚拢，从而可以在查询集上根据与支持集示例的相似性进行分类或回归。
3. **基于优化的元学习：** 这一类别将元学习明确地构建为一个优化问题，通常涉及学习优化器本身或相关的超参数 (hyperparameter)，以促进在新任务上的快速学习。

让我们更详细地分析每个类别。

> 常见的元学习方法分类，基于它们实现快速适应的主要机制。

## 基于梯度的元学习

这一类算法旨在找到一组初始模型参数 (parameter) $\theta$，这些参数对基于新任务数据的变化高度敏感。元目标通常是在使用相应的支持集 $S_i$ 进行少量梯度更新后，最小化查询集 $Q_i$ 上的损失。

最有代表性的例子是**模型无关元学习 (MAML)**。在MAML中，对于元训练期间采样的每个任务 $T_i$，内循环在支持集 $S_i$ 上执行一步或多步标准梯度下降 (gradient descent)，以获得任务适应的参数 $\theta_i'$。


$$
\theta_i' = \theta - \alpha \nabla_{\theta} \mathcal{L}_{S_i}(\theta)
$$


外循环然后通过对这一内部更新步骤进行微分，并使用查询集 $Q_i$ 上的表现，来更新初始参数 $\theta$。这需要计算二阶导数（梯度的梯度），通常称为元梯度。


$$
\theta \leftarrow \theta - \beta \nabla_{\theta} \sum_{T_i} \mathcal{L}_{Q_i}(\theta_i') = \theta - \beta \sum_{T_i} \nabla_{\theta} \mathcal{L}_{Q_i}(\theta - \alpha \nabla_{\theta} \mathcal{L}_{S_i}(\theta))
$$


变体，例如**一阶MAML (FOMAML)** 和 **Reptile**，通过忽略二阶项来简化此过程，大幅降低了计算成本，但可能改变优化动态。**隐式MAML (iMAML)** 使用隐式微分来更稳定和高效地计算元梯度，特别是对于许多内部步骤的情况。

对于基础模型，$\theta$ 的高维度使得计算和存储完整的二阶梯度极其昂贵。这促使使用一阶近似或专门为大规模设计的技术，我们将在第2章分析这些。

## 基于度量的元学习

度量方法不优化参数 (parameter)以进行梯度更新，而是学习一个嵌入 (embedding)函数 $f_{\phi}$，将输入映射到一个空间，在该空间中，相似性对应于类别成员资格或任务关联。适应通常涉及在这个学习到的空间中，使用距离度量（例如，欧几里得距离）或一个学习到的相似性函数来比较查询示例和支持示例。

**原型网络**是一个知名的例子。它们通过平均其示例的嵌入，为支持集 $S_i$ 中存在的每个类别 $k$ 计算一个“原型”表示 $c_k$：$c_k = \frac{1}{|S_{i,k}|} \sum_{x_j \in S_{i,k}} f_{\phi}(x_j)$。查询点 $x_q$ 的分类然后通过对原型距离的softmax进行：


$$
p(y=k | x_q) = \frac{\exp(-d(f_{\phi}(x_q), c_k))}{\sum_{k'} \exp(-d(f_{\phi}(x_q), c_{k'}))}
$$


其他值得注意的方法包括**匹配网络**（使用注意力机制 (attention mechanism)计算支持集示例的加权组合进行查询预测），以及**关系网络**（采用一个独立的神经网络 (neural network)模块来学习查询和支持示例嵌入之间的非线性相似性得分）。

这些方法通常严重依赖于学习到的嵌入函数 $f_{\phi}$ 的质量。当使用基础模型时，可以直接运用强大的预训练 (pre-training)表示，或在元训练期间微调 (fine-tuning)嵌入函数。将这些技术适应到来自基础模型的极高维度嵌入带来独特的挑战，这些将在第3章进一步讨论。

## 基于优化的元学习

这种观点更广泛地将元学习视为学习优化过程本身的各个方面。这些方法不只是学习一个初始参数 (parameter)集 $\theta$，还可能学习更新规则、学习率或其他优化超参数 (hyperparameter)。

一种方法是**学习优化 (L2O)**，其中一个独立的神经网络 (neural network)（元学习器，通常是像LSTM这样的RNN）被训练来输出基础模型（学习器）的参数更新。元学习器观察学习器的梯度和状态，并提出旨在快速最小化任务损失的更新 $\Delta \theta_t$。


$$
\theta_{t+1} = \theta_t + \Delta \theta_t \quad \text{其中} \quad \Delta \theta_t = g_{\psi}(\nabla_{\theta_t} \mathcal{L}(\theta_t), h_t)
$$


在这里，$g_{\psi}$ 是由 $\psi$ 参数化的元学习器，且 $h_t$ 是其内部状态。参数 $\psi$ 在许多任务上训练，以产生高效的优化轨迹。

此外，许多元学习算法，包括MAML，都可以从形式上通过**双层优化**的角度来理解。外层优化调整元参数（例如MAML中的初始权重 (weight) $\theta$，或L2O中的元学习器 $\psi$），以最小化一个外层目标（例如，适应后的查询集损失）。内层优化使参数适应特定任务（例如，最小化支持集损失）。这种观点，将在第4章讨论，为分析和开发元学习算法提供了一个强大的体系。学习最优初始化，尽管通常与基于梯度的方法相关联，但本质上也是一个在这个体系内解决的优化问题。

## 选择方法

每个类别都有优缺点。基于梯度的方法通过梯度直接优化适应性能，但计算量可能很大，且对优化细节敏感。基于度量的方法通常更易于实现，且在测试时计算成本更低，但严重依赖于嵌入 (embedding)空间的质量和所选的度量。基于优化的方法在调整学习过程方面提供灵活性，但训练和分析可能很复杂。

选择通常取决于具体问题、任务的性质、计算预算，以及目标主要是快速适应现有参数 (parameter)还是从头学习有效的表示。在基础模型的背景下，可扩展性和预训练 (pre-training)知识的高效利用成为主要考虑因素，影响不同方法的适用性，并促使采用混合策略或专门的适应技术，正如我们将在后面的章节中看到。理解这些基本类别对于掌握有效适应大型模型所需的先进技术是必要的。

## 参考资料

- [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400) — Chelsea Finn, Pieter Abbeel, Sergey Levine (2017)
  Journal: Proceedings of the 34th International Conference on Machine Learning (ICML); Volume: 70; Pages: 1126-1135; DOI: [10.48550/arXiv.1703.03400](https://doi.org/10.48550/arXiv.1703.03400)
  介绍了模型无关元学习 (MAML)，一种基于梯度的元学习算法。
- [Matching Networks for One Shot Learning](https://arxiv.org/abs/1606.04080) — Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Koray Kavukcuoglu, and Daan Wierstra (2016)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 29; DOI: [10.48550/arXiv.1606.04080](https://doi.org/10.48550/arXiv.1606.04080)
  介绍了匹配网络，一种通过比较查询示例与支持集示例进行少样本学习的基于注意力的方法。
- [Learning to learn by gradient descent by gradient descent](https://arxiv.org/abs/1606.04474) — Marcin Andrychowicz, Misha Denil, Sergio Gomez, Matthew W. Hoffman, David Pfau, Tom Schaul, Brendan Shillingford, Nando de Freitas (2016)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 29; DOI: [10.48550/arXiv.1606.04474](https://doi.org/10.48550/arXiv.1606.04474)
  提出了一种基于优化的元学习方法，其中一个LSTM元学习器为基础模型学习优化算法。
- [On First-Order Meta-Learning Algorithms](https://arxiv.org/abs/1803.02999) — Alex Nichol, Joshua Achiam, and John Schulman (2018)
  Journal: arXiv preprint arXiv:1803.02999; DOI: [10.48550/arXiv.1803.02999](https://doi.org/10.48550/arXiv.1803.02999)
  描述了Reptile，一种简单且可扩展的一阶元学习算法，与模型无关元学习相关。
