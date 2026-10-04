---
course: "bayesian-machine-learning"
chapter: "revisiting-bayesian-foundations"
lesson: "probabilistic-modeling-principles"
sourceId: 3508
sourceUrl: "https://apxml.com/zh/courses/bayesian-machine-learning/chapter-1-revisiting-bayesian-foundations/probabilistic-modeling-principles"
title: "概率建模原理"
description: "理解构建机器学习任务概率模型的核心原理。"
order: 1
plots: []
sourceHash: "bc6d27779d1f9a7af1f7dfc041ac3fb2e900b810e7d56188138dc19340e1b0f8"
sourceCorrections: []
---

正如本章引言所述，我们的目的并非仅仅是罗列定义，而是巩固概率建模的原理，从适合高级贝叶斯机器学习 (machine learning)的角度来看。从根本上说，概率建模就是运用概率论的语言，阐明数据生成方式的假设，并对不确定性进行推理 (inference)。它提供了一个连贯的数学框架，用于表达知识、结合证据和进行预测。

### 生成式视角

贝叶斯建模的一个主要思想是生成式视角。我们不直接对预测函数 $P(y|x)$ 进行建模，而是通常思考*整个数据集* $\mathcal{D}$（可能包含特征 $X$ 和目标 $Y$）是如何产生的。我们假设一个随机过程，由某些参数 (parameter) $\theta$ 和可能的隐（未观测）变量 $z$ 控制，这个过程可能生成了 $\mathcal{D}$。

这种生成式描述涉及指定所有相关量（包括已观测和未观测）的联合概率分布：$P(\mathcal{D}, \theta, z)$。构建模型意味着对此分布做出明确选择。

### 概率模型的组成部分

构建概率模型需要指定几个重要组成部分：

1. **变量：** 我们确定所涉及的不同类型的量：

   - **观测变量：** 这些构成我们的数据 $\mathcal{D}$。这可以是特征、标签、传感器读数、文本文档等。我们将其视为底层生成过程的固定结果。
   - **参数 (parameter) ($\theta$)：** 这些是未观测到的量，它们控制模型中变量之间的关系。例子包括回归系数、分布的均值和方差，或神经网络 (neural network)中的权重 (weight)。在贝叶斯方法中，参数本身被视为随机变量，具有分布。
   - **隐变量 ($z$)：** 这些是其他未观测变量，它们不是参数，而是模型结构的一部分，通常用于捕捉底层结构、簇或状态。主题模型（如第5章讨论的LDA）大量依赖表示主题分配的隐变量。
2. **结构假设：** 我们定义这些变量*之间*的概率关系。哪些变量直接影响其他变量？在给定其他变量的情况下，某些变量是否独立？这些假设定义了模型的结构，并常通过概率图模型（后面会讲到）进行可视化。条件独立性假设对于简化模型和实现高效计算尤为重要。例如，我们通常假设在给定参数的情况下，数据点是独立的。
3. **概率分布：** 我们为连接变量的概率分布选择具体的数学形式。这涉及选择：

   - **似然函数 $P(\mathcal{D} | \theta, z)$：** 此分布量化 (quantization)了在给定参数 $\theta$ 和隐变量 $z$ 的具体值下观测到数据 $\mathcal{D}$ 的概率。似然函数的选择很大程度上取决于数据的性质（例如，连续数据选择正态分布，二元结果选择伯努利分布，计数数据选择泊松分布）。
   - **先验分布 $P(\theta, z)$ 或 $P(\theta)P(z)$：** 此分布表达了我们在观测任何数据*之前*对参数和隐变量的信念。它编码了假设或融入了现有知识。我们将在本章后面讨论先验选择策略。

### 似然函数 $P(\mathcal{D} | \theta)$

我们来关注似然函数（为简洁起见，通常不明确提及隐变量 $z$，记为 $P(\mathcal{D} | \theta)$）。它直接根据在给定参数 (parameter)情况下数据点如何生成的生成式假设得出。例如，如果我们假设 $N$ 个独立同分布（i.i.d.）的数据点 $\mathcal{D} = \{x_1, ..., x_N\}$ 从分布 $f(x | \theta)$ 中抽取，则似然函数为：


$$
P(\mathcal{D} | \theta) = \prod_{i=1}^{N} f(x_i | \theta)
$$


请记住，似然函数在数据 $\mathcal{D}$ 固定时被视为 $\theta$ 的函数。它告诉我们，根据我们实际观测到的数据，不同参数值有多么合理。它*不是* $\theta$ 上的概率分布。选择一个反映数据特征的恰当似然函数是建模的一个根本步骤。与独立同分布情况相关的一个常见假设是*可交换性*，这意味着观测值的顺序不影响联合概率。

### 先验分布 $P(\theta)$

先验分布 $P(\theta)$ 捕捉了我们在考虑特定数据集 $\mathcal{D}$ *之前*对参数 (parameter) $\theta$ 的不确定性或知识。这是贝叶斯推断的一个标志性特征。先验分布可以：

- 融入领域知识。
- 充当正则化 (regularization)项，防止参数取极端值，这在复杂模型或高维场景中对于避免过拟合 (overfitting)尤为重要。
- 反映相对无知状态（使用“无信息”或“客观”先验，尽管真正的客观性存在争议）。

先验分布和似然函数之间通过贝叶斯定理的相互作用，使我们能够正式更新信念。

### 整合：联合分布与推断

完整的概率模型由数据和参数 (parameter)（以及隐变量，如果有的话）的联合分布定义。对于参数 $\theta$ 和数据 $\mathcal{D}$，即为：


$$
P(\mathcal{D}, \theta) = P(\mathcal{D} | \theta) P(\theta)
$$


这个联合分布概括了整个生成式描述。贝叶斯推断中的主要目标是计算*后验分布* $P(\theta | \mathcal{D})$，它表示我们在观测数据*之后*对参数的更新信念。正如本章引言所述，这通过贝叶斯定理实现：


$$
P(\theta | \mathcal{D}) = \frac{P(\mathcal{D} | \theta) P(\theta)}{P(\mathcal{D})}
$$


其中 $P(\mathcal{D}) = \int P(\mathcal{D} | \theta) P(\theta) d\theta$ 是边际似然或证据。

### 一个简单例子：抛掷一枚有偏硬币

考虑对抛掷 $N$ 次可能有偏硬币的结果进行建模。

- **观测数据 $\mathcal{D}$：** 一系列正面 (H) 和反面 (T)，例如 $N_H$ 次正面和 $N_T$ 次反面 ($N = N_H + N_T$)。
- **参数 (parameter) $\theta$：** 未知的出现正面的概率，$0 \le \theta \le 1$。
- **生成过程：**

  1. 根据某些先验信念选择偏置 (bias) $\theta$ 的值。
  2. 对于每次抛掷 $i=1..N$，根据概率 $\theta$ 生成一个结果 $x_i \in \{H, T\}$。
- **似然函数：** 假设抛掷独立，观测到 $N_H$ 次正面和 $N_T$ 次反面的概率由二项式似然给出（差一个常数因子）：

  
  $$
  P(\mathcal{D} | \theta) \propto \theta^{N_H} (1-\theta)^{N_T}
  $$
  
- **先验：** 我们需要为 $\theta$ 指定一个先验分布。概率的常见选择是 Beta 分布，$P(\theta) = \text{Beta}(\theta | \alpha, \beta) \propto \theta^{\alpha-1} (1-\theta)^{\beta-1}$，其中 $\alpha$ 和 $\beta$ 是反映先验信念的超参数 (hyperparameter)（例如，$\alpha=1, \beta=1$ 对应于均匀先验，表示没有偏好）。
- **后验：** 使用贝叶斯定理，$\theta$ 的后验分布也是一个 Beta 分布：

  
  $$
  P(\theta | \mathcal{D}) \propto P(\mathcal{D} | \theta) P(\theta) \propto \theta^{N_H} (1-\theta)^{N_T} \theta^{\alpha-1} (1-\theta)^{\beta-1} = \theta^{N_H + \alpha - 1} (1-\theta)^{N_T + \beta - 1}
  $$
  

  因此，$P(\theta | \mathcal{D}) = \text{Beta}(\theta | N_H + \alpha, N_T + \beta)$。

以下是这个简单生成模型的图形表示：

> 此图展示了硬币抛掷示例的生成过程。超参数 $\alpha, \beta$（固定，由点表示）确定了硬币偏置 $\theta$（未观测参数，红色节点）的先验。偏置 $\theta$ 随后决定了每次观测到的硬币抛掷 $x_i$（观测变量，双环形，蓝色节点）的概率。板（plate）表示此过程重复 $N$ 次。

尽管此示例简单，但其基本原理适用于机器学习 (machine learning)中遇到的更复杂模型。定义清晰的生成过程、指定恰当的似然函数和先验分布，然后运用贝叶斯定理推导后验，这些构成了构建高斯过程和贝叶斯深度学习 (deep learning)等复杂技术的基础。理解这些原理使我们不仅能够应用现有模型，还能根据特定问题构建新颖的概率模型，明确处理和量化 (quantization)不确定性。我们将会看到，挑战通常在于计算后验分布，尤其当证据 $P(\mathcal{D})$ 的积分难以处理时，这促使我们在后续章节中讨论高级推断方法。

## 参考资料

- [Bayesian Data Analysis](https://www.crcpress.com/Bayesian-Data-Analysis/Gelman-Carlin-Stern-Dunson-Vehtari-Rubin/p/book/9781439840955) — Andrew Gelman, John B. Carlin, Hal S. Stern, David B. Dunson, Aki Vehtari, and Donald B. Rubin (2013)
  Publisher: Chapman & Hall; DOI: [10.1201/b16018](https://doi.org/10.1201/b16018)
  一本关于贝叶斯方法的经典综合性教科书，全面阐述了概率建模原理、似然函数、先验分布和后验推断的细节。
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-and-machine-learning-2/) — Christopher M. Bishop (2006)
  Publisher: Springer; Pages: 738; DOI: [10.1007/978-0-387-45528-3](https://doi.org/10.1007/978-0-387-45528-3)
  提供了一种全面的机器学习概率方法，详细描述了生成模型、概率图模型以及构建统计模型的基本组成部分。
- [Probabilistic Graphical Models: Principles and Techniques](https://mitpress.mit.edu/books/probabilistic-graphical-models) — Daphne Koller and Nir Friedman (2009)
  Publisher: MIT Press
  概率图模型领域的权威教科书，阐述了如何在复杂概率模型中编码结构假设和条件独立性。
