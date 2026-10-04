---
course: "vae-representation-learning"
chapter: "probabilistic-models-representation-foundations"
lesson: "probabilistic-models-advanced-perspective"
sourceId: 6267
sourceUrl: "https://apxml.com/zh/courses/vae-representation-learning/chapter-1-probabilistic-models-representation-foundations/probabilistic-models-advanced-perspective"
title: "概率模型：高级视角"
description: "对与生成学习相关的概率建模体系进行细致审视。"
order: 1
plots: []
sourceHash: "7624cb825b29ad4c12f62d48c166f135e538f266b659a707143f1e84bc3f36fd"
sourceCorrections: []
---

概率模型是许多机器学习 (machine learning)方法的根本，尤其是在生成方面。对这些模型，尤其是理解变分自编码器（VAE）等高级生成模型而言，一个高级视角是必要的。主要目的是领会这些模型如何试图描述数据的核心：其潜在的概率分布。

### 核心思想：不确定性建模

从根本上说，概率模型是一个数学构造，它定义了一组可能结果的概率分布。对于数据集 $X = \{x_1, x_2, ..., x_N\}$，每个$x_i$是一个数据点（如图像、句子），概率模型旨在描述$P(X)$。更普遍的是，假设数据点是独立同分布（i.i.d.）的，我们主要关注建模单个数据点的概率分布$P(x)$。

此方法的优点在于其量化 (quantization)不确定性的能力。概率模型可以提供一系列可能性及其相关的似然，而非预测单一的确定性输出。这对于生成任务来说是根本的，因为我们希望合成看起来像是来自原始数据分布的新数据。

### 生成模型与判别模型：快速回顾

机器学习 (machine learning)模型通常大致分为两种类型：判别模型和生成模型。

- **判别模型** 学习根据输入$x$预测标签或输出$y$。它们对条件概率$P(y|x)$进行建模。例子包括逻辑回归和支持向量 (vector)机。
- **生成模型** 另一方面，旨在学习联合分布$P(x,y)$，或者在没有显式标签时，学习数据本身的分布$P(x)$。从$P(x,y)$可以推导出用于预测的$P(y|x)$，但判别模型反之则不然。对我们来说更重要的是，通过学习$P(x)$，生成模型可以通过从这个学到的分布中采样来创建新的数据样本。

VAE坚定地属于生成模型范畴。它们的主要目标是学习一个模型$P_{model}(x)$，以近似真实的、未知的数据分布$P_{data}(x)$。

> 一个示意图，显示了生成模型和判别模型的主要功能。生成模型学习潜在数据分布，使得新样本的生成成为可能。

### 显式密度模型与隐式密度模型

概率模型的一个“高级视角”涉及更细致地查看它们如何表示$P(x)$。我们可以大致将它们分类为：

1. **显式密度模型**：这些模型为 $P_{model}(x; \theta)$ 定义一个显式公式，这里 $\theta$ 代表模型参数 (parameter)。我们可以直接评估任何给定数据点$x$的似然。

   - **易于处理的显式模型**：对于某些显式模型，$P_{model}(x; \theta)$ 可以容易计算。例子包括像高斯分布这样的简单分布，或者像正则流（我们将在课程后期涉及）这样更复杂的分布。
   - **难处理的显式模型**：许多复杂的显式模型定义的 $P_{model}(x; \theta)$ 难以直接计算。这种难处理性通常源于对潜在变量的边缘化，如 $P(x; \theta) = \int P(x, z; \theta) dz$。VAE属于这一类别，并且 $P(x)$（证据）的难处理性是变分推断的一个核心驱动因素。
2. **隐式密度模型**：这些模型不定义显式密度函数。相反，它们提供了一种从 $P_{model}(x; \theta)$ 采样的方法，而无需显式写下密度函数。生成对抗网络（GAN）是一个主要例子。

VAE作为显式（尽管难处理）密度模型，提供了能够近似观测数据似然的优点，这是许多隐式模型中不直接具备的属性。

### 学习目标：最大似然估计

训练概率模型的一个普遍原则是最大似然估计（MLE）。给定一个从未知真实分布 $P_{data}(x)$ 中抽样得到的数据集 $X = \{x_1, ..., x_N\}$，MLE 旨在找到模型 $P_{model}(x; \theta)$ 的参数 (parameter) $\theta$，以使观测到给定数据集的概率（或似然）最大化。假设数据为独立同分布，似然为：


$$
L(\theta | X) = \prod_{i=1}^{N} P_{model}(x_i; \theta)
$$


实际中，为了数值稳定性和数学上的便利，我们使用对数似然：


$$
\mathcal{L}(\theta | X) = \sum_{i=1}^{N} \log P_{model}(x_i; \theta)
$$


那么，MLE 估计值 $\theta^*$ 为：


$$
\theta^* = \arg\max_{\theta} \mathcal{L}(\theta | X)
$$


这等同于最小化经验数据分布 $\hat{P}_{data}(x)$（它将概率 $1/N$ 赋予每个观测到的 $x_i$）与模型分布 $P_{model}(x; \theta)$ 之间的 Kullback-Leibler（KL）散度，表示为 $KL(\hat{P}_{data} || P_{model})$。

### 高维空间 (high-dimensional space)中的挑战

图像、音频或文本等数据通常存在于非常高维的空间中。一个 $100 \times 100$ 像素且有 3 个颜色通道的图像具有 $100 \times 100 \times 3 = 30,000$ 维。由于“维度灾难”，在这种空间中直接建模 $P(x)$ 极具挑战性：

- 空间体积随维度数量指数级增长。
- 数据变得固有的稀疏，使密度估计变得困难。
- $x$ 的可能配置数量非常庞大。

概率生成模型通常通过假设数据（尽管维度很高）实际位于或接近一个低维流形来解决这个问题。这是一个重要的思想，它直接促成了潜在变量的使用，我们将在下一部分进行讨论。这些潜在变量旨在以更紧凑的形式捕获数据的内在结构。

### 模型假设的重要性

每个概率模型都对数据生成过程做出假设。这些假设编码在模型家族的选择中（例如变量的分布、依赖关系的函数形式）。普遍假设包括：

- **分布假设**：例如，假设输出是高斯的，或者像素是伯努利的。
- **独立性假设**：例如，在给定潜在变量的情况下，像素是条件独立的，或者朴素贝叶斯中的特征是独立的。
- **结构假设**：例如，线性关系，或用于参数 (parameter)化分布的特定神经网络 (neural network)架构。

这些假设是一把双刃剑：

- **优点**：它们使学习变得可行，减少参数数量，并可以引入先验知识。
- **缺点**：如果假设过于严格或不正确，模型将存在高偏差，并且无法捕获 $P_{data}(x)$ 的真实复杂度。

在 VAE 中，对潜在变量的先验分布 $P(z)$（通常是标准高斯分布）和近似后验 $q(z|x)$ 的形式做出了重要假设。理解这些假设对于解读模型行为和局限性来说非常重要。

### 展望未来：概率模型与表示学习

构建数据概率模型的努力不仅仅是为了生成。它与表示学习有着内在联系。当模型有效地捕获 $P(x)$ 时，特别是通过潜在变量等机制，这些机制通常会学到数据的有意义的、压缩的表示。例如，如果一个模型学到人脸图像具有姿态、表情和光照等潜在因素，那么模型中对应这些因素的内部状态就形成了有用的表示。

本部分对概率模型进行了高级阐述，强调了它们在密度估计中的作用、模型的类型（显式/隐式，易处理/难处理）以及 MLE 这种普遍的学习方法。我们还指出了高维数据带来的挑战以及模型假设的重要作用。当我们转向理解潜在变量模型，并随后理解变分自编码器时，这些元素是根本的，变分自编码器是设计用于学习分布和有意义表示的复杂概率生成模型。

## 参考资料

- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-and-machine-learning/) — Christopher M. Bishop (2006)
  Publisher: Springer
  一本关于概率模型理论基础、最大似然估计以及生成模型与判别模型区别的教科书。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本关于机器学习概率方法、生成模型和高维数据挑战的书，提供现代背景。
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Diederik P Kingma, Max Welling (2013)
  Journal: International Conference on Learning Representations (ICLR 2014); DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  引入变分自编码器的原始论文，是难处理显式密度生成模型的重要实例，也是本课程的奠基性工作。
- [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio (2014)
  Journal: Advances in Neural Information Processing Systems (NeurIPS 2014); DOI: [10.48550/arXiv.1406.2661](https://doi.org/10.48550/arXiv.1406.2661)
  提出生成对抗网络的开创性论文，是隐式密度生成模型的一个典型例子。
