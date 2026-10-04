---
course: "autoencoders-representation-learning"
chapter: "foundations-representation-learning"
lesson: "information-bottleneck-primer"
sourceId: 2802
sourceUrl: "https://apxml.com/zh/courses/autoencoders-representation-learning/chapter-1-foundations-representation-learning/information-bottleneck-primer"
title: "信息瓶颈理论基础"
description: "介绍信息瓶颈原理的理论知识。"
order: 5
plots: []
sourceHash: "997f6dddad97dc197cfdc125eaabd1a6974958e2e668e0a44d4a224f2dd424a6"
sourceCorrections: []
---

对于从数据中学习有用表示，审视指导此过程的理论架构会很有帮助。由Naftali Tishby、Fernando Pereira和William Bialek提出的信息瓶颈（IB）原理，提供了一种准确思考数据压缩与保留相关信息之间权衡的方法。

设想您有一些输入数据，我们称之为 $X$。您希望创建一个压缩表示 $Z$，它充当一个瓶颈。这个表示 $Z$ 应该对预测或理解某个其他相关变量 $Y$ 有用。在许多无监督学习 (supervised learning) (unsupervised learning)场景中，包括基本的自编码器，相关信息 $Y$ 通常就是输入 $X$ 本身；我们希望将 $X$ 压缩成 $Z$，以便能从 $Z$ 中准确地重构出 $X$。

信息瓶颈的主要思路是找到一个表示 $Z$，它关于原始输入 $X$ 所保留的信息尽可能少，同时关于目标变量 $Y$ 所保留的信息尽可能多。这形成了一个瓶颈，它会滤除 $X$ 中不相关的细节，但保留对 $Y$ 具有预测性或提供信息的那些部分。

### 信息量化 (quantization)：互信息

为了精确地说明这一点，我们使用信息理论中的*互信息*。两个随机变量 $A$ 和 $B$ 之间的互信息，记作 $I(A; B)$，衡量通过观察 $B$ 获得关于 $A$ 的信息量，反之亦然。它量化了在已知一个变量的情况下，另一个变量不确定性的减少量。如果 $A$ 和 $B$ 相互独立，则 $I(A; B) = 0$。如果已知 $B$ 可以完全确定 $A$，则 $I(A; B)$ 等于 $A$ 的熵 $H(A)$。

### 信息瓶颈目标函数

信息瓶颈原理旨在找到一个从 $X$ 到 $Z$ 的概率映射，由条件概率分布 $p(z|x)$ 表示，该映射优化以下目标函数：


$$
\mathcal{L}_{IB} = I(X; Z) - \beta I(Z; Y)
$$


此处：

- $I(X; Z)$ 衡量表示 $Z$ 包含关于输入 $X$ 的信息量。最小化此项可以强制进行压缩。
- $I(Z; Y)$ 衡量表示 $Z$ 包含关于相关变量 $Y$ 的信息量。最大化此项（这等同于在目标函数中最小化其负值）能让 $Z$ 有效地提供关于 $Y$ 的信息。
- $\beta$ 是一个正的拉格朗日乘子，它控制着压缩（低 $I(X; Z)$）与相关性（高 $I(Z; Y)$）之间的权衡。较大的 $\beta$ 会更侧重于保留关于 $Y$ 的信息，这可能以减少压缩为代价。较小的 $\beta$ 则优先考虑压缩。

通常，我们假设存在马尔可夫链结构 $Y \leftrightarrow X \leftrightarrow Z$。这意味着在给定 $X$ 的条件下，$Z$ 与 $Y$ 条件独立；$Z$ 拥有关于 $Y$ 的所有信息都必须通过 $X$ 获得。

> 该图示说明了信息瓶颈原理。输入 $X$ 被编码为一个压缩表示 $Z$，该表示应在最小化关于 $X$ 所保留信息的同时，最大化关于目标 $Y$ 所保留的信息。

### 与自编码器和无监督学习 (supervised learning) (unsupervised learning)的关系

在自编码器的背景下，输入是 $X$，编码器生成瓶颈表示 $Z$，解码器尝试从 $Z$ 重构 $X$。在这里，相关变量 $Y$ 仅仅是输入 $X$ 本身。信息瓶颈的目标变成找到一个 $Z$，它使 $I(X; Z)$ 最小化，同时使 $I(Z; X)$ 最大化。这初看起来似乎是矛盾的，但其限制来自瓶颈层的有限容量（例如，其维度或施加的正则化 (regularization)）。

自编码器旨在学习一种编码 $p(z|x)$ 和解码 $p(x|z)$ 的方式，使得 $Z$ 是 $X$ 的一个压缩版本（通过降维或正则化实现较低的 $I(X; Z)$），但仍能进行准确重构（较高的 $I(Z; X)$）。权衡参数 (parameter) $\beta$ 隐式关联着我们在重构准确性与瓶颈中的压缩或正则化程度之间如何做出偏重。

尽管标准自编码器并未明确优化涉及互信息项（这些项通常难以直接计算）的信息瓶颈目标，但该原理提供了一个有价值的理论视角。它有助于解释为什么诸如降维、稀疏约束或添加噪声（如在去噪自编码器中）等技术可以产生更好的表示。它们迫使模型丢弃不相关信息（最小化 $I(X; Z)$），同时保留重构所需的必要要素（最大化 $I(Z; X)$）。

变分自编码器（VAEs），我们将在第4章中了解，其目标函数（证据下界，即ELBO）可证明与信息瓶颈目标有直接联系，从而为理论与实践之间提供了更具体的关联。

理解信息瓶颈原理为我们思考表示学习的基本目标提供了一种正式方法：即找到能够捕获输入数据中最相关方面的简洁表示。这种视角有助于我们接下来设计和分析各种自编码器架构。您可能会发现重新回顾“数学预备知识回顾”部分中讨论的知识点，有助于更好地掌握互信息和相关的概率思想。

## 参考资料

- [The Information Bottleneck Method](https://proceedings.neurips.cc/paper_files/paper/1997/file/0f9687cd310f8021c38597d34d284a28-Paper.pdf) — Naftali Tishby, Fernando C. Pereira, William Bialek (1998)
  Journal: Advances in Neural Information Processing Systems 10; Publisher: MIT Press; Pages: 584-590
  介绍了信息瓶颈原理的基础，定义了数据压缩与保留相关信息之间的权衡。
- [Deep Variational Information Bottleneck](https://arxiv.org/pdf/1612.00410.pdf) — Alexander A. Alemi, Ian Fischer, Joshua V. Dillon, Kevin Murphy (2017)
  Journal: International Conference on Learning Representations (ICLR 2017); DOI: [10.48550/arXiv.1612.00410](https://doi.org/10.48550/arXiv.1612.00410)
  将信息瓶颈原理应用于深度学习，提出了变分信息瓶颈（VIB）并展示了其与变分自编码器的关联。
- [The Information Bottleneck Method (Chapter 24.3)](https://mitpress.mit.edu/books/probabilistic-machine-learning-advanced-topics) — Kevin P. Murphy (2023)
  Publisher: MIT Press; Pages: 250-252
  提供了信息瓶颈原理的简洁而现代的教科书式解释，包括其数学公式及其在表示学习中的作用，选自《概率机器学习：引论》一书。
