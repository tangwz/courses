---
course: "julia-for-machine-learning"
chapter: "julia-unsupervised-learning-dimensionality"
lesson: "julia-other-dimensionality-reduction"
sourceId: 6880
sourceUrl: "https://apxml.com/zh/courses/julia-for-machine-learning/chapter-4-julia-unsupervised-learning-dimensionality/julia-other-dimensionality-reduction"
title: "其他降维方法"
description: "简要概述其他降维方法，如 t-SNE，用于 Julia 中的数据可视化和分析。"
order: 4
plots: []
sourceHash: "38ac78126eff1e7e47dd32a8532a28ec267af2d404052dad4a3021af3cb0f22e"
sourceCorrections: []
---

降维技术有助于简化复杂数据集。主成分分析（PCA）是一种广泛应用且有效的降维方法，尤其适用于数据具有潜在线性结构的情况。然而，许多数据集，特别是复杂的，表现出非线性模式，而 PCA 可能无法理想地捕获这些模式。为了处理这类非线性或满足不同目标（例如增强数据可视化效果），存在其他降维方法。

### 流形学习：t-SNE 和 UMAP

许多高维数据集被认为位于或接近一个称为流形的低维非线性子空间上。设想一张在三维空间中卷起来的纸张；数据点位于纸张的二维表面（流形），即使它们由三维坐标描述。流形学习算法旨在“展开”这个流形，以找到数据的真实低维表示。

#### t-分布随机邻居嵌入 (embedding) (t-SNE)

t-分布随机邻居嵌入 (t-SNE) 是一种主要用于在二维或三维空间中可视化高维数据集的技术。它将高维数据点之间的相似度建模为条件概率，然后尝试找到一个低维嵌入，使得相似点保持接近，不相似点相互远离。

核心思想是将数据点之间的高维欧几里得距离转换为表示相似度的条件概率。例如，如果数据点 $x_j$ 位于以 $x_i$ 为中心的 Gaussian 分布内，则它是 $x_i$ 的邻居。t-SNE 随后尝试在低维空间中使用 Student's t-分布重现这些概率，这有助于更清晰地分离不同的聚类。

**t-SNE 的优点：**

- 擅长显示局部结构，并在低维可视化中形成分隔良好的聚类。
- 有效地可视化非线性结构。

**t-SNE 的注意事项：**

- **计算成本：** t-SNE 可能计算密集，特别是对于大型数据集。
- **参数 (parameter)敏感性：** 结果对其参数可能相当敏感，尤其是“困惑度”（perplexity），它大致与每个点考虑的最近邻数量有关。通常需要进行实验。
- **全局结构：** 尽管 t-SNE 在局部结构方面表现出色，但结果图中聚类的全局排列及其间距可能并非总是有意义的。t-SNE 图中聚类的大小和距离应谨慎解释。
- **不适用于通用降维：** 它主要是一种可视化工具，通常不用于在训练监督模型之前进行特征提取，而 PCA 则可能用于此目的。

在 Julia 中，你可以使用 `TSne.jl` 等包来实现 t-SNE。

#### 均匀流形逼近与投影 (UMAP)

均匀流形逼近与投影 (UMAP) 是一种更近期的降维方法，它像 t-SNE 一样，非常适合可视化非线性数据结构。然而，它也常被用作更通用的降维工具。UMAP 的理论基础是流形理论和拓扑数据分析。它构建一个表示数据的高维图，然后优化一个低维图，使其在结构上尽可能相似。

**UMAP 的优点：**

- **速度：** 通常比 t-SNE 更快，特别是在大型数据集上。
- **全局结构保留：** 与 t-SNE 相比，UMAP 在保留数据的全局结构方面通常做得更好。这意味着低维嵌入中聚类的相对位置可以更具解释性。
- **可伸缩性：** 可以更有效地扩展到大型数据集。
- **通用性：** 尽管非常适合可视化，UMAP 嵌入有时可用作下游机器学习 (machine learning)任务的输入特征。

**UMAP 的注意事项：**

- **参数：** 像 t-SNE 一样，UMAP 也有参数（例如 `n_neighbors`，`min_dist`）可能会影响生成的嵌入结果。
- **解释：** 尽管在全局结构方面优于 t-SNE，它仍然是一种投影，所有解释都应谨慎进行。

对于 Julia 实现，`UMAP.jl` 包提供了所需的工具。

下图展示了流形学习方法（如 t-SNE 和 UMAP）的总体思想，它们试图找到一种低维表示，以捕获位于流形上的数据的内在结构。

> 该图显示了流形学习算法如何将数据从高维空间 (high-dimensional space)（其中数据可能形成像“瑞士卷”一样的复杂形状）投影到低维空间，目的在于保留数据点之间的局部关系。

### 用于非线性降维的自编码器

自编码器是一种用于无监督学习 (supervised learning) (unsupervised learning)的人工神经网络 (neural network)，它们在降维方面非常有效，特别是对于捕捉复杂的非线性关系。一个自编码器由两个主要部分组成：

1. **编码器：** 网络的这部分将输入数据压缩成低维表示，通常称为“瓶颈”层或“潜在空间表示”。
2. **解码器：** 这部分尝试从编码器生成的压缩表示中重建原始输入数据。

网络被训练以最小化重建误差，即原始输入与重建输出之间的差异。训练完成后，编码器部分可以独立使用，将高维数据转换为低维潜在空间。这提供了一种压缩的、学习到的数据表示。

自编码器具有高度灵活性，并且可以学习比 PCA 等线性方法更复杂的数据结构。它们是通向深度学习 (deep learning)方法的桥梁，在 Julia 中，你通常会使用像 `Flux.jl` 这样的深度学习库（我们将在后续章节中介绍）来构建和训练自编码器。

### 线性判别分析 (LDA)

值得一提的是线性判别分析 (LDA)，尽管它在技术上是一种**监督**学习算法。与 PCA、t-SNE 或 UMAP 不同，LDA 使用类别标签来寻找一个低维子空间，该子空间能最大化类别之间的可分离性。因此，尽管它减少了维度，但其主要目标是找到对分类任务最具判别力的维度。

LDA 常用于分类模型的预处理步骤。它将数据投影到低维空间，使类别尽可能地分离，这可以提高后续分类器的性能和效率。

在 Julia 中，LDA 可通过 `MultivariateStats.jl` 等包获得。请记住，应用 LDA 需要标签数据。

选择合适的降维方法在很大程度上取决于你的具体目标和数据性质。如果你的主要目标是复杂非线性数据的可视化，t-SNE 或 UMAP 是不错的选择。如果你需要通过线性变换在降维的同时尽可能多地保留方差，PCA 是首选。对于非线性降维，特别是当你怀疑存在非常复杂的结构或在深度学习 (deep learning)框架中工作时，自编码器提供了一种多功能的方法。如果你要以最佳方式分离预定义类别来降低维度，那么在拥有标签数据的情况下，LDA 是合适的。每种方法都提供了一个不同的视角来审视和简化你的数据。

## 参考资料

- [Visualizing Data using t-SNE](http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) — Laurens van der Maaten and Geoffrey Hinton (2008)
  Journal: Journal of Machine Learning Research; Volume: 9; Pages: 2579-2605; DOI: [10.5555/1504958.1504968](https://doi.org/10.5555/1504958.1504968)
  介绍t-SNE（t分布随机邻域嵌入）用于高维数据可视化的基础论文。
- [UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction](https://arxiv.org/abs/1802.03426) — Leland McInnes, John Healy, and James Melville (2018)
  Journal: arXiv; DOI: [10.48550/arXiv.1802.03426](https://doi.org/10.48550/arXiv.1802.03426)
  提出UMAP（均匀流形逼近与投影）作为可扩展降维算法的原始论文。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  涵盖自编码器和其他用于表示学习的深度学习技术的综合教科书。
- [Pattern Recognition and Machine Learning](https://link.springer.com/book/9780387310732) — Christopher M. Bishop (2006)
  Publisher: Springer; DOI: [10.1007/bpa2610](https://doi.org/10.1007/bpa2610)
  提供线性判别分析（LDA）及其他经典降维方法详细论述的标准教科书。
