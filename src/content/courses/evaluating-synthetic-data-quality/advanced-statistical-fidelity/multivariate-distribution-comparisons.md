---
course: "evaluating-synthetic-data-quality"
chapter: "advanced-statistical-fidelity"
lesson: "multivariate-distribution-comparisons"
sourceId: 5643
sourceUrl: "https://apxml.com/zh/courses/evaluating-synthetic-data-quality/chapter-2-advanced-statistical-fidelity/multivariate-distribution-comparisons"
title: "多变量分布比较"
description: "除了简单的边际检查之外，比较联合分布的方法。"
order: 1
plots: ["plots/5643-0.json"]
sourceHash: "0d110888b9d9d266cc6e144a61fd07bd923e4dc759e9752b21e85e46d05d4bec"
sourceCorrections: []
---

仅凭均值（$\mu$）或标准差（$\sigma$）等单一特征统计量来评估合成数据，所呈现的情况是不全面的。虽然使每个特征 $X_i$ 的边际分布 $P(X_i)$ 相符是必要的第一步，但这并不能确保特征之间复杂的依赖关系得到保留。数据集的特点是存在复杂的依赖关系和相关性，共同构成多变量结构。高质量的合成数据集必须复现这种联合分布 $P(X_1, X_2, ..., X_n)$，而不仅仅是各个独立的组成部分。评估这种多变量相似性的方法将在此进行介绍。

简单来说，我们需要检验合成数据中变量间的关系是否反映了真实数据中的关系。真实数据中倾向于一同增长的特征，在合成数据中是否也如此？复杂的非线性依赖关系是否得到体现？回答这些问题需要不再局限于单变量比较。

### 多变量结构的可视化观察

在考察定量指标之前，可视化检查通常是一个有益的起点，特别是对于维度数量（$n$）相对较小的数据集。

- **散点图与配对图：** 对于两个变量（$X_i, X_j$），简单的散点图可以显示线性或非线性关系、聚类和异常值。在此基础上，**配对图**（也称散点图矩阵）会显示所有选定特征组合的成对散点图，以及对角线上每个特征的边际分布。将真实数据和合成数据的配对图进行比较，可以定性地了解成对关系保留得如何。Python 中的 Seaborn 等库使得生成这些图表变得简单。



![配对图比较（真实数据与合成数据）](plots/5643-0.json)



> 该配对图比较了真实数据集（蓝色）与合成数据集（橙色）的三个特征。对角线显示了边际分布的直方图，非对角线则显示了成对关系的散点图。理想情况下，合成图表中的形状和趋势应与真实图表中的形状和趋势高度相似。

- **降维：** 对于具有许多特征（$n > 3$）的数据集，直接可视化完整的联合分布是不可能的。主成分分析（PCA）或 t-分布随机近邻嵌入 (embedding)（t-SNE）等降维方法可以将高维数据投影到二维或三维空间。通过将*相同*的投影（从真实数据或组合数据中学习得到）应用于真实数据集和合成数据集，并绘制结果，您可以直观地检查整体结构、聚类和流形是否得到保留。如果合成数据点在低维嵌入中与真实数据点很好地重叠，则表明存在良好的结构相似性。

### 定量多变量比较方法

可视化直观但具有主观性。定量方法为多变量相似性提供了客观评分：

- **距离和散度度量：** 有多种数学度量可以量化 (quantization)两个概率分布之间的“距离”或“散度”。

  - **马氏距离：** 它衡量一个点与一个分布之间的距离，并考虑了协方差结构。它可用于比较两个多变量分布的中心点。
  - **Kullback-Leibler（KL）散度与 Jensen-Shannon（JS）散度：** 这些信息论度量量化了两个概率分布之间的差异。JS 散度是对称的且总是有界，因此常比 KL 散度更受欢迎。在高维空间 (high-dimensional space)中准确计算这些度量可能面临困难，并且可能需要密度估计方法。
  - **最大均值差异（MMD）：** 一种基于核函数的方法，用于衡量两个数据集在高维再生核希尔伯特空间（RKHS）中嵌入 (embedding)之间的距离。它避免了直接的密度估计，并且在高维空间中通常表现良好。
- **假设检验：** 统计检验可以正式评估真实样本和合成样本是否来自同一潜在多变量分布的零假设。例子包括：

  - **Hotelling's T平方检验：** t检验的多变量推广形式，用于比较两个多变量样本的均值。
  - **基于核函数的检验（例如，使用MMD）：** 基于MMD等度量的非参数 (parameter)检验。
    我们将在下一节更详细地讨论假设检验。
- **基于判别器的评估：** 这种方法借鉴了生成对抗网络（GAN）的思路。训练一个分类器（例如，逻辑回归模型、支持向量 (vector)机或神经网络 (neural network)）来区分真实数据集和合成数据集中的样本。如果分类器难以区分它们（即，对于平衡数据集，其准确率接近50%），则表明这些分布在统计上是相似的。本章稍后讨论的倾向性评分方法是这种方法的具体示例。

### 多变量比较中的挑战

评估多变量分布比单变量比较本身更为复杂：

- **维度灾难：** 随着特征数量（$n$）的增加，数据空间的体积呈指数增长。数据点变得稀疏，从而更难估计密度或准确比较分布。可视化效果也会降低。
- **计算成本：** 许多多变量方法（例如，核方法、复杂的假设检验）计算成本可能很高，尤其是在处理大型数据集或高维数据时。
- **可解释性：** 单个距离或检验统计量（如MMD或p值）总结了总体差异，但可能无法显示分布具体*如何*不同（例如，哪些特定相关性不匹配）。

尽管存在这些挑战，评估多变量保真度对于确保合成数据真正体现原始数据特点来说是不可或缺的。后续章节将介绍假设检验、相关性分析、信息论度量和倾向性评分等具体方法，为您提供执行这些严格评估的工具。

## 参考资料

- [A Kernel Two-Sample Test](https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf) — Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard Schölkopf, and Alex Smola (2012)
  Journal: Journal of Machine Learning Research; Volume: 13; Pages: 723-773; DOI: [10.5555/2810848.2810903](https://doi.org/10.5555/2810848.2810903)
  介绍了最大平均差异（MMD）作为一种非参数核函数两样本检验方法，对定量多变量比较至关重要。
- [Generative Adversarial Networks](https://proceedings.neurips.cc/paper_files/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf) — Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio (2014)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; Volume: 27; Pages: 2672-2680; DOI: [10.48550/arXiv.1406.2661](https://doi.org/10.48550/arXiv.1406.2661)
  提出生成对抗网络（GANs）的原始概念，是合成数据质量判别器评估的基础。
- [A Survey of Metrics for Generative Models](https://dl.acm.org/doi/10.1145/3403986) — Zhifeng Wang, Jun Zou (2020)
  Journal: ACM Computing Surveys; Publisher: ACM; Volume: 53; Pages: 1-35; DOI: [10.1145/3403986](https://doi.org/10.1145/3403986)
  全面回顾了生成模型评估指标，涵盖了评估合成数据质量的各种定量方法，包括保真度和多样性方面。
- [Elements of Information Theory](https://www.wiley.com/en-us/Elements+of+Information+Theory%2C+2nd+Edition-p-9780471241959) — Thomas M. Cover and Joy A. Thomas (2006)
  Publisher: Wiley
  信息论的权威教材，对于理解分布比较中使用的 Kullback-Leibler (KL) 和 Jensen-Shannon (JS) 散度等指标至关重要。
