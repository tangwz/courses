---
course: "autoencoders-representation-learning"
chapter: "foundations-representation-learning"
lesson: "linear-dimensionality-reduction-limitations"
sourceId: 2792
sourceUrl: "https://apxml.com/zh/courses/autoencoders-representation-learning/chapter-1-foundations-representation-learning/linear-dimensionality-reduction-limitations"
title: "线性降维的局限性"
description: "分析PCA和SVD在处理复杂数据结构方面的局限。"
order: 2
plots: ["plots/2792-0.json"]
sourceHash: "8df0dccaa9932926cdcc69de4bdd6e76464766a88ee3f3169e2e906c1ed6e7e5"
sourceCorrections: []
---

“尽管降维的目标——减少特征数量同时保留重要信息——很有吸引力，但像主成分分析（PCA）这样的经典线性技术所基于的假设，对于复杂数据集而言往往不成立。理解这些局限对于认识到对包括自编码器在内的更复杂非线性方法的需要是很重要的，我们稍后将讨论这些方法。”

PCA，或许是应用最广的线性降维算法，通过找到一组新的正交轴（称为主成分）来工作，这些轴捕获了数据中的最大方差。它将原始数据投影到一个由具有最大方差的主成分所构成的低维子空间（一条线、一个平面或一个超平面）上。从数学上看，这种投影是一种线性变换。如果$X$是我们的原始数据矩阵，降维后的表示$Z$通过$Z = XW$获得，其中$W$的列是前几个主成分向量 (vector)。

### 线性约束

这个根本局限性直接源于其线性特性。PCA假设数据的内在结构可以通过线性子空间很好地表示。当数据点在高维空间 (high-dimensional space)中围绕一条直线或一个平面聚集时，这种方法效果很好。然而，许多数据集具有固有的非线性复杂结构。

设想数据分布在一个弯曲流形上，例如经典的“瑞士卷”数据集，甚至是嵌入 (embedding)更高维度的简单抛物线。PCA试图通过线性投影来最大化方差，但它无法捕捉到真正的底层结构。它可能会找到一个能捕捉整体分布的投影，但会完全展平曲线，从而丢失数据特定形状中编码的信息。



![非线性数据结构上的PCA](plots/2792-0.json)



> PCA试图通过线性投影（虚线）捕捉最大方差。这未能表示数据的底层弯曲结构（蓝色点）。

### 方差并非全部

PCA的目标只专注于最大化方差。虽然方差通常与信息含量相关，但并非总是如此。设想一个数据集，其中不同的类别沿着一个方差相对较低的方向分开，而最大方差的方向对于区分这些类别来说是无关紧要的。PCA会优先考虑高方差方向，并可能在其低维投影中合并不同的类别，从而丢弃最具区分性的信息。对于保留数据*结构*或*可分离性*最有用的方向，可能与最大全局方差的方向不一致。

### 对尺度的敏感性

另一个实际考量是PCA对输入特征尺度的敏感性。因为它在数据的协方差矩阵上操作，数值范围较大（因此可能方差较大）的特征会不成比例地影响主成分。如果一个特征以千米为单位测量，另一个以毫米为单位测量，那么第一个特征很可能会主导分析，除非数据预先进行了标准化（例如，通过缩放到零均值和单位方差）。尽管标准化是一个标准的预处理步骤，但这表明PCA的结果并非固有的尺度不变。

### 全局结构与局部结构

PCA提供了数据方差结构的全局视角。它识别出整个数据集的主要变化方向。然而，它可能无法很好地保留局部结构。在原始高维空间 (high-dimensional space)中相邻的点，在低维PCA投影中可能会变得遥远，特别是当数据位于复杂流形上时。侧重于保留局部邻域关系的技术，例如接下来要讨论的流形学习方法，通常能提供更有见地的低维嵌入 (embedding)，用于可视化和分析。

这些局限性意味着，对于涉及图像、文本或复杂传感器读数的许多任务，PCA等线性方法通常不足以应对。它们可以作为基线或预处理步骤，但无法捕捉数据中存在的丰富非线性关系。这促使我们转向非线性降维技术，从流形学习开始，并引出自编码器强大的表示学习能力。

## 参考资料

- [Pattern Recognition and Machine Learning](https://link.springer.com/book/9780387310732) — Christopher M. Bishop (2006)
  Publisher: Springer; Pages: Chapter 12; DOI: [10.1007/b139590](https://doi.org/10.1007/b139590)
  这本教科书介绍了PCA的原理和局限性，为降维和相关机器学习概念提供了坚实的基础。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 5, Chapter 14
  本书在表示学习的背景下讨论了PCA，将其与非线性自动编码器进行对比，并强调了其在处理复杂数据时的局限性。
- [A Global Geometric Framework for Nonlinear Dimensionality Reduction](https://pubmed.ncbi.nlm.nih.gov/11125149/) — Joshua B. Tenenbaum, Vin de Silva, and John C. Langford (2000)
  Journal: Science; Publisher: American Association for the Advancement of Science (AAAS); Volume: 290; Pages: 2319-2323; DOI: [10.1126/science.290.5500.2319](https://doi.org/10.1126/science.290.5500.2319)
  本文介绍了Isomap，一种流形学习方法，展示了像PCA这样的线性方法无法捕获弯曲流形（例如“瑞士卷”）结构的局限性。
- [Nonlinear Dimensionality Reduction by Locally Linear Embedding](https://doi.org/10.1126/science.290.5500.2323) — Sam T. Roweis, Lawrence K. Saul (2000)
  Journal: Science; Volume: 290; Pages: 2323-2326; DOI: [10.1126/science.290.5500.2323](https://doi.org/10.1126/science.290.5500.2323)
  这篇论文介绍了局部线性嵌入（LLE），这是另一种流形学习算法，解决了线性方法在保持局部数据结构方面的局限性。
