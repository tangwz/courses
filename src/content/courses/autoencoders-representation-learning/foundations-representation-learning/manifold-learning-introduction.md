---
course: "autoencoders-representation-learning"
chapter: "foundations-representation-learning"
lesson: "manifold-learning-introduction"
sourceId: 2795
sourceUrl: "https://apxml.com/zh/courses/autoencoders-representation-learning/chapter-1-foundations-representation-learning/manifold-learning-introduction"
title: "流形学习方法简介"
description: "非线性方法（如t-SNE和UMAP）在高维数据可视化方面的概述。"
order: 3
plots: ["plots/2795-0.json", "plots/2795-1.json"]
sourceHash: "da78f925fdb5fd00419e3fd72a40b2a7eae201c3c5eda8744ebe74b4e6b99be3"
sourceCorrections: []
---

像主成分分析（PCA）这样的线性方法在捕捉高维数据中的复杂模式时存在局限性。为此，我们将注意力转向专门为非线性结构设计的方法。许多数据集，尽管存在于高维空间 (high-dimensional space)中，却显示出其固有维度低得多的特点。想象一张在三维空间中卷起的纸；它的点有三个坐标，但其潜在结构本质上是二维的。这一理念由**流形假设**正式提出，该假设认为高维数据通常位于或接近嵌入 (embedding)在高维空间中的低维、可能弯曲的流形上。

流形学习算法旨在展现这种潜在的低维结构。与PCA寻找最大全局方差方向不同，这些方法在映射到低维空间时，通常侧重于保留数据点的*局部*邻域结构。这使得它们在可视化复杂数据集和理解线性方法可能遗漏的关系方面特别有效。两种广泛使用的方法是t-分布式随机邻居嵌入（t-SNE）和均匀流形近似与投影（UMAP）。

### t-分布式随机邻居嵌入 (embedding)（t-SNE）

t-SNE主要以其在二维或三维中可视化高维数据的有效性而闻名。其核心思想是模拟高维数据点之间的相似性以及相应低维点之间的相似性，然后最小化这两种相似性分布之间的差异。

1. **高维相似性：** 对于每对高维点 $x_i$ 和 $x_j$，t-SNE计算一个条件概率 $p_{j|i}$，它表示点 $x_j$ 与点 $x_i$ 的相似度。这基于以 $x_i$ 为中心的N分布。离 $x_i$ 越近的点获得更高的概率值。随后，它定义了一个联合概率 $p_{ij} = (p_{j|i} + p_{i|j}) / 2n$。此处的 $n$ 代表数据点的数量。
2. **低维相似性：** 类似地，对于相应的低维映射点 $y_i$ 和 $y_j$，t-SNE使用重尾学生t分布（自由度为一，类似于柯西分布）定义一个联合概率 $q_{ij}$。重尾允许高维空间 (high-dimensional space)中中度不相似的点在低维映射中被模型化为更远的距离，有助于避免拥挤。
3. **最小化散度：** t-SNE优化低维空间中点 $y_i$ 的位置，以最小化联合概率分布 $P$（来自高维数据）和 $Q$（来自低维映射）之间的Kullback-Leibler (KL)散度：
   
   $$
   KL(P || Q) = \sum_{i<j} p_{ij} \log \frac{p_{ij}}{q_{ij}}
   $$
   
   这种最小化通常通过梯度下降 (gradient descent)来执行。

t-SNE中一个重要的参数 (parameter)是**困惑度（perplexity）**，它大致对应于每个点考虑的有效最近邻居数量。调整困惑度会影响最终的可视化效果。

尽管t-SNE擅长可视化和展现聚类结构，但它有一些需要记住的特点：

- **计算成本：** 对于非常大的数据集可能很慢（$O(N^2)$ 或使用近似方法时为 $O(N \log N)$）。
- **非确定性：** 由于优化过程，在相同数据上多次运行t-SNE可能产生略微不同的结果。
- **侧重局部结构：** 它擅长显示局部关系（簇），但t-SNE图中分离簇之间的距离对于它们在高维空间中的实际分离可能没有意义。全局结构保留不是其主要目标。
- **嵌入质量：** 主要为可视化设计；由此产生的低维坐标通常不适合作为下游任务的通用特征。



![t-SNE 映射](plots/2795-0.json)



> 使用t-SNE等流形学习方法投影高维数据点，旨在保留局部聚类结构的二维可视化。颜色代表原始空间中可能的组别。

### 均匀流形近似与投影（UMAP）

UMAP是一种较新的方法，已获得显著流行。它根植于流形理论和拓扑数据分析。与t-SNE一样，它旨在找到一个保留结构信息的低维表示，但它通常在局部细节和全局结构保留之间取得更好的平衡。

UMAP算法可以概括为两个主要步骤：

1. **高维图构建：** UMAP首先构建高维数据的加权图表示。对于每个点，它根据所选距离度量找到其最近邻居。主要的观点是使用模糊拓扑结构（模糊单纯形集）来模拟这种邻域关系。它确保权重 (weight)（代表连接可能性）从每个点向外平滑衰减，有效地创建数据的拓扑“模糊”表示。它还致力于确保连接点之间距离的更均匀分布，因此得名中的“Uniform”。
2. **低维布局优化：** UMAP随后在目标低维空间中初始化点，并优化它们的位置，使相应的低维图结构尽可能与高维图结构相似。这种优化最小化高维和低维模糊拓扑表示之间的交叉熵。

与t-SNE相比，UMAP通常提供几个优点：

- **速度：** 通常更快，尤其适用于大型数据集和高嵌入 (embedding)维度。
- **全局结构：** 与t-SNE相比，通常能更好地保留数据的全局结构。UMAP图中簇之间的距离可以更有意义。
- **通用降维：** 尽管UMAP也擅长可视化，但其产生的嵌入有时被认为比t-SNE嵌入更适合用作下游机器学习 (machine learning)任务的输入特征。
- **更少的超参数 (parameter) (hyperparameter)：** 主要超参数与邻居数量和嵌入点之间的最小距离有关，这有时比t-SNE的困惑度更容易调整。



![UMAP 映射](plots/2795-1.json)



> 使用UMAP等流形学习方法进行的二维可视化。注意与t-SNE相比，簇的位置可能有所不同，可能更准确地反映全局关系，同时仍能分离局部组。

### 流形学习在背景中的作用

t-SNE和UMAP都是有效的算法，用于理解复杂高维数据的结构，特别是通过可视化。它们证实了非线性关系普遍存在的直觉，并表明能够捕捉这些关系的方法是必要的。

然而，这些方法主要提供一种*映射*或*布局*，其优化目标是保留某些结构属性，通常是局部邻域。它们不明确地学习一个*函数*（像神经网络 (neural network)），将数据从高维空间 (high-dimensional space)映射到低维流形并返回。这限制了它们在需要生成能力或现成特征提取函数的任务中的直接使用。

理解这些流形学习方法为自编码器提供了有益的背景。正如我们所见，自编码器学习显式的、参数 (parameter)化的非线性函数（编码器和解码器），尝试发现并表示数据中的这些低维流形。虽然t-SNE和UMAP擅长可视化和数据考察，但自编码器提供了一个学习压缩表示的框架，可用于更广泛的任务，包括降维、特征提取、数据生成和异常检测。流形学习的成功强调了自编码器提供的非线性特征提取能力的必要性。

## 参考资料

- [Visualizing Data using t-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) — Laurens van der Maaten and Geoffrey Hinton (2008)
  Journal: Journal of Machine Learning Research; Volume: 9; Pages: 2579-2605; DOI: [10.5555/1755833.1755834](https://doi.org/10.5555/1755833.1755834)
  介绍t-SNE算法的原始学术论文，详细阐述了其数学基础和在高维数据可视化方面的实证表现。
- [UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction](https://joss.theoj.org/papers/10.21105/joss.00861) — Leland McInnes, John Healy, Nathaniel Saul, Lukas Großberger (2018)
  Journal: Journal of Open Source Software; Publisher: The Open Journal; Volume: 3; Pages: 861; DOI: [10.21105/joss.00861](https://doi.org/10.21105/joss.00861)
  介绍UMAP的开创性论文，这是一种现代流形学习算法，因其高效性以及相比t-SNE更好地保留数据全局结构而闻名。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  这本被广泛引用的教材包含一个专门章节（第15章：表示学习），在深度学习背景下讨论了降维和流形学习，为理解自编码器提供了重要背景知识。
- [UMAP: Uniform Manifold Approximation and Projection (Official Documentation)](https://umap-learn.readthedocs.io/en/latest/) — McInnes, L, Healy, J (2024)
  UMAP Python库的官方文档，由创建者提供实用指南、超参数的详细解释以及其理论基础的见解。
