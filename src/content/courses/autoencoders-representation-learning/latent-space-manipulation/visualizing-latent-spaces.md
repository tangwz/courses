---
course: "autoencoders-representation-learning"
chapter: "latent-space-manipulation"
lesson: "visualizing-latent-spaces"
sourceId: 2892
sourceUrl: "https://apxml.com/zh/courses/autoencoders-representation-learning/chapter-6-latent-space-manipulation/visualizing-latent-spaces"
title: "使用 t-SNE 和 UMAP 可视化潜在空间"
description: "运用降维方法来展示所学潜在空间的结构。"
order: 1
plots: ["plots/2892-0.json", "plots/2892-1.json"]
sourceHash: "cc35795b43acd0cb3ab177e9cd508f102410b7433ef41b5ed77f20401bb64d33"
sourceCorrections: []
---

自动编码器包含一个编码器函数，它将高维输入数据 $x$ 映射到低维潜在表示 $z$，以及一个尝试从 $z$ 重构 $x$ 的解码器。重构损失衡量自动编码器信息保留的程度。然而，这个指标本身并未充分说明潜在空间本身的*结构*。这个空间是否有效地组织了数据？相似的输入是否被映射到潜在空间中的邻近点？

可视化潜在空间能为模型学习到的表示提供有益的直观认识。然而，潜在空间虽然维度低于输入，但通常仍是高维的（例如，16、32、64 或更多维度），无法直接绘图。我们需要方法将这些高维潜在向量 (vector)投影到我们能够方便观察的 2D 或 3D 空间。

为此，两种普遍使用且有效的方法是 t-分布随机邻域嵌入 (embedding)（t-SNE）和均匀流形近似与投影（UMAP）。这些是非线性降维方法，专门设计用于保留数据的局部结构，使其成为观察自动编码器如何组织潜在表示的优秀手段。请记住，这些方法主要用于*可视化*和分析，不一定用于生成下游任务的嵌入，因为它们可能会扭曲整体结构或距离。

### t-分布随机邻域嵌入 (embedding)（t-SNE）

t-SNE 是一种流行的高维数据集可视化方法。它通过将数据点之间的高维欧几里得距离转换为表示相似度的条件概率来实现。具体来说，数据点 $x_j$ 相对于数据点 $x_i$ 的相似度被建模为条件概率 $p_{j|i}$，即如果邻居是根据以 $x_i$ 为中心的某个高斯分布的概率密度按比例选择的，$x_i$ 会选择 $x_j$ 作为其邻居。

在低维空间（通常是 2D）中，t-SNE 使用具有一个自由度的学生 t-分布（这等同于柯西分布）定义了一个类似的条件概率 $q_{j|i}$。t-SNE 的目的随后是找到数据点的低维嵌入，该嵌入能最小化联合概率 $P = (p_{j|i} + p_{i|j}) / 2n$ 和 $Q = (q_{j|i} + q_{i|j}) / 2n$（ $n$ 是数据点数量）之间的 Kullback-Leibler (KL) 散度。


$$
KL(P || Q) = \sum_{i < j} p_{ij} \log \frac{p_{ij}}{q_{ij}}
$$


最小化此 KL 散度会促使高维潜在空间中相似的点（$p_{ij}$ 值高）在低维图中以彼此靠近的点（$q_{ij}$ 值高）表示。在低维空间中使用重尾 t-分布有助于缓解拥挤问题（即点倾向于在图的中心聚集），并使得不相似的点可以被建模得更远。

**t-SNE 的考量：**

- **困惑度（Perplexity）：** 此超参数 (parameter) (hyperparameter)大致对应每个点考虑的最近邻居数量。典型值介于 5 到 50 之间。它影响数据局部和整体方面的平衡。不同的困惑度值可以显示不同的结构。
- **计算开销：** 标准 t-SNE 的计算规模与数据点数量呈二次方关系，这使得它对于非常大的数据集而言速度较慢。存在更快的近似方法，如 Barnes-Hut t-SNE。
- **非确定性：** t-SNE 使用随机初始化和优化，这意味着在相同数据上多次运行它可能会产生不同的可视化结果。通常，运行它几次以确保观察到的结构稳定是有益的。
- **解释：** t-SNE 图中簇的相对大小以及簇*之间*的距离不一定包含有意义的信息。应关注哪些点聚类在一起（局部结构）。

**使用 t-SNE 进行潜在空间可视化：**

1. 训练你的自动编码器。
2. 将你的输入数据集（例如，验证集或测试集）通过训练好的编码器，以获得相应的潜在向量 (vector) $z$。
3. 对这组潜在向量应用 t-SNE（例如，在 Python 中使用 `sklearn.manifold.TSNE`），指定 `n_components=2`。
4. 创建所得 2D 点的散点图。如果你的原始数据有标签（例如，MNIST 的数字类别），根据这些标签为点着色，以查看自动编码器是否已学会在潜在空间中分离它们。



![潜在空间的 t-SNE 投影](plots/2892-0.json)



> 一张带有四种不同类别（颜色所示）数据的潜在空间的 2D t-SNE 可视化图。请注意其分离为不同的簇。

### 均匀流形近似与投影（UMAP）

UMAP 是一种较新的降维方法，它已获得相当大的关注度，与 t-SNE 相比，它通常能生成更高质量的可视化结果，同时计算速度更快。它基于黎曼几何和代数拓扑学。

UMAP 的核心思路包含两个主要步骤：

1. **高维图构建：** UMAP 构建高维数据的加权图表示。对于每个点，它找到其最近邻居，并根据点连接的可能性分配边权重 (weight)，假设流形上局部均匀分布。此步骤旨在对数据的底层流形结构进行建模。
2. **低维布局优化：** UMAP 随后寻求一个低维嵌入 (embedding)（例如 2D），以保留高维图的基本拓扑结构。它优化低维空间中点的位置，以最小化高维图的边权重与为低维嵌入计算的相似权重集之间的交叉熵。

与 t-SNE 相比，UMAP 在保留数据的整体结构方面表现更好，这意味着簇的相对位置可能更有意义。它通常也更快，并且更适用于大型数据集。

**UMAP 的考量：**

- **`n_neighbors`：** 类似于 t-SNE 中的困惑度，此参数 (parameter)控制 UMAP 如何平衡局部与整体结构。较小的值更注重局部结构，而较大的值包含更多整体信息。典型值范围为 5 到 50。
- **`min_dist`：** 此参数控制低维嵌入中点之间的最小距离。值越低，簇越紧密，而值越高，点会更分散。它主要影响嵌入的视觉密度。
- **确定性（固定随机状态下）：** 与 t-SNE 的典型实现不同，如果你固定随机种子，UMAP 的优化过程是确定性的，从而产生可复现的结果。

**使用 UMAP 进行潜在空间可视化：**

该过程与使用 t-SNE 类似：

1. 训练你的自动编码器。
2. 使用编码器获取潜在向量 (vector) $z$。
3. 对潜在向量应用 UMAP（例如，在 Python 中使用 `umap-learn` 库），指定 `n_components=2`。请确保首先安装该库（`pip install umap-learn`）。为了性能，你可能还需要 `pynndescent`。
4. 创建 2D UMAP 嵌入的散点图，如果可用，根据类别标签为点着色。

```python
# umap-learn 使用示例
# 假设 'latent_vectors' 是一个形状为 (样本数量, 潜在维度) 的 NumPy 数组
# 'labels' 是对应类别标签的数组。

import umap
import plotly.graph_objects as go
import numpy as np # 假设 latent_vectors 和 labels 已定义

# 演示用占位数据
np.random.seed(42)
n_samples_per_class = 30
latent_dim = 16
centers = np.random.rand(4, latent_dim) * 20 - 10
latent_vectors = np.vstack([
    np.random.randn(n_samples_per_class, latent_dim) * 1.5 + centers[i] for i in range(4)
])
labels = np.repeat(np.arange(4), n_samples_per_class)

reducer = umap.UMAP(n_neighbors=15,
                    min_dist=0.1,
                    n_components=2,
                    random_state=42)

embedding = reducer.fit_transform(latent_vectors)

# 创建 Plotly 图表（代码嵌入在下面的最终 Plotly JSON 中）
```



![潜在空间的 UMAP 投影](plots/2892-1.json)



> 之前展示的同一潜在空间的 2D UMAP 可视化图。将簇的形状和相对位置与 t-SNE 图进行比较。UMAP 通常能更好地保留整体结构。

### 解释可视化结果

虽然 t-SNE 和 UMAP 是功能强大的工具，但解释时需要谨慎：

- **关注邻域：** 这两种方法都擅长显示高维空间 (high-dimensional space)中哪些点是邻居。如果来自同一类别的点在 2D 图中聚类在一起，这表明自动编码器学习到了有意义的表示，这些表示能捕获类别差异。
- **不要过度解释距离/大小：** 如前所述，2D 图中簇之间的绝对距离以及簇的大小或密度可能是可视化算法及其超参数 (parameter) (hyperparameter)的人为结果。避免仅基于这些方面得出强烈结论。
- **尝试超参数：** 使用不同的 `perplexity` 或 `n_neighbors` 值运行算法。在不同设置下都出现的稳定簇更可能代表数据中的真实结构。
- **与其他分析互补：** 可视化只是一种工具。将 t-SNE/UMAP 图的见解与定量指标和下游任务性能结合起来，以全面了解表示质量。

通过对自动编码器编码器生成的潜在向量 (vector)应用 t-SNE 和 UMAP，你可以获得一个可视化的窗口，以查看所学表示的结构。这有助于理解模型如何组织信息，以及它是否成功捕获了数据中变化的底层因素，从而为接下来讨论的更高级分析和操作创造了条件。

## 参考资料

- [Visualizing Data using t-SNE](http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) — Laurens van der Maaten, Geoffrey Hinton (2008)
  Journal: Journal of Machine Learning Research; Volume: 9; Pages: 2579-2605
  介绍t-SNE的原始研究论文，详细阐述了其数学公式和应用。
- [UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction](https://joss.theoj.org/papers/10.21105/joss.00861) — Leland McInnes, John Healy, Nathaniel Saul, Lukas Großberger (2018)
  Journal: Journal of Open Source Software; Publisher: The Open Journal; Volume: 3; Pages: 861; DOI: [10.21105/joss.00861](https://doi.org/10.21105/joss.00861)
  提出UMAP的开创性论文，解释了其理论基础和实际优势。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本涵盖深度学习基础概念的综合教科书，包括自编码器、表示学习和降维技术。
- [UMAP: Uniform Manifold Approximation and Projection - How UMAP Works](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html) — Leland McInnes, John Healy, James Melville (2018)
  UMAP算法的概念性解释，作为其理论和实现细节的易懂指南。
- [\`sklearn.manifold.TSNE\` - scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) — scikit-learn developers (2023)
  Publisher: scikit-learn
  官方文档，概述了scikit-learn库中t-SNE的参数、用法和注意事项。
