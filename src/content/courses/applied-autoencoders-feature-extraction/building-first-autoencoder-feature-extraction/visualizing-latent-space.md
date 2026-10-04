---
course: "applied-autoencoders-feature-extraction"
chapter: "building-first-autoencoder-feature-extraction"
lesson: "visualizing-latent-space"
sourceId: 6462
sourceUrl: "https://apxml.com/zh/courses/applied-autoencoders-feature-extraction/chapter-3-building-first-autoencoder-feature-extraction/visualizing-latent-space"
title: "潜在空间可视化（适用时）"
description: "可视化潜在空间的技术，特别是当它是二维或三维时，以理解数据结构。"
order: 9
plots: ["plots/6462-0.json"]
sourceHash: "a151a0630240b1e16865bf25546f3261f969c3429c98d4a58e8aa73e6f0c1869"
sourceCorrections: []
---

在成功训练自编码器并从其瓶颈层提取特征后，接下来很自然地会想理解这些学习到的表示是什么样子。可视化潜在空间能提供很有价值的认识，了解自编码器如何组织输入数据，以及提取的特征是否捕获了有意义的结构。

可视化潜在空间最直接的方式是当其维度非常低时，通常是二维或三维。如果你将自编码器设计为2D或3D瓶颈，就可以直接绘制编码表示。

### 二维或三维潜在空间的直接可视化

如果你的自编码器将输入数据压缩到二维潜在空间，每个输入样本会被转换为一对值 $(z_1, z_2)$。你可以创建一个散点图，其中每个点对应一个输入样本，其x坐标是 $z_1$，y坐标是 $z_2$。如果你的原始数据有相关标签（例如，分类问题中的类别标签），你可以根据这些标签对散点图中的点进行颜色编码。这可以帮助你观察自编码器是否学会在潜在空间中将相似项分组或分离不同的类别。

对于三维潜在空间，你可以使用三维散点图。虽然在二维屏幕上解释起来有点困难，但在Matplotlib或Plotly等库中可用的交互式三维绘图工具会非常有用。

假设你已经在包含三个类别的数据集上训练了一个自编码器，并将潜在维度设置为2。提取潜在特征后，你可能会得到一个像这样的图：



![表格数据的二维潜在空间](plots/6462-0.json)



> 一个二维散点图，显示按原始类别颜色编码的潜在特征。明确的分离或聚类表明自编码器已经学习到区分性特征。

在Python中，Matplotlib（`matplotlib.pyplot.scatter`）和Seaborn（`seaborn.scatterplot`）等库常用于创建此类二维图。对于三维图，Matplotlib的`mplot3d`工具包是标准选择。

### 可视化高维潜在空间

如果你的潜在空间有超过三个维度怎么办？你无法直接绘制4D或10D空间。在这种情况下，你可以应用另一种降维技术，*专门用于可视化目的*。这里的目的是将高维潜在向量 (vector)投影到二维或三维，以便你可以绘制它们。记住很重要，这种二次降维仅用于可视化，并不会改变自编码器学到的特征。

针对此，有两种常用技术：

1. **主成分分析（PCA）**：如第一章所述，PCA可以找到潜在数据中的主成分（方差最大的方向），并将数据投影到前两或三个成分上。这是一种线性投影，计算效率高。Scikit-learn的`PCA`类可用于此。
2. **t-分布随机邻域嵌入 (embedding)（t-SNE）**：t-SNE是一种非线性降维技术，特别适合可视化高维数据集。它将高维点之间的相似性建模为概率，并试图找到一个保持这些相似性的低维嵌入。t-SNE在显示聚类和局部结构方面通常非常有效。然而，它比PCA计算量更大，且生成的图可能对其超参数 (parameter) (hyperparameter)（如困惑度和学习率）敏感。全局结构（例如，良好分离的聚类之间的距离）可能并非总是能准确表示。Scikit-learn提供了`TSNE`的实现。

使用这些技术时，你首先会使用训练好的自编码器提取高维潜在特征。然后，你会将这些特征输入到PCA或t-SNE中，以获得二维或三维表示，然后你可以按前面所述绘制它们。

### 解释可视化结果

当你查看这些潜在空间可视化结果时，通常会关注：

- **聚类**：对应相似输入的点是否聚在一起？如果你的数据有标签，同一类别的点是否形成不同的聚类？
- **分离**：不同类别或类型的数据在潜在空间中是否良好分离？
- **流形结构**：数据是否似乎位于潜在空间中的一个低维流形上？这可以说明自编码器已经学习到数据的一些底层结构。
- **异常点**：是否有点远离其他点？这些可能对应于不寻常的输入或异常，我们将在后续章节中提及此话题。

一个结构良好的潜在空间可视化，其中相似数据点彼此靠近，不相似数据点彼此远离，通常表明自编码器已经学习到有意义且可能管用的特征。例如，如果不同颜色（代表原始类别）在你的二维图中形成不同的团块，这是一个好迹象。

### 局限性和注意事项

尽管能提供见解，潜在空间可视化主要是一种定性工具。

- **主观性**：图的“美观度”并不总是与提取特征在下游任务上的表现直接相关。
- **信息损失**：当使用PCA或t-SNE将高维潜在空间投影到二维或三维进行可视化时，一些信息不可避免地会丢失。你看到的是真实潜在结构的一个投影。
- **t-SNE细节**：t-SNE图可能特别复杂。t-SNE图中聚类的大小和聚类之间的距离不总是有直接、可靠的含义。它最好用于理解局部邻域和聚类的存在，而不是它们的全局排列。

尽管有这些局限，可视化潜在空间是构建和理解自编码器过程中的一个有价值的步骤。它提供了对模型学习内容的视觉验证，并可以指导进一步的实验，例如调整潜在空间维度或其他超参数 (parameter) (hyperparameter)。然而，特征质量的最终检验仍在于它们在下游机器学习 (machine learning)模型中的表现，我们将在第七章进一步研究。对于某些高级自编码器变体，如变分自编码器（VAEs），我们将在第六章介绍，检查潜在空间对于理解其生成能力变得更为重要。

## 参考资料

- [Visualizing Data using t-SNE](http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) — Laurens van der Maaten and Geoffrey Hinton (2008)
  Journal: Journal of Machine Learning Research; Publisher: Microtome Publishing; Volume: 9; Pages: 2579-2605
  这篇基础性论文介绍了t-SNE，这是一种非线性降维技术，广泛用于可视化高维数据，包括潜在空间。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本内容全面的教科书，涵盖了深度学习的理论基础，包括自动编码器以及与学习和可视化潜在表示相关的原则。
- [sklearn.manifold.TSNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) — Scikit-learn developers (2023)
  Publisher: Scikit-learn
  Scikit-learn中t-SNE实现的官方文档，提供了在Python中应用t-SNE进行潜在空间可视化的实用细节。
- [sklearn.decomposition.PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) — Scikit-learn developers (2023)
  Scikit-learn中主成分分析（PCA）实现的官方文档，这是一种用于可视化高维潜在空间的线性降维方法。
- [Matplotlib: Visualization with Python](https://matplotlib.org/stable/) — Matplotlib Development Team (2023)
  Matplotlib的官方文档，这是一个用于创建静态、动画和交互式可视化的Python库，包括用于潜在空间的2D和3D散点图。
