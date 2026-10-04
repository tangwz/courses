---
course: "applied-autoencoders-feature-extraction"
chapter: "nn-dimensionality-reduction-recap"
lesson: "hands-on-pca-dimensionality-reduction"
sourceId: 6444
sourceUrl: "https://apxml.com/zh/courses/applied-autoencoders-feature-extraction/chapter-1-nn-dimensionality-reduction-recap/hands-on-pca-dimensionality-reduction"
title: "实践：PCA降维操作"
description: "关于使用Python库实现主成分分析（PCA）进行数据降维的实践环节。"
order: 7
plots: ["plots/6444-0.json", "plots/6444-1.json"]
sourceHash: "05437b94d33f6e4f0160a613a1d5b82f5cc79b4cc9f4edff1368a76611b66080"
sourceCorrections: []
---

主成分分析（PCA）是一种常用的线性降维技术。理解它的应用将为处理更复杂的自编码器模型打下良好基础。本次动手练习将引导你使用Python和Scikit-learn在一个常见数据集上实现PCA。

我们的目标是在数据集里减少特征数量，同时尽可能多地保留原始信息（方差）。我们还会对结果进行可视化，以便直观地了解PCA的功用。

### 准备PCA：鸢尾花数据集

我们将使用Scikit-learn中方便提供的知名鸢尾花数据集。该数据集包含150个鸢尾花样本，每个样本有四个特征：萼片长度、萼片宽度、花瓣长度和花瓣宽度。鸢尾花有三个品种：山鸢尾、变色鸢尾和维吉尼亚鸢尾。

首先，让我们加载数据集：

```python
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
X = iris.data  # 特征
y = iris.target # 目标标签（品种）

print(f"Original data shape: {X.shape}")
# 预期输出：Original data shape: (150, 4)
```

输出显示了150个样本和4个特征，符合预期。

### 特征缩放的重要性

PCA对特征的尺度很敏感。如果某个特征的值范围远大于其他特征，它将在PCA计算中占据主导地位。为避免这种情况，我们应该对特征进行标准化，使其具有零均值和单位方差。Scikit-learn的`StandardScaler`非常适合此用。

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 你可以验证缩放后数据的均值和标准差
# print(f"Mean of scaled data (approx): {np.mean(X_scaled, axis=0)}")
# print(f"Std dev of scaled data (approx): {np.std(X_scaled, axis=0)}")
```

我们的数据经过缩放后，就可以应用PCA了。

### 应用主成分分析

我们将使用Scikit-learn的`PCA`类。PCA的一个主要参数 (parameter)是`n_components`，它指定了我们希望将数据降到的主成分数量（即新的维度）。让我们首先将鸢尾花数据集的4个特征降至2个主成分，这将使我们能够在2D图中可视化数据。

```python
from sklearn.decomposition import PCA

# 使用2个分量初始化PCA
pca = PCA(n_components=2)

# 在缩放后的数据上拟合PCA并进行转换
X_pca = pca.fit_transform(X_scaled)

print(f"Shape after PCA: {X_pca.shape}")
# 预期输出：Shape after PCA: (150, 2)
```

如你所见，数据现在有150个样本和2个特征（我们的主成分）。这两个分量是PCA构建的新特征，表示原始四个特征的线性组合。

### 可视化降维后的数据

既然我们已将维度降至两维，我们可以创建一个散点图，查看不同鸢尾花品种在这个新的2D空间中的分布情况。我们将使用原始目标标签`y`为点进行颜色编码。



![PCA后鸢尾花数据集（2个分量）](plots/6444-0.json)



> 鸢尾花品种在由前两个主成分定义的2D空间中绘制。即使在降维后，也请注意这些类别分离得有多好。

这个可视化效果提供了很多信息。它表明前两个主成分捕获了足够的信息，能很好地分辨出三种鸢尾花。山鸢尾品种（蓝色）与变色鸢尾（绿色）和维吉尼亚鸢尾（橙色）明显分离。

### 理解解释方差

通过降至2个分量，我们保留了多少信息？PCA通过`explained_variance_ratio_`属性帮助我们量化 (quantization)这一点。该属性返回一个数组，其中每个值表示每个选定分量所解释的方差百分比。

```python
print(f"Explained variance ratio per component: {pca.explained_variance_ratio_}")
# 预期输出（约）：Explained variance ratio per component: [0.72962445 0.22850762]

print(f"Total explained variance by 2 components: {np.sum(pca.explained_variance_ratio_):.4f}")
# 预期输出（约）：Total explained variance by 2 components: 0.9581
```

第一个主成分解释了大约73%的方差，第二个解释了大约23%。这两个分量共同捕获了原始4维数据中约95.81%的总方差。这是很不错的；我们将维度减半，同时保留了大部分数据的变异性。

选择合适的分量数量通常涉及降维和信息损失之间的权衡。一个常见的决定方法是绘制累积解释方差与分量数量的关系图。

```python
# 用所有分量拟合PCA以查看完整范围
pca_full = PCA().fit(X_scaled)
cumulative_explained_variance = np.cumsum(pca_full.explained_variance_ratio_)
```

现在，让我们可视化这个累积解释方差。



![主成分的累积解释方差](plots/6444-1.json)



> 此图表显示了随主成分数量增加而解释的累积方差。你可以看出，使用2个分量，超过95%的方差被捕获。

这个图有助于你决定保留多少个分量。例如，如果你的目标是保留95%的方差，那么对于鸢尾花数据集来说，两个分量就足够了。如果你需要99%，你就会选择三个分量。

### 数据重构（简要）

PCA还允许你使用`inverse_transform`方法将降维后的数据转换回原始的高维空间 (high-dimensional space)。重构的数据不会与原始数据完全相同（除非你使用所有分量），因为在降维过程中会丢失一些信息。这种重构的想法是自编码器的核心，我们稍后会看到。

```python
X_reconstructed = pca.inverse_transform(X_pca)
print(f"Shape of reconstructed data: {X_reconstructed.shape}")
# 预期输出：Shape of reconstructed data: (150, 4)
# 注意：X_reconstructed是X_scaled的近似值
```

### 从PCA到自编码器

本次动手练习演示了PCA如何有效地降低维度，同时保留数据中的重要方差。你学会了：

- 通过缩放准备数据。
- 使用Scikit-learn应用PCA。
- 解释解释方差以选择分量数量。
- 可视化转换后的数据。

PCA执行线性变换。尽管它很强大，但它可能无法有效地捕获数据中复杂的非线性关系。这正是本课程主要主题——自编码器发挥作用的地方。自编码器是能够学习非线性降维的神经网络 (neural network)。将数据编码到较低维空间（如PCA的主成分）然后解码回来的原理对于自编码器来说也很根本。练习了PCA后，你现在能更好地理解自编码器用于特征提取的机制和优点。

## 参考资料

- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://link.springer.com/book/10.1007/978-0-387-84858-7) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer; Pages: Chapter 14, pages 497-521; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  一本综合性教科书，从统计和理论角度阐释了PCA及其他降维技术。
- [sklearn.decomposition.PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) — Scikit-learn Developers (2023)
  Scikit-learn中主成分分析（PCA）类的官方文档，详细介绍了其参数、属性和方法。
- [sklearn.preprocessing.StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) — Scikit-learn Developers (2023)
  Scikit-learn中StandardScaler类的官方文档，用于通过移除均值并缩放到单位方差来标准化特征。
- [Hands-On Machine Learning with Scikit-Learn, Keras, & TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125974/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: Chapter 8, pages 231-255
  第3版。一份实践指南，包含使用Scikit-learn逐步实现PCA的教程，并附有可视化示例。
