---
course: "probability-statistics-essentials-ml"
chapter: "common-probability-distributions"
lesson: "normal-gaussian-distribution"
sourceId: 1371
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-2-common-probability-distributions/normal-gaussian-distribution"
title: "正态（高斯）分布"
description: "学习正态分布在统计学和机器学习中的性质与作用。"
order: 4
plots: ["plots/1371-0.json", "plots/1371-1.json"]
sourceHash: "aa5f48bd26ae24ed5a986153cdcb9c321d07591b990912e7f296e297bee779e5"
sourceCorrections: []
---

正态分布，又称高斯分布或钟形曲线，或许是统计学和机器学习 (machine learning)中最广为人知且最常遇到的连续概率分布。它的普遍性不仅源于它能近似描述多种自然现象，还源于它在统计理论中扮演着重要角色，特别是由于中心极限定理（我们将在第4章中介绍）。

“许多测量值，如人类身高、实验中的测量误差或血压，通常趋向于服从正态分布，至少是近似地。这使其成为建模连续数据不可或缺的工具。”

### 定义正态分布

一个连续随机变量 $X$ 服从正态分布，如果其概率密度函数（PDF）由以下给出：


$$
f(x | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$


这种分布完全由两个参数 (parameter)刻画：

1. **均值 ($\mu$)**：这个参数表示分布的中心或峰值。它决定了钟形曲线在水平轴上的位置。
2. **方差 ($\sigma^2$)**：这个参数衡量分布的离散程度或宽度。更大的方差会导致更矮、更宽的曲线，而更小的方差则会形成更高、更窄的曲线。通常，这种分布会使用标准差 ($\sigma = \sqrt{\sigma^2}$) 来参数化，它与随机变量 $X$ 具有相同的单位。

我们将均值为 $\mu$、方差为 $\sigma^2$ 的正态分布记作 $N(\mu, \sigma^2)$。

### 正态分布的性质

正态分布具有几个显著特点：

- **钟形曲线**：其概率密度函数（PDF）的图形呈对称的单峰钟形。
- **对称性**：曲线围绕其均值 $\mu$ 完全对称。
- **均值、中位数和众数**：由于其对称性，正态分布的均值、中位数和众数都相等 ($\mu$)。
- **总面积**：与任何概率密度函数一样，曲线下的总面积等于1。
- **渐近尾部**：曲线渐近地趋向水平轴，这意味着当 $x$ 趋向正无穷或负无穷时，曲线会越来越接近水平轴，但永远不会真正接触到它。



![参数不同的正态分布](plots/1371-0.json)



> 具有不同均值 ($\mu$) 和方差 ($\sigma^2$) 的正态分布。N(0, 1) 以0为中心，标准差为1。N(3, 1) 向右移动，以3为中心。N(0, 4) 以0为中心，但由于标准差为2 ($\sigma^2=4$) 而更宽。

### 经验法则（68-95-99.7 法则）

一个实用的经验法则，用于正态分布，将标准差与数据落在均值周围特定范围内的比例联系起来：

- 大约 **68%** 的数据落在均值的一个标准差范围内 ($\mu \pm \sigma$)。
- 大约 **95%** 的数据落在均值的两个标准差范围内 ($\mu \pm 2\sigma$)。
- 大约 **99.7%** 的数据落在均值的三个标准差范围内 ($\mu \pm 3\sigma$)。

如果数据服从正态分布，此法则提供了一种快速了解数据离散程度的方法。

### 标准正态分布（$Z$-分布）

正态分布的一个特例是标准正态分布，记作 $Z$，它的均值为0，方差（和标准差）为1，即 $Z \sim N(0, 1)$。它的概率密度函数简化为：


$$
\phi(z) = \frac{1}{\sqrt{2\pi}} e^{-\frac{z^2}{2}}
$$


标准正态分布具有特殊重要性，因为任何正态分布 $X \sim N(\mu, \sigma^2)$ 都可以通过一个简单的线性变换，称为标准化或计算 $Z$-分数，转换成标准正态分布：


$$
Z = \frac{X - \mu}{\sigma}
$$


$Z$-分数告诉我们特定值 $X$ 距离均值 $\mu$ 有多少个标准差。这种变换非常有价值，因为它：

1. **比较**：它允许通过将不同正态分布的值放在同一尺度上来进行比较。
2. **概率计算**：任何正态分布的概率都可以使用标准正态分布的累积分布函数（CDF）来计算，通常记作 $\Phi(z)$。标准正态概率表或计算函数（如SciPy中的）广泛可用。对于 $X \sim N(\mu, \sigma^2)$，概率 $P(X \le x)$ 等价于 $P(Z \le \frac{x - \mu}{\sigma}) = \Phi(\frac{x - \mu}{\sigma})$。



![标准正态分布 (N(0,1)) 及经验法则](plots/1371-1.json)



> 标准正态分布 N(0,1)，显示了根据经验法则，距离均值（0）1、2和3个标准差之内的近似面积。

### 在机器学习 (machine learning)中的应用与作用

正态分布在许多统计学和机器学习背景下是根本的：

- **残差建模**：在线性回归（第6章中会介绍）中，一个常见假设是预测值和实际值之间的误差（残差）服从正态分布。
- **特征分布**：当输入特征服从正态分布时，某些算法表现更好。存在将非正态分布特征进行转换的技术。
- **参数 (parameter)初始化**：神经网络 (neural network)中的权重 (weight)通常使用从正态分布中抽取的值进行初始化。
- **算法组成部分**：某些算法，如高斯朴素贝叶斯，明确假设特征在每个类别中服从正态分布。线性判别分析（LDA）也依赖于这个假设。
- **中心极限定理**：如前所述，这个定理指出，无论总体原始分布如何，随着样本量增加，样本均值的分布趋近于正态分布。这证明了在许多情况下使用基于正态分布的推断是合理的。

在接下来的实践部分和后续章节中，你将了解到如何使用Python库，如SciPy (`scipy.stats.norm`)，来计算概率（PDF, CDF）、生成随机样本（`rvs`），并将正态分布拟合到数据。了解它的性质是有效应用许多统计技术的重要一步。

## 参考资料

- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  一本基础书籍，详述机器学习中使用的统计模型和算法，提供免费在线版本。它解释了正态分布在线性判别分析等模型中的作用及其广泛的统计背景。
- [Introduction to Probability and Statistics (MIT OpenCourseWare 18.05)](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2014/) — Jeremy Orloff, Jonathan Bloom (2014)
  Publisher: MIT OpenCourseWare
  麻省理工开放课程提供了涵盖概率基础概念的讲义和材料，其中包括对正态分布及其重要性的详尽解释。
- [\`scipy.stats.norm\` Reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html) — SciPy Developers (2024)
  SciPy中正态（高斯）连续随机变量的官方文档，提供PDF、CDF、随机抽样和参数拟合的函数。上次更新适用于SciPy 1.12.0。
