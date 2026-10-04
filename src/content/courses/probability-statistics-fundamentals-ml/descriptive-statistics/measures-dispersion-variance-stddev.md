---
course: "probability-statistics-fundamentals-ml"
chapter: "descriptive-statistics"
lesson: "measures-dispersion-variance-stddev"
sourceId: 2410
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-fundamentals-ml/chapter-2-descriptive-statistics/measures-dispersion-variance-stddev"
title: "衡量离散程度：方差和标准差"
description: "理解并计算方差和标准差，以衡量数据的离散程度或变异性。"
order: 3
plots: ["plots/2410-0.json"]
sourceHash: "4d52d29307a27ffad27ca9ddbe8b4881580e9d5a988b592a2277e956fad50684"
sourceCorrections: []
---

了解数据的中心（平均值、中位数或众数）能提供一个有用的参考点。然而，这并不能说明全部情况。考虑两组考试分数：两组的平均分都可能是75分，但其中一组的分数可能紧密地集中在70到80之间，而另一组的分数可能从40到100大幅波动。单凭集中趋势度量无法反映这种分散度或变异性的差异。

为了量化 (quantization)数据点的分散程度，我们使用离散程度度量。最常用且重要的度量是**方差**和**标准差**。它们告诉我们数据点平均偏离其平均值的程度。

### 方差：平均平方差

假设你有一个数据集，并且已经计算出它的平均值（总体的为$\mu$，样本的为$\bar{x}$）。衡量离散程度的一个自然起始点可能是计算每个数据点与平均值的偏差（$x_i - \mu$或$x_i - \bar{x}$），然后对这些偏差求平均。然而，存在一个问题：正偏差（高于平均值的点）和负偏差（低于平均值的点）总是会相互抵消，导致平均偏差为零。这没什么用！

为了克服这种抵消问题，我们在求平均之前将每个偏差平方。平方使得所有偏差都变为正值，确保它们不会相互抵消。这种*平方*偏差的平均值称为**方差**。

#### 总体方差 ($\sigma^2$)

如果你的数据集代表了感兴趣的整个总体，那么方差（用希腊字母sigma的平方$\sigma^2$表示）的计算公式如下：


$$
\sigma^2 = \frac{\sum_{i=1}^{N} (x_i - \mu)^2}{N}
$$


其中：

- $N$ 是总体中数据点的总数。
- $x_i$ 是每个单独的数据点。
- $\mu$ 是总体平均值。
- $\sum$ 表示将从 $i=1$ 到 $N$ 的所有数据点的平方差求和。

#### 样本方差 ($s^2$)

通常，我们处理的是*样本*，即总体的子集。当从样本估计总体方差时，我们使用样本方差（用$s^2$表示）的公式略有不同：


$$
s^2 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n-1}
$$


其中：

- $n$ 是样本中数据点的数量。
- $x_i$ 是样本中每个单独的数据点。
- $\bar{x}$ 是样本平均值。
- $\sum$ 表示将从 $i=1$ 到 $n$ 的所有数据点的平方差求和。

请注意分母是 $n-1$ 而不是 $n$。这被称为**贝塞尔校正**。使用 $n-1$ 使得样本方差成为总体方差的无偏估计量，这意味着如果你抽取许多样本并使用 $n-1$ 计算它们的方差，这些样本方差的平均值会更接近真实的总体方差。直观地说，样本平均值$\bar{x}$是*从样本本身*计算得出的，这使得样本数据点平均而言比它们与真实总体平均值$\mu$的距离稍微近一些。除以一个较小的数（$n-1$）会略微增大方差估计值，从而抵消这种效应。对于初学者而言，主要内容是标准统计软件和库在从样本数据计算方差时使用 $n-1$。

更大的方差表示数据点平均而言距离平均值更远（更分散）。更小的方差表示数据点更紧密地聚集在平均值周围。

方差的一个缺点是它的单位。如果你的数据表示身高（单位为厘米，cm），那么方差的单位将是平方厘米（cm²），这在原始数据背景下难以直接理解。这将我们引向标准差。

### 标准差：回归原始单位

**标准差**就是方差的平方根。它通常比方差更受青睐，因为它将离散程度度量恢复到与原始数据相同的单位，使其更易于理解。

#### 总体标准差 ($\sigma$)

对于总体，标准差（用$\sigma$表示）为：


$$
\sigma = \sqrt{\sigma^2} = \sqrt{\frac{\sum_{i=1}^{N} (x_i - \mu)^2}{N}}
$$


#### 样本标准差 ($s$)

对于样本，标准差（用$s$表示）为：


$$
s = \sqrt{s^2} = \sqrt{\frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n-1}}
$$


标准差提供了数据点与平均值的*典型*或*平均距离*的度量。

- **低**标准差意味着数据点倾向于非常接近平均值。分布曲线（如直方图）将是高而窄的。
- **高**标准差意味着数据点分布在更宽的取值范围内。分布曲线将是矮而宽的。

让我们看看两组具有相同平均值但不同标准差的数据集。



![平均值相同、标准差不同的分布](plots/2410-0.json)



> 两组数据集的平均值都可能在75左右。数据集A（蓝色）的值紧密聚集，导致标准差较低。数据集B（粉色）的值分布范围更广，导致标准差较高。

理解方差和标准差在统计学和机器学习 (machine learning)中很基本。它们用于：

- **数据分析：** 量化 (quantization)特征中的变异性。
- **特征缩放：** 像标准化这样的方法使用平均值和标准差来重新缩放特征，这对于对特征大小敏感的算法（如支持向量 (vector)机或主成分分析）可能很重要。
- **统计推断：** 估计置信区间和执行假设检验通常涉及标准差。
- **异常检测：** 偏离平均值很多个标准差的点可能被认为是异常值。

尽管极差可以快速了解总体分散程度，但方差和标准差提供了更详细、更广泛使用的衡量数据如何围绕中心值聚集的指标。你将在本章后面“用Python计算描述性统计量”一节中学习如何使用Python高效计算这些量。

## 参考资料

- [Probability & Statistics for Engineers & Scientists](https://books.google.com/books/about/Probability_Statistics_for_Engineers_Scie.html?id=z1zEDgAAQBAJ) — Ronald E. Walpole, Raymond H. Myers, Sharon L. Myers, Keying E. Ye (2016)
  Publisher: Pearson Education; Pages: 816
  这是一本经典而全面的教材，对描述性统计学（包括方差和标准差的理论基础和计算）进行了严谨而易懂的讲解。
- [Practical Statistics for Data Scientists: 50 Essential Concepts](https://www.oreilly.com/library/view/practical-statistics-for/9781492072935/) — Peter Bruce, Andrew Bruce, Peter Gedeck (2020)
  Publisher: O'Reilly Media
  本书为机器学习实践者提供了统计概念的实用视角，清晰地解释了数据科学背景下的方差和标准差。
- [Introduction to Probability and Statistics (18.05)](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2014/) — Jeremy Orloff, Jonathan Bloom (2014)
  Publisher: MIT OpenCourseWare
  一门免费优质的大学课程，涵盖了基本的统计度量，提供视频讲座、笔记和习题集，有助于深入理解。
