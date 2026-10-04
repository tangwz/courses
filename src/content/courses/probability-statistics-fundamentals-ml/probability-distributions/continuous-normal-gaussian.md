---
course: "probability-statistics-fundamentals-ml"
chapter: "probability-distributions"
lesson: "continuous-normal-gaussian"
sourceId: 2452
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-fundamentals-ml/chapter-4-probability-distributions/continuous-normal-gaussian"
title: "连续型分布：正态（高斯）分布"
description: "了解正态（高斯）分布在统计学和机器学习中的特性及其作用。"
order: 7
plots: ["plots/2452-0.json"]
sourceHash: "4407ce83a0b5037e9153b2f480071788d70137d86834a683464bc1f85e5f4ccc"
sourceCorrections: []
---

在了解了均匀分布（其中给定范围内的每个结果都等可能）之后，我们现在来看概率和统计学中最常见且很有意义的连续型分布：正态分布，它也被广泛地称为高斯分布或钟形曲线。

它的普遍性并非偶然。许多自然现象，如人类身高、测量误差和血压等，都趋向于服从正态分布。此外，它在许多统计理论和机器学习 (machine learning)算法中起着基本作用，部分原因在于中心极限定理，我们将在本章后面部分讨论这一内容。

### 正态分布的形态

正态分布的特点是其对称的钟形。曲线以其均值为中心，其分布的宽度或范围由其标准差决定。

### 参数 (parameter)：均值与标准差

一个特定的正态分布由两个参数来确定：

1. **均值 ($\mu$)**: 这个参数表示分布的中心。它是钟形曲线的顶点，也是随机变量的平均值。改变均值会将整个曲线沿数轴向左或向右移动，而不改变其形态。
2. **标准差 ($\sigma$)**: 这个参数控制分布的扩散程度或离散程度。较小的标准差会形成更高、更窄的曲线，表明数据点紧密聚集在均值附近。较大的标准差会形成更矮、更宽的曲线，表示数据有更大的变异性。方差 ($\sigma^2$) 也常用于其定义中。

### 概率密度函数 (PDF)

对于连续型分布，概率密度函数 (PDF) 用于描述变量取特定范围内的值的可能性，由曲线下的面积表示。正态分布的 PDF 由以下公式给出：


$$
f(x | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$


其中：

- $x$ 是随机变量的值。
- $\mu$ 是均值。
- $\sigma^2$ 是方差 ($\sigma$ 是标准差)。
- $\pi$ 是数学常数圆周率 (大约 3.14159)。
- $e$ 是自然对数的底数 (大约 2.71828)。

虽然这个公式可能看起来复杂，但主要需要记住的是，这条曲线的形态完全由均值 $\mu$ 和标准差 $\sigma$ 决定。这条曲线下的总面积，与任何 PDF 一样，总是等于 1。

### 正态分布的可视化

下面的图表显示了标准正态分布（其中 $\mu=0$ 且 $\sigma=1$）的标志性钟形，以及另一个具有不同均值和标准差（$\mu=2, \sigma=1.5$）的正态分布。



![正态分布 PDF](plots/2452-0.json)



> 两种正态分布 PDF 的比较。蓝色曲线 (μ=0, σ=1) 是标准正态分布。绿色曲线 (μ=2, σ=1.5) 以 2 为中心，并由于标准差较大而更宽。

### 经验法则 (68-95-99.7 法则)

理解正态分布扩散程度的有用指导是经验法则：

- 大约 **68%** 的数据落在距均值一个标准差的范围内（即在 $\mu - \sigma$ 和 $\mu + \sigma$ 之间）。
- 大约 **95%** 的数据落在距均值两个标准差的范围内（即在 $\mu - 2\sigma$ 和 $\mu + 2\sigma$ 之间）。
- 大约 **99.7%** 的数据落在距均值三个标准差的范围内（即在 $\mu - 3\sigma$ 和 $\mu + 3\sigma$ 之间）。

这个法则提供了一种快速估算服从正态分布的数据在特定范围内预期比例的方法。

### 正态分布在机器学习 (machine learning)中有何作用？

正态分布经常出现在机器学习的背景中：

1. **残差建模**：在许多回归模型（如线性回归）中，假设误差（或残差，即预测值与实际值之间的差）服从正态分布。
2. **算法假定**：一些算法，如高斯朴素贝叶斯，明确假设特征服从正态分布。线性判别分析 (LDA) 也假设每个类别内的数据服从正态分布。
3. **中心极限定理**：如前所述，这个定理（稍后讨论）指出，无论原始总体分布如何，随着样本量增大，样本均值的分布趋近于正态分布。这对于统计推断很根本。
4. **参数 (parameter)初始化**：神经网络 (neural network)中的权重 (weight)常用从正态分布中抽取的值进行初始化。
   "5. **自然过程**：对由许多小型独立效应总和产生的过程或数据进行建模时，正态分布通常能提供一个很好的近似。"

因此，理解正态分布的特性对于正确应用和解释许多统计和机器学习方法是必要的。在接下来的部分，我们将看到如何使用 Python 生成服从这种分布的数据点。

## 参考资料

- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://hastie.su.domains/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  一本关于统计学习方法的奠基性著作，详细阐述了正态分布在模型中的应用、假设以及各种算法的理论基础。
- [Lecture Notes for MIT 6.041/6.431: Introduction to Probability](https://ocw.mit.edu/courses/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/) — John Tsitsiklis (2010)
  Publisher: MIT OpenCourseWare
  著名大学课程的易于理解的讲义，解释了概率分布，包括正态分布及其特征。
