---
course: "probability-statistics-essentials-ml"
chapter: "common-probability-distributions"
lesson: "uniform-distribution"
sourceId: 1365
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-2-common-probability-distributions/uniform-distribution"
title: "均匀分布"
description: "了解连续均匀分布的特性和用途。"
order: 3
plots: ["plots/1365-0.json", "plots/1365-1.json"]
sourceHash: "17e4c8922dbf6a696cd8692ff546aadfa1d2954c0cd3a99b1e35da4f583e9bcf"
sourceCorrections: []
---

在了解了离散结果的分布之后，我们现在将注意力转向连续随机变量。最简单的连续分布是均匀分布。它描述了一种情况，即给定范围内所有可能的结果都等可能出现。想象一下生成一个介于0和1之间的随机数；该区间内的任何值出现的可能性都相同。

### 连续均匀分布

连续均匀分布定义在特定区间 $[a, b]$ 上，其中 $a$ 是最小值， $b$ 是最大值。遵循此分布的随机变量 $X$ 表示为 $X \sim U(a, b)$。

#### 概率密度函数 (PDF)

与使用概率质量函数 (PMF) 的离散分布不同，连续分布使用概率密度函数 (PDF)，表示为 $f(x)$。PDF 不给出*某个特定*值的概率（对于任何连续变量，这在技术上是零），而是给出该值周围概率的*密度*。随机变量 $X$ 落入某个范围内的概率，通过计算该范围内 PDF 曲线下的面积得到。

对于均匀分布 $U(a, b)$，PDF 在区间 $[a, b]$ 内是常数，在其他地方为零。这个常数值确保曲线下的总面积等于1，这是任何有效 PDF 的必要条件。

PDF 的公式如下：


$$
f(x; a, b) =
\begin{cases}
\frac{1}{b-a} & \text{当 } a \le x \le b \\
0 & \text{当 } x < a \text{ 或 } x > b
\end{cases}
$$


PDF 的高度是 $1 / (b-a)$，即区间长度的倒数。这个恒定高度反映了分布的“均匀”特性，其中概率均匀地分布在整个区间上。



![均匀分布 U(2, 8) 的 PDF](plots/1365-0.json)



> 均匀分布 $U(2, 8)$ 的概率密度函数 (PDF)。在 $x=2$ 和 $x=8$ 之间，密度常数为 $1/(8-2) = 1/6 \approx 0.167$，在其他地方为零。曲线下的总面积为 $(8-2) \times (1/6) = 1$。

#### 累积分布函数 (CDF)

累积分布函数 (CDF)，$F(x)$，给出随机变量 $X$ 取值小于或等于 $x$ 的概率，即 $P(X \le x)$。

对于均匀分布 $U(a, b)$，CDF 在区间 $[a, b]$ 上从0线性增加到1。

CDF 的公式如下：


$$
F(x; a, b) =
\begin{cases}
0 & \text{当 } x < a \\
\frac{x-a}{b-a} & \text{当 } a \le x \le b \\
1 & \text{当 } x > b
\end{cases}
$$


当 $x=a$ 时，CDF 是 $F(a) = (a-a)/(b-a) = 0$。当 $x=b$ 时，CDF 是 $F(b) = (b-a)/(b-a) = 1$。在 $a$ 和 $b$ 之间，概率线性累积。



![均匀分布 U(2, 8) 的 CDF](plots/1365-1.json)



> 均匀分布 $U(2, 8)$ 的累积分布函数 (CDF)。它显示了概率 $P(X \le x)$ 从 $x=2$ 处的0线性增加到 $x=8$ 处的1。

#### 均值和方差

期望值（均值）和方差是任何分布的重要统计量。

- **均值（期望值）：** $U(a, b)$ 分布的均值是区间的中心点：
  $E[X] = \frac{a+b}{2}$
  这很直观，因为概率对称地分布在区间的中心周围。
- **方差：** 方差衡量分布的离散程度。对于 $U(a, b)$，它为：
  $Var(X) = \frac{(b-a)^2}{12}$
  方差仅取决于区间 $(b-a)$ 的宽度。更宽的区间导致更大的方差，表示可能值的离散程度更大。

### 特性和应用

均匀分布的主要特点是，它表示对已知范围 $[a, b]$ 内变量值完全不确定，假设该范围内所有值具有相同的可能性。

- **随机数生成：** 大多数编程语言提供生成遵循 $U(0, 1)$ 分布的伪随机数的函数。这些函数通常是使用逆变换采样或拒绝采样等技术从其他更复杂的分布生成随机样本的基础。
- **模拟：** 当已知输入参数 (parameter)在一个范围内，但其确切值或在该范围内的分布未知或假设为均匀时（例如，模拟某个时间窗口内的到达时间），它在模拟中很有用。
- **贝叶斯统计中的先验：** 在贝叶斯推断中，如果我们没有先验知识偏向范围 $[a, b]$ 内参数的任何特定值，我们可以为其分配一个 $U(a, b)$ 先验分布。这有时被称为“无信息”先验，尽管其性质需要仔细考量。
- **机器学习 (machine learning)初始化：** 有时，神经网络 (neural network)中的初始权重 (weight)是从零附近的一个小区间（例如 $U(-c, c)$）的均匀分布中采样的。

值得一提的是，还存在**离散均匀分布**，其中有有限数量 $n$ 的结果，每个结果的概率都是 $1/n$。一个经典例子是掷一个公正的六面骰子，其中每个结果 $\{1, 2, 3, 4, 5, 6\}$ 的概率都是 $1/6$。然而，我们在此关注的是连续版本，它在机器学习环境中建模连续量时更常遇到。

连续均匀分布虽然简单，但它是概率论及其应用中的一个基本组成部分，尤其是在模拟中，并作为理解随机过程的依据，其中一个范围内的结果是等可能的。在接下来的章节中，我们将研究正态分布和指数分布等，它们对结果不完全等可能的现象进行建模。

## 参考资料

- [Pattern Recognition and Machine Learning](https://link.springer.com/book/9780387310732) — Christopher M. Bishop (2006)
  Publisher: Springer; Pages: 738; DOI: [10.1007/b13994](https://doi.org/10.1007/b13994)
  一本综合性著作，在机器学习和模式识别的背景下介绍了概率分布，包括连续均匀分布。
- [All of Statistics: A Concise Course in Statistical Inference](https://link.springer.com/book/10.1007/978-0-387-21736-9) — Larry Wasserman (2004)
  Publisher: Springer; DOI: [10.1007/978-0-387-21736-9](https://doi.org/10.1007/978-0-387-21736-9)
  提供了机器学习所需的统计理论的严谨而简明的概述，涵盖基本分布、其性质和统计推断。
