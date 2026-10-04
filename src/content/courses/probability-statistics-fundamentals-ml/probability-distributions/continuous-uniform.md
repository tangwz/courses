---
course: "probability-statistics-fundamentals-ml"
chapter: "probability-distributions"
lesson: "continuous-uniform"
sourceId: 2450
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-fundamentals-ml/chapter-4-probability-distributions/continuous-uniform"
title: "连续分布：均匀分布"
description: "理解均匀分布，其中某个范围内的所有结果出现可能性均等。"
order: 6
plots: ["plots/2450-0.json"]
sourceHash: "f16a09ffa248e070c64935e629cc4d0a2c5fff7b64635451f07a70f5d777a696"
sourceCorrections: []
---

均匀分布是一种最简单的连续概率分布。对于连续变量，概率由概率密度函数 (PDF) 表示。

设想一个随机过程，其中在某个特定范围内，任何值出现的可能性都与其他值相同。比如一个完美的随机数生成器，它生成0到1之间的数字；该区间内的任何数字都有相同的生成机会。这种情况可以通过连续均匀分布来描述。

### 均匀分布的定义

一个连续随机变量 $X$ 服从均匀分布，如果它在一个给定区间内取值，例如从 $a$ 到 $b$，并且概率密度在该区间内保持不变。在此区间之外，概率密度为零。

此分布由两个参数 (parameter)决定：

- $a$：最小可能值（下限）。
- $b$：最大可能值（上限），且 $b > a$。

### 概率密度函数 (PDF)

均匀分布的 PDF，记作 $U(a, b)$，在区间 $[a, b]$ 内定义了这种恒定的概率密度。因为任何 PDF 曲线下的总面积必须等于 1（代表 100% 概率），且区间宽度为 $b-a$，所以 PDF 的高度必须是 $\frac{1}{b-a}$。

PDF 的公式如下：


$$
f(x; a, b) = \begin{cases} \frac{1}{b-a} & \text{当 } a \le x \le b \\ 0 & \text{其他情况} \end{cases}
$$


这个公式告诉我们，如果一个值 $x$ 落入范围 $[a, b]$ 内，概率密度是一个常数正值。如果 $x$ 在此范围之外，概率密度为零，这表示这样的值不会出现。PDF 的图看起来像一个简单的矩形。



![均匀分布PDF (a=2, b=8)](plots/2450-0.json)



> 在区间 $[2, 8]$ 上定义的均匀分布的概率密度函数 (PDF)。在此区间内，密度恒定为 $1/(8-2) = 1/6$，其他地方为零。曲线下的总面积是 $(8-2) \times (1/6) = 1$。

### 概率计算

对于连续分布，概率对应于 PDF 曲线下的面积。对于均匀分布，计算 $X$ 落入子区间 $[c, d]$（在 $a \le c \le d \le b$ 条件下）的概率是很直接的。这就是宽度为 $d-c$、高度为 $\frac{1}{b-a}$ 的矩形面积。


$$
P(c \le X \le d) = \text{宽度} \times \text{高度} = (d-c) \times \frac{1}{b-a} = \frac{d-c}{b-a}
$$


例如，使用我们的 $U(2, 8)$ 分布：

- 获得介于3和5之间的值的概率是 $P(3 \le X \le 5) = \frac{5-3}{8-2} = \frac{2}{6} = \frac{1}{3}$。
- 获得小于或等于4的值的概率是 $P(X \le 4) = P(2 \le X \le 4) = \frac{4-2}{8-2} = \frac{2}{6} = \frac{1}{3}$。（请记住，最小值为 $a=2$）。
- 获得一个恰好等于5的值的概率 $P(X=5)$ 为 0。这对于连续分布中的任何单个点都是成立的，因为单个点没有对应的“面积”（宽度为零）。

### 特性

- **均值（期望值）**：您期望的平均值是区间的中心点：

  
  $$
  \mu = E[X] = \frac{a+b}{2}
  $$
  

  对于 $U(2, 8)$，均值为 $\frac{2+8}{2} = 5$。
- **方差**：方差衡量了分布的离散程度：

  
  $$
  \sigma^2 = Var(X) = \frac{(b-a)^2}{12}
  $$
  

  方差只取决于区间宽度 ($b-a$)。更宽的区间意味着更大的方差。对于 $U(2, 8)$，方差是 $\frac{(8-2)^2}{12} = \frac{6^2}{12} = \frac{36}{12} = 3$。标准差是 $\sigma = \sqrt{3} \approx 1.732$。

### 应用场景

均匀分布常用于我们对某个过程信息了解很少时，除了知道值被限制在某个范围外，且没有理由相信该范围内的任何值比其他值更可能出现。

- 它是计算机中随机数生成的核心（通常均匀生成0到1之间的数字）。
- 它可以用作量化 (quantization)误差的简单模型。
- 在贝叶斯统计中，它有时可用作已知在特定范围内的参数 (parameter)的无信息先验分布。

尽管在机器学习 (machine learning)模型直接输出中，它可能不如正态分布等其他分布常见，但理解均匀分布对于掌握连续概率的原理很重要，同时了解它在模拟和不确定性表示中的作用。稍后，我们将学习如何使用 Python 从这种和其他分布中生成样本。

## 参考资料

- [Pattern Recognition and Machine Learning](https://www.springer.com/gp/book/9780387310732) — Christopher M. Bishop (2006)
  Publisher: Springer; DOI: [10.1007/b139930](https://doi.org/10.1007/b139930)
  一本机器学习综合教材，提供了必要的概率和统计基础，包括连续分布。
- [Introduction to Probability](https://ocw.mit.edu/courses/res-6-012-introduction-to-probability-spring-2018/) — John Tsitsiklis, Patrick Jaillet (2018)
  Publisher: MIT OpenCourseWare
  麻省理工学院在线课程，提供概率论基础讲座、笔记和作业，涵盖连续分布。
- [All of Statistics: A Concise Course in Statistical Inference](https://www.springer.com/gp/book/9780387402727) — Larry Wasserman (2004)
  Publisher: Springer; DOI: [10.1007/978-0-387-21765-7](https://doi.org/10.1007/978-0-387-21765-7)
  一本简明严谨的教材，涵盖了概率和统计推断的核心概念，包括均匀分布。
