---
course: "probability-statistics-fundamentals-ml"
chapter: "probability-distributions"
lesson: "probability-density-function"
sourceId: 2448
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-fundamentals-ml/chapter-4-probability-distributions/probability-density-function"
title: "连续分布的概率密度函数 (PDF)"
description: "了解用于描述连续概率分布的概率密度函数 (PDF)。"
order: 5
plots: ["plots/2448-0.json"]
sourceHash: "3faa480a1b64c4ebda9d47f06f945c23db5281d7757b429dadb8e8291f79cd9e"
sourceCorrections: []
---

概率质量函数 (PMF) 描述离散随机变量，例如三次掷硬币中正面朝上的次数。对于离散变量，可以列出所有可能的结果，并为每个结果指定一个特定的概率。PMF 给出 $P(X=x)$，即随机变量 $X$ 取精确值 $x$ 的概率。

但是，当变量可以在连续范围内取 *任何* 值时，会发生什么呢？考虑测量某人的精确身高、准确温度，或者某个过程完成所需的时间。这些都是连续随机变量。如果我们尝试为单个无限精确的值（例如 *精确* 到 175.0000... 厘米的身高）指定一个概率，那么这个概率实际上将是零。可能性实在太多了！

对于连续变量，我们不关注单个点的概率，而是讨论变量落入某个特定 *区间* 的概率。这就是 **概率密度函数 (PDF)** 的作用。

### 理解概率密度函数 (PDF)

PDF，通常记作 $f(x)$，是一个函数，它描述了连续随机变量取某个给定值的相对可能性。与 PMF 不同，PDF 在特定点 $x$ 上的值 $f(x)$ 本身 *不是* 概率。相反，PDF 曲线在两点之间（例如 $a$ 和 $b$ 之间）的 **曲线下面积** 代表了随机变量 $X$ 落入该区间 $[a, b]$ 的概率。

从数学上讲，这可以用积分表示：

$P(a \le X \le b) = \int_{a}^{b} f(x) dx$

如果你以前没有见过 $\int$ 符号或微积分，请不要担心。其重要思想是 **面积 = 概率**。对于我们将遇到的许多标准分布，我们不需要手动执行积分；我们可以使用表格或软件函数。

可以这样想：想象一个具有连续值的大型数据集的直方图。当你把直方图的条形（bins）变得越来越窄时，条形的顶部会开始形成一条平滑的曲线。这条平滑曲线就代表了 PDF。曲线上任何点的高度都表明了数值在哪里更密集地聚集。

### PDF 的性质

一个有效的 PDF 必须满足两个主要性质：

1. **非负性：** 密度函数对于 $x$ 的所有可能值必须始终大于或等于零。不可能出现负的可能性。
   $f(x) \ge 0 \text{ 对于所有 } x$
2. **总面积为 1：** 由于随机变量必然取 *某个* 值，因此 PDF 整个曲线下的总面积必须等于 1。这表示 100% 的概率。
   $\int_{-\infty}^{\infty} f(x) dx = 1$

### 理解 PDF

需要记住的是：

- 值 $f(x)$ 表示密度，而非概率。如果 $f(x)$ 的值高于 $f(y)$，这意味着变量在 $x$ 附近的小区间内出现的可能性比在 $y$ 附近同样大小的小区间内出现的可能性更大。对于某些分布，即使 $f(x)$ 的值可能大于 1，只要曲线下的总面积保持为 1，也是可能的。
- 对于连续变量 $X$，它取任何 *单个特定* 值的概率为零：$P(X = c) = 0$。这是因为在单个点处曲线下没有“面积”（从 $c$ 到 $c$ 的区间宽度为零）。这也意味着 $P(a \le X \le b)$ 与 $P(a < X < b)$ 相同。

我们将其可视化。考虑一个通用 PDF 曲线：



![概率密度函数 (PDF) 示意图](plots/2448-0.json)



> 阴影区域表示概率 $P(2 \le X \le 4)$。该概率等于 $x=2$ 和 $x=4$ 之间蓝色曲线下的面积。曲线 $f(x)$ 的高度表明了值 $x$ 周围的概率密度。

### PDF 与 PMF：快速比较

| 特征 | PMF (离散) | PDF (连续) |
| --- | --- | --- |
| **适用范围** | 离散随机变量 | 连续随机变量 |
| **函数值** | $P(X=x)$ (点概率) | $f(x)$ (点密度) |
| **概率** | 集合上 PMF 值的求和 | 区间上 PDF 曲线下的面积（积分） |
| **取值范围** | $0 \le P(X=x) \le 1$ | $f(x) \ge 0$ (可大于 1) |
| **求和/积分** | $\sum_{\text{所有 } x} P(X=x) = 1$ | $\int_{-\infty}^{\infty} f(x) dx = 1$ |

理解 PDF 在处理机器学习 (machine learning)中的连续测量数据时非常重要。许多算法假定特征服从特定的分布（例如我们接下来将要了解的正态分布），PDF 允许我们有效地建模和处理这些连续量。它们帮助我们量化 (quantization)不确定性，并对连续结果进行概率表述。

## 参考资料

- [Applied Statistics and Probability for Engineers](https://www.wiley.com/en-us/Applied+Statistics+and+Probability+for+Engineers,+7th+Edition-p-9781119400301) — Douglas C. Montgomery, George C. Runger (2018)
  Publisher: Wiley
  本书涵盖了概率分布的基础知识，包括连续随机变量及其概率密度函数，并强调数据科学相关的应用。
- [All of Statistics: A Concise Course in Statistical Inference](https://www.stat.cmu.edu/~larry/all-of-statistics/) — Larry Wasserman (2004)
  Publisher: Springer
  这本精炼而严谨的教材涵盖了统计推断的概念，包括对概率密度函数的详尽讨论，适合对数学有浓厚兴趣的读者。
- [Lecture Notes on Continuous Random Variables and PDFs](https://ocw.mit.edu/courses/mathematics/18-440-probability-and-random-variables-spring-2014/lecture-notes/) — Scott Sheffield (2014)
  Publisher: MIT OpenCourseWare
  源自麻省理工学院基础课程的讲义，清晰地解释了连续随机变量和概率密度函数，并附有示例和插图。
