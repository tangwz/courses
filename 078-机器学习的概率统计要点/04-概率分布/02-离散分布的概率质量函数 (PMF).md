# 离散分布的概率质量函数 (PMF)

来源：[原文](https://apxml.com/zh/courses/probability-statistics-fundamentals-ml/chapter-4-probability-distributions/probability-mass-function)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在上一节中，我们明确了概率分布描述了随机变量不同结果的发生可能性。我们还指出了能取离散、独立值的变量（离散型）与能在某个范围内取任意值的变量（连续型）之间的主要区别。现在，我们来关注如何精确定义离散随机变量的概率。

想象一个随机过程，它有有限或可数无限个可能结果，例如掷骰子或计算广告点击次数。我们如何为每个具体结果指定概率呢？这正是\*\*概率质量函数 (PMF)\*\*的作用所在。

## 什么是概率质量函数 (PMF)？

概率质量函数，常缩写为 PMF，是一个给出离散随机变量 $X$ 恰好等于某个特定值 $x$ 的概率的函数。可以把它看作是将总概率（总是 1）分配给所有可能的离散结果，为每个值分配一定的“概率质量”。

在数学上，我们用 $P(X=x)$ 或有时用 $p(x)$ 来表示 PMF。此函数必须满足两个主要条件：

1. **非负性：** 赋予任何特定值 $x$ 的概率必须大于或等于零。事情发生的几率不可能是负数。
   $P(X=x) \ge 0 \quad \text{对于所有可能的 } x \text{ 值}$
2. **总和为一：** 如果将离散随机变量 $X$ 可以取的所有可能值的概率加起来，总和必须等于 1。这表示我们已考虑了所有可能的结果。
   $\sum_{x} P(X=x) = 1$
   其中求和是对 $X$ 的样本空间中所有可能的 $x$ 值进行的。

## 例子：掷一个公平的六面骰子

让我们使用一个常见的例子：掷一个标准的、公平的六面骰子。
随机变量，我们称之为 $X$，表示掷骰子的结果。
$X$ 的可能取值是 1 到 6 的整数：$\{1, 2, 3, 4, 5, 6\}$。这是一个离散的结果集。

由于骰子是公平的，每个结果发生的概率均等。有 6 种可能的结果，所以掷出任何特定数字的概率是 $1/6$。

随机变量 $X$ 的 PMF 可以写为：


$$
P(X=x) = \begin{cases}
1/6 & \text{如果 } x \in \{1, 2, 3, 4, 5, 6\} \\
0 & \text{否则}
\end{cases}
$$


让我们检查这两个条件：

1. **非负性：** 对于所有 $x$，$P(X=x) \ge 0$ 吗？是的，对于可能的结果，概率是 $1/6$，否则是 $0$，两者都是非负的。
2. **总和为一：** 概率之和等于 1 吗？
   $\sum_{x} P(X=x) = P(X=1) + P(X=2) + P(X=3) + P(X=4) + P(X=5) + P(X=6)$
   $= \frac{1}{6} + \frac{1}{6} + \frac{1}{6} + \frac{1}{6} + \frac{1}{6} + \frac{1}{6} = 6 \times \frac{1}{6} = 1$
   是的，概率总和为 1。

因此，这个函数 $P(X=x) = 1/6$（对于 $x=1..6$）确实是公平掷骰子的 PMF。

## PMF 的可视化

我们可以使用条形图来可视化 PMF。每个条形的高度代表了每个可能结果 $x$ 的概率 $P(X=x)$。



[交互图表：公平六面骰子掷出结果的 PMF](https://apxml.com/zh/courses/probability-statistics-fundamentals-ml/chapter-4-probability-distributions/probability-mass-function#plot-baytvx)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "公平六面骰子掷出结果的 PMF",
    "xaxis": {
      "title": "结果 (x)",
      "dtick": 1
    },
    "yaxis": {
      "title": "概率 P(X=x)",
      "range": [
        0,
        0.2
      ]
    }
  },
  "data": [
    {
      "type": "bar",
      "x": [
        1,
        2,
        3,
        4,
        5,
        6
      ],
      "y": [
        0.16666666666666666,
        0.16666666666666666,
        0.16666666666666666,
        0.16666666666666666,
        0.16666666666666666,
        0.16666666666666666
      ],
      "marker": {
        "color": "#228be6"
      }
    }
  ]
}
```

</details>



> 概率质量平均分布在掷骰子的六个可能结果上。每个结果的概率质量为 1/6。

PMF 清楚地展示了概率是如何分布在可能的离散值上的。记住，PMF 给出的是随机变量 *恰好* 等于某个特定值的概率，这一点很重要。这与连续分布不同，在连续分布中，命中任何一个精确值的概率为零，我们而是使用概率密度函数 (PDF) 来讨论区间上的概率，我们很快会讲到这一点。

理解 PMF 是处理特定离散概率分布的第一步，例如伯努利分布和二项分布，这些在机器学习 (machine learning)中建模二元结果或计数数据时经常会遇到。

## 参考资料

- [Applied Statistics and Probability for Engineers](https://www.wiley.com/en-us/Applied+Statistics+and+Probability+for+Engineers%2C+7th+Edition-p-9781119747900) — Douglas C. Montgomery, George C. Runger (2018)
  Publisher: John Wiley & Sons
  一本基础教材，涵盖应用概率和统计学，清晰解释了工程和科学应用中的离散分布和概率质量函数。
- [Introduction to Probability and Statistics (18.05)](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2014/) — Jeremy Orloff, Jonathan Bloom (2014)
  Publisher: MIT OpenCourseWare
  来自知名大学课程的在线材料，提供概率学讲座和资源，包括离散随机变量及其概率质量函数。

---

[上一节](01-%E4%BB%80%E4%B9%88%E6%98%AF%E6%A6%82%E7%8E%87%E5%88%86%E5%B8%83%EF%BC%9F.md) · [下一节](03-%E7%A6%BB%E6%95%A3%E5%88%86%E5%B8%83%EF%BC%9A%E4%BC%AF%E5%8A%AA%E5%88%A9.md)
