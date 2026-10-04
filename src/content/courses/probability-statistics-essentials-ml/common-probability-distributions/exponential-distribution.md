---
course: "probability-statistics-essentials-ml"
chapter: "common-probability-distributions"
lesson: "exponential-distribution"
sourceId: 1373
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-2-common-probability-distributions/exponential-distribution"
title: "指数分布"
description: "了解指数分布如何模拟事件发生时间。"
order: 5
plots: ["plots/1373-0.json"]
sourceHash: "23162c2361fd5f2c6f1f688afb493812a6a2669226eb2424a2b2a16c7e648ef5"
sourceCorrections: []
---

继伯努利、二项和泊松等离散分布，以及连续均匀和正态分布之后，我们现在考察另一种基本的连续概率分布：指数分布。这种分布常用于模拟在一个过程中，事件以恒定平均速率独立发生，直到某个特定事件出现所需的时间。

### 指数分布

设想您正在追踪那些随机但以稳定的平均速率随时间发生的事件，例如顾客抵达服务台或放射性衰变事件。泊松分布有助于模拟在固定时间间隔内事件发生的*数量*，而指数分布则模拟连续事件*之间的时间*，或等待下一个事件发生的时间。

指数分布由一个正参数 (parameter) $\lambda$（lambda）表示，称为速率参数。这个参数表示每单位时间（或空间，或其他连续介质）的平均事件数。

#### 概率密度函数 (PDF)

指数分布随机变量 $T$（表示时间）的概率密度函数（PDF）如下：


$$
f(t; \lambda) = \begin{cases} \lambda e^{-\lambda t} & \text{当 } t \ge 0 \\ 0 & \text{当 } t < 0 \end{cases}
$$


其中：

- $t$ 是时间变量（必须非负）。
- $\lambda$ 是速率参数（$\lambda > 0$）。
- $e$ 是自然对数的底数（约 2.71828）。

PDF $f(t; \lambda)$ 描述了事件在特定时间 $t$ 发生的相对可能性。请注意，概率密度在 $t=0$ 时最高，并随着 $t$ 的增加呈指数下降。更高的速率 $\lambda$ 会导致更快的下降，表示预期的等待时间更短。相反，较低的 $\lambda$ 会导致较慢的下降和更长的预期等待时间。



![指数分布PDF](plots/1373-0.json)



> 速率参数 $\lambda = 0.5$（蓝色）和 $\lambda = 1.5$（青色）的指数分布概率密度函数。速率越高，衰减越快。

#### 累积分布函数 (CDF)

累积分布函数（CDF），$F(t; \lambda)$，表示事件在时间 $t$ 或之前发生的概率，即 $P(T \le t)$。它通过对PDF从0到 $t$ 进行积分计算得出：


$$
F(t; \lambda) = \int_{0}^{t} \lambda e^{-\lambda x} dx = 1 - e^{-\lambda t} \quad \text{当 } t \ge 0
$$


当 $t < 0$ 时，$F(t; \lambda) = 0$。

CDF从0开始，随着 $t$ 趋近无穷大，趋近于1。它表示事件随时间推进而累积的概率。

### 特性

#### 无记忆性

指数分布具有一种独特而重要的特性，称为无记忆性。数学上表示为：


$$
P(T > s + t \mid T > s) = P(T > t) \quad \text{对于所有 } s, t \ge 0
$$


简单来说，如果一个事件在时间 $s$ 之前没有发生，它在至少额外时间 $t$ 内不会发生的概率，与它最初在时间 $t$ 内不会发生的概率相同。该过程基本上“忘记”了它已经等待了多久。

考虑模拟不会磨损的组件（故障是纯随机的）的寿命。如果该组件已运行100小时，它再运行50小时的概率，与一个新组件能够运行50小时的概率相同。这个特性使得指数分布适合模拟那些其过去状态不影响事件在下一时刻发生概率的现象。

#### 均值和方差

指数分布随机变量 $T$ 的期望值（均值）和方差与速率参数 (parameter) $\lambda$ 直接相关：

- **均值（期望值）：** $E[T] = \frac{1}{\lambda}$
- **方差：** $Var(T) = \frac{1}{\lambda^2}$

均值 $1/\lambda$ 表示事件发生的平均等待时间。这与直觉相符：如果事件的速率（$\lambda$）高，则事件之间的平均时间（$1/\lambda$）应低，反之亦然。标准差也是 $1/\lambda$，这意味着分布的离散程度等于其均值。

### 应用及与泊松分布的关联

指数分布广泛应用于各个方面，包括：

- **排队论：** 模拟顾客的到达间隔时间或服务时间。
- **可靠性工程：** 模拟电子组件或系统在恒定故障率假设下的寿命。
- **物理学：** 模拟放射性衰变时间。
- **金融：** 模拟大市场波动之间的时间。

指数分布与泊松分布之间存在直接关系。如果事件根据泊松过程发生，平均速率为每单位时间 $\lambda$ 个事件，那么连续事件之间的等待时间是独立同分布的指数随机变量，具有相同的速率参数 (parameter) $\lambda$。这种二元性很有用：如果您知道事件发生的平均速率（泊松分布），您就知道了等待时间（指数分布）的分布，反之亦然。

总之，指数分布为那些以恒定平均速率和无记忆性为特点的过程，提供了模拟事件发生时间的简单而有效的模型。理解其PDF、CDF和特性对于模拟数据分析和机器学习 (machine learning)中常见的时间-事件数据很有用。

## 参考资料

- [Mathematics for Machine Learning](https://mml-book.github.io/) — Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong (2020)
  Publisher: Cambridge University Press; DOI: [10.1017/9781108679904](https://doi.org/10.1017/9781108679904)
  涵盖机器学习的基本数学概念，包括指数分布等概率分布，并以与机器学习从业者相关的背景进行解释。
- [All of Statistics: A Concise Course in Statistical Inference](https://link.springer.com/book/10.1007/978-0-387-21736-9) — Larry Wasserman (2004)
  Publisher: Springer; DOI: [10.1007/978-0-387-21736-9](https://doi.org/10.1007/978-0-387-21736-9)
  一本简洁而全面的统计推断资源，为数据科学领域的学生和研究人员提供了对概率分布（包括指数分布）的清晰解释。
- [Introduction to Probability and Statistics (Course 6.041SC)](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-041sc-probabilistic-systems-analysis-and-applied-probability-fall-2010/) — John Tsitsiklis (2010)
  Publisher: MIT OpenCourseWare
  麻省理工学院开放课程的官方教材，从工程和应用科学的角度提供关于概率分布（包括指数分布）的基础讲座和笔记。
