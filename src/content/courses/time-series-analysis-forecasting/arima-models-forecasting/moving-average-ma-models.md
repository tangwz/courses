---
course: "time-series-analysis-forecasting"
chapter: "arima-models-forecasting"
lesson: "moving-average-ma-models"
sourceId: 3423
sourceUrl: "https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-4-arima-models-forecasting/moving-average-ma-models"
title: "移动平均 (MA) 模型"
description: "自回归 (AR) 模型关注过去值对当前值的直接作用，而移动平均 (MA) 模型则侧重于另一种时间依赖来源：过去预测误差的影响。想象一下"
order: 2
plots: ["plots/3423-0.json"]
sourceHash: "5670304abe7a07091a7fe2f07dd16db23f648007587d7497c7eb3fb3f847ed47"
sourceCorrections: []
---

自回归 (autoregressive) (AR) 模型捕捉过去*值*对当前值的直接作用。相比之下，移动平均 (MA) 模型则关注另一种时间依赖来源：过去*预测误差*的影响。这可以想象成驾驶一艘船；有时你的调整不仅取决于你之前的位置，还取决于你之前的修正偏离了多少。

MA 模型提出，当前观测值 ($Y\_t$) 是当前误差项 ($\\epsilon\_t$) 和一个或多个过去误差项的线性组合。这些误差项表示过去影响序列的随机冲击或不可预测的分量。

### MA(q) 模型方程

一个 $q$ 阶的移动平均模型，记作 MA(q)，数学定义为：

$Y_t = \mu + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \dots + \theta_q \epsilon_{t-q}$

让我们分解一下这些分量：

- $Y\_t$: 时间序列在当前时间点 $t$ 的值。
- $\\mu$: 序列的均值或基准水平。对于零均值过程，此项可能被省略。
- $\\epsilon\_t$: 时间 $t$ 的白噪声误差项。这代表当前时间步的不可预测冲击。我们假设这些误差是独立同分布的，通常服从均值为零、方差为常数 ($\\sigma^2$) 的正态分布。
- $\\epsilon\_{t-1}, \\epsilon\_{t-2}, \\dots, \\epsilon\_{t-q}$: 前 $q$ 个时间段的误差项。这些是*过去*的预测误差。请注意，这些通常无法直接观测，而是作为模型拟合过程的一部分进行估计。
- $\\theta\_1, \\theta\_2, \\dots, \\theta\_q$: MA 模型的参数 (parameter)。这些系数决定了每个过去误差项对当前观测值 $Y\_t$ 的权重 (weight)或影响。
- $q$: MA 模型的阶数，表明模型中包含多少个过去的误差项。

本质上，MA(q) 模型表明，过去 $q$ 个时期的随机冲击或误差会继续回响并影响序列的当前值。

### MA 模型的特点

- **平稳性：** 与 AR 模型不同，AR 模型根据其参数 (parameter)可能不平稳，而有限阶 MA(q) 模型*总是*弱平稳的。这是因为 $Y\_t$ 被定义为白噪声项的有限线性组合，而白噪声项具有常数均值（零）和常数方差。$Y\_t$ 的均值为 $\\mu$，其方差和自协方差结构不依赖于时间 $t$。
- **有限记忆：** 其主要特点是过去误差项的影响仅持续 $q$ 个周期。误差 $\\epsilon\_{t-k}$ 直接影响 $Y\_{t-k}, Y\_{t-k+1}, \\dots, Y\_{t-k+q}$，但对 $Y\_{t-k+q+1}$ 或后续值没有*直接*影响。这与 AR 模型形成对比，AR 模型中过去值的影响理论上可以无限期持续，尽管通常随时间衰减。

### 使用 ACF 图确定 MA 阶数 (q)

如第 3 章所述，自相关函数 (ACF) 图有助于确定潜在 MA 模型的阶数 $q$。对于一个真实的 MA(q) 过程：

- ACF 在滞后 1 到 $q$ 处将出现统计上显著的尖峰。
- ACF 将在滞后 $q$ 之后突然**截断**。这意味着所有大于 $q$ 的滞后 ($k \> q$) 的自相关将统计上不显著（即，落在零附近的置信区间内）。

考虑 MA(2) 过程的理论 ACF：



![MA(2) 的理论 ACF](plots/3423-0.json)



> MA(2) 过程的理论 ACF 图。在滞后 1 和 2 处存在显著相关性，之后相关性急剧下降到不显著范围（虚线表示置信区间）。

如果您在您的*平稳*时间序列数据（如果需要，在差分之后）的 ACF 图中观察到这种急剧截断的模式，这表明 MA(q) 模型可能适用，其中 $q$ 是 ACF 截断的滞后阶数。反之，MA(q) 过程的偏自相关函数 (PACF) 通常呈“拖尾”现象，意味着它会更缓慢地衰减到零。

### 作为构成要素的 MA 模型

虽然纯 MA 模型有时用于预测，但它们更常作为更一般的 ARMA 和 ARIMA 框架中的组成部分出现。理解 MA 模型如何捕捉对过去误差的依赖，是理解 ARMA 模型如何结合过去值和过去误差信息，以及 ARIMA 模型如何将其扩展到非平稳数据的根本。

在接下来的章节中，我们将了解如何结合 AR 和 MA 分量形成 ARMA 模型，然后加入差分，得到功能多样的 ARIMA 模型类。

## 参考资料

- [Time Series Analysis: Forecasting and Control](https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118675021) — George E. P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, and Greta M. Ljung (2015)
  Publisher: Wiley
  关于ARIMA模型的基础教材，详细介绍了MA模型的理论、性质和识别方法。
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/) — Rob J Hyndman and George Athanasopoulos (2021)
  Publisher: OTexts
  一本易于理解的教材，提供了MA模型及其特性的实用解释和示例。
- [Time Series Analysis](http://pup.princeton.edu) — James Douglas Hamilton (1994)
  Publisher: Princeton University Press; Pages: 799
  一本学术教材，阐述了MA模型和时间序列的数学基础。
