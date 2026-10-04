---
course: "time-series-analysis-forecasting"
chapter: "arima-models-forecasting"
lesson: "autoregressive-ar-models"
sourceId: 3421
sourceUrl: "https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-4-arima-models-forecasting/autoregressive-ar-models"
title: "自回归 (AR) 模型"
description: "自回归 (AR) 模型是 ARIMA 家族的基本组成部分之一。AR 模型的核心思想简单而有效：时间序列的当前值可以通过其自身先前值的线性组合来预测。这"
order: 1
plots: ["plots/3421-0.json"]
sourceHash: "f1d08a39346591d537d750b00e0f4b7108260c0cad79d54bd228b45025a9b0a9"
sourceCorrections: []
---

自回归 (autoregressive) (AR) 模型是 ARIMA 家族的基本组成部分之一。AR 模型的核心思想简单而有效：时间序列的当前值可以通过其自身先前值的线性组合来预测。这直接模拟了序列对其过往值的依赖性。

### 定义与结构

一个 $p$ 阶的自回归 (autoregressive)模型，表示为 AR(p)，根据前 $p$ 个值（$Y\_{t-1}, Y\_{t-2}, \\dots, Y\_{t-p}$）预测当前值 $Y\_t$。其数学表达式为：

$Y_t = c + \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + \dots + \phi_p Y_{t-p} + \epsilon_t$

我们来分析一下这些项：

- $Y\_t$: 时间序列在当前时间点 $t$ 的值。
- $c$: 常数项（截距）。
- $\\phi\_1, \\phi\_2, \\dots, \\phi\_p$: 这些是自回归系数。它们表示应用于过往值的权重 (weight)。例如，$\\phi\_1$ 表示前一个值（$Y\_{t-1}$）对当前值（$Y\_t$）的影响程度。
- $Y\_{t-1}, Y\_{t-2}, \\dots, Y\_{t-p}$: 时间序列在之前时间点（滞后 1, 2, ..., p）的值。
- $p$: AR 模型的阶数，表示回归中包含多少个过往值。
- $\\epsilon\_t$: 在时间 $t$ 的误差项。这表示 $Y\_t$ 中未被过往值解释的部分。对于标准 AR 模型，此项假定为白噪声，这意味着它具有零均值、常数方差，并且在时间上不相关。

### 直观理解与阶数选择

可以把 AR(1) 模型（$p=1$）看作：它表明当前值 $Y\_t$ 主要依赖于紧邻的前一个值 $Y\_{t-1}$，加上一个常数和一些随机噪声：

$Y_t = c + \phi_1 Y_{t-1} + \epsilon_t$

如果 $\\phi\_1$ 为正，昨天的高值表明今天也可能是高值。如果为负，昨天的高值表明今天可能是低值。AR(2) 模型会使用最近的两个过往值（$Y\_{t-1}$ 和 $Y\_{t-2}$），依此类推。

我们如何确定合适的阶数 $p$ 呢？正如第 3 章所讨论的，偏自相关函数（PACF）图在这里非常有帮助。对于一个纯 AR(p) 过程，PACF 图通常表现出：

1. 直至滞后 $p$ 的显著相关性。
2. 在滞后 $p$ 之后急剧截断，随后的偏自相关值接近于零（在置信区间内）。

在你的（平稳）时间序列的 PACF 中观察到这种模式，表明 AR(p) 模型可能是一个好的起点。



![AR(1) 过程示例 (phi=0.7)](plots/3421-0.json)



> 一个表现出 AR(1) 行为的示例时间序列。请注意，连续点在数值上趋于相对接近，反映了前一点的影响。

### 平稳性要求

需要记住，标准 AR 模型假定时间序列 $Y\_t$ 是平稳的。它的均值、方差和自相关结构不应随时间变化。如果你的数据是非平稳的（例如，表现出趋势或季节性），在应用 AR 模型之前，你通常需要对其进行变换，通常通过差分（如第 2 章所述）。这个差分步骤是 ARIMA 中由 'I' 表示的“积分”部分，我们将在本章稍后将其组合。

AR 模型捕获时间序列数据中一种特定类型的依赖结构。它们通过解释过往*值*如何影响当前来为更复杂的模型奠定基础。接下来，我们将查看移动平均 (MA) 模型，这类模型侧重于过往*预测误差*的作用。

## 参考资料

- [Time Series Analysis: Forecasting and Control](https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118675021) — George E. P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, and Greta M. Ljung (2015)
  Publisher: Wiley; DOI: [10.1002/9781118675021](https://doi.org/10.1002/9781118675021)
  一本经典且权威的教科书，为时间序列分析提供了全面的基础，包括AR模型、平稳性和使用PACF进行模型识别。
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/) — Rob J Hyndman and George Athanasopoulos (2021)
  Publisher: OTexts
  一本高度易懂且实用的指南，涵盖AR模型、用于阶数选择的PACF以及时间序列预测的广泛背景下的平稳性。可在线免费获取。
- [Time Series Analysis and Its Applications: With R Examples](https://doi.org/10.1007/978-3-319-52452-8) — Robert H. Shumway, David S. Stoffer (2017)
  Publisher: Springer; DOI: [10.1007/978-3-319-52452-8](https://doi.org/10.1007/978-3-319-52452-8)
  一本全面的学术教科书，提供了时间序列模型的详细理论解释和实际应用，包括AR过程、平稳性和模型识别。
