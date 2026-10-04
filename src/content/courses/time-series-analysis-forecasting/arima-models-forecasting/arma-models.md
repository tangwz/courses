---
course: "time-series-analysis-forecasting"
chapter: "arima-models-forecasting"
lesson: "arma-models"
sourceId: 3426
sourceUrl: "https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-4-arima-models-forecasting/arma-models"
title: "结合AR和MA：ARMA模型"
description: "你已经了解了自回归（AR）模型（根据过去值预测未来值）和移动平均（MA）模型（根据过去预测误差预测未来值）。虽然它们单独使用有帮助，但许多实际中的平稳时间序列"
order: 3
plots: []
sourceHash: "edf2a841eac3cabe0658302e6c6778b1f332e7f6da049db9b54b1dabdf618a70"
sourceCorrections: []
---

自回归 (autoregressive)（AR）模型根据过去值预测未来值，而移动平均（MA）模型根据过去的预测误差预测未来值。尽管这些模型单独使用时有帮助，但许多实际中的平稳时间序列所呈现的相关结构，并非纯粹的AR或纯粹的MA过程单独能够完全捕捉。其变化可能同时取决于近期值和近期的冲击或误差。

在这种情况下，自回归移动平均（ARMA）模型便派上用场。它通过将AR和MA两个部分结合到一个模型中，提供了一个更灵活的框架。ARMA模型假设序列的当前值与其自身的先前值和先前的误差项线性相关。

### ARMA(p, q)模型的结构

ARMA模型表示为ARMA(p, q)，其中：

- `p` 是自回归 (autoregressive)（AR）部分的阶数。
- `q` 是移动平均（MA）部分的阶数。

时间序列 $Y\_t$ 的ARMA(p, q)模型方程为：

$Y_t = c + \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + ... + \phi_p Y_{t-p} + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + ... + \theta_q \epsilon_{t-q} + \epsilon_t$

我们来分解一下：

- $Y\_t$：时间序列在时间 $t$ 的值。
- $c$：常数项（基线或截距）。
- $\\phi\_1, ..., \\phi\_p$：AR项的系数（$p$个过去值 $Y\_{t-1}$ 到 $Y\_{t-p}$）。
- $\\theta\_1, ..., \\theta\_q$：MA项的系数（$q$个过去误差项 $\\epsilon\_{t-1}$ 到 $\\epsilon\_{t-q}$）。
- $\\epsilon\_t$：时间 $t$ 的白噪声误差项。它表示序列在时间 $t$ 的不可预测部分。
- $\\epsilon\_{t-1}, ..., \\epsilon\_{t-q}$：过去误差项。它们是实际值与模型在先前时间点的预测值之间的差异。

本质上，该模型说明当前观测值 $Y\_t$ 是 $p$ 个过去观测值（AR部分）、 $q$ 个过去预测误差（MA部分）、一个常数项以及当前误差项的加权和。

> ARMA(p, q)模型的流程，展示了过去值和过去误差如何帮助预测当前值。

### 平稳性假设

直接应用ARMA模型的一个重要前提是，时间序列 $Y\_t$ 必须是**平稳的**。回顾第二章，平稳性意味着序列具有恒定的均值、恒定的方差和随时间不变的自相关结构。如果你的序列表现出趋势或季节性，它就是非平稳的，在这种情况下，在未将数据转换为平稳状态之前，ARMA模型并不适用（这会引出我们接下来要讨论的ARIMA模型）。

### ARMA与ACF和PACF的关联

在第三章中，你了解了自相关函数（ACF）图和偏自相关函数（PACF）图如何帮助识别平稳时间序列的结构。让我们回顾一下它们在ARMA模型背景下的典型模式：

1. **纯AR(p)模型：**
   - ACF：逐渐衰减（呈指数或正弦曲线状）。
   - PACF：在滞后 `p` 后突然截断。
2. **纯MA(q)模型：**
   - ACF：在滞后 `q` 后突然截断。
   - PACF：逐渐衰减。
3. **ARMA(p, q)模型：**
   - ACF：在滞后 `q` 后逐渐衰减。
   - PACF：在滞后 `p` 后逐渐衰减。

对于混合ARMA过程，ACF和PACF通常都会逐渐衰减至零。具体的衰减速度可以提供关于 `p` 和 `q` 的线索，但与纯AR或MA过程相比，解释ARMA模型的这些模式通常更具挑战性。通常，你会使用这些图来初步确定可能的 `p` 和 `q` 值，然后依靠模型诊断和信息准则（如AIC或BIC，第六章会提到）在拟合过程中调整模型阶数。

通过结合AR和MA项，ARMA模型能够比单独的AR或MA模型捕捉到更广泛的平稳数据时间依赖性。这使得它们成为你时间序列分析工具箱中的一种有用工具。然而，对平稳性的要求是一个局限性，我们接下来将通过引入“差分”部分来解决，从而引出完整的ARIMA模型。

## 参考资料

- [Time Series Analysis: Forecasting and Control](https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118675021) — George E. P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, Greta M. Ljung (2015)
  Publisher: John Wiley & Sons; DOI: [10.1002/9781118675021](https://doi.org/10.1002/9781118675021)
  关于ARIMA模型的经典教材，涵盖AR、MA和ARMA的理论、识别、估计和预测。
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/) — Rob J Hyndman and George Athanasopoulos (2021)
  Publisher: OTexts
  一本全面且易懂的在线教材，在时间序列预测的背景下提供了ARMA模型的实用指导。
- [Time Series Analysis with Applications in R](https://doi.org/10.1007/978-0-387-75933-1) — Jonathan D. Cryer and Kung-Sik Chan (2008)
  Publisher: Springer; Pages: 491; DOI: [10.1007/978-0-387-75933-1](https://doi.org/10.1007/978-0-387-75933-1)
  结合了ARMA模型的理论基础和实际应用，并提供了R语言实现的示例。
