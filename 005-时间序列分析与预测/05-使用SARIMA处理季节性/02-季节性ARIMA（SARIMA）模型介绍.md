# 季节性ARIMA（SARIMA）模型介绍

来源：[原文](https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-5-seasonal-arima-sarima/sarima-introduction)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管ARIMA模型是进行时间序列建模的强大工具，但在处理具有明显重复季节性模式的数据时，它们存在局限。仅通过非季节性AR或MA项来捕捉季节性，通常需要非常高的阶数，从而导致模型过于复杂，可能无法准确反映其固有的季节性结构。

为了解决这个问题，我们引入了\*\*季节性自回归 (autoregressive)积分移动平均（SARIMA）\*\*模型。SARIMA通过明确地在模型结构中加入季节性成分，扩展了基本的ARIMA框架。这使得它特别适合于模式在固定周期内重复的时间序列数据，例如，显示年度峰值的月销售数据或表现出每周循环的每日网站流量。

### SARIMA表示法理解

SARIMA模型通常表示为 $SARIMA(p, d, q)(P, D, Q)_m$。我们来具体看看这种表示法：

- $(p, d, q)$: 这些是**非季节性**参数 (parameter)，与你在上一章学到的标准ARIMA模型中的参数完全相同。
  - $p$: 非季节性自回归 (autoregressive)（AR）阶数。
  - $d$: 非季节性差分阶数。
  - $q$: 非季节性移动平均（MA）阶数。
- $(P, D, Q)$: 这些表示模型的**季节性**成分。它们与非季节性成分相似，但在季节性滞后上发挥作用。
  - $P$: 季节性自回归（AR）阶数。它捕获当前观测值与前几个季节观测值之间的关系。
  - $D$: 季节性差分阶数。它通过减去相隔一个完整季节的观测值来处理季节性趋势。
  - $Q$: 季节性移动平均（MA）阶数。它模拟当前误差与前几个季节误差之间的关系。
- $m$: 这是一个重要的参数，表示**季节周期**或频率。它是完整季节周期中的时间步长数量。例如：
  - $m = 12$：对于具有年度季节性的月度数据。
  - $m = 4$：对于具有年度季节性的季度数据。
  - $m = 7$：对于具有每周季节性的每日数据。
  - $m = 52$：对于具有年度季节性的每周数据。

### SARIMA的工作原理

可以将SARIMA模型看作是结合了两个过程：一个建模非季节性动态，另一个建模季节性动态。

1. **季节性差分：** 如果 $D > 0$，模型首先应用季节性差分（$y_t - y_{t-m}$）来消除或减少季节性趋势。这有助于稳定序列的季节性成分。
2. **非季节性差分：** 如果 $d > 0$，模型接着应用非季节性差分（$y'_t - y'_{t-1}$，其中 $y'$ 是当 $D>0$ 时的季节性差分序列，否则 $y'=y$）来处理非季节性趋势，并使序列在常规意义上平稳。
3. **ARMA建模：** 最后，将类似ARMA的结构应用于差分后的序列。这种结构包括：
   - 非季节性AR（$p$）和MA（$q$）项，捕捉短期滞后（例如，滞后1，滞后2）处的关联。
   - 季节性AR（$P$）和MA（$Q$）项，捕捉季节性滞后（例如，滞后 $m$，滞后 $2m$）处的关联。

例如，对于月度数据（$m=12$），一个 $SARIMA(1, 1, 1)(1, 1, 1)_{12}$ 模型表示：

- 非季节性：一个AR(1)项，一阶常规差分（$d=1$），以及一个MA(1)项。
- 季节性：一个季节性AR(1)项（与滞后12的关联），一阶季节性差分（$D=1$，$y_t - y_{t-12}$），以及一个季节性MA(1)项（滞后12处误差之间的关联）。

通过加入这些独特的季节性参数 (parameter) $(P, D, Q)_m$，SARIMA提供了一种更有条理、更易于理解的方式来建模受可预测的周期性模式影响的时间序列。后续部分将指导您使用ACF/PACF图确定合适的季节性阶数，并使用`statsmodels`库在Python中实现这些模型。

## 参考资料

- Time Series Analysis: Forecasting and Control — George E. P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, and Greta M. Ljung (2015)
  Publisher: Wiley; Pages: 712
  这是一本经典权威著作，全面阐述了ARIMA和SARIMA模型的理论及其应用。
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/) — Rob J Hyndman and George Athanasopoulos (2021)
  Publisher: OTexts
  这是一本易于理解的在线教科书，提供时间序列预测的实用指导，其中包含关于SARIMA等季节性模型的专门章节。
- [statsmodels.tsa.statespace.sarimax.SARIMAX](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html) — statsmodels developers (2023)
  Python statsmodels库中SARIMAX类的官方文档，它实现了SARIMA模型。该文档详细说明了参数和使用方法，以便于实际实现。

---

[上一节](01-ARIMA%E6%A8%A1%E5%9E%8B%E5%A4%84%E7%90%86%E5%AD%A3%E8%8A%82%E6%80%A7%E6%95%B0%E6%8D%AE%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7.md) · [下一节](03-%E8%AF%86%E5%88%AB%E5%AD%A3%E8%8A%82%E6%80%A7%E6%88%90%E5%88%86%20%28ACF-PACF%29.md)
