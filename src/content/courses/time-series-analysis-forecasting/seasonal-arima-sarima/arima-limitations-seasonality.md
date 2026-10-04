---
course: "time-series-analysis-forecasting"
chapter: "seasonal-arima-sarima"
lesson: "arima-limitations-seasonality"
sourceId: 3447
sourceUrl: "https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-5-seasonal-arima-sarima/arima-limitations-seasonality"
title: "ARIMA模型处理季节性数据的局限性"
description: "了解为什么标准ARIMA模型可能无法充分捕捉强烈的季节性模式。"
order: 1
plots: ["plots/3447-0.json", "plots/3447-1.json"]
sourceHash: "4a775df27c7c541faebc1a69063adce9641b54f47825c2daefc5aa563b0c5919"
sourceCorrections: []
---

自回归 (autoregressive)积分滑动平均（ARIMA）模型，记作$ARIMA(p, d, q)$，是捕捉时间序列数据中序列相关性和趋势的有用工具。这些模型通过将序列的当前值与其自身的历史值（AR部分）以及过去的预测误差（MA部分）关联起来发挥作用，通常在应用差分以实现平稳性之后（I部分）。

"然而，许多时间序列都表现出**季节性**，这是一种在固定已知周期内重复出现的模式。可以想想月度零售销售数据常在节假日之前达到峰值、季度公司财报，或者每日网站流量显示出工作日/周末的变化。这些周期以规律的间隔出现（例如，每12个月、4个季度或7天）。"

标准的$ARIMA(p, d, q)$模型在处理强烈的季节性时面临挑战。原因如下：

1. **侧重于短期滞后：** ARIMA中的$p$和$q$参数 (parameter)主要捕捉观测值与紧随其前的观测值之间的依赖关系（滞后1、滞后2等）。尽管差分（$d$）处理趋势，但它不能固有地考虑到观测值与恰好一季前的另一个观测值之间的关系（例如，今年12月与去年12月）。
2. **忽视季节性依赖：** 强烈的季节性模式意味着时间$t$的值通常与时间$t-m$的值高度相关，其中$m$是季节周期（例如，月度数据$m=12$）。$ARIMA(p, d, q)$模型没有内置机制来直接纳入这种滞后$m$依赖关系，而不会使$p$或$q$变得非常大。
3. **模型复杂性：** 要强制标准ARIMA模型捕捉滞后$m$处的季节性效应，你可能需要将$p$或$q$设置为至少$m$。对于月度数据（$m=12$），这可能意味着包含12个或更多AR或MA参数。这类模型变得笨重，难以可靠地估计，并且容易过度拟合数据中的非季节性波动。大量参数无法有效地针对特定的季节性结构。
4. **差分效率低下：** 尽管差分阶数$d$有助于消除趋势，但它不能有效地处理季节性模式。一种不同类型的差分，即**季节性差分**，它计算一个观测值与上一季节相应观测值之间的差值（$y_t - y_{t-m}$），通常是必需的。这不属于标准ARIMA框架的一部分。

考虑一个代表空调月度销售额的时间序列，其销售额可能每年在夏季月份达到峰值。



![月度空调销售额（示例）](plots/3447-0.json)



> 示例月度销售数据显示出明显的年度季节性模式。销售额每年在6-8月（夏季）达到峰值。

一个尝试预测明年7月销售额的ARIMA模型将主要查看6月、5月、4月等的销售额（滞后1、2、3...）。尽管这些近期值提供了一些信息，但由于需求的季节性，*去年*7月的销售额（滞后12）通常是更强的预测因素。标准ARIMA模型不明确使用这个滞后12的信息，除非$p$或$q$设置为12或更高，这，如前所述，会创建一个效率低下的模型结构。

此外，如果我们查看这类季节性数据（在可能使其平稳之后）的自相关函数（ACF）图，我们经常会看到明显的自相关性，这不仅仅在短期滞后处，而且在与季节频率对应的滞后处（例如，月度数据的12、24、36）。



![季节性数据ACF示例](plots/3447-1.json)



> 具有月度季节性数据ACF图示例。请注意滞后12和滞后24处的明显峰值，这表明除了短期滞后相关性之外，在季节频率处存在强相关性。虚线表示近似置信区间。

标准ARIMA模型没有结构能有效地捕捉ACF中这些独特的季节性峰值。它们尝试从滞后1开始模拟衰减模式，但没有专门针对滞后$m, 2m, 3m$等的特定项。

由于这些局限性，仅依靠$ARIMA(p, d, q)$处理强季节性数据通常会导致次优的预测结果。模型可能无法准确捕捉重复出现的峰谷，或需要过多的参数。这使得我们需要一种扩展，它能明确地纳入季节性成分，从而引出季节性ARIMA（SARIMA）模型。

## 参考资料

- [Time Series Analysis: Forecasting and Control](https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118675021) — George E. P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, Greta M. Ljung (2015)
  Publisher: Wiley
  ARIMA和SARIMA模型的奠基性著作，阐述了时间序列建模中季节性成分的必要性。
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/) — Rob J Hyndman, George Athanasopoulos (2021)
  Publisher: OTexts
  以易于理解的方式阐述了标准ARIMA模型处理季节性数据的挑战以及使用SARIMA模型的原因。
- [Introduction to Time Series and Forecasting](https://doi.org/10.1007/978-3-319-29854-2) — Brockwell, P. J., and Davis, R. A. (2016)
  Publisher: Springer; DOI: [10.1007/978-3-319-29854-2](https://doi.org/10.1007/978-3-319-29854-2)
  介绍了时间序列模型的理论基础，包括对季节性ARIMA过程的严谨处理。
