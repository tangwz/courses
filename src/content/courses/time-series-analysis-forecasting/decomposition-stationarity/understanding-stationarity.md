---
course: "time-series-analysis-forecasting"
chapter: "decomposition-stationarity"
lesson: "understanding-stationarity"
sourceId: 3358
sourceUrl: "https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-2-decomposition-stationarity/understanding-stationarity"
title: "理解平稳性"
description: "定义平稳性并解释其为何是许多时间序列模型的重要前提。"
order: 1
plots: ["plots/3358-0.json"]
sourceHash: "a7372256acd19aa4e63d27a6e9571df2845a1798047f4500579a7e53e4f8c10f"
sourceCorrections: []
---

许多时间序列模型，特别是我们将考察的经典统计模型，都依赖于数据随时间变化的表现的一个**基本前提**：**平稳性**。直观地说，如果一个时间序列的统计特性不依赖于观察序列的时间点，那么它就是平稳的。举个例子：如果你从序列的开头取一段，再从末尾取一段相同长度的序列，它们的基本统计特征（如平均值或数据的分散程度）应该大致相同。

**弱平稳性**（或二阶平稳性）是一种常用的平稳性定义。一个时间序列{$Y_t$}是弱平稳的，如果它满足以下三个条件：

1. **均值不变：** 序列的期望值（均值）随时间保持不变。
   $E[Y_t] = \mu \quad \text{对于所有 } t$
2. **方差不变：** 序列的方差随时间保持不变且有限。
   $Var(Y_t) = E[(Y_t - \mu)^2] = \sigma^2 < \infty \quad \text{对于所有 } t$
3. **自协方差不变：** 两个时间点的值之间的协方差仅取决于这些时间点之间的距离（滞后），而不取决于具体的时间点。
   $Cov(Y_t, Y_{t+h}) = E[(Y_t - \mu)(Y_{t+h} - \mu)] = \gamma_h \quad \text{对于所有 } t \text{ 和滞后 } h$

简单来说：

- 序列的平均水平不会系统性地增加或减少（没有趋势）。
- 平均水平周围的波动宽度一致（方差不变）。
- 观测值与其滞后值之间的关系在序列中的任何位置都是一致的。

思考与非平稳数据的对比。具有明显上升趋势的序列不满足均值不变的条件。波动随时间变宽的序列不满足方差不变的条件。具有明显季节性的数据通常不满足均值不变和自协方差不变的条件，因为平均水平和点之间的关系取决于一年中的时间。



![平稳与非平稳时间序列示例](plots/3358-0.json)



> 上方序列围绕恒定均值波动，方差不变，这是平稳数据的**特点**。下方序列呈现明显的上升趋势，不满足平稳性的均值不变条件。

### 为什么平稳性对建模很重要？

平稳性的前提**很重要**，因为它大大简化了建模过程。

1. **可预测性：** 如果一个序列是平稳的，那么从历史数据中获得的其统计特性（均值、方差、相关性）更可能与未来相关。这使得预测更可靠，因为数据生成过程被假定随时间保持稳定。
2. **模型适用性：** 许多**基本**的时间序列模型，如ARMA（自回归 (autoregressive)滑动平均），都是为平稳数据设计的。这些模型试图根据过去的值和过去的误差来解释围绕恒定均值的波动。直接将它们应用于非平稳数据可能导致无效的统计推断、模型拟合不佳以及不可靠的预测。估计的参数 (parameter)可能没有意义。
3. **避免虚假结果：** 分析非平稳时间序列之间的关系时，可能会遇到“**虚假回归**”。这意味着你可能会发现变量之间存在统计上**有意义的**关系，而这些变量实际上是独立的，只是碰巧具有相似的趋势。使用平稳数据有助于避免这些误导性结果。

识别非平稳性是处理它的第一步。我们将接下来介绍的分解等方法，有助于确定导致非平稳性的分量，如趋势和季节性。本章稍后，我们将讨论差分等方法，这些方法旨在将非平稳数据转换为适合ARIMA（自回归差分滑动平均）等模型的平稳形式，其中“**差分**”部分专门处理由趋势引起的非平稳性。

## 参考资料

- [Time Series Analysis: Forecasting and Control](https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118675021) — George E. P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, and Greta M. Ljung (2015)
  Publisher: John Wiley & Sons
  阐释平稳性及其在ARIMA等经典时间序列模型中作用的权威教材。
- [Time Series Analysis and Its Applications: With R Examples](https://link.springer.com/book/10.1007/978-3-319-52455-1) — Robert H. Shumway and David S. Stoffer (2017)
  Publisher: Springer; Pages: 562; DOI: [10.1007/978-3-319-52455-1](https://doi.org/10.1007/978-3-319-52455-1)
  为平稳性提供清晰实用的介绍，包含示例和计算方面的内容。
- [Time Series Analysis](https://press.princeton.edu/books/hardcover/9780691042893/time-series-analysis) — James D. Hamilton (1994)
  Publisher: Princeton University Press
  一本全面而严谨的研究生教材，涵盖平稳性的理论基础。
