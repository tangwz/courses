---
course: "time-series-analysis-forecasting"
chapter: "arima-models-forecasting"
lesson: "arima-models-introduction"
sourceId: 3428
sourceUrl: "https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-4-arima-models-forecasting/arima-models-introduction"
title: "引入整合：ARIMA 模型"
description: "在之前的章节中，我们学习了运用过往值的自回归 (AR) 模型和使用过往误差的移动平均 (MA) 模型。我们还了解到 ARMA 模型如何结合这两种思路。然而，AR、MA 和 ARM"
order: 4
plots: []
sourceHash: "76d657b77892b6a37afe75e367536710385e2f772922a49d877d6e080b1d73ef"
sourceCorrections: []
---

自回归 (autoregressive) (AR) 模型运用过往值，而移动平均 (MA) 模型则使用过往误差。ARMA 模型结合了这两种思路。然而，AR、MA 和 ARMA 模型的一个重要假设是，基础的时间序列数据必须是**平稳的**。其统计特征，如均值和方差，不应随时间变化。

当我们遇到实际数据时会怎样？这些数据经常显示出趋势或其他形式的非平稳性。将 ARMA 模型直接应用于这类数据会导致不可靠的结果。这就是 ARIMA 中“I”的作用所在。

ARIMA 代表**差分整合自回归移动平均**。“整合”部分解决了非平稳性问题。回顾第2章“时间序列分解与平稳性”，**差分**是一种常用方法，用于将非平稳序列转换为平稳序列。我们通过计算连续观测值之间的差值来实现这一点。有时，如果第一次差分未能得到平稳序列，我们可能需要多次应用差分。

“整合”一词表明建模过程包含了这个差分步骤。它指的是一个思路，即原始非平稳序列可以被认为是差分的逆过程，本质上是平稳差分序列的求和或（在离散意义上的）积分。

ARIMA 模型的特点是三个参数 (parameter)：$(p, d, q)$。

- **p (AR 阶数):** 模型中包含的滞后观测值数量。这与 AR 和 ARMA 模型中的 `p` 相同，但它是根据*差分后*的序列确定的。
- **d (差分次数):** 原始观测值进行差分以达到平稳性的次数。如果原始序列已经平稳，则 $d=0$。
- **q (MA 阶数):** 移动平均窗口的大小，表示预测方程中滞后预测误差的数量。这与 MA 和 ARMA 模型中的 `q` 相同，应用于*差分后*的序列。

因此，ARIMA(p, d, q) 模型实际上是将 ARMA(p, q) 模型应用于时间序列在经过 `d` 次差分*之后*。

> 流程图说明了差分 (d) 如何使数据平稳，从而可以确定 ARMA(p, q) 结构，最终将 ARIMA(p, d, q) 模型应用于原始数据。

如果时间序列已经平稳，我们设置 $d=0$，ARIMA(p, 0, q) 模型将直接简化为 ARMA(p, q) 模型。ARIMA 的优点在于它能够处理更广泛的时间序列，特别是那些经过一次或多次差分后变得平稳的序列。

在使用像 Python 中的 `statsmodels` 这样的库时，您通常指定 $(p, d, q)$ 阶数并提供*原始的*非平稳时间序列。库会在模型拟合过程中在内部处理差分。通常不需要在将数据传递给 ARIMA 函数之前手动进行差分，尽管理解差分步骤 (`d`) 对于选择正确的模型阶数很重要。

在接下来的章节中，我们将讨论选择 $p$、$d$ 和 $q$ 适当值的策略，如何使用 Python 拟合模型，以及如何评估其性能。

## 参考资料

- [Time Series Analysis: Forecasting and Control](https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118675021) — George E. P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, and Greta M. Ljung (2015)
  Publisher: John Wiley & Sons; DOI: [10.1002/9781118675008](https://doi.org/10.1002/9781118675008)
  这是一本关于ARIMA模型的权威基础性著作，详细阐述了其理论基础、模型识别、参数估计和诊断检验。
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/) — Rob J Hyndman and George Athanasopoulos (2021)
  Publisher: OTexts
  这是一本广受好评且免费提供的在线教材，它以现代实用方法讲解时间序列预测，包括对平稳性、差分和ARIMA模型的详细讨论。
