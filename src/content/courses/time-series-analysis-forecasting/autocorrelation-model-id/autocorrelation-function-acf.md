---
course: "time-series-analysis-forecasting"
chapter: "autocorrelation-model-id"
lesson: "autocorrelation-function-acf"
sourceId: 3404
sourceUrl: "https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-3-autocorrelation-model-id/autocorrelation-function-acf"
title: "自动相关函数 (ACF)"
description: "理解自动相关性，以及 ACF 如何衡量不同时间滞后观测值之间的相关性。"
order: 1
plots: ["plots/3404-0.json"]
sourceHash: "9b00c8e5f93fc089f8022a0fd10c71e8e1f579d524f9a707016caedd5403162e"
sourceCorrections: []
---

自动相关函数（ACF）是量化 (quantization)平稳时间序列内部*相关结构*的主要方法。它有助于理解给定时间 $t$ 的序列值 $y_t$ 与其过去值 $y_{t-1}$、$y_{t-2}$ 等之间的关系。理解这种内部依赖性是分析时间序列数据的基础。

### 理解自动相关性

自动相关性简单来说就是“自身相关”。它衡量时间序列滞后值之间的线性关系。简单来说，它告诉我们时间 $t$ 的序列值与 $k$ 个周期前（即时间 $t-k$）的值相关程度如何。这个滞后 $k$ 可以是 1、2、3 等。

特定滞后 $k$ 的 ACF 值，通常表示为 $\rho_k$，其计算方式类似于标准相关系数，但在所有可用的 $t$ 上针对 $y_t$ 和 $y_{t-k}$ 进行计算。


$$
\rho_k = \frac{\text{协方差}(y_t, y_{t-k})}{\text{方差}(y_t)}
$$


此处，$\text{协方差}(y_t, y_{t-k})$ 是序列与其滞后版本之间的协方差，而 $\text{方差}(y_t)$ 是序列的方差。由于我们假设序列是平稳的，$\text{方差}(y_t)$ 随时间保持不变，并且$\text{协方差}(y_t, y_{t-k})$ 仅取决于滞后 $k$，而非特定时间 $t$。

ACF 值范围从 -1 到 1：

- **+1:** 完全正相关。如果序列增加，则 $k$ 个周期前的值也倾向于增加。
- **-1:** 完全负相关。如果序列增加，则 $k$ 个周期前的值倾向于减少。
- **0:** 序列与其 $k$ 个周期前的值之间无线性相关。

根据定义，滞后 0 的自动相关性 $\rho_0$ 始终为 1，因为任何序列在无滞后时都与自身完全相关。

### 计算和绘制 ACF

我们很少手动计算这些值。Python 中的 `statsmodels` 等统计库提供了计算和绘制 ACF 的函数。标准的可视化形式是“相关图”或 ACF 图。此图在 Y 轴上显示自动相关值 $\rho_k$，在 X 轴上显示不同的滞后 $k$（通常从滞后 1 开始，但有时会包含滞后 0）。

统计软件包生成的 ACF 图的一个重要特点是包含**显著性边界**。这些通常表示为阴影区域（通常为蓝色）。自动相关性条形图超出此边界的滞后被认为是统计上显著的（通常在 5% 的显著性水平下）。这表明，假设该滞后的真实自动相关性为零，则在该滞后观察到的相关性不太可能是由于随机机会单独引起的。

让我们看一个平稳时间序列的 ACF 图示例：



![自动相关函数 (ACF)](plots/3404-0.json)



> ACF 图显示了滞后 1 到 20 的自动相关值。蓝色阴影区域代表 95% 置信区间。延伸到此区域之外的条形图表明统计上显著的自动相关性。

### 解释 ACF 图

在上面的图中：

- 滞后 0 的自动相关性为 1（根据定义，尽管在侧重于模型识别的图中常被省略）。
- 滞后 1、2、3 和 4 呈现显著的正自动相关性，因为它们的条形图明显超出阴影置信带。
- 自动相关性随着滞后的增加而逐渐减小。这种缓慢衰减的 ACF 模式是某些类型时间序列过程的特点，例如我们将在后面讨论的自回归 (autoregressive)（AR）过程。
- 在滞后 4 或 5 之后，自动相关性通常在置信边界内，表明它们在统计上不显著，可能由随机噪声引起。

研究 ACF 图是时间序列分析中的一个基本步骤。它帮助我们理解过程的“记忆”。过去的值能回溯多远并显著影响当前值？ACF 中的衰减模式（例如，突然截断与缓慢衰减）提供了关于数据潜在结构的线索，并指导我们选择合适的模型，如移动平均（MA）或自回归（AR）模型。我们将在“解释 ACF/PACF 以进行模型选择”一节中研究 ACF 模式与模型识别之间的这种联系。目前，重点是理解 ACF 衡量什么以及如何解读其图。

## 参考资料

- [Time Series Analysis: Forecasting and Control](https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118675021) — George E. P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, Greta M. Ljung (2015)
  Publisher: Wiley; DOI: [10.1002/9781118619193](https://doi.org/10.1002/9781118619193)
  经典时间序列方法的入门教材，详细介绍了ACF的理论及其在模型识别中的应用。
- [Time Series Analysis and Its Applications: With R Examples](https://doi.org/10.1007/978-3-319-52455-8) — Robert H. Shumway, David S. Stoffer (2017)
  Publisher: Springer; Pages: 562; DOI: [10.1007/978-3-319-52455-8](https://doi.org/10.1007/978-3-319-52455-8)
  一本内容全面的教材，涵盖时间序列分析的理论方面和实际应用，其中包含ACF的专门章节。
- [statsmodels.graphics.tsaplots.plot_acf](https://www.statsmodels.org/stable/generated/statsmodels.graphics.tsaplots.plot_acf.html) — statsmodels developers (2024)
  用于计算和绘制自相关函数（ACF）的Python函数官方文档，包含了其参数和使用方法。
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/) — Rob J Hyndman, George Athanasopoulos (2021)
  Publisher: OTexts
  一本在线教材，提供实用的预测指南，清晰解释了ACF在理解时间序列模式和模型选择中的作用。
