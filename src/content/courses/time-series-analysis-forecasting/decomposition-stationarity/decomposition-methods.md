---
course: "time-series-analysis-forecasting"
chapter: "decomposition-stationarity"
lesson: "decomposition-methods"
sourceId: 3361
sourceUrl: "https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-2-decomposition-stationarity/decomposition-methods"
title: "时间序列分解方法"
description: "学习加法和乘法模型，以将时间序列分解为其潜在组成部分。"
order: 2
plots: ["plots/3361-0.json"]
sourceHash: "7383b89577a472b0fa7f93ea9e4616197173c85453a9e88ab287c9a994a38df2"
sourceCorrections: []
---

时间序列分解是一种基本技术，用以将时间序列 $Y_t$ 拆分成其潜在的、不可观测的组成部分。可以将其比作拆解一台机器，以明白每个齿轮和杠杆如何对整体功能起作用。目的是分离出趋势、季节性以及剩余的不规则波动（残差或噪声）等模式。这个过程能提供对数据结构的有益理解，并有助于为建模做好准备，尤其是在识别和去除导致非平稳性的部分时。

有两种主要的结构模型用于分解，基于各部分如何组合：

### 加法分解

加法模型假设组成部分相加形成观测到的时间序列。它表示为：

$Y_t = T_t + S_t + R_t$

各部分含义：

- $Y_t$是时间$t$的观测值。
- $T_t$是时间$t$的趋势部分。
- $S_t$是时间$t$的季节部分。
- $R_t$是时间$t$的残差（或不规则/随机）部分。

当季节性波动的幅度或残差围绕趋势的方差随时间保持相对稳定时，加法模型最为合适。如果您绘制数据图，且季节性波动似乎并未随着序列整体水平的升高而变大，那么加法方法可能适用。假设有月度销售数据，其中假期季节在每年12月为销售额增加大约*相同的金额*（例如10,000美元），无论基准销售额是高是低。

### 乘法分解

乘法模型假设组成部分相乘：

$Y_t = T_t \times S_t \times R_t$

当季节性变动或残差波动似乎与时间序列的水平成比例时，此模型通常更适用。随着趋势的增加，季节性波动的幅度或随机噪声的幅度也倾向于增加。例如，如果假期销售额占月度基准销售额的*百分比*增长（例如20%），那么随着整体销售趋势上升，12月销售高峰的绝对值也将增长。在这种情况下，乘法模型能提供更好的描述。

将乘法关系通过对时间序列取对数转换为加法关系也是常见做法：

$\log(Y_t) = \log(T_t) + \log(S_t) + \log(R_t)$

这使得加法分解方法可以应用于对数转换后的数据，从而简化分析。

### 常用分解技术

有多种算法可以执行分解。以下是两种常用的：

1. **经典分解：** 这是一种相对简单的方法，通常基于移动平均线。

   - **估计趋势 ($T_t$)：** 计算一个窗口大小与季节周期匹配的移动平均线（例如，月度数据为12），以平滑季节性和噪声。通常使用中心移动平均线。
   - **去除趋势：** 从原始序列中移除趋势估计值。对于加法模型，计算 $Y_t - T_t$。对于乘法模型，计算 $Y_t / T_t$。
   - **估计季节性 ($S_t$)：** 对去趋势后的序列中每个季节（如所有1月的值、所有2月的值等）在所有年份中的数值进行平均。调整这些季节性平均值，使它们在一个完整周期内总和为0（加法）或平均为1（乘法）。
   - **计算残差 ($R_t$)：** 从原始序列中减去（加法）或除以（乘法）估计的趋势和季节部分：$R_t = Y_t - T_t - S_t$（加法）或 $R_t = Y_t / (T_t \times S_t)$（乘法）。

   虽然易于理解，但经典分解存在缺点。它在序列的开头和结尾处表现不佳（中心移动平均线在没有假设的情况下无法计算），假设季节性在每个周期完全相同地重复，并且可能对异常值敏感。
2. **STL 分解（使用Loess的季节和趋势分解）：** 这是一种由Cleveland等人（1990）开发，更精细、更通用的方法。STL 使用**Loess（局部加权散点平滑）**，一种非参数 (parameter)回归技术，迭代地估计趋势和季节部分。

   - 它执行一个内循环，估计并去除季节性，然后从经过季节性调整的数据中估计并去除趋势。
   - 一个外循环细化估计值，并计算鲁棒性权重 (weight)以减少异常值的影响。
   - STL的主要优点包括其处理任何类型季节性（不仅仅是12个月等固定周期）的能力，用户可以控制趋势和季节部分的平滑度，以及对异常值的抵抗力。在大多数实际应用中，它通常优于经典分解。

### 分解结果的可视化

分解在可视化时效果最好。通常，您会将原始时间序列及其估计的趋势、季节和残差部分绘制在不同的面板上。这有助于轻松观察潜在模式。



![时间序列分解示例（加法）](plots/3361-0.json)



> 加法时间序列分解的示例输出，展示了原始观测数据、估计趋势、重复季节模式和剩余残差部分。

理解这些组成部分是重要的。趋势和季节性通常代表了序列的非平稳部分。通过分解识别它们，我们可以采取差分（我们将在下一节讨论）等步骤来消除它们，从而达到许多预测模型（如ARIMA）所需的平稳性。残差部分理想情况下代表平稳噪声；检查其属性有助于验证分解和模型假设。

## 参考资料

- [Forecasting: Principles and Practice](https://otexts.com/fpp3/) — Rob J Hyndman and George Athanasopoulos (2021)
  Publisher: OTexts
  一本全面的时间序列预测教科书，包含对经典分解和STL等分解方法的详细解释及实际案例。
- [STL: A Seasonal-Trend Decomposition Procedure Based on Loess](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGJwWjVIWPqdbZXkKkOK1ISopOHaNBYFbU277-ke3bKRT2Gr4FPBkltvk5IrZyevIVfkxYavSqnDSkQ4QnmehcvnOUigzdt5TmTSfn4NgbkQoNOGtU9xpAHt7RyFEyvhrUU4WoTU65YBq0AiaT0zXkL18DgCskUEBmqFb-vp-itbdabkfcu_eM=) — Robert B. Cleveland, William S. Cleveland, Jean E. McRae, and Irma Terpenning (1990)
  Journal: Journal of Official Statistics; Volume: 6; Pages: 3-73
  介绍了基于Loess的季节性趋势分解（STL）方法的论文，详细阐述了其算法和优点。
- [Time Series Analysis: Forecasting and Control](https://www.wiley.com/en-us/Time+Series+Analysis:+Forecasting+and+Control,+5th+Edition-p-9781118675021) — George E. P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, and Greta M. Ljung (2015)
  Publisher: Wiley
  一本经典而全面的时间序列分析教科书，涵盖了理论基础、分解方法和各种预测模型。
