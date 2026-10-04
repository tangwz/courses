---
course: "time-series-analysis-forecasting"
chapter: "arima-models-forecasting"
lesson: "selecting-arima-order"
sourceId: 3431
sourceUrl: "https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-4-arima-models-forecasting/selecting-arima-order"
title: "选择ARIMA模型阶数 (p, d, q)"
description: "选择ARIMA模型中p、d、q适当参数的策略，基于ACF/PACF和信息准则。"
order: 5
plots: ["plots/3431-0.json", "plots/3431-1.json"]
sourceHash: "8e4d0c8d9f4d8488a8d434518efe760d41fe174313bb68c2f4c5d21f2dc5bf0c"
sourceCorrections: []
---

选择ARIMA模型组成部分（AR、I、MA）的适当阶数，它们由参数 (parameter)$(p, d, q)$表示，是构建有效ARIMA模型的根本。这个选择过程涉及确定所需的差分次数（'I'部分），然后根据平稳序列的自相关模式，确定AR和MA部分的结构。

### 确定差分阶数（`d`）

ARIMA中的'I'代表'差分整合'。这个组成部分处理时间序列中的非平稳性，特别是与趋势或水平变化相关的非平稳性。参数 (parameter)$d$表示时间序列需要进行差分的次数，以达到平稳。

1. **评估初始平稳性：** 首先，检查您的原始时间序列。使用第2章中讨论的目视检查方法（绘制序列图、移动平均/方差）和统计检验（如增广迪基-富勒检验）。

   - 如果序列看起来是平稳的（均值和方差恒定，ADF检验拒绝非平稳的零假设），则不需要差分，此时$d = 0$。
   - 如果序列是非平稳的，则进行差分。
2. **应用差分：** 计算序列的一阶差分：$\Delta y_t = y_t - y_{t-1}$。现在，检验这个差分序列的平稳性。

   - 如果一阶差分序列是平稳的，则$d = 1$。这对于表现出线性趋势的序列来说非常常见。
3. **考虑二阶差分（如果需要）：** 如果一阶差分序列仍然非平稳（可能是由于二次趋势或变化趋势），计算二阶差分：$\Delta^2 y_t = \Delta y_t - \Delta y_{t-1} = (y_t - y_{t-1}) - (y_{t-1} - y_{t-2})$。检验这个二阶差分序列的平稳性。

   - 如果二阶差分序列是平稳的，则$d = 2$。

**重要指导原则：** 使用使序列平稳所需的最少差分阶数。过度差分会向数据中引入人为模式和依赖关系，从而不必要地使模型复杂化。很少需要$d > 2$。如果在两轮差分后仍未达到平稳，您可能需要考虑其他变换（如取对数）或替代的建模方法。

一旦您确定了$d$的值并获得了平稳时间序列（我们称之为$y'_t$，如果$d=0$它可能是原始序列，如果$d>0$则是差分后的版本），您就可以接着确定AR和MA的阶数（$p$和$q$）。

### 使用ACF和PACF确定AR（`p`）和MA（`q`）阶数

自相关函数（ACF）和偏自相关函数（PACF）图（在第3章中讨论过）是推断模型ARMA部分阶数$p$和$q$的主要工具。记住要使用**平稳（差分后）时间序列**$y'_t$来生成这些图。

以下是要查找的一般模式：

- **AR(p) 模型特点：**

  - **ACF：** 缓慢衰减或指数式衰减趋近于零。它可能显示振荡。
  - **PACF：** 在滞后$p$后急剧截断。这意味着偏自相关在滞后$p$之前是统计显著的，然后急剧降至接近零（在显著性边界内）。
  - *解释：* 如果您看到这种模式，它表明AR($p$)模型可能适合，这意味着$q=0$。$p$的值由PACF截断的滞后决定。
- **MA(q) 模型特点：**

  - **ACF：** 在滞后$q$后急剧截断。自相关在滞后$q$之前是显著的，然后急剧下降。
  - **PACF：** 缓慢衰减或指数式衰减趋近于零。
  - *解释：* 这种模式表明MA($q$)模型，这意味着$p=0$。$q$的值由ACF截断的滞后决定。
- **ARMA(p, q) 模型特点：**

  - **ACF：** 在滞后$q$后缓慢衰减。
  - **PACF：** 在滞后$p$后缓慢衰减。
  - *解释：* 如果两张图都显示衰减行为，这表明AR和MA项都必要。确定确切的$p$和$q$可能更具挑战性。通常，您可能会在两张图中观察到指数衰减或阻尼正弦波。在这种情况下，您可以寻找衰减有效开始的滞后，或者尝试将$p$和$q$设为小值（例如，$p=1, q=1$）作为起点。

**ACF/PACF图解释的可视化：**

让我们设想平稳序列的两种情形（$d$已确定）：

**情形1：潜在的AR(2)模型**



![ACF和PACF图（情形1：表明AR(2)）](plots/3431-0.json)



> 在情形1中，ACF图显示出缓慢、有些指数式的衰减。PACF图在滞后1和2处显示显著峰值，然后急剧截断（滞后2之后的峰值都在显著性边界内，由虚线表示）。这强烈表明是一个AR(2)模型，所以$p=2$且$q=0$。完整的模型将是ARIMA(2, d, 0)。

**情形2：潜在的MA(1)模型**



![ACF和PACF图（情形2：表明MA(1)）](plots/3431-1.json)



> 在情形2中，ACF图在滞后1处显示单个显著峰值，然后截断。PACF图衰减得更慢，可能呈几何式或振荡式。这种模式强烈表明是一个MA(1)模型，暗示$p=0$且$q=1$。完整的模型将是ARIMA(0, d, 1)。

### 迭代选择和信息准则

ACF和PACF图很少像理想化例子那样清晰。数据中的噪声会模糊模式。因此，选择$(p, d, q)$通常是一个迭代过程：

1. **确定`d`：** 找到使序列平稳所需的最小差分次数。
2. **分析ACF/PACF：** 检查平稳序列的图。根据截断或衰减行为，确定$p$和$q$的潜在候选阶数。从更简单的模型开始（小的$p, q$值）。
3. **拟合候选模型：** 使用`statsmodels`等库拟合ARIMA($p, d, q$)模型（接下来会讲到）。
4. **检查诊断结果：** 分析拟合模型的残差。如果模型良好，残差应类似于白噪声（没有剩余的显著自相关）。这将在“模型诊断”部分进行介绍。
5. **比较模型：** 如果多个候选模型看起来合理（例如，ARIMA(1, d, 0) 对比 ARIMA(0, d, 1) 对比 ARIMA(1, d, 1)），您可以使用像AIC（赤池信息准则）或BIC（贝叶斯信息准则）这样的信息准则来帮助选择。这些指标在模型拟合度（它解释数据的程度）和模型复杂度（参数 (parameter)$p+q$的数量）之间取得平衡。通常，AIC或BIC值较低的模型更受青睐。我们将在第6章更详细地讨论评估指标，包括AIC/BIC。
6. **完善：** 根据诊断结果和比较指标，完善您对$p$和$q$的选择。您可能需要尝试略有不同的阶数，并重复步骤3-5。

这种结合分析ACF/PACF图、拟合模型、检查残差以及潜在地使用信息准则的方法，提供了一种系统途径，以得到适合您时间序列的ARIMA($p, d, q$)设定。

## 参考资料

- [Time Series Analysis: Forecasting and Control](https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118675021) — George E. P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, Greta M. Ljung (2015)
  Publisher: Wiley
  关于ARIMA模型的权威著作，详细介绍了平稳性、差分以及使用ACF/PACF进行模型识别。
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/) — Rob J Hyndman and George Athanasopoulos (2021)
  Publisher: OTexts
  提供了一本关于时间序列预测的易读现代指南，包含使用ACF/PACF和信息准则识别ARIMA阶数的实用指导。
- [Time Series Analysis and Its Applications: With R Examples](https://www.springer.com/book/9783319524511) — Robert H. Shumway and David S. Stoffer (2017)
  Publisher: Springer; DOI: [10.1007/978-3-319-52452-8](https://doi.org/10.1007/978-3-319-52452-8)
  一本标准教材，介绍了时间序列统计方法，包括关于平稳性、ARIMA模型识别和诊断的详细章节。
