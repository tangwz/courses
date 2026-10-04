---
course: "time-series-analysis-forecasting"
chapter: "model-evaluation-selection"
lesson: "comparing-forecasts"
sourceId: 3475
sourceUrl: "https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-6-model-evaluation-selection/comparing-forecasts"
title: "比较不同模型的预测结果"
description: "Strategies for comparing the performance of different forecasting models (e.g., ARIMA vs. SARIMA) on the test set."
order: 5
plots: ["plots/3475-0.json"]
sourceHash: "33ec5e342660f037ef6b5579ee0cdb85498699755ea631e711f4b62b5e4e8be4"
sourceCorrections: []
---

在拟合了多个备选模型（例如不同的ARIMA配置，或比较ARIMA与SARIMA）之后，目标是选择在未见过的数据上表现最好的模型。这种比较需要使用评估指标和标准，并且要对所有竞争模型一致地使用。

基本流程包括以下步骤：

1. **一致的数据划分：** 确保所有模型都在完全相同的训练数据集上训练，并在完全相同的测试数据集上评估。为不同模型使用不同的数据子集会使比较无效。请记住使用之前讨论过的合适的时间序列划分方法。
2. **指标计算：** 对于每个模型，生成测试集覆盖期间的预测。通过比较这些预测与测试集中的实际值，计算选定的误差指标（MAE、MSE、RMSE、MAPE）。
3. **信息准则回顾：** 回顾在模型拟合阶段（在训练数据上）计算的AIC和BIC值。虽然测试集上的误差指标衡量预测准确性，但AIC和BIC有助于根据模型的拟合优度并考虑模型复杂度的惩罚来比较模型。

我们假设已经为训练数据拟合了两个模型：一个非季节性`ARIMA(1,1,1)`和一个`SARIMA(1,1,1)(1,1,0,12)`，后者旨在捕捉年度季节性（m=12）。我们现在在测试集上评估它们。

**比较测试集上的指标**

在为测试期生成两个模型的预测后，我们计算误差指标。假设我们得到以下结果：

| 模型 | MAE | RMSE | MAPE (%) | AIC（来自训练数据） | BIC（来自训练数据） |
| --- | --- | --- | --- | --- | --- |
| `ARIMA(1,1,1)` | 15.2 | 19.8 | 8.5 | 950.5 | 962.1 |
| `SARIMA(1,1,1)(1,1,0,12)` | 9.8 | 12.5 | 5.1 | 885.2 | 902.6 |

解读这些结果：

- **误差指标（MAE、RMSE、MAPE）：** SARIMA模型在测试集上的所有误差指标值均低于ARIMA模型。这表明其预测在测试期间平均而言更接近实际值。较低的RMSE表示它在避免大误差方面表现优于ARIMA模型。较低的MAPE表示平均百分比误差较小，这对于相对比较很有用。
- **信息准则（AIC、BIC）：** SARIMA模型也具有较低的AIC和BIC值。这表明，即使考虑到其更高的复杂性（更多参数 (parameter)），在训练数据上拟合的显著改进（这些准则所衡量的）也证明了增加参数是合理的。较低的AIC/BIC值通常表示在平衡拟合与简洁性方面表现更好的模型。

在这种情况下，测试集表现和信息准则都指向`SARIMA(1,1,1)(1,1,0,12)`模型是该数据集的更好选择，这可能是因为原始数据表现出季节性，而简单的ARIMA模型无法有效捕捉。

**可视化比较**

除了数值指标之外，将不同模型的预测结果与测试集中的实际值绘制在一起，可以提供有价值的视觉洞察。



![测试集上的模型预测比较](plots/3475-0.json)



> 实际测试数据与ARIMA和SARIMA模型预测结果的比较。SARIMA预测更贴近实际数据。

这种可视化方式让您可以看到每个模型在*哪些地方*表现良好或不佳。一个模型是否持续高估或低估？它是否能更好地捕捉转折点？上图中SARIMA的预测结果似乎比简单的ARIMA预测更贴近实际数据，这进一步证实了从指标得出的结论。

**选择“最佳”模型**

通常，一个模型会在大多数指标上明显优于其他模型，这使得选择变得直接明了。然而，有时您可能会面临权衡：

- 模型A的MAE较低，但模型B的RMSE较低。
- 模型C的测试集指标表现最好，但模型D的AIC/BIC显著较低，且简单得多。

在这种情况下，请考虑：

- **应用场景：** 是最小化平均误差更重要（倾向MAE）还是避免大而不常出现的误差更重要（倾向RMSE）？如果百分比误差对利益相关者更有意义，则侧重于MAPE（同时注意其在实际值为零或接近零时的局限性）。
- **简洁性：** 如果简单模型（参数更少）的性能仅比复杂模型略差，通常会更受青睐。它们通常更容易理解，计算速度更快，并且对数据模式的微小变化可能更具鲁棒性（不易过拟合 (overfitting)）。AIC和BIC明确地惩罚了模型的复杂度。
- **残差分析：** 确保所选模型的残差（在训练数据上）满足模型假设（例如，看起来是白噪声）。一个模型可能具有良好的预测指标，但未能通过诊断检查，这可能存在潜在问题。

模型比较是一个迭代过程。通过系统地计算指标、审查信息准则以及直观地查看保留测试集上的预测结果，您可以为您的特定时间序列问题做出关于哪个模型提供最可靠和准确预测的明智决定。

## 参考资料

- [Forecasting: Principles and Practice](https://otexts.com/fpp3/) — Rob J Hyndman and George Athanasopoulos (2021)
  Publisher: OTexts
  一本优秀的实用指南，涵盖时间序列预测，包括模型评估、选择准则（AIC/BIC）、各种度量（MAE、RMSE、MAPE）以及ARIMA/SARIMA模型的比较。
- [Time Series Analysis: Forecasting and Control](https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118675021) — George E. P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, and Greta M. Ljung (2015)
  Publisher: Wiley
  ARIMA和SARIMA模型的基础著作，提供了关于模型构建、拟合和诊断检查的深刻理论和实践见解，对于理解所比较的模型至关重要。
- [Elements of Forecasting](https://www.amazon.com/Elements-Forecasting-Francis-Diebold/dp/0324359049) — Francis X. Diebold (2007)
  Publisher: South-Western Cengage Learning
  全面概述了预测方法和原理，包括对预测评估、不同模型比较以及样本外性能重要性的严谨讨论。
