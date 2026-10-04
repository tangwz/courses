---
course: "time-series-analysis-forecasting"
chapter: "model-evaluation-selection"
lesson: "visualizing-forecast-performance"
sourceId: 3477
sourceUrl: "https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-6-model-evaluation-selection/visualizing-forecast-performance"
title: "预测表现的可视化"
description: "将实际值与模型预测值进行绘图，以视觉评估表现的方法。"
order: 6
plots: ["plots/3477-0.json"]
sourceHash: "349a8ca3cdb77c4e696cb50f2176f569fbd19631dea27cae7d800a1d2bf4dd49"
sourceCorrections: []
---

虽然像平均绝对误差（$MAE$）或均方根误差（$RMSE$）这样的数值指标能简洁概括预测的准确度，但它们未能呈现全部情况。低$RMSE$可能掩盖了预测表现不佳的时期，或者未能显示预测中的系统性偏差。将预测值与实际观测值进行可视化，是模型评估中重要一步。它有助于评估表现、发现误差中的模式，并增强对模型的信心。

### 绘制预测值与实际值

最直接的评估表现的方法，是将模型预测值与测试集中的实际值随时间变化绘制在一起。这种比较能立即显示预测与真实数据吻合的程度。

在检查这类图表时，请留意：

- **整体拟合度：** 预测值是否普遍遵循实际值的模式？
- **数值大小：** 预测值是否持续偏高或偏低（偏差）？
- **转折点：** 模型是否成功预测了时间序列方向的变化？
- **特定事件：** 模型在数据异常时期或出现峰值时表现如何？

假设我们训练了一个模型，并为保留的测试期生成了预测。我们可以将这些预测与真实值绘制在一起。

```python
# 假设 'test_data' 是包含实际值的 pandas Series
# 'forecasts' 是包含预测值的 pandas Series，
# 两者都具有相同的 DatetimeIndex。

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data, label='实际值', color='#495057') # Gray
plt.plot(forecasts.index, forecasts, label='预测值', color='#fd7e14', linestyle='--') # Orange

# 可选：如果可用，添加置信区间
# plt.fill_between(forecast_index, conf_int_lower, conf_int_upper, color='orange', alpha=0.2, label='置信区间')

plt.title('预测值与实际值比较')
plt.xlabel('时间')
plt.ylabel('数值')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
```

一个典型的输出可能如下所示：



![预测值与实际值对比](plots/3477-0.json)



> 对比图显示了测试期内实际时间序列值（实心蓝线）与模型预测值（虚线橙线）的对照。

此图提供了比单独指标更丰富的理解。我们可以看到预测在哪里偏差最大，以及它是否捕捉了整体趋势和波动。如果模型生成了置信区间（ARIMA/SARIMA模型常见），绘制它们能提供预测不确定性的重要视角。

### 残差可视化

残差，计算方式为 $Residual_t = Actual_t - Forecast_t$，表示每个时间点的预测误差。绘制残差是一个基本的诊断步骤，常在模型拟合后进行（如第四章和第五章所示），但在测试集上的最终评估中同样有价值。分析残差有助于验证模型假设是否成立，以及是否存在未被解释的系统性模式。

重要的残差图包括：

- **残差随时间变化图：** 将残差绘制在时间索引上。理想情况下，此图应类似白噪声。这意味着残差应围绕零点分布，具有恒定方差，并且没有可辨识的模式（如趋势或季节性）。此处出现的模式表明模型未能捕捉到数据结构的某些方面。
- **残差直方图和密度图：** 这用于检查残差是否近似正态分布，这是许多时间序列模型（如ARIMA）进行统计推断所依据的假设。显著的偏度或峰度可能表明存在问题。
- **残差ACF图：** 残差的自相关函数图检查是否存在剩余的时间依赖性。ACF中显著的尖峰表明模型未能完全捕捉自相关结构，可能需要不同的模型阶数或类型。

将残差的这些方面可视化，能更透彻地理解模型的不足之处和潜在的改进空间。例如，如果看到残差随时间方差增大，可能表明需要进行变换或采用不同的建模方法。

### 多个模型的视觉比较

当您有多个候选模型（例如，不同ARIMA阶数、ARIMA与SARIMA，或统计模型与机器学习 (machine learning)模型）的预测时，将它们与实际值一起绘制在同一个坐标轴上，是非常有效的方法。

```python
# 假设 'test_data'、'forecast_model1'、'forecast_model2' 都是 pandas Series

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data, label='实际值', color='#495057', linewidth=2) # Gray
plt.plot(forecast_model1.index, forecast_model1, label='模型1预测', color='#f76707', linestyle='--') # Orange
plt.plot(forecast_model2.index, forecast_model2, label='模型2预测', color='#1098ad', linestyle=':') # Cyan

plt.title('不同模型预测值的比较')
plt.xlabel('时间')
plt.ylabel('数值')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
```

这种直接比较可以轻松看出哪个模型更紧密地跟踪实际值，更好地处理季节性或趋势等特定特征，或者在序列不同部分显示更小的偏差。它通过展示预测行为的定性差异，补充了对$AIC$、$BIC$或$RMSE$等数值指标的比较。

总而言之，虽然定量指标对于概括表现是必要的，但将预测及其误差可视化能提供重要背景。这些图表能帮助您了解模型成功或失败的方式和地点，诊断潜在问题，有效地比较备选方案，并最终构建更可靠的预测系统。

## 参考资料

- [Forecasting: Principles and Practice (3rd Edition)](https://otexts.com/fpp3/) — Rob J Hyndman and George Athanasopoulos (2021)
  Publisher: OTexts
  关于时间序列预测的综合在线教材，包含模型评估和诊断图表的章节。
- [Time Series Analysis and Its Applications: With R Examples](https://tsa5.github.io/) — Robert H. Shumway, David S. Stoffer (2024)
  Publisher: Springer
  关于时间序列统计方法的教材，包含模型诊断和残差分析的讨论。
- [Python Data Science Handbook: Essential Tools for Working with Data](https://jakevdp.github.io/PythonDataScienceHandbook/) — Jake VanderPlas (2016)
  Publisher: O'Reilly Media
  关于Python数据科学库的指南，涵盖用于可视化的Matplotlib和用于数据处理的Pandas。
