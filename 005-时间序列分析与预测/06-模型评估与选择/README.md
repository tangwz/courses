# 第 6 章：模型评估与选择

来源：[原章节](https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-6-model-evaluation-selection)

[返回课程目录](../README.md)

在之前的章节中构建了ARIMA和SARIMA等预测模型后，下一步是评估它们的表现。建立模型只是流程的一部分；我们需要客观的方法来确定它预测未来值的准确程度，以及它与替代方案相比如何。

在本章中，您将学习针对时间序列数据的有效模型评估方法。我们将从如何正确地将数据分成训练集和测试集开始，同时遵守时间顺序以避免“预见未来”。然后，您将学习计算和解释常见的预测准确性指标，包括：

*   平均绝对误差 ($MAE$)
*   均方误差 ($MSE$)
*   均方根误差 ($RMSE$)
*   平均绝对百分比误差 ($MAPE$)

例如，$MAE$ 表示预测值与实际值之间的平均绝对差：
$$MAE = \frac{1}{n} \sum_{i=1}^{n} |Actual_i - Forecast_i|$$

我们还将考察像赤池信息准则 ($AIC$) 和贝叶斯信息准则 ($BIC$) 这样的信息准则，它们通过平衡模型拟合度和复杂度来辅助模型选择。在本章结束时，您将能够应用这些指标和准则来比较不同的预测模型，并可视化它们与实际数据的表现。

## 小节

- 1. [模型评估的必要性](01-%E6%A8%A1%E5%9E%8B%E8%AF%84%E4%BC%B0%E7%9A%84%E5%BF%85%E8%A6%81%E6%80%A7.md)
- 2. [时间序列的训练-测试分离](02-%E6%97%B6%E9%97%B4%E5%BA%8F%E5%88%97%E7%9A%84%E8%AE%AD%E7%BB%83-%E6%B5%8B%E8%AF%95%E5%88%86%E7%A6%BB.md)
- 3. [常用评估指标（MAE、MSE、RMSE、MAPE）](03-%E5%B8%B8%E7%94%A8%E8%AF%84%E4%BC%B0%E6%8C%87%E6%A0%87%EF%BC%88MAE%E3%80%81MSE%E3%80%81RMSE%E3%80%81MAPE%EF%BC%89.md)
- 4. [信息准则 (AIC, BIC)](04-%E4%BF%A1%E6%81%AF%E5%87%86%E5%88%99%20%28AIC%2C%20BIC%29.md)
- 5. [比较不同模型的预测结果](05-%E6%AF%94%E8%BE%83%E4%B8%8D%E5%90%8C%E6%A8%A1%E5%9E%8B%E7%9A%84%E9%A2%84%E6%B5%8B%E7%BB%93%E6%9E%9C.md)
- 6. [预测表现的可视化](06-%E9%A2%84%E6%B5%8B%E8%A1%A8%E7%8E%B0%E7%9A%84%E5%8F%AF%E8%A7%86%E5%8C%96.md)
- 7. [动手练习：评估预测](07-%E5%8A%A8%E6%89%8B%E7%BB%83%E4%B9%A0%EF%BC%9A%E8%AF%84%E4%BC%B0%E9%A2%84%E6%B5%8B.md)

章节测验：[在线测验](https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-6-model-evaluation-selection/quiz)
