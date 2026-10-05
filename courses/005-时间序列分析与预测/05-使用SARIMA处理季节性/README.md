# 第 5 章：使用SARIMA处理季节性

来源：[原章节](https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-5-seasonal-arima-sarima)

[返回课程目录](../README.md)

尽管ARIMA模型对许多时间序列都很有效，但在处理具有强烈季节性模式（即在固定周期内重复出现的可预测周期，如月度或季度变化）的数据时，它们常常显得不足。

本章将介绍季节性ARIMA（SARIMA），它是专门为具有季节性组成部分的时间序列数据建模和预测而设计的扩展。我们将介绍SARIMA如何将季节性项与您在ARIMA中学到的非季节性部分相结合。

您将学习如何：

*   理解SARIMA模型的结构，其表示形式为 $ SARIMA(p, d, q)(P, D, Q)_m $，其中 $ (P, D, Q) $ 代表季节性自回归、差分和移动平均阶数，$ m $ 是季节性周期。
*   使用自相关函数（ACF）图和偏自相关函数（PACF）图来识别潜在的季节性阶数。
*   为非季节性 $ (p, d, q) $ 和季节性 $ (P, D, Q)_m $ 参数选择恰当的数值。
*   使用Python的`statsmodels`库将SARIMA模型拟合到数据。
*   通过检验残差来诊断已拟合的模型。
*   生成明确考虑季节性影响的预测。

在本章结束时，您将能够应用SARIMA模型，有效地分析和预测受季节性影响的时间序列。

## 小节

- 1. [ARIMA模型处理季节性数据的局限性](01-ARIMA%E6%A8%A1%E5%9E%8B%E5%A4%84%E7%90%86%E5%AD%A3%E8%8A%82%E6%80%A7%E6%95%B0%E6%8D%AE%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7.md)
- 2. [季节性ARIMA（SARIMA）模型介绍](02-%E5%AD%A3%E8%8A%82%E6%80%A7ARIMA%EF%BC%88SARIMA%EF%BC%89%E6%A8%A1%E5%9E%8B%E4%BB%8B%E7%BB%8D.md)
- 3. [识别季节性成分 (ACF/PACF)](03-%E8%AF%86%E5%88%AB%E5%AD%A3%E8%8A%82%E6%80%A7%E6%88%90%E5%88%86%20%28ACF-PACF%29.md)
- 4. [选择 SARIMA 阶数 (p, d, q)(P, D, Q)m](04-%E9%80%89%E6%8B%A9%20SARIMA%20%E9%98%B6%E6%95%B0%20%28p%2C%20d%2C%20q%29%28P%2C%20D%2C%20Q%29m.md)
- 5. [在 Python 中拟合 SARIMA 模型](05-%E5%9C%A8%20Python%20%E4%B8%AD%E6%8B%9F%E5%90%88%20SARIMA%20%E6%A8%A1%E5%9E%8B.md)
- 6. [SARIMA模型诊断](06-SARIMA%E6%A8%A1%E5%9E%8B%E8%AF%8A%E6%96%AD.md)
- 7. [使用 SARIMA 进行预测](07-%E4%BD%BF%E7%94%A8%20SARIMA%20%E8%BF%9B%E8%A1%8C%E9%A2%84%E6%B5%8B.md)
- 8. [动手实践：构建SARIMA模型](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BASARIMA%E6%A8%A1%E5%9E%8B.md)

章节测验：[在线测验](https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-5-seasonal-arima-sarima/quiz)
