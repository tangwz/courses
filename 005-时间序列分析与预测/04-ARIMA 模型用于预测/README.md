# 第 4 章：ARIMA 模型用于预测

来源：[原章节](https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-4-arima-models-forecasting)

[返回课程目录](../README.md)

在前几章中，我们已经确定了如何分解时间序列、检验平稳性，并借助 ACF 和 PACF 图识别可能的模型结构，现在我们将侧重于构建统计预测模型。本章介绍自回归整合移动平均 (ARIMA) 模型系列，这是一类广泛应用于分析和预测平稳时间序列数据的模型。

你将了解其核心组成部分：
* **自回归 (AR) 模型：** 过去的数值如何影响当前数值。
* **移动平均 (MA) 模型：** 过去的预测误差如何影响当前数值。
* **ARMA 模型：** 结合 AR 和 MA 分量用于平稳序列。
* **ARIMA 模型：** 引入差分（“整合”部分）以处理非平稳序列，再应用 ARMA 建模。

我们将讲解如何确定 ARIMA 模型阶数（表示为 ($p, d, q$)），依据 ACF/PACF 分析所得。你将学习如何使用 Python 的 `statsmodels` 库将这些模型拟合到数据，通过检查残差诊断模型的拟合程度，并为未来时间点生成预测。本章最后将通过动手实践，构建和评估一个完整的 ARIMA 预测工作流程。

## 小节

- 1. [自回归 (AR) 模型](01-%E8%87%AA%E5%9B%9E%E5%BD%92%20%28AR%29%20%E6%A8%A1%E5%9E%8B.md)
- 2. [移动平均 (MA) 模型](02-%E7%A7%BB%E5%8A%A8%E5%B9%B3%E5%9D%87%20%28MA%29%20%E6%A8%A1%E5%9E%8B.md)
- 3. [结合AR和MA：ARMA模型](03-%E7%BB%93%E5%90%88AR%E5%92%8CMA%EF%BC%9AARMA%E6%A8%A1%E5%9E%8B.md)
- 4. [引入整合：ARIMA 模型](04-%E5%BC%95%E5%85%A5%E6%95%B4%E5%90%88%EF%BC%9AARIMA%20%E6%A8%A1%E5%9E%8B.md)
- 5. [选择ARIMA模型阶数 (p, d, q)](05-%E9%80%89%E6%8B%A9ARIMA%E6%A8%A1%E5%9E%8B%E9%98%B6%E6%95%B0%20%28p%2C%20d%2C%20q%29.md)
- 6. [在Python中使用statsmodels拟合ARIMA模型](06-%E5%9C%A8Python%E4%B8%AD%E4%BD%BF%E7%94%A8statsmodels%E6%8B%9F%E5%90%88ARIMA%E6%A8%A1%E5%9E%8B.md)
- 7. [模型诊断与残差分析](07-%E6%A8%A1%E5%9E%8B%E8%AF%8A%E6%96%AD%E4%B8%8E%E6%AE%8B%E5%B7%AE%E5%88%86%E6%9E%90.md)
- 8. [使用ARIMA模型进行预测](08-%E4%BD%BF%E7%94%A8ARIMA%E6%A8%A1%E5%9E%8B%E8%BF%9B%E8%A1%8C%E9%A2%84%E6%B5%8B.md)
- 9. [动手实践：构建ARIMA模型](09-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BAARIMA%E6%A8%A1%E5%9E%8B.md)

章节测验：[在线测验](https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-4-arima-models-forecasting/quiz)
