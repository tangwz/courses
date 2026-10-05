# 在 Python 中拟合 SARIMA 模型

来源：[原文](https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-5-seasonal-arima-sarima/fitting-sarima-python)

[返回章节目录](README.md) · [返回课程目录](../README.md)

使用时间序列数据估计 SARIMA 模型的参数 (parameter)是时间序列分析的**主要**步骤。此过程涉及处理非季节性阶数 $(p, d, q)$ 和季节性阶数 $(P, D, Q)_m$ 以准确拟合模型。Python 的 `statsmodels` 库为此目的提供了一个方便且功能强大的实现方式。

我们将使用的主要工具是位于 `statsmodels.tsa.statespace.sarimax` 中的 `SARIMAX` 类。此类旨在处理包含季节性 (Seasonal)、自回归 (autoregressive)成分 (AR)、差分 (I)、移动平均成分 (MA) 以及外生回归变量 (X) 的模型——尽管在本节中我们不会侧重于“X”部分。

要拟合 SARIMA 模型，您首先需要实例化 `SARIMAX` 类，并提供您的时间序列数据和选定的模型阶数。

您将使用的主要参数有：

- `endog`：这是您的时间序列数据，通常是 Pandas Series 或 NumPy 数组。`endog` 代表内生变量，意为您尝试建模和预测的变量。
- `order`：一个表示非季节性阶数 $(p, d, q)$ 的元组。
- `seasonal_order`：一个表示季节性阶数 $(P, D, Q, m)$ 的元组，其中 $m$ 是单个季节周期中的时间步数（例如，对于具有年度周期的月度数据为 12，对于季度数据为 4）。

假设您已确定数据的合适阶数。例如，假设您正在处理月度数据 ($m=12$)，并决定使用 $SARIMA(1, 1, 1)(1, 1, 0)_{12}$ 模型。以下是如何在 Python 中实例化并拟合它：

```python
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# 假设 'series_train' 是您的训练时间序列数据 (Pandas Series)
# 示例：月度数据，因此 m = 12
# 假设已确定这些阶数：
p, d, q = 1, 1, 1       # 非季节性阶数
P, D, Q, m = 1, 1, 0, 12 # 季节性阶数

# 1. 实例化 SARIMAX 模型
model = SARIMAX(series_train,
                order=(p, d, q),
                seasonal_order=(P, D, Q, m),
                enforce_stationarity=False, # 当进行差分时 (d>0 或 D>0) 通常设置为 False
                enforce_invertibility=False) # 当进行差分时 (d>0 或 D>0) 通常设置为 False

# 2. 将模型拟合到数据
# 此步骤使用最大似然估计 (MLE) 来估计模型参数
model_fit = model.fit(disp=False) # disp=False 隐藏收敛消息

# 3. 打印模型概要
print(model_fit.summary())
```

`enforce_stationarity` 和 `enforce_invertibility` 参数分别与 AR 和 MA 成分的属性相关。虽然对于纯 ARMA 模型通常会强制执行，但当涉及差分时 ($d>0$ 或 $D>0$)，将它们设置为 `False` 有时可以避免估计问题，因为差分本身处理了非平稳性。

`.fit()` 方法执行核心估计过程。它使用数值优化技术，通常是最大似然估计 (MLE)，以找到指定 AR、MA、季节性 AR 和季节性 MA 项的系数值，从而最好地匹配观测数据。这对于复杂模型或大型数据集而言可能计算量大。将 `disp=False` 设置为 `False` 会抑制优化器发出的收敛输出消息。

模型拟合完成后，`model_fit` 对象包含结果，包括估计的参数、标准误差、拟合优度统计量等。`.summary()` 方法提供全面概览，格式良好，便于查看。

典型的概要输出如下所示（数值仅供说明）：

```
                                     SARIMAX Results
==========================================================================================
因变量:                              y   观测数量:                  132
模型:             SARIMAX(1, 1, 1)x(1, 1, 0, 12)   对数似然                -150.450
日期:                            Thu, 15 Aug 2024   AIC                            308.900
时间:                                    10:30:00   BIC                            319.500
样本:                                01-01-2010   HQIC                           313.200
                                     - 12-01-2020
协方差类型:                              opg
==============================================================================
                 系数    标准误差          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1          0.3500      0.080      4.375      0.000       0.193       0.507
ma.L1         -0.8500      0.060    -14.167      0.000      -0.968      -0.732
ar.S.L12       0.1500      0.090      1.667      0.096      -0.026       0.326
sigma2         0.9500      0.150      6.333      0.000       0.656       1.244
===================================================================================
Ljung-Box (L1) (Q):                   0.02   Jarque-Bera (JB):                 2.50
Prob(Q):                              0.88   Prob(JB):                         0.29
异方差性 (H):               1.10   偏度:                             0.20
Prob(H) (双边):                  0.75   峰度:                             3.50
===================================================================================

警告:
[1] 协方差矩阵使用梯度外积（复数步长）计算。
```

在此概要中：

- `coef`：显示参数的估计值（非季节性 AR(1) 项 `ar.L1`、非季节性 MA(1) 项 `ma.L1`、季节性 AR(12) 项 `ar.S.L12`）。
- `std err`：系数估计值的标准误差。
- `z`：z 统计量 (`coef` / `std err`)。
- `P>|z|`：与 z 统计量相关的 p 值。较低的 p 值（通常 < 0.05）表明系数具有统计显著性。
- `[0.025 0.975]`：系数的 95% 置信区间。
- `sigma2`：残差的估计方差。
- `Log Likelihood`、`AIC`、`BIC`、`HQIC`：用于模型比较的信息准则（将在第 6 章中进一步讨论）。
- 底部表格提供了对残差进行诊断测试的结果（我们将在下一部分介绍）。

拟合模型是重要步骤。获得 `model_fit` 后，您可以继续通过检查模型的残差来诊断模型的充分性，然后使用它来生成未来时间点的预测。

## 参考资料

- [statsmodels.tsa.statespace.sarimax.SARIMAX](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html) — statsmodels developers (2024)
  Python `statsmodels` 库中 SARIMAX 类的官方文档，详细说明了其参数和用法。
- [Time Series Analysis: Forecasting and Control](https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118675021) — George E. P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, and Greta M. Ljung (2015)
  Publisher: Wiley
  关于 ARIMA 和 SARIMA 模型的权威教科书，为模型估计提供了全面的理论基础。
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/) — Rob J Hyndman and George Athanasopoulos (2021)
  Publisher: CRC Press
  一本实用且广泛使用的时序预测指南，包含 SARIMA 模型的易懂解释及其应用。

---

[上一节](04-%E9%80%89%E6%8B%A9%20SARIMA%20%E9%98%B6%E6%95%B0%20%28p%2C%20d%2C%20q%29%28P%2C%20D%2C%20Q%29m.md) · [下一节](06-SARIMA%E6%A8%A1%E5%9E%8B%E8%AF%8A%E6%96%AD.md)
