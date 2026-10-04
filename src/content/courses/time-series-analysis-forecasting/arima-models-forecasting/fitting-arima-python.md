---
course: "time-series-analysis-forecasting"
chapter: "arima-models-forecasting"
lesson: "fitting-arima-python"
sourceId: 3433
sourceUrl: "https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-4-arima-models-forecasting/fitting-arima-python"
title: "在Python中使用statsmodels拟合ARIMA模型"
description: "在Python中使用statsmodels库来拟合时间序列数据的ARIMA模型。"
order: 6
plots: ["plots/3433-0.json"]
sourceHash: "bc09c63d4cdc91f9448ef1e8524bfdac00faa1208b4e2de255a33cdce6693267"
sourceCorrections: []
---

时间序列预测中的ARIMA模型构建涉及多个步骤：准备时间序列数据，通过差分使其平稳（确定$d$参数 (parameter)），并使用ACF/PACF图获得自回归 (autoregressive)($p$)和移动平均($q$)阶数的初步估计。接下来，使用Python的`statsmodels`库对所选的ARIMA($p, d, q$)模型进行参数估计。

`statsmodels`中实现此功能的主要工具是位于`statsmodels.tsa.arima.model`模块中的`ARIMA`类。此实现基于状态空间方法，为处理ARIMA建模提供了一种灵活且高效的途径。

### 导入和准备数据

首先，请确保您已导入所需的库，并将数据加载到带有`DatetimeIndex`的pandas Series中。尽管`statsmodels`有时可以使用简单的数值索引，但对于时间序列分析来说，使用`DatetimeIndex`是最佳实践，对于有效解释结果和预测也非常重要。

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm # 经常用于数据集或其他工具

# 假设 'series_original' 是包含时间序列数据的 pandas Series
# 它应该有一个 DatetimeIndex。例如：
# rng = pd.date_range('2020-01-01', periods=100, freq='D')
# series_original = pd.Series(np.random.randn(100).cumsum() + 50, index=rng)

# 基于之前的分析（平稳性检验、ACF/PACF），我们假设
# 决定采用 ARIMA(1, 1, 1) 模型。
p = 1
d = 1
q = 1
```

### 实例化和拟合模型

要拟合ARIMA模型，您首先创建`ARIMA`类的一个实例，传入您的时间序列数据（`endog`表示内生变量）和所选的阶数$(p, d, q)$。`order`参数 (parameter)接受一个包含这三个整数的元组。

```python
# 1. 实例化 ARIMA 模型
# 提供原始序列和阶数 (p, d, q)
# 模型会根据 d 内部处理差分
model = ARIMA(series_original, order=(p, d, q))

# 2. 拟合模型
# 这一步执行参数估计，通常使用
# 最大似然估计 (MLE)。
results = model.fit()
```

`fit()`方法承担了大部分繁重的工作。它估计AR系数($\phi_1, ..., \phi_p$)、MA系数($\theta_1, ..., \theta_q$)以及误差项的方差($\sigma^2$)。它使用数值优化技术来寻找参数值，使在给定模型结构的情况下观察到实际数据的可能性最大化。

### 检查拟合结果

模型拟合完成后，`results`对象（通常是`ARIMAResultsWrapper`的一个实例）包含了关于估计模型的丰富信息。查看主要发现最便捷的方式是使用`summary()`方法。

```python
# 3. 打印拟合模型的摘要
print(results.summary())
```

`summary()`的输出通常包括：

- **模型信息：** 有关模型类型（ARIMA）、因变量、拟合日期/时间以及样本范围的详细信息。
- **系数表：** 这是一个重要的部分。它列出了每个参数 (parameter)的估计值（例如，`ar.L1`表示第一个AR滞后系数$\phi_1$，`ma.L1`表示第一个MA滞后系数$\theta_1$，`sigma2`表示误差方差）。在每个系数旁边，您会看到：
  - `std err`：系数估计值的标准误差，表示其精确度。
  - `z`：z统计量（系数除以标准误差），用于假设检验。
  - `P>|z|`：与z统计量相关的p值。较小的p值（通常< 0.05）表明该系数在统计上与零有显著差异。
  - `[0.025 0.975]`：系数的95%置信区间。
- **模型诊断：** 提供了AIC（赤池信息准则）和BIC（贝叶斯信息准则）等信息准则。这些对于比较不同的ARIMA阶数很有用（较低的值通常表示模型拟合和复杂性之间有更好的平衡）。您通常还会看到残差检验的结果（例如Ljung-Box用于自相关性，Jarque-Bera用于正态性），我们将在模型诊断部分详细讨论。

这是一个解释系数行的例子：

```
        coef    std err          z      P>|z|      [0.025      0.975]
-----------------------------------------------------------------------
ar.L1   0.650      0.080      8.125      0.000       0.493       0.807
```

这表明第一个AR项的估计系数($\hat{\phi}_1$)是0.650。p值为0.000，强烈表明该项在统计上具有显著性。95%置信区间范围从0.493到0.807。

### 访问拟合值

`results`对象还允许您访问模型在训练数据上的拟合值。这些是模型在样本内会做出的单步超前预测。将这些与实际数据进行比较可以直观地感受模型的拟合情况。请记住，如果$d > 0$，拟合值对应于*差分后*的序列。

```python
# 访问拟合值
fitted_values = results.fittedvalues

# 如果 d=1，拟合值对应于差分序列
# 如果 d=0，拟合值对应于原始序列
```

让我们可视化拟合值与实际数据（如果适用，经过差分后）的比较情况。对于ARIMA(1, 1, 1)的例子，我们会将拟合值与一阶差分序列进行比较。

```python
# 示例：假设 d=1
series_diff1 = series_original.diff().dropna()

# 生成一些用于绘图的示例数据
np.random.seed(42)
n_points = 100
rng = pd.date_range('2020-01-01', periods=n_points, freq='D')
true_ar = [0.7]
true_ma = [-0.4]
errors = np.random.normal(0, 1, n_points)
y = np.zeros(n_points)
for t in range(1, n_points):
    y[t] = true_ar[0] * y[t-1] + errors[t] + true_ma[0] * errors[t-1]
series_arma = pd.Series(y, index=rng) # 这是一个平稳的 ARMA(1,1)
# 为了使其成为 ARIMA(1,1,1)，我们进行积分
series_original_example = series_arma.cumsum() + 50
series_original_example = series_original_example.asfreq('D') # 确保频率

# 对此示例数据拟合 ARIMA(1,1,1)
model_example = ARIMA(series_original_example, order=(1, 1, 1))
results_example = model_example.fit()
fitted_vals_example = results_example.fittedvalues

# 获取实际的差分序列进行比较
actual_diff_example = series_original_example.diff().dropna()

# 确保索引对齐以便绘图（拟合值可能会缺少前 d 个点）
comparison_df = pd.DataFrame({
    'Actual Differenced': actual_diff_example,
    'Fitted Values': fitted_vals_example
}).dropna()
```



![ARIMA(1,1,1): 实际值 vs. 拟合值 (差分数据)](plots/3433-0.json)



> 示例数据的一小段中，ARIMA(1,1,1)模型的实际一阶差分值与单步超前拟合值之间的比较。

此图有助于直观地评估模型在训练期内捕获（差分后）序列动态的良好程度。显著的偏差可能表示模型存在错误设定。

拟合模型是一个核心步骤，但并非终点。您现在已经根据所选阶数估计了参数 (parameter)。接下来的步骤包括通过分析残差来仔细诊断模型拟合情况，然后使用经过验证的模型来生成预测。

## 参考资料

- [statsmodels.tsa.arima.model.ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html) — statsmodels Developers (2023)
  `statsmodels`中ARIMA类的官方文档，详细介绍了其用法、参数和结果对象。
- [Time Series Analysis: Forecasting and Control](https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118675021) — George E. P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, Greta M. Ljung (2015)
  Publisher: John Wiley & Sons; Pages: 720
  一本具有开创性的书籍，提供了ARIMA模型的理论和实践基础，包括识别、最大似然估计以及诊断检查。
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/) — Rob J Hyndman, George Athanasopoulos (2021)
  Publisher: OTexts
  一本高度易读且全面的在线教科书，涵盖了现代预测方法，包括对ARIMA模型及其应用的详细处理。
- [Mastering Time Series Analysis with Python](https://www.packtpub.com/product/mastering-time-series-analysis-with-python/9781838644061) — Gabriel A. Papa (2020)
  Publisher: Packt Publishing
  一本使用`statsmodels`等Python库实现各种时间序列模型（包括ARIMA）的实践指南，并附有代码示例。
