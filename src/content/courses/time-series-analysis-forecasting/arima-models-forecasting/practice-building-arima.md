---
course: "time-series-analysis-forecasting"
chapter: "arima-models-forecasting"
lesson: "practice-building-arima"
sourceId: 3444
sourceUrl: "https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-4-arima-models-forecasting/practice-building-arima"
title: "动手实践：构建ARIMA模型"
description: "使用Python拟合ARIMA模型到样本数据，进行诊断，并生成预测。"
order: 9
plots: ["plots/3444-0.json", "plots/3444-1.json"]
sourceHash: "d3cf046bef39bb9b1373639a55b5a196594731492da28883692682c854378994"
sourceCorrections: []
---

提供了一个构建ARIMA模型的完整示例，从数据准备开始，到生成预测结束。使用一个遵循已知ARIMA过程的合成数据集，有助于验证这些步骤。

### 环境设置

首先，确保已安装所需库并导入它们：

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# 配置绘图风格（可选）
plt.style.use('seaborn-v0_8-whitegrid')
```

### 生成样本数据

在此练习中，我们将生成遵循ARIMA(1,1,1)过程的数据。这意味着序列的一阶差分遵循ARMA(1,1)过程。了解潜在过程使我们能够检查分析是否能返回正确的模型结构。

```python
# 定义ARIMA(1,1,1)参数
np.random.seed(42) # 用于结果复现
n_sample = 250
ar_params = np.array([0.6])  # ARMA部分的AR(1)系数
ma_params = np.array([0.4])  # ARMA部分的MA(1)系数
ar = np.r_[1, -ar_params]    # AR的滞后多项式表示
ma = np.r_[1, ma_params]     # MA的滞后多项式表示

# 生成平稳的ARMA(1,1)分量
arma_process = ArmaProcess(ar, ma)
arma_data = arma_process.generate_sample(nsample=n_sample)

# 积分ARMA数据以获得ARIMA(1,1,1)
# 积分（d=1）通过计算累积和实现
y = np.cumsum(arma_data) + 100 # 添加一个常数以模拟起始水平

# 创建一个带日期时间索引的pandas Series
timestamps = pd.date_range(start='2021-01-01', periods=n_sample, freq='D')
ts_data = pd.Series(y, index=timestamps, name='Value')

print("前5个数据点：\n", ts_data.head())
```

### 步骤1：数据查看与平稳性检查

我们来可视化生成序列并检查其平稳性。

```python
# 绘制原始时间序列
plt.figure(figsize=(10, 4))
plt.plot(ts_data, color='#1c7ed6')
plt.title('生成的ARIMA(1,1,1)时间序列')
plt.xlabel('日期')
plt.ylabel('值')
plt.show()
```



![生成的ARIMA(1,1,1)时间序列](plots/3444-0.json)



> 生成的时间序列数据显示波动，但没有明显的恒定均值，这表明其不具备平稳性。

该图显示出一种游走行为，这是随机游走或积分过程的特征。我们用增广迪基-富勒（ADF）检验来确认这一点。ADF检验的原假设（$H_0$）是序列存在单位根（即非平稳）。

```python
# 对原始序列进行ADF检验
adf_result = adfuller(ts_data)

print(f'ADF统计量: {adf_result[0]:.4f}')
print(f'p值: {adf_result[1]:.4f}')
print('临界值:')
for key, value in adf_result[4].items():
    print(f'\t{key}: {value:.4f}')

if adf_result[1] > 0.05:
    print("\n结果：该序列可能非平稳（未能拒绝H0）。")
else:
    print("\n结果：该序列可能平稳（拒绝H0）。")
```

ADF检验结果可能显示较高的p值（例如，> 0.05），证实了我们视觉上对非平稳性的判断。我们需要对序列进行差分以使其平稳。鉴于我们怀疑 $d=1$（根据我们生成数据和绘图的方式），我们进行一阶差分。

```python
# 计算一阶差分
ts_diff = ts_data.diff().dropna() # dropna() 删除第一个NaN值

# 绘制差分序列
plt.figure(figsize=(10, 4))
plt.plot(ts_diff, color='#40c057')
plt.title('时间序列的一阶差分')
plt.xlabel('日期')
plt.ylabel('差分值')
plt.show()

# 对差分序列进行ADF检验
adf_result_diff = adfuller(ts_diff)

print("\n差分序列的ADF检验:")
print(f'ADF统计量: {adf_result_diff[0]:.4f}')
print(f'p值: {adf_result_diff[1]:.4f}')

if adf_result_diff[1] <= 0.05:
    print("结果：差分序列可能平稳（拒绝H0）。")
else:
    print("结果：差分序列可能非平稳（未能拒绝H0）。")
```

差分序列的图现在应该看起来更平稳，在恒定均值（接近零）附近波动，并具有恒定方差。对 `ts_diff` 的ADF检验应产生非常小的p值（例如，< 0.01），这强烈表明了平稳性。这证实了 $d=1$ 对于我们的ARIMA模型是合适的。

### 步骤2：确定AR和MA阶数 (p, q)

现在我们分析*平稳*差分序列（`ts_diff`）的自相关函数（ACF）图和偏自相关函数（PACF）图，以估计AR阶数（$p$）和MA阶数（$q$）。

- **ACF图：** 有助于确定MA阶数（$q$）。寻找滞后 $q$ 后的明显截断。
- **PACF图：** 有助于确定AR阶数（$p$）。寻找滞后 $p$ 后的明显截断。

```python
# 绘制差分序列的ACF和PACF图
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# ACF图
plot_acf(ts_diff, ax=axes[0], lags=20, color='#228be6', vlines_kwargs={"colors": '#228be6'})
axes[0].set_title('自相关函数 (ACF)')
axes[0].set_xlabel('滞后')
axes[0].set_ylabel('ACF')

# PACF图
plot_pacf(ts_diff, ax=axes[1], lags=20, method='ywm', color='#15aabf', vlines_kwargs={"colors": '#15aabf'})
axes[1].set_title('偏自相关函数 (PACF)')
axes[1].set_xlabel('滞后')
axes[1].set_ylabel('PACF')

plt.tight_layout()
plt.show()
```

**图形判读：**

- **PACF：** 我们预期PACF图在滞后1处显示一个显著的峰值，然后急剧截断（落入置信区间带内）。这表明AR阶数为 $p=1$。
- **ACF：** 我们预期ACF图在滞后1处显示一个显著的峰值，然后逐渐衰减，或者在滞后1后截断。滞后1后的截断表明MA阶数为 $q=1$。逐渐衰减也可以是AR过程的迹象，但结合滞后1处的PACF截断，差分序列的ARMA(1,1)结构看起来可能。

基于此分析，一个合理的模型阶数是 $(p, d, q) = (1, 1, 1)$。这与我们生成数据时使用的参数 (parameter)相符。

### 步骤3：拟合ARIMA模型

现在我们拟合一个ARIMA(1,1,1)模型到*原始*时间序列数据（`ts_data`）。当你指定 $d=1$ 时，`statsmodels`库会在内部处理差分。

```python
# 定义ARIMA模型阶数
p, d, q = 1, 1, 1

# 创建并拟合ARIMA模型
# 注意：我们是在原始ts_data上拟合的，并指定d=1
model = ARIMA(ts_data, order=(p, d, q))
model_fit = model.fit()

# 打印模型汇总信息
print(model_fit.summary())
```

汇总信息提供了丰富的内容：

- **系数 (coef)：** AR项（ar.L1）、MA项（ma.L1）以及可能的常数项或漂移项的估计值。将这些与我们生成数据时使用的 `ar_params` (0.6) 和 `ma_params` (0.4) 进行比较。它们应该合理接近。
- **标准误差 (std err)：** 与系数估计值相关的不确定性。
- **p值 (P>|z|) :** 系数的显著性检验。小的p值（例如，< 0.05）表明这些项具有统计显著性。
- **对数似然、AIC、BIC：** 模型拟合的衡量指标，用于比较不同模型阶数。较低的值通常表示更好的拟合，并对复杂性进行惩罚。
- **残差诊断：** 诸如Ljung-Box (Prob(Q)) 检验残差中剩余的自相关性，Jarque-Bera (Prob(JB)) 检验正态性。

### 步骤4：模型诊断

一个好的ARIMA模型，其残差应类似于白噪声：零均值、恒定方差且无自相关性。

```python
# 获取模型残差
residuals = model_fit.resid

# 绘制残差
plt.figure(figsize=(10, 4))
plt.plot(residuals, color='#be4bdb')
plt.title('ARIMA(1,1,1)模型残差')
plt.xlabel('日期')
plt.ylabel('残差值')
plt.show()

# 绘制残差的ACF和PACF图
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(residuals, ax=axes[0], lags=20, color='#f76707', vlines_kwargs={"colors": '#f76707'})
axes[0].set_title('残差ACF')
plot_pacf(residuals, ax=axes[1], lags=20, method='ywm', color='#f59f00', vlines_kwargs={"colors": '#f59f00'})
axes[1].set_title('残差PACF')
plt.tight_layout()
plt.show()
```

残差图应围绕零点波动，没有明显的模式。残差的ACF和PACF图理想情况下在滞后 > 0 时不应显示置信区间外的显著峰值。模型汇总中的Ljung-Box检验结果（Prob(Q)）的p值应大于0.05，这表明残差中没有剩余的显著自相关性。Jarque-Bera检验（Prob(JB)）检查正态性；如果p值较低，残差可能不服从正态分布，但ARIMA有时对轻微偏差具有容忍性。

如果诊断结果良好，我们就可以进行预测。如果不好，你可能需要重新考虑模型阶数 (p, d, q) 或检查是否存在其他因素（例如季节性，将在下一章介绍）。

### 步骤5：预测

我们使用已拟合的模型进行预测。我们可以在原始数据结束后为未来的时间点生成预测。

```python
# 定义向前预测的步数
n_forecast_steps = 30

# 生成预测
forecast_result = model_fit.get_forecast(steps=n_forecast_steps)
forecast_mean = forecast_result.predicted_mean
forecast_ci = forecast_result.conf_int(alpha=0.05) # 95% 置信区间

# 为预测期创建日期索引
last_date = ts_data.index[-1]
forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_forecast_steps, freq='D')

# 结合原始数据和预测进行绘图
plt.figure(figsize=(12, 5))
plt.plot(ts_data.index, ts_data, label='观测数据', color='#495057')
plt.plot(forecast_index, forecast_mean, label='预测', color='#f03e3e')
plt.fill_between(forecast_index,
                 forecast_ci.iloc[:, 0], # 置信区间下限
                 forecast_ci.iloc[:, 1], # 置信区间上限
                 color='#ffc9c9', alpha=0.5, label='95% 置信区间')

plt.title('ARIMA(1,1,1)预测')
plt.xlabel('日期')
plt.ylabel('值')
plt.legend()
plt.show()
```



![ARIMA(1,1,1)预测与观测值对比](plots/3444-1.json)



> 使用ARIMA(1,1,1)模型得到的预测值，以及95%置信区间，延伸到观测数据期之后。

该图显示原始数据，然后是预测值。请注意，随着预测期限的增加，置信区间会变宽，反映出未来不确定性的增加。对于 $d \ge 1$ 的ARIMA(p,d,q)模型，长期预测将趋于一个恒定水平（如果 $d=0$），或者遵循线性趋势（如果 $d=1$ 且包含常数项/漂移项）。

"这结束了我们构建基本ARIMA模型的动手实践。你已经了解了整个流程：检查平稳性，如有必要进行差分，使用ACF/PACF确定 $p$ 和 $q$，拟合模型，检查残差，最后生成预测。请记住，数据通常更为复杂，可能需要考量季节性（下一章将介绍）或其他外部因素。"

## 参考资料

- [Forecasting: Principles and Practice](https://otexts.com/fpp3/) — Rob J Hyndman and George Athanasopoulos (2021)
  Publisher: OTexts; Pages: Chapters 8-9
  一本全面且实用的在线时间序列预测教材，详细涵盖了ARIMA模型及其应用。
- [Time Series Analysis: Forecasting and Control](https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118675021) — George E. P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, Greta M. Ljung (2015)
  Publisher: Wiley; Pages: 720; DOI: [10.1002/9781118619193](https://doi.org/10.1002/9781118619193)
  ARIMA模型的奠基性教材，为时间序列分析的Box-Jenkins方法提供了理论和方法学基础。
- [ARIMA: Autoregressive Integrated Moving Average (ARIMA) Models](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html) — statsmodels developers (2023)
  `statsmodels`中ARIMA模型实现的官方文档，详细说明了其在Python中的使用方法、参数和结果。
