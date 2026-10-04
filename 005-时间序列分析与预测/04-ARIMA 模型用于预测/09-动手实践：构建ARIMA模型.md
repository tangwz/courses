# 动手实践：构建ARIMA模型

来源：[原文](https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-4-arima-models-forecasting/practice-building-arima)

[返回章节目录](README.md) · [返回课程目录](../README.md)

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



[交互图表：生成的ARIMA(1,1,1)时间序列](https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-4-arima-models-forecasting/practice-building-arima#plot-1bm6b8u)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "生成的ARIMA(1,1,1)时间序列",
    "xaxis": {
      "title": "日期"
    },
    "yaxis": {
      "title": "值"
    },
    "showlegend": false,
    "width": 600,
    "height": 400,
    "template": "seaborn"
  },
  "data": [
    {
      "type": "scatter",
      "x": [
        "2021-01-01",
        "2021-01-02",
        "2021-01-03",
        "2021-01-04",
        "2021-01-05",
        "2021-01-06",
        "2021-01-07",
        "2021-01-08",
        "2021-01-09",
        "2021-01-10",
        "2021-01-11",
        "2021-01-12",
        "2021-01-13",
        "2021-01-14",
        "2021-01-15",
        "2021-01-16",
        "2021-01-17",
        "2021-01-18",
        "2021-01-19",
        "2021-01-20",
        "2021-01-21",
        "2021-01-22",
        "2021-01-23",
        "2021-01-24",
        "2021-01-25",
        "2021-01-26",
        "2021-01-27",
        "2021-01-28",
        "2021-01-29",
        "2021-01-30",
        "2021-01-31",
        "2021-02-01",
        "2021-02-02",
        "2021-02-03",
        "2021-02-04",
        "2021-02-05",
        "2021-02-06",
        "2021-02-07",
        "2021-02-08",
        "2021-02-09",
        "2021-02-10",
        "2021-02-11",
        "2021-02-12",
        "2021-02-13",
        "2021-02-14",
        "2021-02-15",
        "2021-02-16",
        "2021-02-17",
        "2021-02-18",
        "2021-02-19"
      ],
      "y": [
        101.76,
        100.56,
        101.03,
        101.66,
        100.92,
        100.89,
        100.48,
        100.77,
        101.76,
        101.38,
        101.44,
        101.99,
        102.29,
        101.54,
        101.03,
        100.87,
        101.65,
        101.38,
        102.37,
        103.46,
        103.53,
        103.69,
        103.82,
        103.74,
        103.97,
        103.93,
        104.85,
        104.82,
        105.43,
        106.19,
        105.78,
        105.17,
        105.81,
        105.94,
        106.61,
        106.47,
        106.46,
        107.02,
        108.13,
        107.77,
        108.0,
        107.99,
        108.29,
        108.49,
        108.47,
        109.05,
        108.45,
        109.02,
        108.54,
        108.3
      ],
      "mode": "lines",
      "line": {
        "color": "#1c7ed6"
      }
    }
  ]
}
```

</details>



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



[交互图表：ARIMA(1,1,1)预测与观测值对比](https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-4-arima-models-forecasting/practice-building-arima#plot-xpbfhn)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "ARIMA(1,1,1)预测与观测值对比",
    "xaxis": {
      "title": "日期",
      "range": [
        "2021-07-01",
        "2021-10-07"
      ]
    },
    "yaxis": {
      "title": "值"
    },
    "legend": {
      "title": {
        "text": "图例"
      }
    },
    "width": 700,
    "height": 450,
    "template": "seaborn"
  },
  "data": [
    {
      "type": "scatter",
      "x": [
        "2021-08-01",
        "2021-08-02",
        "2021-08-03",
        "2021-08-04",
        "2021-08-05",
        "2021-08-06",
        "2021-08-07",
        "2021-08-08",
        "2021-08-09",
        "2021-08-10",
        "2021-08-11",
        "2021-08-12",
        "2021-08-13",
        "2021-08-14",
        "2021-08-15",
        "2021-08-16",
        "2021-08-17",
        "2021-08-18",
        "2021-08-19",
        "2021-08-20",
        "2021-08-21",
        "2021-08-22",
        "2021-08-23",
        "2021-08-24",
        "2021-08-25",
        "2021-08-26",
        "2021-08-27",
        "2021-08-28",
        "2021-08-29",
        "2021-09-07"
      ],
      "y": [
        119.4,
        119.7,
        118.4,
        117.4,
        117.2,
        118.3,
        117.9,
        118.8,
        118.8,
        118.8,
        118.8,
        118.3,
        118.3,
        119.1,
        119.4,
        119.4,
        118.9,
        119.1,
        118.3,
        117.9,
        118.0,
        117.7,
        118.1,
        118.3,
        118.7,
        119.0,
        119.4,
        119.6,
        119.7,
        119.8
      ],
      "mode": "lines",
      "name": "观测数据",
      "line": {
        "color": "#495057"
      }
    },
    {
      "type": "scatter",
      "x": [
        "2021-09-08",
        "2021-09-09",
        "2021-09-10",
        "2021-09-11",
        "2021-09-12",
        "2021-09-13",
        "2021-09-14",
        "2021-09-15",
        "2021-09-16",
        "2021-09-17",
        "2021-09-18",
        "2021-09-19",
        "2021-09-20",
        "2021-09-21",
        "2021-09-22",
        "2021-09-23",
        "2021-09-24",
        "2021-09-25",
        "2021-09-26",
        "2021-09-27",
        "2021-09-28",
        "2021-09-29",
        "2021-09-30",
        "2021-10-01",
        "2021-10-02",
        "2021-10-03",
        "2021-10-04",
        "2021-10-05",
        "2021-10-06",
        "2021-10-07"
      ],
      "y": [
        119.83,
        119.84,
        119.84,
        119.85,
        119.85,
        119.85,
        119.85,
        119.85,
        119.85,
        119.85,
        119.85,
        119.85,
        119.85,
        119.85,
        119.85,
        119.85,
        119.85,
        119.85,
        119.85,
        119.85,
        119.85,
        119.85,
        119.85,
        119.85,
        119.85,
        119.85,
        119.85,
        119.85,
        119.85,
        119.85
      ],
      "mode": "lines",
      "name": "预测",
      "line": {
        "color": "#f03e3e"
      }
    },
    {
      "type": "scatter",
      "x": [
        "2021-09-08",
        "2021-09-09",
        "2021-09-10",
        "2021-09-11",
        "2021-09-12",
        "2021-09-13",
        "2021-09-14",
        "2021-09-15",
        "2021-09-16",
        "2021-09-17",
        "2021-09-18",
        "2021-09-19",
        "2021-09-20",
        "2021-09-21",
        "2021-09-22",
        "2021-09-23",
        "2021-09-24",
        "2021-09-25",
        "2021-09-26",
        "2021-09-27",
        "2021-09-28",
        "2021-09-29",
        "2021-09-30",
        "2021-10-01",
        "2021-10-02",
        "2021-10-03",
        "2021-10-04",
        "2021-10-05",
        "2021-10-06",
        "2021-10-07",
        "2021-10-07",
        "2021-10-06",
        "2021-10-05",
        "2021-10-04",
        "2021-10-03",
        "2021-10-02",
        "2021-10-01",
        "2021-09-30",
        "2021-09-29",
        "2021-09-28",
        "2021-09-27",
        "2021-09-26",
        "2021-09-25",
        "2021-09-24",
        "2021-09-23",
        "2021-09-22",
        "2021-09-21",
        "2021-09-20",
        "2021-09-19",
        "2021-09-18",
        "2021-09-17",
        "2021-09-16",
        "2021-09-15",
        "2021-09-14",
        "2021-09-13",
        "2021-09-12",
        "2021-09-11",
        "2021-09-10",
        "2021-09-09",
        "2021-09-08"
      ],
      "y": [
        117.87,
        117.17,
        116.66,
        116.26,
        115.94,
        115.67,
        115.43,
        115.23,
        115.04,
        114.87,
        114.72,
        114.57,
        114.44,
        114.31,
        114.19,
        114.08,
        113.97,
        113.87,
        113.78,
        113.69,
        113.6,
        113.52,
        113.44,
        113.37,
        113.3,
        113.23,
        113.16,
        113.1,
        113.04,
        112.99,
        121.9,
        121.87,
        121.83,
        121.79,
        121.74,
        121.69,
        121.64,
        121.59,
        121.55,
        121.51,
        121.47,
        121.43,
        121.39,
        121.35,
        121.31,
        121.28,
        121.24,
        121.21,
        121.17,
        121.14,
        121.11,
        121.08,
        121.05,
        121.02,
        120.99,
        120.96,
        120.93,
        120.89,
        120.84,
        121.79
      ],
      "fill": "toself",
      "fillcolor": "rgba(240, 62, 62, 0.3)",
      "line": {
        "color": "rgba(255, 255, 255, 0)"
      },
      "hoverinfo": "skip",
      "showlegend": false,
      "name": "置信区间下限"
    },
    {
      "type": "scatter",
      "x": [
        "2021-09-08",
        "2021-09-09",
        "2021-09-10",
        "2021-09-11",
        "2021-09-12",
        "2021-09-13",
        "2021-09-14",
        "2021-09-15",
        "2021-09-16",
        "2021-09-17",
        "2021-09-18",
        "2021-09-19",
        "2021-09-20",
        "2021-09-21",
        "2021-09-22",
        "2021-09-23",
        "2021-09-24",
        "2021-09-25",
        "2021-09-26",
        "2021-09-27",
        "2021-09-28",
        "2021-09-29",
        "2021-09-30",
        "2021-10-01",
        "2021-10-02",
        "2021-10-03",
        "2021-10-04",
        "2021-10-05",
        "2021-10-06",
        "2021-10-07"
      ],
      "y": [
        121.79,
        122.5,
        123.02,
        123.43,
        123.76,
        124.03,
        124.27,
        124.48,
        124.67,
        124.84,
        124.99,
        125.14,
        125.27,
        125.4,
        125.52,
        125.63,
        125.74,
        125.84,
        125.93,
        126.02,
        126.11,
        126.19,
        126.27,
        126.34,
        126.41,
        126.48,
        126.55,
        126.61,
        126.67,
        126.72
      ],
      "mode": "lines",
      "name": "95% 置信区间",
      "line": {
        "color": "rgba(255, 255, 255, 0)"
      },
      "fill": "tonexty",
      "fillcolor": "rgba(240, 62, 62, 0.3)",
      "hoverinfo": "skip",
      "showlegend": true
    }
  ]
}
```

</details>



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

---

[上一节](08-%E4%BD%BF%E7%94%A8ARIMA%E6%A8%A1%E5%9E%8B%E8%BF%9B%E8%A1%8C%E9%A2%84%E6%B5%8B.md) · [下一节](../05-%E4%BD%BF%E7%94%A8SARIMA%E5%A4%84%E7%90%86%E5%AD%A3%E8%8A%82%E6%80%A7/01-ARIMA%E6%A8%A1%E5%9E%8B%E5%A4%84%E7%90%86%E5%AD%A3%E8%8A%82%E6%80%A7%E6%95%B0%E6%8D%AE%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7.md)
