---
course: "time-series-analysis-forecasting"
chapter: "autocorrelation-model-id"
lesson: "plotting-acf-pacf-python"
sourceId: 3408
sourceUrl: "https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-3-autocorrelation-model-id/plotting-acf-pacf-python"
title: "在 Python 中绘制 ACF 和 PACF"
description: "使用 Python 的 statsmodels 库生成并理解 ACF 和 PACF 图。"
order: 3
plots: ["plots/3408-0.json", "plots/3408-1.json"]
sourceHash: "f19f3cd644ca25a98d85a62ffb92c535094d49d296eaee227186447021b9a88c"
sourceCorrections: []
---

自相关函数 (ACF) 和偏自相关函数 (PACF) 的生成和显示在 Python 中是时间序列分析的一个主要步骤。这些图是直观查看时间序列数据相关结构的基本工具，有助于识别潜在模型参数 (parameter)。为此，我们将主要使用 `statsmodels` 库。

请记住，ACF 和 PACF 分析通常在**平稳**的时间序列数据上进行。如果你的数据表现出趋势或季节性，你应该在生成这些图*之前*，应用差分等转换（如第 2 章所讨论的）来达到平稳。在非平稳数据上操作可能导致误导性的 ACF/PACF 图，因为潜在的相关性会被趋势或季节性掩盖。

### 生成自相关函数 (ACF) 图

`statsmodels` 库在其 `graphics.tsaplots` 模块中提供了一个方便的函数 `plot_acf`，用于计算和绘制 ACF。

假设你有一个存储在名为 `stationary_series` 的 Pandas Series 对象中的平稳时间序列。你可以这样生成 ACF 图：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_process import ArmaProcess # 用于生成样本数据

# --- 生成样本平稳数据 (例如，AR(1) 过程) ---
# 这模拟了你可能在差分后得到的数据
np.random.seed(42)
ar_coeffs = np.array([1, -0.7]) # AR 参数 = 0.7
ma_coeffs = np.array([1])      # 没有 MA 部分
ar_process = ArmaProcess(ar_coeffs, ma_coeffs)
sample_data = ar_process.generate_sample(nsample=500)
stationary_series = pd.Series(sample_data)
# ------------------------------------------------------------

# 绘制 ACF
fig, ax = plt.subplots(figsize=(10, 5))
plot_acf(stationary_series, lags=20, ax=ax) # 绘制前 20 个滞后
ax.set_xlabel("滞后")
ax.set_ylabel("自相关性")
ax.set_title("自相关函数 (ACF)")
plt.show() # 显示图表
```

这段代码首先生成一些样本平稳数据（模拟一个 AR(1) 过程用于演示）。核心部分是对 `plot_acf(stationary_series, lags=20, ax=ax)` 的调用。

- 第一个参数 (parameter)是你的平稳时间序列数据（一个 Pandas Series 或 NumPy 数组）。
- `lags` 参数指定了要计算和显示的自相关滞后数量。选择合适的数量取决于你的数据频率和预期相关长度，但 20-40 个滞后通常是一个合理的起点。
- 我们传入一个 Matplotlib `Axes` 对象 (`ax`) 以更好地控制图表外观，尽管 `plot_acf` 在需要时可以生成自己的图。

生成的图表通常如下所示：



![自相关函数 (ACF) 示例](plots/3408-0.json)



> ACF 图显示了 y 轴上的相关系数，对应 x 轴上的不同时间滞后。蓝色的阴影区域表示置信区间（通常为 95%）。超出此范围的相关性被认为具有统计显著性。请注意，滞后为 0 时的 ACF 总是 1，因为任何序列都与自身完全相关。此示例显示了 AR 过程的典型模式：相关性呈指数衰减趋近于零。

### 生成偏自相关函数 (PACF) 图

类似地，`statsmodels` 提供了 `plot_pacf` 函数来计算和绘制 PACF。请记住，PACF 衡量的是序列与其滞后 $k$ 之间的相关性，同时去除了中间滞后 ($1, 2, ..., k-1$) 的线性影响。

用法与 `plot_acf` 类似：

```python
# 假设 'stationary_series' 包含你的平稳数据
from statsmodels.graphics.tsaplots import plot_pacf

# 绘制 PACF
fig, ax = plt.subplots(figsize=(10, 5))
plot_pacf(stationary_series, lags=20, ax=ax, method='ywm') # 绘制前 20 个滞后
ax.set_xlabel("滞后")
ax.set_ylabel("偏自相关性")
ax.set_title("偏自相关函数 (PACF)")
plt.show() # 显示图表
```

我们再次使用 `stationary_series`。`lags` 参数 (parameter)的工作方式相同。`method` 参数指定了 PACF 的估计方法；'ywm' (Yule-Walker Modified) 是一种常用且通常是默认的方法。

生成的图表可以帮助我们理解观测值与其滞后之间的直接关系，同时排除了间接相关性：



![偏自相关函数 (PACF) 示例](plots/3408-1.json)



> PACF 图的结构与 ACF 图类似，x 轴为滞后，y 轴为偏相关性，并有一个置信区间带。同样，超出范围的值具有统计显著性。这个 PACF 示例图显示在滞后 1 之后有一个明显的截断，这是 AR(1) 过程的典型特征。所有滞后大于 1 的偏自相关性都接近于零并落在置信区间内。

### 初步解读

生成这些图是第一步。真正的价值在于解读显著峰值的*模式*：

- **自回归 (autoregressive) (AR) 过程**通常显示 ACF 图中的相关性逐渐（通常呈指数或正弦曲线）衰减至零，而 PACF 图在某个滞后 $p$ 之后会急剧截断。PACF 截断的滞后数指示了 AR($p$) 模型的阶数 $p$。
- **移动平均 (MA) 过程**通常显示相反的情况：ACF 图在某个滞后 $q$ 之后会急剧截断，而 PACF 图中的相关性逐渐衰减至零。ACF 截断的滞后数指示了 MA($q$) 模型的阶数 $q$。
- **自回归移动平均 (ARMA) 过程**在 ACF 和 PACF 图中都表现出逐渐衰减。

上面生成的图表，基于模拟的 AR(1) 数据 ($y_t = 0.7 y_{t-1} + \epsilon_t$)，清晰地显示了这种 AR 模式：ACF 呈指数衰减，而 PACF 仅在滞后 1 处有显著峰值，然后截断（落在置信区间内）。

通过生成和检查这些平稳时间序列的图表，你可以获得关于数据生成过程的重要线索。这种视觉检查，结合稍后讨论的定量测量，有助于指导 ARIMA 模型中 $p$ 和 $q$ 阶数的选择，我们将在下一章中介绍。下一节将详细说明如何更系统地利用这些模式进行模型识别。

## 参考资料

- [Time Series Analysis: Forecasting and Control](https://www.wiley.com/en-us/Time+Series+Analysis:+Forecasting+and+Control,+5th+Edition-p-9781118675021) — George E. P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, Greta M. Ljung (2015)
  Publisher: Wiley
  一本基础性著作，全面讲解ARIMA模型的理论与实践，包括用于模型识别的自相关和偏自相关函数的详细解释。
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/) — Rob J Hyndman, George Athanasopoulos (2021)
  Publisher: OTexts
  一本易于理解的在线教科书，涵盖时间序列分析的基本概念，包括平稳性、ACF和PACF，并附有实际示例和清晰的解释。
- [statsmodels.graphics.tsaplots.plot_acf Documentation](https://www.statsmodels.org/stable/generated/statsmodels.graphics.tsaplots.plot_acf.html) — statsmodels Developers (2024)
  Python `statsmodels`库中`plot_acf`函数的官方文档，详细说明了其用于生成自相关图的用法。`plot_pacf`函数用法类似。
