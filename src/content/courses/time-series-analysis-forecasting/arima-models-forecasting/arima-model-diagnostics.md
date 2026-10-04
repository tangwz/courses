---
course: "time-series-analysis-forecasting"
chapter: "arima-models-forecasting"
lesson: "arima-model-diagnostics"
sourceId: 3436
sourceUrl: "https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-4-arima-models-forecasting/arima-model-diagnostics"
title: "模型诊断与残差分析"
description: "通过检查残差（例如，正态性、自相关性）来验证拟合的 ARIMA 模型的假设。"
order: 7
plots: []
sourceHash: "ae54c615fd6f4e2eb310d36d43f312c337112d5aa8084b5c3b83e469c2276f0b"
sourceCorrections: []
---

将 ARIMA 模型拟合到时间序列数据，通常使用 `statsmodels`，是预测工作中的一个重大步骤。然而，在生成预测结果之前，有必要停下来评估模型实际拟合数据的效果。拟合模型相对容易；但要确保它是*好*模型，则需要进行一些诊断工作。残差分析可满足这一需求。

### 残差的含义

残差就是观测值与您的模型对训练数据预测值之间的差异。对于时间 $t$，残差 $e_t$ 的计算方式如下：


$$
e_t = y_t - \hat{y}_t
$$


其中 $y_t$ 是时间 $t$ 时的实际观测值，而 $\hat{y}_t$ 是您的 ARIMA 模型在时间 $t$ 拟合的值。

可以将残差视为“剩余部分”——模型无法解释的数据部分。如果您的 ARIMA 模型已有效地反映了内部规律（自回归 (autoregressive)和移动平均分量，并通过差分处理了非平稳性），那么残差理想情况下应表现得像随机噪声。具体来说，它们应类似于**白噪声**。

### 良好残差的特征

一个拟合得好的 ARIMA 模型应产生满足以下条件的残差：

1. **零均值：** 残差的平均值应接近零。如果均值明显不同于零，则表明预测结果存在偏差。
2. **方差恒定（同方差性）：** 残差的分布应随时间保持一致。如果方差发生变化（例如，残差随时间变大），则表明存在异方差性，这可能需要对数据进行变换（如取对数）或使用更高级的模型（如 GARCH）。
3. **无自相关：** 残差在不同滞后处不应相互关联。残差中存在显著的自相关，表明数据中仍存在模型尚未捕获的预测模式。这通常意味着 AR ($p$) 或 MA ($q$) 阶数可能需要调整。
4. **正态性（通常是期望的）：** 理想情况下，残差应呈正态分布。虽然 ARIMA 对于点预测不严格要求残差呈正态分布，但它是构建精确预测区间和执行某些统计检验所需的假设。明显偏离正态性可能表明存在异常值或模型构建不完全正确。

### 在 Python 中进行诊断检查

`statsmodels` 库使进行这些诊断检查变得简单易行。当您拟合 ARIMA 模型时，结果对象包含有用的信息和诊断方法。

假设您已拟合了这样的模型：

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

# 假设 'data' 是您的 pandas Series（平稳或已差分）
# 假设 'p', 'd', 'q' 是您选择的阶数
# 示例拟合过程（请替换为您实际的数据和阶数）
# model = ARIMA(your_endog_data, order=(p, d, q)) # 如果 d > 0 则使用
# model_fit = model.fit()

# 让我们创建一些虚拟结果以进行演示
# 请替换为您实际的模型拟合
np.random.seed(42)
dummy_data = sm.tsa.arma_generate_sample(ar=[0.75], ma=[0], nsample=200)
dummy_series = pd.Series(dummy_data)
model = ARIMA(dummy_series, order=(1, 0, 0)) # 拟合一个 AR(1)
model_fit = model.fit()

# model_fit 对象现在包含结果和诊断工具
print(model_fit.summary())
```

`model_fit.summary()` 输出本身提供了一些初步的诊断信息，通常包含：

- 对数似然 (Log Likelihood), AIC, BIC: 用于比较模型很有帮助（更多内容会在第 6 章讨论）。
- 系数的标准误差和 p 值：表明 AR/MA 项在统计上是否具有重要性。
- Ljung-Box (L1) (Q)：一个用于检验残差在指定滞后处（通常是滞后 1）是否存在自相关的测试。`Prob(Q)` 值（p 值）理想情况下应大于 0.05，表明在该滞后处无重要自相关。
- Jarque-Bera (JB)：一个用于检验正态性的测试。`Prob(JB)` 值（p 值）理想情况下应大于 0.05，支持残差呈正态分布的原假设。

虽然摘要很有用，但视觉诊断通常提供更多信息。`statsmodels` 提供了一个实用的方法，名为 `plot_diagnostics`。

```python
import matplotlib.pyplot as plt

# 生成标准诊断图
fig = model_fit.plot_diagnostics(figsize=(12, 8))
plt.tight_layout() # 调整布局以防止重叠
plt.show()
```

该命令通常会生成一个 2x2 的图表网格：

1. **标准化残差图：** 显示随时间变化的残差图。观察：

   - 它们是否围绕零居中？
   - 方差是否恒定，或者是否呈扇形散开或收窄？
   - 是否存在明显模式或异常值？理想情况下，它应看起来像围绕零的随机散点。
2. **直方图加估计密度：** 显示残差分布与正态分布的比较（通常带有 KDE - 核密度估计）。观察：

   - 直方图是否类似钟形？
   - KDE 线（通常是橙色或蓝色）是否紧密贴合标准正态分布线（N(0,1)，通常是灰色或黑色）？摘要中报告的 Jarque-Bera 统计量量化 (quantization)了这一点。
3. **正态 Q-Q 图：** 将残差分布的分位数与标准正态分布的分位数进行比较。观察：

   - 点是否大致落在对角参考线（通常是红色）上？系统性偏差（如 S 形或弯曲）表明偏离正态性。异常值将表现为远离线的点。
4. **相关图（ACF 图）：** 显示残差的自相关函数（ACF）。观察：

   - 大多数相关尖峰是否在阴影置信区间内（通常是蓝色）？一个好的模型应具有无重要自相关性的残差（滞后 0 除外，它总是 1）。特定滞后处的重要尖峰可能表示需要调整 AR 或 MA 阶数，如果尖峰出现在季节性滞后处，则可能需要考虑季节性。Ljung-Box 检验检查多个滞后处相关性的整体重要性。

### 结果解读

- **残差未围绕零居中：** 这在包含常数/截距的标准 ARIMA 拟合过程中很少见，但如果常数被错误地省略，则可能表明存在问题。
- **方差变化（异方差性）：** 如果方差变化是主要关注点，请考虑在建模前对原始数据进行变换（例如 Box-Cox 或对数变换），或考虑使用 ARCH/GARCH 模型。
- **残差中存在重要自相关：** 这是一个常见问题。这意味着您的模型未能完全捕获所有线性依赖性。
  - 如果残差的 ACF/PACF 中在滞后 `k` 处存在重要尖峰，请尝试增加与该滞后对应的 AR 或 MA 阶数（例如，添加 AR(k) 或 MA(k) 项）。
  - 如果存在表明季节性的模式（在滞后 `m`、`2m` 等处出现尖峰），您可能需要一个 SARIMA 模型（下一章将涵盖）。
- **非正态性：**
  - 检查原始数据或残差中是否存在异常值。有时少数极端点会扭曲正态性。
  - 虽然并非总是对预测结果具有决定性作用，但严重的非正态性可能会轻微影响预测区间的精确度。变换可能有所帮助，但请评估改善正态性是否超过潜在的解释难题。

模型诊断是一个反复过程。您拟合模型，检查残差，如果它们显示出问题，则调整模型（例如，更改阶数 $p, d, q$，应用变换），并重复该过程，直到残差看起来足够随机且无结构。只有到那时，您才能对模型生成的预测结果有信心。

## 参考资料

- [Forecasting: Principles and Practice](https://otexts.com/fpp3/) — Rob J Hyndman and George Athanasopoulos (2021)
  Publisher: OTexts
  为时间序列预测提供了全面且易于理解的指南，包括对ARIMA模型构建、残差诊断和解释的详细讨论。
- [statsmodels.tsa.arima.model.ARIMAResults](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMAResults.html) — statsmodels Developers (2023)
  `statsmodels`中`ARIMAResults`对象的官方文档，对理解Python中用于模型诊断的方法和属性至关重要。
- [Time Series Analysis: Forecasting and Control](https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118675021) — George E. P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, and Greta M. Ljung (2015)
  Publisher: John Wiley & Sons; Pages: 712
  时间序列分析的基础教材，提供了ARIMA模型的理论深度、模型识别、估计以及对残差的严格诊断检查。
- [Time Series Analysis and Its Applications: With R Examples](https://books.google.com/books?id=T4QkDwAAQBAJ) — Robert H. Shumway, David S. Stoffer (2017)
  Publisher: Springer; Pages: 562
  对时间序列分析进行了详细且统计学上严谨的处理，包括对ARIMA模型以及残差的统计特性和诊断检查的深度介绍。
