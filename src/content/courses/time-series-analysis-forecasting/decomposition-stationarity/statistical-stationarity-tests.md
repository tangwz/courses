---
course: "time-series-analysis-forecasting"
chapter: "decomposition-stationarity"
lesson: "statistical-stationarity-tests"
sourceId: 3395
sourceUrl: "https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-2-decomposition-stationarity/statistical-stationarity-tests"
title: "平稳性的统计检验 (ADF 检验)"
description: "介绍并应用增广迪基-富勒 (ADF) 检验以严谨地检验平稳性。"
order: 5
plots: []
sourceHash: "93ba00cd66567810b4dbec8fd24589069dba8b7d0e7e393dfe92f2bb075948f0"
sourceCorrections: []
---

虽然肉眼观察能为我们提供关于平稳性的线索，但这可能带有主观性。细微的趋势或缓慢的均值回归可能难以察觉，我们需要一种更客观、量化 (quantization)的方法来验证我们的观察。统计假设检验提供了这种严谨性。它们使我们能够严谨地检验时间序列是否符合平稳性条件。

平稳性检验中最常用的方法之一是**增广迪基-富勒 (ADF) 检验**。

### 增广迪基-富勒 (ADF) 检验

ADF 检验是一种统计检验，被称为单位根检验。“单位根”的存在是说明时间序列非平稳的一种正式说法。具体来说，这表明冲击或过去的值对序列的未来值有持续影响，这是非平稳过程（如随机游走）的特点。

核心思想是建立一个假设检验：

- **零假设 ($H_0$):** 时间序列*存在*单位根（即非平稳）。
- **备择假设 ($H_a$):** 时间序列*不存在*单位根（即平稳）。

我们的目标通常是找到*反对*零假设的证据。如果我们能拒绝 $H_0$，我们就能更有把握地认为我们的序列是平稳的（或已成功转换为平稳的）。

### 解释 ADF 检验结果

ADF 检验会生成多个输出，但对我们的解释来说，最重要的是**ADF 检验统计量**和**p 值**。

1. **ADF 检验统计量:** 这是一个负数。这个统计量越负，反对零假设的证据就越强（即，平稳性的证据越强）。
2. **p 值:** 这个值表示如果零假设为真，观察到给定检验统计量（或更极端的值）的概率。
   - **如果 p 值很小（通常 $\le 0.05$)**: 我们**拒绝**零假设 ($H_0$)。这表明有强烈证据表明该序列是平稳的。
   - **如果 p 值很大（通常 $> 0.05$)**: 我们**未能拒绝**零假设 ($H_0$)。这表明没有足够的证据断定该序列是平稳的；我们假定它是非平稳的。
3. **临界值:** 检验还提供了在不同显著性水平（例如 1%、5%、10%）下的检验统计量临界值。如果 ADF 检验统计量小于（即更负于）某个显著性水平（例如 5%）的临界值，我们就可以在该水平上拒绝零假设。将检验统计量与这些临界值进行比较，提供了另一种评估假设的方式。

### 在 Python 中实现 ADF 检验

`statsmodels` 库提供了一个 ADF 检验的便捷实现。让我们看看如何使用它。假设你有一个包含时间序列值的 Pandas Series `series_data`：

```python
# 导入 adfuller 函数
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np

# 例子：生成非平稳数据（随机游走）
np.random.seed(42)
random_walk = np.random.randn(500).cumsum()
series_data_non_stationary = pd.Series(random_walk)

# 例子：生成平稳数据（白噪声）
white_noise = np.random.randn(500)
series_data_stationary = pd.Series(white_noise)


# 定义一个函数来执行并打印 ADF 检验结果
def perform_adf_test(series, series_name=""):
    print(f"--- {series_name} 的 ADF 检验结果 ---")
    # 执行 ADF 检验
    result = adfuller(series.dropna()) # dropna() 处理可能因差分产生的 NaN

    # 提取并打印结果
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print('临界值:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.4f}')

    # 解释 p 值
    if result[1] <= 0.05:
        print("\n结论：拒绝零假设 (H0)。数据很可能是平稳的。")
    else:
        print("\n结论：未能拒绝零假设 (H0)。数据很可能是非平稳的。")
    print("-"*(30 + len(series_name))) # 分隔符

# 检验非平稳数据
perform_adf_test(series_data_non_stationary, "随机游走 (非平稳示例)")

print("\n") # 在输出之间添加空格

# 检验平稳数据
perform_adf_test(series_data_stationary, "白噪声 (平稳示例)")
```

**代码输出:**

```text
--- 随机游走 (非平稳示例) 的 ADF 检验结果 ---
ADF 统计量: -1.1272
p 值: 0.6964
临界值:
	1%: -3.4436
	5%: -2.8674
	10%: -2.5699

结论：未能拒绝零假设 (H0)。数据很可能是非平稳的。
-------------------------------------------------------------

--- 白噪声 (平稳示例) 的 ADF 检验结果 ---
ADF 统计量: -22.0378
p 值: 0.0000
临界值:
	1%: -3.4436
	5%: -2.8674
	10%: -2.5699

结论：拒绝零假设 (H0)。数据很可能是平稳的。
--------------------------------------------------------
```

**解释:**

- 对于**随机游走**数据，p 值 (0.6964) 远大于 0.05。此外，ADF 统计量 (-1.1272) 大于（即负值较小）所有临界值。因此，我们未能拒绝零假设，并得出结论：该序列很可能是非平稳的，正如预期。
- 对于**白噪声**数据，p 值非常小（报告为 0.0000），小于 0.05。ADF 统计量 (-22.0378) 远比所有临界值更负。我们强烈拒绝零假设，并得出结论：该序列很可能是平稳的。

ADF 检验是时间序列分析工具集中的一种标准工具。它提供了一种正式方法来检查平稳性假设，这在进行 ARIMA 等依赖于此的模型处理之前是需要的。如果检验表明非平稳性，我们将需要应用变换，例如差分，我们将在下文介绍。

## 参考资料

- [Time Series Analysis](https://press.princeton.edu/books/hardcover/9780691042893/time-series-analysis) — James D. Hamilton (1994)
  Publisher: Princeton University Press
  一本涵盖时间序列分析理论基础的综合学术著作，包括对单位根和ADF检验等平稳性检验的详细讨论。
- [Forecasting: Principles and Practice (3rd edition)](https://otexts.com/fpp3/stationarity.html) — Rob J. Hyndman and George Athanasopoulos (2021)
  Publisher: OTexts
  一本易于理解且实用的时间序列预测在线教科书，清晰解释了平稳性和单位根检验（包括ADF检验），并提供了实际案例。
- [statsmodels.tsa.stattools.adfuller](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html) — statsmodels developers (2024)
  statsmodels Python 库中Augmented Dickey-Fuller检验实现的官方文档，提供了参数、返回值和用法细节。
