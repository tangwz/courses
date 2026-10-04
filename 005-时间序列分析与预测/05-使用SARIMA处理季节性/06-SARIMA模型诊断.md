# SARIMA模型诊断

来源：[原文](https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-5-seasonal-arima-sarima/diagnosing-sarima-models)

[返回章节目录](README.md) · [返回课程目录](../README.md)

拟合SARIMA模型后，过程并未结束。进行诊断检查很重要，以评估所选模型是否充分拟合数据。这一步有助于判断模型假设是否合理满足，以及它是否捕获了时间序列中存在的主要模式，包括季节性。未能诊断模型可能导致预测不可靠和结论错误。

诊断时间序列模型（包括SARIMA）的主要工具是分析模型的*残差*。残差是训练数据中实际观测值与模型预测值之间的差异：

$\text{残差}_t = \text{实际值}_t - \text{预测值}_t$

如果SARIMA模型拟合良好，残差理想情况下应类似*白噪声*。这意味着残差应具有：

1. 均值恒定，围绕零点。
2. 方差随时间保持不变（同方差性）。
3. 没有显著自相关；不同时间步的残差应相互独立。

接下来我们看看常用的诊断方法。

### 绘制残差随时间的变化图

简单的第一步是绘制残差与时间的关系图。此图有助于直观检查模型未能捕获的任何剩余模式、异常值、非恒定方差或趋势。理想情况下，图上应显示点随机分布在零点附近，没有明显结构。



[交互图表：SARIMA模型残差随时间的变化](https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-5-seasonal-arima-sarima/diagnosing-sarima-models#plot-1odi6nc)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "SARIMA模型残差随时间的变化",
    "xaxis": {
      "title": "时间"
    },
    "yaxis": {
      "title": "残差值"
    },
    "template": "plotly_white",
    "legend": {
      "traceorder": "normal"
    }
  },
  "data": [
    {
      "type": "scatter",
      "name": "残差",
      "x": [
        "2022-01-01",
        "2022-01-02",
        "2022-01-03",
        "2022-01-04",
        "2022-01-05",
        "2022-01-06",
        "2022-01-07",
        "2022-01-08",
        "2022-01-09",
        "2022-01-10",
        "2022-01-11",
        "2022-01-12",
        "2022-01-13",
        "2022-01-14",
        "2022-01-15",
        "2022-01-16",
        "2022-01-17",
        "2022-01-18",
        "2022-01-19",
        "2022-01-20",
        "2022-01-21",
        "2022-01-22",
        "2022-01-23",
        "2022-01-24",
        "2022-01-25",
        "2022-01-26",
        "2022-01-27",
        "2022-01-28",
        "2022-01-29",
        "2022-01-30"
      ],
      "y": [
        -0.5,
        0.2,
        1.1,
        -0.8,
        0.4,
        -0.1,
        0.9,
        -1.2,
        0.7,
        0.0,
        -0.3,
        1.5,
        -0.6,
        0.3,
        -0.9,
        1.3,
        -0.4,
        0.1,
        -1.0,
        0.8,
        -0.2,
        0.6,
        -1.4,
        0.5,
        0.0,
        -0.7,
        1.2,
        -0.3,
        0.8,
        -1.1
      ],
      "mode": "lines+markers",
      "line": {
        "color": "#339af0"
      },
      "marker": {
        "color": "#339af0",
        "size": 5
      }
    },
    {
      "type": "scatter",
      "name": "零线",
      "x": [
        "2022-01-01",
        "2022-01-30"
      ],
      "y": [
        0,
        0
      ],
      "mode": "lines",
      "line": {
        "color": "#adb5bd",
        "dash": "dash"
      }
    }
  ]
}
```

</details>



> 残差随时间的变化图。理想情况下，它们应在零线附近随机波动，不显示明显模式或趋势。

如果您观察到明确模式，例如持续的季节性或趋势，则表明模型阶数（非季节性或季节性）可能需要调整。如果残差的方差随时间显著变化（异方差性），则在建模前可能需要考虑数据变换（如对数或Box-Cox变换），或者可能需要更复杂的模型。

### 检查残差的正态性

虽然并非严格要求模型系数有意义，但残差服从正态分布的假设对于构建准确的预测区间通常很重要。两种常用的正态性评估图是直方图和分位数-分位数（QQ）图。

- **直方图：** 此图显示残差值的分布。对于正态分布的残差，直方图应类似以零为中心的钟形。
- **QQ图：** 此图绘制残差分布的分位数与理论正态分布的分位数。如果残差服从正态分布，QQ图中的点应大致落在一条直线对角线上。



[交互图表：交互图表](https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-5-seasonal-arima-sarima/diagnosing-sarima-models#plot-1d626z)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "xaxis1": {
      "title": "残差值",
      "domain": [
        0.0,
        0.45
      ]
    },
    "yaxis1": {
      "title": "频率"
    },
    "xaxis2": {
      "title": "理论分位数",
      "domain": [
        0.55,
        1.0
      ]
    },
    "yaxis2": {
      "title": "样本分位数"
    },
    "showlegend": false
  },
  "data": [
    {
      "type": "histogram",
      "name": "直方图",
      "x": [
        -0.5,
        0.2,
        1.1,
        -0.8,
        0.4,
        -0.1,
        0.9,
        -1.2,
        0.7,
        0.0,
        -0.3,
        1.5,
        -0.6,
        0.3,
        -0.9,
        1.3,
        -0.4,
        0.1,
        -1.0,
        0.8,
        -0.2,
        0.6,
        -1.4,
        0.5,
        0.0,
        -0.7,
        1.2,
        -0.3,
        0.8,
        -1.1,
        0.1,
        -0.4,
        0.5,
        -1.0,
        0.7,
        0.2,
        -0.6,
        1.0,
        -0.9,
        0.3
      ],
      "marker": {
        "color": "#4dabf7"
      },
      "xaxis": "x1",
      "yaxis": "y1"
    },
    {
      "type": "scatter",
      "name": "QQ图",
      "x": [
        -2.15,
        -1.64,
        -1.34,
        -1.15,
        -0.99,
        -0.86,
        -0.74,
        -0.63,
        -0.53,
        -0.43,
        -0.34,
        -0.25,
        -0.17,
        -0.08,
        0.0,
        0.08,
        0.17,
        0.25,
        0.34,
        0.43,
        0.53,
        0.63,
        0.74,
        0.86,
        0.99,
        1.15,
        1.34,
        1.64,
        2.15
      ],
      "y": [
        -1.4,
        -1.2,
        -1.1,
        -1.0,
        -0.9,
        -0.8,
        -0.7,
        -0.6,
        -0.5,
        -0.4,
        -0.3,
        -0.2,
        -0.1,
        0.0,
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.5
      ],
      "mode": "markers",
      "marker": {
        "color": "#4c6ef5"
      },
      "xaxis": "x2",
      "yaxis": "y2"
    },
    {
      "type": "scatter",
      "name": "参考线",
      "x": [
        -2.5,
        2.5
      ],
      "y": [
        -2.5,
        2.5
      ],
      "mode": "lines",
      "line": {
        "color": "#adb5bd",
        "dash": "dash"
      },
      "xaxis": "x2",
      "yaxis": "y2"
    }
  ]
}
```

</details>



> 模型残差的直方图（左）和QQ图（右）。直方图近似于钟形曲线，QQ图中的点接近对角线，表明残差与正态分布相当接近。

偏离正态性，尤其是重尾（QQ图中两端点明显偏离直线），可能表明预测区间不准确。

### 检查残差的自相关性（ACF/PACF）

也许时间序列模型最重要的诊断方法是检查*残差*中的自相关性。如果模型已成功捕获了时间依赖性（包括非季节性和季节性），则残差应不相关。我们使用残差的ACF图进行此检查。

残差ACF图中显著的尖峰，尤其是在较低滞后（如1、2、3...）或在季节性滞后（$m, 2m, 3m, \dots$，其中$m$是季节周期），表明模型尚未完全捕获相关结构。



[交互图表：SARIMA残差的ACF图](https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-5-seasonal-arima-sarima/diagnosing-sarima-models#plot-17i5ell)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "SARIMA残差的ACF图",
    "xaxis": {
      "title": "滞后阶数"
    },
    "yaxis": {
      "title": "自相关性",
      "range": [
        -1,
        1
      ]
    },
    "template": "plotly_white"
  },
  "data": [
    {
      "type": "bar",
      "name": "ACF",
      "x": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24
      ],
      "y": [
        0.08,
        -0.05,
        0.12,
        0.03,
        -0.09,
        0.06,
        -0.02,
        0.15,
        0.01,
        -0.07,
        0.04,
        -0.11,
        0.09,
        0.0,
        -0.06,
        0.13,
        -0.03,
        0.05,
        -0.1,
        0.07,
        -0.01,
        0.1,
        -0.04,
        0.08
      ],
      "marker": {
        "color": "#1c7ed6"
      }
    },
    {
      "type": "scatter",
      "name": "置信区间",
      "x": [
        0.5,
        24.5,
        24.5,
        0.5,
        0.5
      ],
      "y": [
        0.2,
        0.2,
        -0.2,
        -0.2,
        0.2
      ],
      "mode": "lines",
      "line": {
        "color": "#adb5bd",
        "dash": "dash"
      },
      "fill": "toself",
      "fillcolor": "rgba(173, 181, 189, 0.1)"
    }
  ]
}
```

</details>



> 残差的ACF图。大多数尖峰都在置信区间内（虚线），表明没有显著自相关性残留。明显超出范围的尖峰可能表明存在问题。对于具有月度季节性（$m=12$）的SARIMA模型，要特别注意滞后12和24。

如果发现显著自相关性：

- 在非季节性滞后：考虑调整非季节性阶数（$p$或$q$）。例如，滞后1处的尖峰可能表明增加$q$或$p$。
- 在季节性滞后：考虑调整季节性阶数（$P$或$Q$）。滞后$m$处的尖峰可能表明增加$P$或$Q$。

### 正式统计检验：Ljung-Box检验

除了视觉检查，正式统计检验可以检查残差中的自相关性。Ljung-Box检验常用。它检验残差的前$k$个自相关系数联合为零的原假设。

$H_0$: 残差相互独立（无自相关）。
$H_a$: 残差表现出自相关性。

该检验产生一个统计量和一个p值。如果p值很小（通常小于0.05这样的显著性水平），我们拒绝原假设，认定残差中仍存在显著自相关性，表明模型可能存在设定错误。

在Python的`statsmodels`库中，已拟合SARIMA模型的`summary()`方法通常包含Ljung-Box检验结果（常报告为“Prob(Q)”）。寻找大于0.05的p值，以支持残差独立的说法。

### 使用模型摘要输出

`statsmodels`等库在拟合SARIMA模型后提供的`summary()`输出是诊断信息的丰富来源。在Ljung-Box检验之后，它通常包含：

- **系数估计和P值：** 检查估计系数（AR、MA、季节性AR、季节性MA）是否具有统计学意义（p值 < 0.05）。不显著的系数可能表明可以通过移除相应项（降低阶数）来简化模型。
- **标准误差：** 给出系数估计不确定性的一个大概判断。
- **对数似然、AIC、BIC：** 可用于比较不同候选SARIMA模型的信息准则（通常倾向于选择较低值），也是模型整体拟合情况的指标。
- **残差汇总统计：** 通常包括残差的标准差或方差等度量。

仔细审查此汇总表，为视觉诊断提供了量化 (quantization)补充。

通过系统地应用这些诊断检查——检查残差图、测试正态性、检查残差的ACF/PACF，以及解释Ljung-Box等正式检验——您可以对您拟合的SARIMA模型更有信心。如果诊断结果显示问题，请重新查看模型识别和阶数选择步骤（第3章和第5章），以调整模型结构，直到残差接近白噪声。只有在进行了全面诊断之后，才能继续使用模型进行预测。

## 参考资料

- [Time Series Analysis: Forecasting and Control](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHz3ZpcZVP-j_K6Wtsg7OUE2d6QRbrdrFWRbUnzzbgXvRSU-fTs4rQItKBazsl14HFD_Dm4R_rr_gdAJJ88sYWYmoLvP7f25uoSLhoRByRBbY58uTJ-15o8dVSs0Scplk8caTw9hpR0j-WnU_EyPQtmWok7bsUivTFyILCYuUhiPchfjRJVfQKpY-fCMvgjQgom-fiN9ua8bFH62WYWKuIqO3w9A==) — George E. P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, Greta M. Ljung (2015)
  Publisher: Wiley; Pages: 712
  一本关于ARIMA/SARIMA模型的权威教科书，全面涵盖模型诊断、残差分析和白噪声检验。
- [Forecasting: Principles and Practice (3rd edition)](https://otexts.com/fpp3/) — Rob J Hyndman and George Athanasopoulos (2021)
  Publisher: OTexts
  这本免费在线书籍提供了时间序列预测的实用指南，包含SARIMA诊断、残差图和形式化检验的专门章节。
- [\`statsmodels.tsa.statespace.sarimax.SARIMAXResults.plot_diagnostics\`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.plot_diagnostics.html) — `statsmodels` developers (2024)
  官方文档，详细说明了如何使用和解释`statsmodels` SARIMA模型结果生成的诊断图（残差图、直方图、QQ图、ACF图）。
- [On a Measure of Lack of Fit in Time Series Models](https://doi.org/10.1093/biomet/65.2.297) — Greta M. Ljung, George E. P. Box (1978)
  Journal: Biometrika; Volume: 65; Pages: 297-303; DOI: [10.1093/biomet/65.2.297](https://doi.org/10.1093/biomet/65.2.297)
  引入Ljung-Box检验的原始研究论文，这是一种用于评估时间序列残差中是否存在自相关的形式统计检验。

---

[上一节](05-%E5%9C%A8%20Python%20%E4%B8%AD%E6%8B%9F%E5%90%88%20SARIMA%20%E6%A8%A1%E5%9E%8B.md) · [下一节](07-%E4%BD%BF%E7%94%A8%20SARIMA%20%E8%BF%9B%E8%A1%8C%E9%A2%84%E6%B5%8B.md)
