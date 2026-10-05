# 识别季节性成分 (ACF/PACF)

来源：[原文](https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-5-seasonal-arima-sarima/identifying-seasonal-components)

[返回章节目录](README.md) · [返回课程目录](../README.md)

自相关函数（ACF）和偏自相关函数（PACF）图是识别SARIMA模型季节性成分（$P, Q$）可能阶数的有效工具。这些图同样可以帮助确定ARIMA模型中的非季节性成分（$p, q$）。在分析季节性成分时，一个主要的区别在于需要关注哪些滞后。

### 关注季节性滞后

处理季节性数据时，我们关注的是与季节频率对应的滞后处的关联性。如果时间序列有一个季节周期$m$，这意味着相隔$m$个时间步长的观测值之间存在关联。例如：

- **月度数据** 通常有 $m=12$（例如，一月销售额与之前的一月销售额相关）。
- **季度数据** 通常有 $m=4$。
- **日度数据** 可能有 $m=7$（工作日模式）。

季节周期$m$通常根据数据性质和对时间序列图的目视检查来确定，时间序列图通常会显示固定间隔的重复模式。

为了识别季节性阶数$P$和$Q$，我们专门检查ACF和PACF图中季节周期的倍数滞后处：$m, 2m, 3m, \dots$

### 解释ACF/PACF中的季节性模式

解释规则与非季节性成分的规则相似，但应用于季节性滞后：

1. **季节性自回归 (autoregressive)（AR）阶数（P）：**

   - 查看**PACF**图在季节性滞后处（$m, 2m, \dots$）。
   - PACF在滞后$P \times m$后突然截断，同时ACF在季节性滞后处呈现较慢的、通常是正弦或指数型衰减，这表明存在一个$P$阶季节性AR过程。例如，如果$m=12$且PACF仅在滞后12处有显著尖峰，而在24、36等处没有，则表明$P=1$。
2. **季节性移动平均（MA）阶数（Q）：**

   - 查看**ACF**图在季节性滞后处（$m, 2m, \dots$）。
   - ACF在滞后$Q \times m$后突然截断，同时PACF在季节性滞后处呈现较慢的衰减，这表明存在一个$Q$阶季节性MA过程。例如，如果$m=12$且ACF仅在滞后12处有显著尖峰，但在之后截断，则表明$Q=1$。
3. **季节性差分（D）：**

   - ACF中季节性滞后处（例如，滞后$m, 2m, 3m$处的值较高）存在强而持续的正相关性，通常表明需要进行季节性差分（$D > 0$）。如果您怀疑存在季节性非平稳性，应应用季节性差分（从$m$个周期前的观测值中减去当前观测值，即$y'_t = y_t - y_{t-m}$），然后重新检查*差分后*序列的ACF/PACF图以确定$P$和$Q$。如果季节性随着时间推移在水平或振幅上发生明显变化，即使在查看ACF/PACF图之前，通常也能清楚地看出需要季节性差分。

### 示例：带有季节性的ACF/PACF图

假设我们有月度数据（$m=12$），我们怀疑它需要季节性差分（$D=1$）。应用此差分后，我们生成差分序列的ACF和PACF图。



[交互图表：季节性差分数据 (m=12) 的 ACF](https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-5-seasonal-arima-sarima/identifying-seasonal-components#plot-1m3mud4)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "季节性差分数据 (m=12) 的 ACF",
    "xaxis": {
      "title": "滞后"
    },
    "yaxis": {
      "title": "自相关",
      "range": [
        -0.6,
        1.1
      ]
    },
    "showlegend": false,
    "width": 600,
    "height": 350
  },
  "data": [
    {
      "type": "bar",
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
        0.3,
        -0.1,
        0.1,
        0.05,
        -0.02,
        0.15,
        0.08,
        -0.05,
        0.1,
        0.0,
        -0.1,
        -0.4,
        -0.1,
        0.05,
        -0.02,
        0.03,
        0.0,
        -0.08,
        0.02,
        0.0,
        0.01,
        0.05,
        -0.03,
        0.15
      ],
      "marker": {
        "color": "#228be6"
      }
    },
    {
      "type": "scatter",
      "mode": "lines",
      "x": [
        0,
        25
      ],
      "y": [
        0.15,
        0.15
      ],
      "line": {
        "dash": "dash",
        "color": "#adb5bd"
      }
    },
    {
      "type": "scatter",
      "mode": "lines",
      "x": [
        0,
        25
      ],
      "y": [
        -0.15,
        -0.15
      ],
      "line": {
        "dash": "dash",
        "color": "#adb5bd"
      }
    }
  ]
}
```

</details>



> ACF图显示在滞后12（$m=12$）处有一个显著的负向尖峰，而在滞后24处的值较小且不显著。非季节性滞后（例如，滞后1）也可能显著。虚线表示置信区间。



[交互图表：季节性差分数据 (m=12) 的 PACF](https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-5-seasonal-arima-sarima/identifying-seasonal-components#plot-1mdp2s3)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "季节性差分数据 (m=12) 的 PACF",
    "xaxis": {
      "title": "滞后"
    },
    "yaxis": {
      "title": "偏自相关",
      "range": [
        -0.6,
        1.1
      ]
    },
    "showlegend": false,
    "width": 600,
    "height": 350
  },
  "data": [
    {
      "type": "bar",
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
        0.3,
        -0.15,
        0.12,
        0.08,
        -0.05,
        0.16,
        0.06,
        -0.07,
        0.11,
        0.02,
        -0.12,
        -0.35,
        0.05,
        -0.03,
        0.04,
        0.01,
        0.02,
        -0.06,
        0.01,
        0.03,
        -0.02,
        0.04,
        -0.04,
        -0.08
      ],
      "marker": {
        "color": "#228be6"
      }
    },
    {
      "type": "scatter",
      "mode": "lines",
      "x": [
        0,
        25
      ],
      "y": [
        0.15,
        0.15
      ],
      "line": {
        "dash": "dash",
        "color": "#adb5bd"
      }
    },
    {
      "type": "scatter",
      "mode": "lines",
      "x": [
        0,
        25
      ],
      "y": [
        -0.15,
        -0.15
      ],
      "line": {
        "dash": "dash",
        "color": "#adb5bd"
      }
    }
  ]
}
```

</details>



> PACF图显示在滞后12处有一个显著的负向尖峰，并且在更远的季节性滞后处（滞后24较小）值逐渐衰减。非季节性滞后也可能显示显著性。

**示例图的解释：**

- **ACF：** 滞后12处的显著尖峰立即截断（滞后24不显著），这有力地表明存在一个$Q=1$阶的季节性MA成分。
- **PACF：** 滞后12处的尖峰显著，但滞后24处的尖峰小得多且可能不显著，这表明衰减。此模式与季节性MA(1)过程一致。
- **非季节性：** 我们还在非季节性滞后处（例如，ACF和PACF中的滞后1）观察到显著性。这些将用于确定非季节性阶数$p$和$q$，就像第3章中一样。

根据这些针对季节性差分数据（意味着$D=1$）的图，一个可能的季节性阶数可以是$(P=0, D=1, Q=1)_{12}$。非季节性阶数 $(p, d, q)$ 将通过检查初始滞后（1, 2, 3, ...）处的尖峰来确定。

请记住，这些图提供指导，而非明确答案。通常，您可能需要基于对ACF/PACF图稍有不同的理解，尝试几种候选模型，尤其是在模式不完全清晰时。下一节将讨论结合这些观察结果的方式，以选择完整的$SARIMA(p, d, q)(P, D, Q)_m$阶数。

## 参考资料

- [Time Series Analysis: Forecasting and Control](https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118675021) — George E. P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, Greta M. Ljung (2015)
  Publisher: John Wiley & Sons
  这是Box-Jenkins方法的权威参考文献，为识别ARIMA和季节性ARIMA（SARIMA）模型组件提供了基础理解，包括对ACF和PACF图季节性模式的解释。
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/) — Rob J Hyndman and George Athanasopoulos (2021)
  Publisher: OTexts
  提供实用且易于理解的时间序列预测指南，清晰地解释并举例说明如何使用ACF和PACF图来识别SARIMA模型的季节性和非季节性成分。
- [Introduction to Time Series and Forecasting](https://doi.org/10.1007/978-3-319-29318-8) — Peter J. Brockwell, Richard A. Davis (2016)
  Publisher: Springer; DOI: [10.1007/978-3-319-29318-8](https://doi.org/10.1007/978-3-319-29318-8)
  一本严谨而易懂的教科书，涵盖了时间序列分析的理论基础和实际应用，包括对ARIMA和季节性模型及其通过ACF/PACF进行识别的详细讨论。

---

[上一节](02-%E5%AD%A3%E8%8A%82%E6%80%A7ARIMA%EF%BC%88SARIMA%EF%BC%89%E6%A8%A1%E5%9E%8B%E4%BB%8B%E7%BB%8D.md) · [下一节](04-%E9%80%89%E6%8B%A9%20SARIMA%20%E9%98%B6%E6%95%B0%20%28p%2C%20d%2C%20q%29%28P%2C%20D%2C%20Q%29m.md)
