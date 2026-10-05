# 解读ACF/PACF以选择模型

来源：[原文](https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-3-autocorrelation-model-id/interpreting-acf-pacf)

[返回章节目录](README.md) · [返回课程目录](../README.md)

自相关函数（ACF）和偏自相关函数（PACF）是时间序列分析中的基本工具，用于理解数据结构并指导模型选择。这些图揭示了表明平稳时间序列数据具有自回归 (autoregressive)（AR）、移动平均（MA）或组合ARMA特征的模式。ACF和PACF的解读通常在确保时间序列平稳之后进行，这通常通过差分来实现。

核心思想是，纯AR和MA过程在其ACF和PACF图上具有明显的理论特征。通过将您数据图中的模式与这些理论特征进行匹配，您可以提出所需模型的阶数假设。

### 阅读图表：显著性边界

在解读模式之前，请回顾ACF/PACF图通常包含表示显著性边界的阴影区域（通常是95%置信区间）。落在此边界之外的值被认为在统计上显著异于零。边界之内的值通常被认为在统计上不显著（与零相关性一致）。我们关注那些相关性峰值*超出*这些边界的滞后阶数。

### 识别AR(p)模型

阶数为$p$的自回归 (autoregressive)模型，记作AR(p)，表明序列的当前值与其前$p$个值呈线性关系。


$$
Y_t = c + \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + \dots + \phi_p Y_{t-p} + \epsilon_t
$$


其中 $\epsilon_t$ 是白噪声。

一个AR(p)过程的理论特征是：

1. **ACF：** 逐渐拖尾。随着滞后阶数的增加，自相关性呈指数衰减或通过阻尼正弦波模式减弱。它不会突然降至零。
2. **PACF：** 在滞后阶数$p$之后急剧截断。偏自相关性在前$p$个滞后阶数上显著，然后对于随后的滞后阶数急剧降至统计上不显著（在边界内）。

因此，如果您观察到：

- ACF图中相关性缓慢衰减。
- PACF图在滞后阶数$p$之前有显著峰值，然后立即落入显著性边界内。

这种模式强烈表明\*\*AR(p)\*\*模型是合适的。阶数$p$由PACF图中最后一个显著滞后阶数决定。



[交互图表：AR(2)过程的理论PACF](https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-3-autocorrelation-model-id/interpreting-acf-pacf#plot-1sfbcve)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "AR(2)过程的理论PACF",
    "xaxis": {
      "title": "滞后阶数"
    },
    "yaxis": {
      "title": "偏自相关性",
      "range": [
        -1,
        1
      ]
    },
    "shapes": [
      {
        "type": "line",
        "x0": 0.5,
        "y0": 0.1,
        "x1": 10.5,
        "y1": 0.1,
        "line": {
          "color": "#a5d8ff",
          "width": 1,
          "dash": "dash"
        }
      },
      {
        "type": "line",
        "x0": 0.5,
        "y0": -0.1,
        "x1": 10.5,
        "y1": -0.1,
        "line": {
          "color": "#a5d8ff",
          "width": 1,
          "dash": "dash"
        }
      }
    ],
    "template": "plotly_white"
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
        10
      ],
      "y": [
        0.7,
        -0.4,
        0.05,
        -0.02,
        0.01,
        -0.005,
        0.002,
        -0.001,
        0.0005,
        -0.0002
      ],
      "name": "PACF",
      "marker": {
        "color": "#228be6"
      }
    },
    {
      "type": "scatter",
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
        10
      ],
      "y": [
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1
      ],
      "mode": "lines",
      "line": {
        "color": "rgba(0,0,0,0)"
      },
      "showlegend": false
    },
    {
      "type": "scatter",
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
        10
      ],
      "y": [
        -0.1,
        -0.1,
        -0.1,
        -0.1,
        -0.1,
        -0.1,
        -0.1,
        -0.1,
        -0.1,
        -0.1
      ],
      "mode": "lines",
      "line": {
        "color": "rgba(0,0,0,0)"
      },
      "fill": "tonexty",
      "fillcolor": "rgba(165, 216, 255, 0.2)",
      "showlegend": false
    }
  ]
}
```

</details>



> AR(2)过程的PACF图示例。注意在滞后阶数1和2处的显著峰值，随后值位于显著性边界内（虚线/阴影区域）。这表明$p=2$。

### 识别MA(q)模型

阶数为$q$的移动平均模型，记作MA(q)，表明序列的当前值与当前及前$q$个白噪声误差项呈线性关系。


$$
Y_t = c + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \dots + \theta_q \epsilon_{t-q}
$$


一个MA(q)过程的理论特征与AR(p)过程本质上相反：

1. **ACF：** 在滞后阶数$q$之后急剧截断。自相关性在前$q$个滞后阶数上显著，然后急剧降至统计上不显著。
2. **PACF：** 逐渐拖尾。随着滞后阶数的增加，偏自相关性减弱。

所以，如果您的图表显示：

- ACF图在滞后阶数$q$之前有显著峰值，然后立即落入显著性边界内。
- PACF图中相关性缓慢衰减。

这种模式强烈表明\*\*MA(q)\*\*模型是合适的。阶数$q$由ACF图中最后一个显著滞后阶数决定。



[交互图表：MA(1)过程的理论ACF](https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-3-autocorrelation-model-id/interpreting-acf-pacf#plot-18zvm2d)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "MA(1)过程的理论ACF",
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
    "shapes": [
      {
        "type": "line",
        "x0": 0.5,
        "y0": 0.1,
        "x1": 10.5,
        "y1": 0.1,
        "line": {
          "color": "#a5d8ff",
          "width": 1,
          "dash": "dash"
        }
      },
      {
        "type": "line",
        "x0": 0.5,
        "y0": -0.1,
        "x1": 10.5,
        "y1": -0.1,
        "line": {
          "color": "#a5d8ff",
          "width": 1,
          "dash": "dash"
        }
      }
    ],
    "template": "plotly_white"
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
        10
      ],
      "y": [
        0.6,
        0.08,
        -0.03,
        0.01,
        0.005,
        -0.002,
        0.001,
        -0.0005,
        0.0002,
        -0.0001
      ],
      "name": "ACF",
      "marker": {
        "color": "#fa5252"
      }
    },
    {
      "type": "scatter",
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
        10
      ],
      "y": [
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1
      ],
      "mode": "lines",
      "line": {
        "color": "rgba(0,0,0,0)"
      },
      "showlegend": false
    },
    {
      "type": "scatter",
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
        10
      ],
      "y": [
        -0.1,
        -0.1,
        -0.1,
        -0.1,
        -0.1,
        -0.1,
        -0.1,
        -0.1,
        -0.1,
        -0.1
      ],
      "mode": "lines",
      "line": {
        "color": "rgba(0,0,0,0)"
      },
      "fill": "tonexty",
      "fillcolor": "rgba(165, 216, 255, 0.2)",
      "showlegend": false
    }
  ]
}
```

</details>



> MA(1)过程的ACF图示例。仅在滞后阶数1处出现显著峰值，随后值为不显著。这表明$q=1$。

### 识别ARMA(p,q)模型

自回归 (autoregressive)移动平均模型ARMA(p,q)结合了AR和MA两部分。


$$
Y_t = c + \phi_1 Y_{t-1} + \dots + \phi_p Y_{t-p} + \epsilon_t + \theta_1 \epsilon_{t-1} + \dots + \theta_q \epsilon_{t-q}
$$


一个ARMA(p,q)过程的理论特征不那么明显：

1. **ACF：** 在滞后阶数$q$之后逐渐拖尾。
2. **PACF：** 在滞后阶数$p$之后逐渐拖尾。

如果ACF和PACF图都显示出逐渐衰减的模式（无论是指数型还是正弦型），这表明\*\*ARMA(p,q)\*\*模型可能适用。

直接从图表中识别具体的阶数$p$和$q$比纯AR或MA模型更具挑战性。“拖尾”行为不会给出精确的截断点。在实际应用中，如果两个图都拖尾，您可能会假设较低阶模型，如ARMA(1,1)、ARMA(2,1)或ARMA(1,2)，然后使用模型选择准则（如AIC或BIC，第六章讨论）和残差分析（第四章）来选择最合适的模型。



[交互图表：ARMA(1,1)过程的理论ACF和PACF](https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-3-autocorrelation-model-id/interpreting-acf-pacf#plot-16h33sl)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "ARMA(1,1)过程的理论ACF和PACF",
    "xaxis": {
      "title": "滞后阶数"
    },
    "yaxis": {
      "title": "相关性",
      "range": [
        -1,
        1
      ]
    },
    "shapes": [
      {
        "type": "line",
        "x0": 0.5,
        "y0": 0.1,
        "x1": 10.5,
        "y1": 0.1,
        "line": {
          "color": "#a5d8ff",
          "width": 1,
          "dash": "dash"
        }
      },
      {
        "type": "line",
        "x0": 0.5,
        "y0": -0.1,
        "x1": 10.5,
        "y1": -0.1,
        "line": {
          "color": "#a5d8ff",
          "width": 1,
          "dash": "dash"
        }
      }
    ],
    "template": "plotly_white",
    "barmode": "group"
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
        10
      ],
      "y": [
        0.7,
        0.5,
        0.35,
        0.25,
        0.18,
        0.13,
        0.09,
        0.06,
        0.04,
        0.03
      ],
      "name": "ACF",
      "marker": {
        "color": "#fa5252"
      }
    },
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
        10
      ],
      "y": [
        0.7,
        -0.3,
        0.15,
        -0.08,
        0.04,
        -0.02,
        0.01,
        -0.005,
        0.002,
        -0.001
      ],
      "name": "PACF",
      "marker": {
        "color": "#228be6"
      }
    },
    {
      "type": "scatter",
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
        10
      ],
      "y": [
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1
      ],
      "mode": "lines",
      "line": {
        "color": "rgba(0,0,0,0)"
      },
      "showlegend": false
    },
    {
      "type": "scatter",
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
        10
      ],
      "y": [
        -0.1,
        -0.1,
        -0.1,
        -0.1,
        -0.1,
        -0.1,
        -0.1,
        -0.1,
        -0.1,
        -0.1
      ],
      "mode": "lines",
      "line": {
        "color": "rgba(0,0,0,0)"
      },
      "fill": "tonexty",
      "fillcolor": "rgba(165, 216, 255, 0.2)",
      "showlegend": false
    }
  ]
}
```

</details>



> ARMA(1,1)过程的ACF和PACF图示例。两个函数都表现出拖尾行为，仅凭这些图难以精确识别阶数。

### 解读指导总结

| 模型 | ACF模式 | PACF模式 |
| --- | --- | --- |
| **AR(p)** | 逐渐拖尾 | 在滞后阶数$p$后截断 |
| **MA(q)** | 在滞后阶数$q$后截断 | 逐渐拖尾 |
| **ARMA(p,q)** | 在滞后阶数$q$后拖尾 | 在滞后阶数$p$后拖尾 |

### 实际考量

- **需要平稳性：** 这些解读适用于*平稳*时间序列。非平稳序列的ACF图通常显示非常缓慢的衰减，这种衰减会覆盖任何其他模式。始终首先确保平稳性。
  "\* **真实数据存在噪声：** 理论模式是清晰的。数据图表可能会显示一些因随机因素而略微超出边界的峰值，尤其是在较高滞后阶数处。请寻找清晰的截断点或明显的拖尾模式。不要过度解读微小偏差。"
- **季节性：** 如果您的数据具有季节性（例如，月度销售数据），您通常会在季节性滞后（例如12、24、36）处看到ACF/PACF中出现显著峰值。这表明需要季节性模型，如SARIMA，我们将在第五章讨论。非季节性阶数（$p, q$）有时仍可从初始非季节性滞后的行为中推断出来。
- **起点：** 使用这些图表作为指导，提出候选模型阶数（$p, q$）。差分阶数$d$来自平稳性分析（第二章）。它们共同构成了尝试ARIMA(p, d, q)模型的基础，我们将在第四章中进一步分析。

通过仔细检查平稳时间序列的ACF和PACF图，您可以对其底层结构获得有价值的理解，为构建有效的ARIMA预测模型提供了坚实的基础。

## 参考资料

- [Time Series Analysis: Forecasting and Control](https://www.vitalsource.com/products/time-series-analysis-forecasting-and-control-george-e-p-box-gwilym-m-jenkins-gregory-c-v9781119070320) — George E.P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, and Greta M. Ljung (2015)
  Publisher: Wiley; Pages: 712; DOI: [10.1002/9781119054710](https://doi.org/10.1002/9781119054710)
  时间序列分析的经典著作，为ARMA模型以及使用ACF/PACF进行模型识别提供了理论背景和实践指导。
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/) — Rob J Hyndman and George Athanasopoulos (2021)
  Publisher: OTexts
  一本广泛使用的易懂的时间序列预测教科书，对ACF和PACF图的解读以进行模型识别提供了清晰的解释和实践示例。
- [Time Series Analysis and Its Applications: With R Examples](https://books.google.com/books/about/Time_Series_Analysis_and_Its_Applications.html?id=z1JDDgAAQBAJ) — Robert H. Shumway, David S. Stoffer (2017)
  Publisher: Springer; DOI: [10.1007/978-3-319-52447-8](https://doi.org/10.1007/978-3-319-52447-8)
  一本综合性教科书，平衡了理论概念和实际应用，包括使用ACF和PACF进行ARMA模型识别的详细讨论。

---

[上一节](03-%E5%9C%A8%20Python%20%E4%B8%AD%E7%BB%98%E5%88%B6%20ACF%20%E5%92%8C%20PACF.md) · [下一节](05-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9AACF-PACF%20%E5%9B%BE%E7%9A%84%E7%BB%98%E5%88%B6%E4%B8%8E%E8%A7%A3%E8%AF%BB.md)
