# 早期方法：高斯混合模型 (GMMs)

来源：[原文](https://apxml.com/zh/courses/introduction-to-speech-recognition/chapter-3-acoustic-modeling/gmm-acoustic-models)

[返回章节目录](README.md) · [返回课程目录](../README.md)

收集了数千个特定音素（如/t/）的例子，并提取出它们的MFCC特征向量 (vector)后，你会发现所有这些向量并非完全相同。相反，你会看到一团数据点。这种差异源于音高、语速、口音以及相邻音的影响。声学模型的任务是为每个音素的这团点创建数学描述。

一种简单的方法是使用单个**高斯分布**（常称为正态分布或钟形曲线）来建模这团点。高斯分布描述了围绕平均值或均值聚集的数据。虽然有用，但单个高斯分布通常过于死板，无法捕获音素发音方式的复杂变化。例如，单词“top”开头的/t/音可能产生与“stop”中的/t/音略有不同的特征，从而在数据团中形成多个簇。

### 建模特征变化

一种更强大且灵活的方法是使用**高斯混合模型 (GMM)**。GMM不只是一种高斯分布，而是多种高斯分布的组合。可以把它想象成使用多个较小的钟形曲线来近似更复杂的形状。通过组合它们，GMM可以有效地对具有多个峰值或不对称的分布进行建模。

混合模型中的每个高斯分布称为一个“分量”，它有自己的均值和方差。GMM还包括每个分量的“权重 (weight)”，这表示该特定高斯分布对整体模型的贡献程度。这使得GMM能够比单个高斯分布更准确地拟合单个音素特征向量 (vector)的复杂分布。

下图说明了具有三个分量的GMM如何对分组为不同簇的特征向量散点图进行建模。



[交互图表：GMM对特征向量簇进行建模](https://apxml.com/zh/courses/introduction-to-speech-recognition/chapter-3-acoustic-modeling/gmm-acoustic-models#plot-w0wiyt)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": {
      "text": "GMM对特征向量簇进行建模"
    },
    "xaxis": {
      "title": "特征1 (例如, MFCC_1)",
      "zeroline": false
    },
    "yaxis": {
      "title": "特征2 (例如, MFCC_2)",
      "zeroline": false
    },
    "shapes": [
      {
        "type": "circle",
        "xref": "x",
        "yref": "y",
        "x0": -2.7,
        "y0": -2.7,
        "x1": 2.7,
        "y1": 2.7,
        "opacity": 0.2,
        "fillcolor": "#748ffc",
        "line": {
          "color": "#4c6ef5"
        }
      },
      {
        "type": "circle",
        "xref": "x",
        "yref": "y",
        "x0": 2.3,
        "y0": -3.7,
        "x1": 7.7,
        "y1": 1.7,
        "opacity": 0.2,
        "fillcolor": "#69db7c",
        "line": {
          "color": "#40c057"
        }
      },
      {
        "type": "circle",
        "xref": "x",
        "yref": "y",
        "x0": 1.3,
        "y0": 1.8,
        "x1": 6.7,
        "y1": 7.2,
        "opacity": 0.2,
        "fillcolor": "#ff8787",
        "line": {
          "color": "#fa5252"
        }
      }
    ],
    "showlegend": false,
    "plot_bgcolor": "#e9ecef",
    "paper_bgcolor": "white",
    "margin": {
      "t": 40,
      "b": 40,
      "l": 40,
      "r": 20
    }
  },
  "data": [
    {
      "x": [
        -0.5,
        0.2,
        -1.1,
        0.8,
        1.2,
        -1.5,
        -0.3,
        0.5,
        -0.8,
        0.1,
        1.0,
        -0.7,
        0.3
      ],
      "y": [
        -0.8,
        0.5,
        -1.2,
        1.5,
        0.9,
        -1.8,
        -0.5,
        0.2,
        -1.0,
        0.6,
        1.3,
        -0.9,
        0.1
      ],
      "mode": "markers",
      "type": "scatter",
      "marker": {
        "color": "#4c6ef5",
        "size": 8
      }
    },
    {
      "x": [
        5,
        4.5,
        6,
        4,
        5.5,
        4.8,
        6.2,
        3.8,
        5.1,
        4.2,
        5.8,
        4.6,
        5.3
      ],
      "y": [
        -1,
        -2,
        0.1,
        -1.5,
        -0.5,
        -2.2,
        -0.8,
        -1.8,
        -1.2,
        -2.5,
        -1.1,
        -1.9,
        -1.4
      ],
      "mode": "markers",
      "type": "scatter",
      "marker": {
        "color": "#40c057",
        "size": 8
      }
    },
    {
      "x": [
        4,
        3.5,
        5,
        3,
        4.5,
        3.8,
        5.2,
        2.8,
        4.1,
        3.2,
        4.8,
        3.6,
        4.3
      ],
      "y": [
        4.5,
        5,
        3.9,
        5.5,
        4.1,
        5.8,
        3.5,
        5.2,
        4.8,
        6.0,
        4.3,
        5.6,
        4.9
      ],
      "mode": "markers",
      "type": "scatter",
      "marker": {
        "color": "#fa5252",
        "size": 8
      }
    }
  ]
}
```

</details>



> 高斯混合模型使用多个分布（由彩色椭圆形表示）来捕获单个音素特征向量的复杂模式。

### 为每个音素训练GMM

在传统的ASR系统中，这个过程会为语言中的每个音素重复。系统会为/a/训练一个独特的GMM，为/b/训练一个，为/k/训练一个，依此类推。每个GMM从大量的转录音频数据集中学习与其音素对应的特征向量 (vector)的特定统计分布。

训练完成后，这些GMM就可以用于识别了。当系统处理一段新的未知音频并提取其特征向量时，它会向每个GMM提出一个简单的问题：“你生成这个向量的概率是多少？”

如果输入的向量落入其学习到的分布范围内，音素/t/的GMM会计算出一个高似然分数。相反，音素/d/或/a/的GMM会计算出非常低的分数。这为所有可能的音素提供了我们期望的概率，$P(\text{特征} | \text{音素})$。对于给定的特征向量，输出可能如下所示：

- $P(\text{特征} | \text{/t/}) = 0.82$
- $P(\text{特征} | \text{/d/}) = 0.11$
- $P(\text{特征} | \text{/s/}) = 0.04$
- $P(\text{特征} | \text{/a/}) = 0.001$

这组概率正是声学模型需要生成的。然而，GMMs本身有一个重要局限：它们孤立地分析每个音频帧。它们没有固有的时间或序列的理解。语音本质上是声音的序列，而不是随机集合。为了对这种时间流程进行建模，GMM与另一种统计工具——隐马尔可夫模型 (HMM) 结合使用，我们将在下一节讨论。

## 参考资料

- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky, James H. Martin (2025)
  Publisher: Stanford University
  标准教材的最新版本，详细介绍了用于语音识别声学建模的GMM和HMM。
- [Fundamentals of Speech Recognition](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEXCOyqV6oF1PrtHmLgEiAwzclbFEJ_NJ5dMTRM_zHNJXN3TWJEadDildH2QGktBh6gN5dGoIgIW7rR5EQ-ytT--m30K_u0X05ZNkVZnC5ATwN529CPrO0QYUlGx22eRnVSvQuU4KuR6EOFgPEF0wd6bA56IOBnstgJkwATvY7_a1ihqDhIHe251fYQChkauX59JUK7xKCA) — Lawrence R. Rabiner, Biing-Hwang Juang (1993)
  Publisher: Prentice Hall PTR
  经典著作，为统计语音识别（包括GMM和HMM）提供了数学和算法基础。
- [Pattern Recognition and Machine Learning](https://www.springer.com/book/9780387310732) — Christopher M. Bishop (2006)
  Publisher: Springer; DOI: [10.1007/b139367](https://doi.org/10.1007/b139367)
  经典教材，阐述了高斯混合模型作为通用概率模型的理论基础。

---

[上一节](02-%E5%B0%86%E5%A3%B0%E9%9F%B3%E6%98%A0%E5%B0%84%E5%88%B0%E9%9F%B3%E7%B4%A0.md) · [下一节](04-%E9%9A%90%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E6%A8%A1%E5%9E%8B%20%28HMM%29%20%E5%9C%A8%E5%BA%8F%E5%88%97%E6%95%B0%E6%8D%AE%E4%B8%AD%E7%9A%84%E5%BA%94%E7%94%A8.md)
