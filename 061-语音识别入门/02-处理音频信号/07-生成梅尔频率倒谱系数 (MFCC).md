# 生成梅尔频率倒谱系数 (MFCC)

来源：[原文](https://apxml.com/zh/courses/introduction-to-speech-recognition/chapter-2-processing-audio-signals/creating-mfccs)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在将音频处理成帧并应用窗函数后，我们得到一系列短音频片段。正如我们所见，我们可以使用傅里叶变换来生成频谱图，它显示了语音信号的频率内容如何随时间变化。虽然频谱图是一种丰富的数据形式，但它包含大量对于语音理解而言并非同等重要的信息。此外，其高维度使得机器学习 (machine learning)模型直接处理起来计算量很大。

为了从音频信号中提取出最重要的特征，并同时丢弃冗余信息，需要一种特定的方法。实现这种方法是特征提取的主要目标。在语音识别领域，梅尔频率倒谱系数（简称MFCC）是使用最广泛且具有历史意义的特征。MFCC旨在以一种更符合人类听觉感知的方式来表示音频。

### 梅尔刻度：模拟人类听觉

MFCC的一个基本特点是**梅尔刻度**，它是一种听觉上的音高刻度，听众认为其音高间距是相等的。人类在低频下比在高频下更能分辨音高的小变化。例如，100赫兹和200赫兹之间感知的差异远大于10,000赫兹和10,100赫兹之间的差异，尽管它们的绝对差异是相同的。

梅尔刻度将线性频率（以赫兹为单位测量）重新映射，以更好地反映人类听觉的这一特性。将频率 $f$（赫兹）转换为梅尔的公式是：


$$
m = 2595 \cdot \log_{10}(1 + \frac{f}{700})
$$


这种重映射为较低频率提供了更高的分辨率，而为较高频率提供了较低的分辨率，从而有效侧重于频谱中与语音最相关的部分。



[交互图表：交互图表](https://apxml.com/zh/courses/introduction-to-speech-recognition/chapter-2-processing-audio-signals/creating-mfccs#plot-1am9h16)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "xaxis": {
      "title": "频率 (赫兹)",
      "gridcolor": "#e9ecef"
    },
    "yaxis": {
      "title": "梅尔刻度",
      "gridcolor": "#e9ecef"
    },
    "paper_bgcolor": "#ffffff",
    "plot_bgcolor": "#ffffff",
    "font": {
      "color": "#495057"
    },
    "margin": {
      "l": 60,
      "r": 20,
      "t": 20,
      "b": 50
    }
  },
  "data": [
    {
      "x": [
        0,
        500,
        1000,
        2000,
        4000,
        8000
      ],
      "y": [
        0,
        607,
        969,
        1500,
        2146,
        2835
      ],
      "type": "scatter",
      "mode": "lines",
      "name": "梅尔刻度",
      "line": {
        "color": "#4263eb",
        "width": 3
      }
    },
    {
      "x": [
        0,
        500,
        1000,
        2000,
        4000,
        8000
      ],
      "y": [
        0,
        500,
        1000,
        2000,
        4000,
        8000
      ],
      "type": "scatter",
      "mode": "lines",
      "name": "线性刻度",
      "line": {
        "color": "#adb5bd",
        "width": 2,
        "dash": "dash"
      }
    }
  ]
}
```

</details>



> 线性频率（赫兹）与感知梅尔刻度之间的关系。请注意，梅尔刻度在低频处急剧上升，然后趋于平缓，这表明在该较低范围内，感知的音高变化更为敏感。

### 计算MFCC：分步流程

生成MFCC需要对每个音频帧应用一系列变换。每个步骤都旨在逐步分离和压缩定义语音声音的信息。

> 从单个音频帧计算梅尔频率倒谱系数的流程。

让我们逐一查看这个流程中每个阶段发生的情况。

#### 1. 计算功率谱

对于每个加窗的音频帧，我们进行快速傅里叶变换（FFT）以获得其频率谱。然后，我们通过取FFT复数幅度的平方来计算功率谱。这为我们提供了该特定帧在每个频带中存在的能量度量。这与用于生成频谱图一个时间片的信息相同。

#### 2. 应用梅尔滤波器组

这是梅尔刻度发挥作用的地方。我们创建一个**梅尔滤波器组**，它由20到40个三角形滤波器组成。这些滤波器在低频处较窄且间隔紧密，在高频处较宽且分布更广，与梅尔刻度相匹配。



[交互图表：交互图表](https://apxml.com/zh/courses/introduction-to-speech-recognition/chapter-2-processing-audio-signals/creating-mfccs#plot-zr3l6p)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "xaxis": {
      "title": "频率 (赫兹)",
      "gridcolor": "#e9ecef"
    },
    "yaxis": {
      "title": "权重",
      "gridcolor": "#e9ecef",
      "showticklabels": false
    },
    "paper_bgcolor": "#ffffff",
    "plot_bgcolor": "#ffffff",
    "font": {
      "color": "#495057"
    },
    "showlegend": false,
    "margin": {
      "l": 20,
      "r": 20,
      "t": 20,
      "b": 50
    }
  },
  "data": [
    {
      "x": [
        0,
        100,
        200
      ],
      "y": [
        0,
        1,
        0
      ],
      "type": "scatter",
      "mode": "lines",
      "fill": "tozeroy",
      "fillcolor": "#d0bfff",
      "line": {
        "color": "#845ef7"
      }
    },
    {
      "x": [
        100,
        250,
        400
      ],
      "y": [
        0,
        1,
        0
      ],
      "type": "scatter",
      "mode": "lines",
      "fill": "tozeroy",
      "fillcolor": "#d0bfff",
      "line": {
        "color": "#845ef7"
      }
    },
    {
      "x": [
        250,
        450,
        650
      ],
      "y": [
        0,
        1,
        0
      ],
      "type": "scatter",
      "mode": "lines",
      "fill": "tozeroy",
      "fillcolor": "#d0bfff",
      "line": {
        "color": "#845ef7"
      }
    },
    {
      "x": [
        450,
        750,
        1050
      ],
      "y": [
        0,
        1,
        0
      ],
      "type": "scatter",
      "mode": "lines",
      "fill": "tozeroy",
      "fillcolor": "#d0bfff",
      "line": {
        "color": "#845ef7"
      }
    },
    {
      "x": [
        750,
        1200,
        1650
      ],
      "y": [
        0,
        1,
        0
      ],
      "type": "scatter",
      "mode": "lines",
      "fill": "tozeroy",
      "fillcolor": "#d0bfff",
      "line": {
        "color": "#845ef7"
      }
    },
    {
      "x": [
        1200,
        1800,
        2400
      ],
      "y": [
        0,
        1,
        0
      ],
      "type": "scatter",
      "mode": "lines",
      "fill": "tozeroy",
      "fillcolor": "#d0bfff",
      "line": {
        "color": "#845ef7"
      }
    },
    {
      "x": [
        1800,
        2600,
        3400
      ],
      "y": [
        0,
        1,
        0
      ],
      "type": "scatter",
      "mode": "lines",
      "fill": "tozeroy",
      "fillcolor": "#d0bfff",
      "line": {
        "color": "#845ef7"
      }
    },
    {
      "x": [
        2600,
        3600,
        4600
      ],
      "y": [
        0,
        1,
        0
      ],
      "type": "scatter",
      "mode": "lines",
      "fill": "tozeroy",
      "fillcolor": "#d0bfff",
      "line": {
        "color": "#845ef7"
      }
    }
  ]
}
```

</details>



> 一个带有三角形滤波器的梅尔滤波器组。请注意，滤波器在较低频率处较窄且更密集，而在较高频率处变得更宽。

我们将功率谱与这些三角形滤波器中的每一个相乘，并加总每个频带中的能量。结果是一个数字列表，每个滤波器对应一个数字，表示感知刻度频谱不同区域的能量大小。这一步骤有效地将我们数据的维度从数百或数千个频率分箱减少到仅20-40个滤波器组能量。

#### 3. 对滤波器组能量取对数

应用滤波器组后，我们对所有滤波器组能量取对数。这一步骤也与人类感知有关，因为我们对响度的响应是对数而非线性的。它有助于压缩值的动态范围，使特征对信号能量的变化不那么敏感。

#### 4. 进行离散余弦变换 (DCT)

最后一个主要步骤是计算对数滤波器组能量的离散余弦变换（DCT）。滤波器组得到的对数能量彼此之间高度相关。DCT是一种数学运算，它对这些能量进行去相关处理，意味着将其分离成更独立的成分。

这个过程非常有效地将最主要的信息集中到前几个系数中。可以将其想象成图像压缩，其中复杂的图像由更小、更有效的一组值表示。最终的系数就是**梅尔频率倒谱系数**。

### 最终特征向量 (vector)

通常，我们只保留这些DCT系数的前12到13个。较高的系数表示滤波器组能量的非常快速的变化，这些变化通常对语音的信息量较少，并且可能对噪声敏感。第0个系数表示帧的平均能量，有时会被丢弃或单独处理。

完成整个过程后，每个25毫秒的音频帧从数千个原始采样点转换为一个仅包含12或13个数字的小向量。当我们把音频片段中所有帧的这些向量堆叠起来时，就得到一个特征矩阵。这个矩阵，每一行对应一个时间帧，每一列是一个系数，是我们输入到声学模型中的最终表示。它是原始音频信号的紧凑且与感知相关的概括。

## 参考资料

- [Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Sentences](https://doi.org/10.1109/TASSP.1980.1163420) — S. B. Davis and P. Mermelstein (1980)
  Journal: IEEE Transactions on Acoustics, Speech, and Signal Processing; Publisher: IEEE; Volume: 28; Pages: 357-366; DOI: [10.1109/TASSP.1980.1163420](https://doi.org/10.1109/TASSP.1980.1163420)
  这篇开创性论文介绍了梅尔频率倒谱系数（MFCCs）作为语音识别的有效特征。
- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  一本广泛使用的教材，全面介绍了语音处理，包括对MFCCs及其在语音识别中作用的详细解释。（第三版在线草稿）

---

[上一节](06-%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96%E7%AE%80%E4%BB%8B.md) · [下一节](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E9%9F%B3%E9%A2%91%E6%96%87%E4%BB%B6%E5%8F%AF%E8%A7%86%E5%8C%96%E4%B8%8E%E5%A4%84%E7%90%86.md)
