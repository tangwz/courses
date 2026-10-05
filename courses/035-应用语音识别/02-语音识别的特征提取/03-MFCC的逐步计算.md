# MFCC的逐步计算

来源：[原文](https://apxml.com/zh/courses/applied-speech-recognition/chapter-2-feature-extraction-for-speech/calculating-mfccs)

[返回章节目录](README.md) · [返回课程目录](../README.md)

梅尔频率倒谱系数（MFCC）是语音识别中一项重要的特性，旨在以突出人类语音特性方式来表示音频。其计算过程包含多个信号处理阶段，每个阶段都用于将原始音频信号转换为一组紧凑且包含丰富信息的系数。

> 从原始音频信号生成MFCC的完整流程。

### 步骤1：预加重

第一步是对音频信号应用预加重滤波器。人类语音的频谱自然偏斜；其在低频段的能量高于高频段。这对于对输入动态范围敏感的模型来说，可能造成问题。预加重是一种高通滤波器，它提高高频段的能量。

这主要有两个作用：它平衡频谱，使其不再倾斜，并且可以通过放大重要的高频共振峰来提升信噪比（SNR）。该滤波器通常通过一阶差分方程实现：


$$
y[n] = x[n] - \alpha x[n-1]
$$


这里，$x[n]$ 是输入信号的一个样本，$y[n]$ 是输出样本，$\alpha$ 是滤波器系数。$\alpha$ 的典型值在0.95到0.97之间。

### 步骤2：分帧

语音是非平稳信号，表示其统计特性随时间变化。然而，在非常短的持续时间（例如20-30毫秒）内，信号可以被视为准平稳。因此，我们将预加重信号分割成短的、相互重叠的帧。

标准配置是使用25毫秒的帧长和10毫秒的步长（或跳跃长度）。这意味着每帧长25毫秒，我们向前推进10毫秒以创建下一帧，从而产生15毫秒的重叠。这种重叠确保当我们稍后应用加窗函数时，不会在每帧边缘丢失信息。

> 连续音频信号被分割成重叠的帧。

### 步骤3：加窗

分帧后，对每一帧应用窗函数。当我们从信号中提取有限的帧时，会在其边缘产生尖锐的不连续性。对这个块进行傅里叶变换会引入高频伪影，这种现象被称为频谱泄漏。

为了最大限度地减少这一点，我们将每一帧乘以一个窗函数，例如汉明窗。这个函数将帧的两端逐渐衰减到零，平滑信号并减少边缘不连续性。

### 步骤4：快速傅里叶变换（FFT）

对每帧加窗后，我们现在可以将其从时域转换为频域。我们通过对每帧应用快速傅里叶变换（FFT）来实现这一点。FFT计算离散傅里叶变换，并提供帧的频谱。

FFT的输出是一组复数。对于ASR，我们主要关注这些频率的幅度，因此我们计算功率谱，通常将其计算为FFT输出的平方幅值：


$$
P = \frac{|FFT(frame)|^2}{N}
$$


其中 $N$ 是FFT的长度。结果是针对一系列频率分箱的能量值数组。

### 步骤5：应用梅尔滤波器组

FFT产生的线性频率尺度与人类感知声音的方式不符。我们更容易分辨低频中的微小变化，而不是高频。梅尔尺度是一种由听者判断音高间距相等的感知尺度。

为了将功率谱映射到梅尔尺度，我们使用一组三角形滤波器，称为梅尔滤波器组。这个滤波器组通常由20-40个重叠滤波器组成。每个滤波器在低频处较窄，在高频处较宽，反映了人耳听觉的非线性特点。我们将功率谱乘以每个三角形滤波器并求和能量，以获得该特定梅尔尺度频带的滤波器组能量。



[交互图表：交互图表](https://apxml.com/zh/courses/applied-speech-recognition/chapter-2-feature-extraction-for-speech/calculating-mfccs#plot-sr7xvq)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "xaxis": {
      "title": "频率 (赫兹)"
    },
    "yaxis": {
      "title": "幅度"
    },
    "margin": {
      "l": 50,
      "r": 20,
      "t": 20,
      "b": 40
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
      "name": "滤波器 1",
      "fillcolor": "#a5d8ff",
      "line": {
        "color": "#1c7ed6"
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
      "name": "滤波器 2",
      "fillcolor": "#96f2d7",
      "line": {
        "color": "#0ca678"
      }
    },
    {
      "x": [
        250,
        500,
        750
      ],
      "y": [
        0,
        1,
        0
      ],
      "type": "scatter",
      "mode": "lines",
      "name": "滤波器 3",
      "fillcolor": "#ffc9c9",
      "line": {
        "color": "#f03e3e"
      }
    },
    {
      "x": [
        500,
        1000,
        1500
      ],
      "y": [
        0,
        1,
        0
      ],
      "type": "scatter",
      "mode": "lines",
      "name": "滤波器 4",
      "fillcolor": "#d0bfff",
      "line": {
        "color": "#7048e8"
      }
    }
  ],
  "showlegend": false
}
```

</details>



> 梅尔滤波器组中的一小组三角形滤波器。请注意，在较高频率下，滤波器如何变得更宽、更分散。

### 步骤6：滤波器组能量的对数

上一步的输出是一组滤波器组能量。与频率类似，人类对响度的感知是呈对数关系的。为了模仿这一点，我们对所有滤波器组能量取对数。

这一步有一个有益的附带作用：它有助于压缩特征的动态范围，使其对信号幅度的变化不那么敏感。此时的结果是**对数梅尔谱图**，它本身就是一种有用的特性，并且通常直接用作现代深度学习 (deep learning)模型的输入。

### 步骤7：离散余弦变换（DCT）

计算MFCC的最后一步是将离散余弦变换（DCT）应用于对数滤波器组能量。滤波器组能量通常彼此高度相关，因为三角形滤波器重叠。DCT是一种数学运算，用于将这些能量去相关，类似于主成分分析（PCA）的工作方式。

DCT的结果是一组系数，信号的大部分信息集中在前几个分量中。


$$
c_i = \sqrt{\frac{2}{N}} \sum_{j=1}^{N} m_j \cos\left(\frac{\pi i}{N}(j-0.5)\right)
$$


其中 $c_i$ 是第 $i$ 个MFCC，$m_j$ 是第 $j$ 个滤波器组的对数能量，$N$ 是滤波器组的总数。

我们通常只保留少量这些系数，例如前13个（系数2到13，因为第一个通常代表信号偏移）。这种减少作为一种压缩形式，将最有用的频谱信息保留在一个紧凑的向量 (vector)中。每一帧最终得到的系数集就是梅尔频率倒谱系数。

在实际应用中，您很少需要从头开始实现这些步骤。Python中的`librosa`等库提供了高度优化的函数，可以用一行代码完成整个计算。例如：

```python
import librosa

# 假设 'y' 是音频时间序列，'sr' 是采样率
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
```

然而，理解这个逐步过程对于做出明智的特征工程决策和诊断ASR流程中的问题是很重要的。从预加重到最终DCT的每一步都在塑造您的声学模型最终将学习到的特征中起着作用。

## 参考资料

- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/8.pdf) — Daniel Jurafsky and James H. Martin (2018)
  Publisher: Pearson; Pages: Chapter 8, "Feature Extraction for Speech Recognition," pp. 208-219
  一本全面的语音和语言处理教科书，详细阐述了MFCC的计算过程及其在语音识别声学建模中的重要性。
- [Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Sentences](https://ieeexplore.ieee.org/document/1163420) — Steven B. Davis and Paul Mermelstein (1980)
  Journal: IEEE Transactions on Acoustics, Speech, and Signal Processing; Publisher: IEEE; Volume: 28; Pages: 357-366; DOI: [10.1109/TASSP.1980.1163420](https://doi.org/10.1109/TASSP.1980.1163420)
  这篇开创性论文介绍了梅尔频率倒谱系数并评估了其有效性，确立了其作为语音识别特征的地位。
- [Discrete-Time Signal Processing](https://library.rice.edu/catalog/9924452) — Alan V. Oppenheim, Ronald W. Schafer, John R. Buck (1999)
  Publisher: Prentice Hall; Pages: Relevant chapters on Fourier transforms, digital filters, and windowing
  一本离散时间信号处理的基础教科书，详细介绍了傅里叶变换、数字滤波器和窗函数等概念的理论，这些都是MFCC计算中的组成部分。
- [librosa.feature.mfcc](https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html) — Brian McFee and the librosa developers (2024)
  Publisher: librosa project
  `librosa`库MFCC函数的官方文档，展示了在Python中计算MFCC的实际实现和参数。

---

[上一节](02-%E6%A2%85%E5%B0%94%E9%A2%91%E7%8E%87%E5%80%92%E8%B0%B1%E7%B3%BB%E6%95%B0%20%28MFCC%29.md) · [下一节](04-%E6%BB%A4%E6%B3%A2%E5%99%A8%E7%BB%84%E5%92%8C%E5%AF%B9%E6%95%B0%E6%A2%85%E5%B0%94%E9%A2%91%E8%B0%B1%E5%9B%BE.md)
