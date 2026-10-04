---
course: "applied-speech-recognition"
chapter: "feature-extraction-for-speech"
lesson: "mel-frequency-cepstral-coefficients"
sourceId: 7109
sourceUrl: "https://apxml.com/zh/courses/applied-speech-recognition/chapter-2-feature-extraction-for-speech/mel-frequency-cepstral-coefficients"
title: "梅尔频率倒谱系数 (MFCC)"
description: "详细阐述梅尔频率倒谱系数 (MFCC)，一种用于语音识别任务的标准特征集合。"
order: 2
plots: ["plots/7109-0.json"]
sourceHash: "6044d23d8a942bd76a3e104379968c6311a7cd109f4bb6eb23314a5032485c6f"
sourceCorrections: []
---

多年来，梅尔频率倒谱系数 (MFCC) 一直是语音识别系统中表示音频的行业通用方法。它们的设计直接受到人类感知声音方式的启发，这使得它们成为一种有效且紧凑的方式，能表示语音中对区分不同音素的主要部分。即使有新型特征类型出现，了解 MFCC 对任何从事语音技术工作的人来说都不可或缺，因为它们提供了一个强大的参考，并阐明了音频特征工程的根本原则。

MFCC 的主要思路是将信号的频率表示进行转换，以更好地匹配人耳的非线性响应。我们的听觉对低频变化比高频变化更敏感。例如，100 Hz 和 200 Hz 之间的感知差异远大于 10,000 Hz 和 10,100 Hz 之间的差异，即使它们的绝对差异相同。梅尔刻度是一种感知音高刻度，它将这种观察形式化。

### MFCC 计算流程

概括来说，生成 MFCC 涉及一个多步骤过程，将原始音频信号转换为特征向量 (vector)序列。该流程中的每一步都旨在提取和提炼语音学相关信息，同时丢弃噪声和说话人基频（音高）等其他不相关细节。

下图显示了计算 MFCC 的标准流程。

> 将原始音频信号转换为梅尔频率倒谱系数的过程。每一步都细化表示，从时域转向一组紧凑的特征。

我们将在下一节学习每个步骤的实现细节，但首先让我们对该流程中两个最重要的转换：梅尔刻度与倒谱，建立直观理解。

### 用梅尔刻度扭曲频率

梅尔刻度将以赫兹 (Hz) 为单位的物理频率转换为与人类听觉感知对齐 (alignment)的刻度。转换公式为：


$$
m = 2595 \cdot \log_{10}(1 + \frac{f}{700})
$$


这里，$f$ 是以赫兹为单位的物理频率，$m$ 是以梅尔为单位的感知频率。这种对数关系使得低于 1000 Hz 的频率更分散，而高于 1000 Hz 的频率则被压缩。

为了将此应用于我们的信号，我们使用一组称为**梅尔滤波器组**的三角形滤波器，它们分布在整个频谱上。这些滤波器在低频时窄而密集，但在高频时变得更宽更分散，有效地模仿了人耳耳蜗的响应。



![频率 (Hz) 与梅尔刻度](plots/7109-0.json)



> 线性赫兹刻度与感知梅尔刻度之间的关系。梅尔刻度的曲线显示了它如何将更多分辨率分配给低频，模仿人类听觉。

当我们将语音帧的功率谱通过此滤波器组时，会得到一个能量值向量 (vector)，每个滤波器对应一个值。这被称为对数梅尔谱图，它本身是语音识别中流行的特征，也是 MFCC 的前身。

### 用倒谱分离声源与滤波器

应用梅尔滤波器组并取对数后，我们得到频谱包络的平滑表示。然而，所得的特征向量 (vector)彼此之间通常高度相关。MFCC 流程的最后一步，**离散余弦变换 (DCT)**，用于解相关这些向量，并将信息压缩成几个主要系数。

将 DCT 应用于对数梅尔谱图是一种数学技巧，它将我们带入**倒谱域**。“倒谱”（cepstrum）一词是“频谱”（spectrum）的颠倒字母顺序的词，它是通过对对数频谱进行傅里叶变换（或在此例中，高度相关的 DCT）计算得出的。此操作的结果具有一个独特属性：它将声源（声带振动，产生音高）的特征与滤波器（声道形状，产生音素）的特征分离开来。

低阶 DCT 系数表示频谱包络的慢变形状，这对应于音素的身份（例如，/a/、/t/、/sh/）。高阶系数表示快变细节，这通常与音高和激励相关。对于语音识别，语音学信息是最重要的。

因此，我们通常只保留前 13 到 40 个系数，并丢弃其余部分。所得向量是单帧音频的一个 MFCC 特征。通过堆叠所有帧的这些向量，我们创建了最终的特征矩阵，作为我们声学模型的输入。总之，MFCC 之所以有效，是因为它们提供了语音音素内容的一种紧凑、解相关且感知上具有意义的表示。

## 参考资料

- [Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Sentences](https://ieeexplore.ieee.org/document/1163420) — S. B. Davis and P. Mermelstein (1980)
  Journal: IEEE Transactions on Acoustics, Speech, and Signal Processing; Publisher: IEEE; Volume: 28; Pages: 357-366; DOI: [10.1109/TASSP.1980.1163420](https://doi.org/10.1109/TASSP.1980.1163420)
  这篇有影响力的论文介绍了梅尔频率倒谱系数（MFCCs），并展示了它们在语音识别任务中的有效性。
- [Discrete-Time Processing of Speech Signals](https://books.google.com/books/about/Discrete_time_Processing_of_Speech_Signa.html?id=c1-4QgAACAAJ) — John R. Deller, John H. L. Hansen, and John G. Proakis (2000)
  Publisher: Wiley-IEEE Press
  这本书为专门应用于语音的数字信号处理技术提供了坚实的基础，涵盖了MFCC计算中所用变换和滤波的基本原理。
- [A Scale for the Measurement of the Psychological Magnitude of Pitch](https://doi.org/10.1121/1.1915893) — S. S. Stevens, J. Volkmann, and E. B. Newman (1937)
  Journal: Journal of the Acoustical Society of America; Publisher: The Acoustical Society of America; Volume: 8; Pages: 185-190; DOI: [10.1121/1.1915893](https://doi.org/10.1121/1.1915893)
  这篇经典论文引入了梅尔刻度，它是一种感知音高刻度，由于与人类听觉感知的对齐，成为梅尔频率倒谱系数的基本组成部分。
