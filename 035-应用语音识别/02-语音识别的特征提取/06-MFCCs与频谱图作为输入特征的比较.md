# MFCCs与频谱图作为输入特征的比较

来源：[原文](https://apxml.com/zh/courses/applied-speech-recognition/chapter-2-feature-extraction-for-speech/comparing-mfccs-and-spectrograms)

[返回章节目录](README.md) · [返回课程目录](../README.md)

梅尔频率倒谱系数（MFCCs）和对数梅尔频谱图都是语音识别模型的输入特征。一个实际问题是：你的模型应该选择哪一个？虽然MFCCs几十年来一直是公认的标准，但深度学习 (deep learning)的兴起改变了这一共识。对于大多数现代语音识别系统，对数梅尔频谱图是更受青睐的输入特征。将分析它们的权衡，以明白其中缘由。

根本区别在于MFCC计算的最后一步：离散余弦变换（DCT）。这一步旨在消除滤波器组能量之间的相关性，并将最重要的信息压缩到前几个系数中。

### 信息、压缩与模型架构

MFCC处理流程中的DCT是一种有损压缩。它舍弃了一些关于频谱结构的更细致的信息，以换取一种紧凑、去相关化的表示。这对于高斯混合模型（GMMs）等经典机器学习 (machine learning)模型非常有利，因为当输入特征彼此不相关时，这些模型的表现更好。通过将最重要的信号信息集中到少量系数中，MFCCs为这些早期系统提供了高效且实用的输入。

然而，深度神经网络 (neural network)，特别是卷积神经网络 (CNN)（CNNs），运作方式不同。它们非常擅长从高维、相关联的数据中学习相关模式。对于CNN而言，对数梅尔频谱图类似于单通道图像，其中横轴是时间，纵轴是频率。

- **对数梅尔频谱图** 保留了相邻频率 bins 之间的相关性。CNN可以应用其卷积滤波器来学习检测此“图像”中的形状和模式，例如共振峰（频谱图中的暗带）及其随时间的移动。频率之间的这种空间关系是CNN可以有效处理的有价值信息。
- **MFCCs** 通过应用DCT，实际上打乱了这种频谱信息。第一个系数（$c_0$）表示总能量，下一个表示大致的频谱斜率，随后的系数表示更精细的细节。相邻梅尔滤波器之间的直接“空间”关系消失了。虽然神经网络仍然可以从MFCCs中学习，但它无法像处理频谱图那样，以同样直观的方式使用其卷积结构来学习局部频率模式。

下图说明了MFCC处理流程如何包含一个额外的DCT步骤，而对数梅尔频谱图处理流程则省略了该步骤。

> 主要区别在于DCT步骤。对数梅尔频谱图直接提供梅尔滤波器组的输出，保留了局部频谱结构，而MFCCs则对该输出进行压缩和去相关处理。

### 维度与计算成本

典型的MFCC特征向量 (vector)可能有13、20或40个维度。相比之下，对数梅尔频谱图通常使用80或128个梅尔频段，导致每个时间步的特征向量具有80或128个维度。

过去，MFCCs较低的维度是一个很大的优势。它减少了内存需求和计算负荷，这些都是当时重要的限制。如今，随着GPU加速，深度学习 (deep learning)模型可以轻松处理对数梅尔频谱图的更高维度。这些更大的特征向量中包含的额外信息通常会带来更好的模型性能，从而证明增加的计算成本是合理的。

### 主要区别概述

这两种特征类型的选择涉及信息保留与维度之间的权衡。下表概述了主要的比较点。

| 特征 | MFCCs | 对数梅尔频谱图 |
| --- | --- | --- |
| **最终步骤** | 离散余弦变换（DCT） | 梅尔滤波器组能量的对数 |
| **信息** | 压缩的、去相关系数 | 丰富的、相关的频谱结构 |
| **维度** | 低（通常13-40） | 更高（通常64-128） |
| **主要用途** | 传统GMM-HMM系统，资源受限应用 | 现代CNN、RNN和基于Transformer的模型 |
| **核心理念** | 为较简单模型提供紧凑、高效的表示 | 为强大模型提供丰富、图像般的表示 |

### 现代语音识别的建议

对于本课程中我们将构建的模型，例如LSTM、Transformer和Conformer，**对数梅尔频谱图是推荐的输入特征。** 这些架构能够处理高维输入，并且专门设计用于发现数据中复杂、分层的模式。通过为它们提供对数梅尔频谱图，您可以让模型直接从音频的丰富表示中学习最相关的声学特征，而不是依赖于MFCCs的手动工程压缩。

虽然理解MFCCs对于了解语音识别的历史和某些特定应用具有意义，但对数梅尔频谱图是构建高性能、现代语音识别系统的优选特征。

## 参考资料

- [Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Sentences](https://ieeexplore.ieee.org/document/1163420) — S. B. Davis and P. Mermelstein (1980)
  Journal: IEEE Transactions on Acoustics, Speech, and Signal Processing; Publisher: IEEE; Volume: 28; Pages: 357-366; DOI: [10.1109/TASSP.1980.1163420](https://doi.org/10.1109/TASSP.1980.1163420)
  介绍梅尔频率倒谱系数（MFCCs）的基础论文，阐述了其推导过程以及在语音识别中的应用。
- [Fundamentals of Speech Recognition](https://books.google.com/books/about/Fundamentals_of_Speech_Recognition.html?id=c14uAQAAIAAJ) — Lawrence R. Rabiner, Biing-Hwang Juang (1993)
  Publisher: Prentice Hall
  一本经典的教材，全面涵盖了传统语音识别技术，包括MFCCs与GMM-HMM系统相结合的详细理论和应用。
- [Convolutional Neural Networks for Large-Scale Speech Recognition](https://www.researchgate.net/publication/264560935_Convolutional_Neural_Networks_for_Large-Scale_Speech_Recognition) — Osama Abdel-Hamid, Abdel-Rahman Mohamed, Hui Jiang, Li Deng, George Penn, and Dong Yu (2014)
  Journal: IEEE/ACM Transactions on Audio, Speech, and Language Processing; Publisher: IEEE; Volume: 22; Pages: 1585-1598; DOI: [10.1109/TASLP.2014.2339736](https://doi.org/10.1109/TASLP.2014.2339736)
  一篇开创性论文，展示了卷积神经网络（CNNs）在自动语音识别（ASR）中的有效性，这些网络通常使用对数梅尔谱图作为输入，并强调了CNN从其局部模式中学习的能力。
- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  一本权威的在线教材，持续更新，涵盖现代语音和语言处理，包括基于深度学习的自动语音识别的特征提取（MFCCs和谱图）的讨论。

---

[上一节](05-%E7%89%B9%E5%BE%81%E5%BD%92%E4%B8%80%E5%8C%96%E6%8A%80%E6%9C%AF.md) · [下一节](07-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BB%8E%E6%95%B0%E6%8D%AE%E9%9B%86%E4%B8%AD%E6%8F%90%E5%8F%96%E5%92%8C%E5%BD%92%E4%B8%80%E5%8C%96%E7%89%B9%E5%BE%81.md)
