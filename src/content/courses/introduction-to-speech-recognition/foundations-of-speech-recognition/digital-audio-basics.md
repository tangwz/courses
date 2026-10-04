---
course: "introduction-to-speech-recognition"
chapter: "foundations-of-speech-recognition"
lesson: "digital-audio-basics"
sourceId: 7089
sourceUrl: "https://apxml.com/zh/courses/introduction-to-speech-recognition/chapter-1-foundations-of-speech-recognition/digital-audio-basics"
title: "计算机如何处理声音：数字音频基础"
description: "了解声音如何以数字形式表示的基本原理，这是计算机处理的必要步骤。"
order: 6
plots: ["plots/7089-0.json"]
sourceHash: "cdefa39901a21203929c1948c804e5c7256b262aa2f957376b813f286b3d2439"
sourceCorrections: []
---

在计算机理解语音内容之前，它首先需要将声音的物理现象转换成它能理解的语言：数字。我们发出的声音是连续的模拟波，在空气中传播时压力有无数种变化。然而，计算机处理的是离散的、有限的数据。将模拟声音转换为数字信息的流程是所有语音识别的根本。这种转换包含两个主要步骤：采样和量化 (quantization)。

### 从模拟波形到数字信号

设想一个模拟声波就像一条平滑、连续的线，表示声音压力随时间的变化。要将其转换为数字格式，我们无法存储构成该线的无限个点。相反，我们必须通过在固定、离散的时间间隔内进行测量来近似它。



![模拟波形与数字采样点对比](plots/7089-0.json)



> 模拟波形是连续信号。数字化是在离散的时间点捕获其振幅。

### 采样：在时间中截取片段

第一步，**采样**，是在固定的时间间隔内测量模拟波形振幅的行为。把它想象成电影摄影机快速捕捉一系列静态照片，以制造运动的错觉。在音频中，我们捕获的是一系列快速的振幅“快照”。

在计算机能理解所说的内容之前，它必须首先将声音的物理现象转换为它能理解的语言：数字。我们发出的声音是连续的模拟波，在空气中以无限的压力变化传播。然而，计算机处理的是离散的有限数据。将模拟声音转换为数字信息是语音识别的一个主要过程。此转换主要包括两个步骤：采样和量化 (quantization)。

采样率很重要，因为它决定了可以准确捕获的频率范围。根据奈奎斯特-香农采样定理，采样率必须至少是信号中最高频率的两倍。由于人类语音的频率范围通常低于8 kHz，16 kHz的采样率在语音识别中很常用，因为它提供了足够的余量。

### 量化 (quantization)：测量振幅

采样告诉我们*何时*测量之后，**量化**告诉我们为每个测量值分配*什么*数值。模拟波形的振幅仍然是连续的，这表示它可以在一个范围内取任何值。量化通过将其映射到有限离散电平集合中最接近的值来近似这种连续振幅。

可用电平的数量由**位深**决定。更高的位深提供更多电平，从而更准确地表示振幅。

- 一个**8位**音频文件使用$2^8 = 256$个离散电平来表示振幅。
- 一个**16位**音频文件使用$2^{16} = 65,536$个离散电平，提供更高的保真度。这是大多数ASR应用的通用标准。

> 量化将连续的振幅值映射到可用的最接近的离散电平，这由位深决定。

### 结果：一串数字

采样和量化 (quantization)的结合将连续的声波转换成一串数字。序列中的每个数字代表声音在某个特定时间点上的量化振幅。对于一个1秒钟、以16 kHz采样率和16位深采样的音频片段，结果是一个包含16,000个整数的数组，每个整数的值介于-32,768和32,767之间。

这一串数字，通常被称为**波形**，是计算机可以存储和处理的原始数字音频。它是我们ASR流程的第一个输入。尽管它是声音的忠实数字表示，但它尚未处于适合的格式，以便机器学习 (machine learning)模型找出模式。在下一章中，我们将学习如何处理这种原始波形，成为更有用的特征。

## 参考资料

- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky, James H. Martin (2025)
  这本语音与语言处理基础教科书的第三版草稿，详细介绍了语音的声学语音学和数字表示。
- [The Scientist and Engineer's Guide to Digital Signal Processing](http://www.dspguide.com/) — Steven W. Smith (1997)
  Publisher: California Technical Publishing
  这本易于理解的在线教材，实践性地介绍了数字信号处理概念，如采样、混叠和量化，适合自学。
