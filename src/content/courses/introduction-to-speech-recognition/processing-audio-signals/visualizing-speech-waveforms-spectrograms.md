---
course: "introduction-to-speech-recognition"
chapter: "processing-audio-signals"
lesson: "visualizing-speech-waveforms-spectrograms"
sourceId: 7094
sourceUrl: "https://apxml.com/zh/courses/introduction-to-speech-recognition/chapter-2-processing-audio-signals/visualizing-speech-waveforms-spectrograms"
title: "语音可视化：波形图与频谱图"
description: "了解如何使用波形图呈现随时间变化的音频数据振幅，并使用频谱图显示频率内容。"
order: 3
plots: ["plots/7094-0.json", "plots/7094-1.json"]
sourceHash: "7009ec6d3ba21e209dab7742fd6b622abfbd4a5f80fc96b1c8515f5d7c2fcb6d"
sourceCorrections: []
---

音频数字化后，它以一长串代表振幅值的数字序列形式存在。尽管这种格式对计算机来说很理想，但对人来说并不直观。为了了解声音的结构，我们需要将其呈现出来。呈现方式不只为了我们自身的便利；它们构成了机器学习 (machine learning)模型理解语音所用特征的基础。两种最常见的音频呈现方式是波形图和频谱图。

### 波形图：时间中的声音

呈现数字音频最直接的方式是使用波形图。波形图是一种简单的二维图，其中水平轴表示时间，垂直轴表示振幅。振幅对应声音在每个时刻的强度或“响度”。正值和负值表示声波的振动，而接近零的值表示静音。

“Hello world.”这句话的波形图。



![Interactive chart](plots/7094-0.json)



> “Hello world.”短语的波形图。两个明显的能量爆发对应着两个单词，中间由短暂的停顿隔开。

从波形图中，可以轻松分辨出录音中包含声音的部分与包含静音的部分。波峰和波谷展示了声音的强度。然而，波形图有一个重要的局限性：它没有告诉我们任何关于声音*频率*内容的信息。我们能看到有声音发出，但仅仅通过查看振幅，我们无法区分高音的“eee”声和低音的“ooo”声。对于ASR系统来说，这种频率信息对于区分音素是必要的。

### 频谱图：为画面添加频率

为了查看音频信号的频率内容，我们使用一个**频谱图**。频谱图是一种更为丰富的呈现方式，它显示了声音中存在的频率如何随时间变化。可以把它想象成一系列堆叠在一起的频率快照。

为了生成频谱图，音频信号被分解成小的、相互重叠的时间段，称为帧。对于每个帧，使用一种名为\*\*快速傅里叶变换（FFT）\*\*的数学运算来确定不同频带中存在的能量大小。结果是一个二维图，其中时间在水平轴上，频率在垂直轴上，颜色表示在每个时间点上各频率的能量或振幅。



![Interactive chart](plots/7094-1.json)



> 相同“Hello world”短语的频谱图。颜色强度表示能量，黄色表示高能量，蓝色表示低能量。

这种呈现方式提供了更多信息。现在你可以看到明显的模式：

- **元音**，例如“hello”中的“e”和“world”中的“o”，通常在特定的中低频段表现为强的水平能量带。这些频带被称为**共振峰**，是元音的区分特征。
- **擦音**，指通过狭窄通道迫使气流产生辅音（如“s”或“sh”），通常在高频区域显示为嘈杂、分散的能量。
- **爆破音**，指“p”或“t”等辅音，可能会表现为短暂的静音，随后在很宽的频率范围内突然出现能量爆发。

### 从呈现到特征

频谱图不只是一种对人类分析有用的工具。它们是ASR系统从中学习特征的基础。我们在频谱图中可以看到的视觉模式，与机器学习 (machine learning)模型需要识别的声学模式相对应。关于预加重、分帧、加窗以及尤其是MFCC创建的后续部分，都是从这种类似频谱图的表示开始，为声学模型生成一组紧凑有效的特征的过程中的步骤。

## 参考资料

- [Theory and Application of Digital Speech Processing](https://www.pearson.com/us/higher-education/program/Rabiner-Theory-and-Applications-of-Digital-Speech-Processing/PGM111453.html) — Lawrence R. Rabiner, Ronald W. Schafer (2011)
  Publisher: Pearson
  一本专注于数字语音信号处理的奠基性教科书。它深入探讨了语音信号的时域和频域分析，这对于理解波形图和语谱图至关重要。
- [Discrete-Time Signal Processing](https://books.google.com/books/about/Discrete_time_Signal_Processing.html?id=z1YWAQAAMAAJ) — Alan V. Oppenheim, Ronald W. Schafer, and John R. Buck (1999)
  Publisher: Pearson
  一本经典的数字信号处理教科书，提供了采样、傅里叶变换和频谱分析等主题的理论框架，这些是语谱图生成和解读的基础。
- [6.007 Digital Signal Processing, Spring 2011](https://ocw.mit.edu/courses/6-007-digital-signal-processing-spring-2011/) — Alan V. Oppenheim (2011)
  Publisher: MIT OpenCourseWare
  一门提供数字信号处理基础知识的大学课程，包含讲义和作业，涵盖了快速傅里叶变换和频谱分析等概念，这些直接应用于语谱图的创建。
