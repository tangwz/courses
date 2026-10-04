---
course: "introduction-to-speech-recognition"
chapter: "processing-audio-signals"
lesson: "windowing-functions-explained"
sourceId: 7096
sourceUrl: "https://apxml.com/zh/courses/introduction-to-speech-recognition/chapter-2-processing-audio-signals/windowing-functions-explained"
title: "窗口函数详解"
description: "了解为什么像汉明窗这样的窗口函数会应用于音频帧，以减少频谱泄漏。"
order: 5
plots: ["plots/7096-0.json", "plots/7096-1.json", "plots/7096-2.json"]
sourceHash: "3c3c175cc9dc29add3263dabfa7279bac6c0786e4fe717e3c8c48ec8af85b352"
sourceCorrections: []
---

将音频信号分帧后，我们会得到一系列短音频片段。然而，这种分帧处理会带来一个问题。每个帧都存在突然、尖锐的起始和结束，这是原始连续声波中不存在的人为不连续性。

如果我们直接分析这些帧中的频率，这些尖锐的边缘会引入大量原始语音中没有的高频噪声。这种现象称为**频谱泄漏**，即来自特定频率的能量“泄漏”到其他频率中，扭曲了信号的真实频率成分。为了获得准确的表示，我们必须先平滑这些边缘。

### 窗口函数的作用

窗口函数是一种数学函数，我们将其应用于每个帧，以解决频谱泄漏问题。其目的是减小信号在帧的起始和结束处的振幅，使其平滑地趋近于零。你可以将其理解为让每个帧在开始时轻柔地淡入，在结束时轻柔地淡出。

通过将帧的音频数据与窗口函数相乘，我们最大程度地减小边界处的尖锐不连续性。这会得到一个更适合频率分析的信号，这是特征提取中重要的一步。

### 汉明窗

尽管存在多种类型的窗口函数，例如汉宁窗和布莱克曼窗，但对于语音识别而言，**汉明窗**是一种非常常用且有效的选择。汉明窗的形状是中间值接近一，并在边缘处平滑地趋向于小的非零值。

过程很简单：音频帧中的每个采样点都乘以窗口函数中对应的采样点。

让我们把这个过程可视化。首先，想象我们有一个具有尖锐边缘的音频帧。



![Interactive chart](plots/7096-0.json)



> 从信号中分出的一个音频帧。注意其非零的突然起始和结束值。

接下来，我们有与帧长度相同的汉明窗。



![Interactive chart](plots/7096-1.json)



> 一个汉明窗。其值在中间最高，并趋向于边缘。

最后，我们对帧和窗进行逐点相乘。得到的“加窗”帧现在在接近零处开始和结束，形成了一个更平滑的片段。



![Interactive chart](plots/7096-2.json)



> 应用汉明窗后的音频帧。信号现在在两端平滑地趋于零。

### 与重叠帧的关联

加窗处理有助于说明我们为什么要使用**重叠帧**，这是上一节的一个内容。由于窗口函数会减小每个帧边缘处信号的振幅，我们有丢失这些部分所含信息的风险。

通过重叠帧，我们确保在一个帧末尾被弱化的信息在下一个帧的中间部分得到充分的体现。这个过程保证音频信号的任何部分在分析过程中都不会被忽略。

> 帧之间的重叠确保了一个加窗帧末尾被减弱的信息在后续帧中得到适当分析。

我们的音频现在已经分帧并加窗，信号已妥善准备好进行预处理的下一个也是最重要阶段：提取机器学习 (machine learning)模型可以用来区分不同声音的特征。

## 参考资料

- [Discrete-Time Signal Processing](https://www.pearson.com/us/higher-education/program/Oppenheim-Discrete-Time-Signal-Processing-3rd-Edition/PGM216091.html) — Alan V. Oppenheim, Ronald W. Schafer (2010)
  Publisher: Prentice Hall
  提供数字信号处理基础的全面论述，包括窗函数、其特性以及频谱泄漏现象的详细解释。
- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://books.google.com/books/about/Speech_and_Language_Processing.html?id=lI9Tf3vXj0oC) — Daniel Jurafsky and James H. Martin (2009)
  Publisher: Prentice Hall
  一本标准的语音识别教科书，为音频信号预处理（包括分帧、加窗以及使用汉明窗为特征提取做准备）提供了背景知识。
- [Digital Signal Processing](https://ocw.mit.edu/courses/res-6-008-digital-signal-processing-spring-2011/) — Prof. Alan V. Oppenheim (2011)
  Journal: MIT OpenCourseWare, Course RES.6-008; Publisher: Massachusetts Institute of Technology
  包含讲义和习题，涵盖离散时间傅里叶分析，包括有限持续时间信号的影响以及窗函数在减轻频谱泄漏方面的必要性。
