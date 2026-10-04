---
course: "applied-speech-recognition"
chapter: "foundations-digital-audio-speech"
lesson: "introduction-to-spectrograms"
sourceId: 7102
sourceUrl: "https://apxml.com/zh/courses/applied-speech-recognition/chapter-1-foundations-digital-audio-speech/introduction-to-spectrograms"
title: "语音可视化中的语谱图入门"
description: "学习如何创建和解读语谱图，以可视化语音的频率内容随时间的变化。"
order: 6
plots: ["plots/7102-0.json"]
sourceHash: "a757120cb707b865477c48e5ff257d3dd9f1f88e81e923debbe4002fc85d218e"
sourceCorrections: []
---

波形图显示了信号随时间变化的幅度，但它隐藏了区分不同声音所必需的频率信息。另一方面，标准傅里叶变换提供的是整个音频片段中所有频率的概览，但它抹去了这些频率何时出现的信息。对于像语音这样的信号来说，这是一个很大的问题，因为其频率内容是时刻变化的。

为了有效地分析语音，我们需要一种能同时保留时间和频率信息的方法。这正是**语谱图**的作用，它是一种视觉表示，呈现了信号的频率内容如何随时间变化。它是语音处理中最基本的工具之一。

### 短时傅里叶变换 (STFT)

语谱图是通过一种称为**短时傅里叶变换 (STFT)** 的过程生成的。STFT 不是一次性分析整个音频信号，而是将信号分解成短的、重叠的帧或窗口，通常长度为 20-30 毫秒。对于这些短帧中的每一个，我们可以假设信号是平稳的，这意味着其频率属性在该小时间段内没有太大变化。

该过程如下：

1. **加窗：** 音频信号被分成小的、重叠的帧。对每个帧应用一个窗函数（如汉宁窗或汉明窗），以减少频谱泄漏，这是一种可能在帧边界出现的伪影。
2. **傅里叶变换：** 对每个加窗帧计算快速傅里叶变换 (FFT)。这将短时域信号转换为其频域表示，显示该特定帧中存在的每个频率的功率。
3. **堆叠：** 将所有帧得到的频率频谱并排放置，形成一个二维图像。

这有效地创建了一个三维表示，其中x轴是时间，y轴是频率，每个点的颜色或强度代表特定时刻给定频率的幅度或功率。

> 音频信号被分段成重叠的帧。对每个帧计算 FFT，然后将结果堆叠以形成最终的语谱图。

### 在语谱图中读取语音

语谱图将音频信号变成图像，使我们能够看到语音独特的模式。让我们看一下单词“speech”的语谱图并识别其组成部分。



![示例语谱图](plots/7102-0.json)



> 单词“speech”的语谱图显示了频率随时间的变化。

- **元音：** 语谱图最显著的特征是被称为**共振峰**的水平条。它们是声道的共振频率，在语谱图中显示为深色水平条带。共振峰的模式和间距定义了不同的元音声音。例如，在单词“speech”中，初始元音 /i/ 以一组特定的共振峰频率为特征。
- **擦音：** 单词“speech”中的 /s/ 和 /ch/ 音是擦音的例子。这些辅音是通过将空气强行通过狭窄通道产生的，形成湍流、高频噪声。在语谱图上，它们显示为分散在高频范围内的弥散、混乱的能量团。
- **塞音：** 塞音是像 /p/、/t/ 和 /k/ 这样的辅音，它们是通过阻塞气流然后突然放开产生的。这会在语谱图上产生短暂而尖锐的能量爆发。

理解语谱图为我们提供了可视化语音语音学特征的强大工具。元音的水平共振峰、擦音的嘈杂云团以及塞音的突然爆发为每种声音提供了独特的视觉特征，使我们能够分析和处理语音信号。

## 参考资料

- [Discrete-Time Signal Processing](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHqFn6tteD_tULXOXRHIdlcl75wvdcA-xdSvxLAeh0Mhn2moD1cQffF3YWf_ac-dZhT70vP0jMFFvynAsWIs-t35Qvsz1XtG3Ue_6A_dsclhCMsZhEV3Dq8EM27_Tjm) — Alan V. Oppenheim, Ronald W. Schafer, and John R. Buck (2008)
  Publisher: Pearson; Pages: 1144
  这本基础教材对数字信号处理进行了全面的理论和实践阐述，包括傅里叶变换、短时傅里叶变换和窗函数技术的详细解释，这些对生成声谱图至关重要。
- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  一本现代权威教材，广泛涵盖语音处理，清晰解释了声谱图、通过短时傅里叶变换的生成过程，以及它们在可视化和解释语音音素特征（如元音共振峰和辅音特征）中的应用。
- [Automatic Speech Recognition - Lecture 3: Acoustic Features: MFCC and Spectrograms](https://ocw.mit.edu/courses/6-345-automatic-speech-recognition-spring-2003/lecture-notes/lecture-3/) — James R. Glass (2003)
  Publisher: Massachusetts Institute of Technology (MIT OpenCourseWare)
  这门麻省理工学院开放课程的讲座在语音识别背景下，对包括声谱图在内的声学特征进行了出色的学术介绍。它提供了对该主题的结构化教育视角。
