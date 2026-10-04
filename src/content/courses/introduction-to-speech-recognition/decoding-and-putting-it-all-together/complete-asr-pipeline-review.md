---
course: "introduction-to-speech-recognition"
chapter: "decoding-and-putting-it-all-together"
lesson: "complete-asr-pipeline-review"
sourceId: 7133
sourceUrl: "https://apxml.com/zh/courses/introduction-to-speech-recognition/chapter-5-decoding-and-putting-it-all-together/complete-asr-pipeline-review"
title: "完整的ASR流程：回顾"
description: "对所有组成部分（特征提取、声学模型、语言模型和解码器）如何共同运作的全面回顾。"
order: 5
plots: []
sourceHash: "03b94c9c903de0fca10a267b27e3f90bfc9cc1855e2607e50fd4ef64fc1c5a82"
sourceCorrections: []
---

在单独了解了各个组成部分之后，我们现在可以组建完整的自动语音识别流程。理解这些部分如何共同运作是了解机器如何将语音转换为文字的根本。每个阶段都有其独特的职责，一个阶段的输出会成为下一个阶段的输入，最终形成转录文本。

整个过程是一个复杂的概率估计与查找序列。其核心在于，系统试图回答一个问题：“给定此音频，最有可能的词语序列是什么？”

让我们再次回顾整个工作流程，从声波进入系统的那一刻起，直到句子出现。

### ASR流程：从声音到文字

下图说明了语音识别系统的标准架构。我们将在后续部分逐一分解每个步骤。

> 数据在标准自动语音识别流程中的流动，从初始音频到最终文本输出。

#### 1. 音频输入与特征提取

该过程始于原始音频信号，它是声波的数字表示。正如我们在第2章所见，这种原始数据不适合机器学习 (machine learning)模型。它包含过多信息，其中大部分与区分语音无关。

自动语音识别的完整流程将各个组件整合在一起。一个主要组件是**特征提取**，它将音频转换为紧凑且信息丰富的表示。音频被分割成短而重叠的帧（通常长25毫秒）。对于每个帧，系统计算一组特征，其中梅尔频率倒谱系数（MFCCs）是最常见的选择。结果是一个特征向量 (vector)序列，其中每个向量都是音频小片段频率内容的数值概括。

#### 2. 声学模型：听起来像什么？

特征向量序列被输入到**声学模型（AM）**。声学模型的任务（我们在第3章中已讲到）是确定在给定特定语言单位（如音素）的情况下，这些音频特征出现的可能性。

对于每个音频帧，AM计算该语言中所有可能音素的概率。例如，对于一个10毫秒的音频片段，它可能会给出：

- 此为音素 /s/ 的概率是 0.8
- 此为音素 /t/ 的概率是 0.1
- 此为音素 /f/ 的概率是 0.05
- ...等等。

输出是一个概率流，它代表着所说声音的基本构成。

#### 3. 语言模型：是否合理？

同时，\*\*语言模型（LM）\*\*提供了一个完全不同的信息来源。正如第4章所讨论的，语言模型对音频一无所知。它唯一的目的是计算给定词语序列的概率。它在大量文本上训练，并学习哪些词语可能出现在其他词语之后。

例如，N-gram语言模型会给"recognize speech"这个短语赋予比发音相似但无意义的"wreck a nice beach"更高的概率。它提供了解决歧义所需的语言背景。

#### 4. 解码器：结合依据并寻找最佳路径

所有部分在此汇合。**解码器**是驱动最终转录查找的引擎。它接收三个输入：

1. 来自声学模型的**声学概率**。
2. 来自语言模型的**语言概率**。
3. 一个**发音词典**，它是一个将词语映射到其音素序列的词典（例如，"cat" -> /k/ /ae/ /t/）。

解码器的任务是找到最能解释输入音频特征（$O$）的单个词语序列（$W$）。它通过找到最大化声学模型和语言模型分数乘积的序列来完成此操作，这在语音识别的基本方程中有所体现：


$$
\hat{W} = \underset{W}{\mathrm{argmax}} \, P(O|W) \times P(W)
$$


- $P(O|W)$ 是声学模型分数：“给定此词语序列（及其词典中对应的音素），听到这些音频特征的概率是多少？”
- $P(W)$ 是语言模型分数：“此词语序列在语言中出现的概率是多少？”

由于可能的词语序列数量非常多，解码器使用维特比算法等高效搜索算法来剪除不太可能的路径，并在不评估每一种可能性的情况下找到最优结果。

#### 5. 最终输出

查找完成后，解码器输出单个最有可能的词语序列。这是用户看到的最终转录文本。该序列是竞争的胜利者，是声学模型和语言模型综合得分最高的假设。从声波到句子，该流程已成功将语音转换为文字。

## 参考资料

- [Fundamentals of Speech Recognition](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHMCvNrEqJupELfx8vIKCD-YaBJ0PL_fz-dajIdQydnPOmbs4llXqfcHJG_KbGMV3IibTuR2k4caWIHJRrE1GKI1u1AkHXXrrAguzgMdfxbVq6uHqufkodOg5nLxT70MXkA_Mp01iHeMzETRl7Ym3iY4ogsKFnMXa8gBfCinKbJ62Dy9hHpzq6uiMsL9JnVB7ybzD-F833F) — Lawrence R. Rabiner, Biing-Hwang Juang (1993)
  Publisher: Prentice Hall; Pages: 507
  一本基础性教科书，严格地从数学角度处理统计语音识别，特别是隐马尔可夫模型（HMMs）和维特比算法。
