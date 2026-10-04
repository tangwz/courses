---
course: "introduction-to-speech-recognition"
chapter: "foundations-of-speech-recognition"
lesson: "components-of-a-speech-recognition-pipeline"
sourceId: 7083
sourceUrl: "https://apxml.com/zh/courses/introduction-to-speech-recognition/chapter-1-foundations-of-speech-recognition/components-of-a-speech-recognition-pipeline"
title: "语音识别处理流程的组成要素"
description: "对ASR流程中标准组成部分的概述：特征提取、声学模型、语言模型和解码器。"
order: 3
plots: []
sourceHash: "00d3596c1f12244cf03b824c94343a07ce194808de45f9b95dcdc991835df1dc"
sourceCorrections: []
---

自动语音识别系统并非单一实体，而是由不同处理阶段协调运作构成的流程。可以将其想象成工厂的装配线，原材料从一端进入，成品从另一端产出。在ASR中，原材料是声波，成品是文本。装配线上的每个工位都有其专门任务，理解这些作用是理解ASR的第一步。

标准的ASR流程包含四个主要组成部分：特征提取、声学模型、语言模型和解码器。这些部分依次工作，系统地将复杂的音频信号转换为连贯的词语序列。

> 标准ASR流程图。音频数据从左到右流动，在每个阶段进行转换。

### 特征提取

流程的第一阶段处理原始音频信号。数字音频录音包含大量数据，这些数据里，许多对识别语音没有直接帮助。例如，它包括背景噪音、说话者的情感语调或录音伪影。特征提取部分的任务是处理这些原始音频，并将其提炼为紧凑而信息丰富的表示形式。

此过程将音频波形转换为一系列数值向量 (vector)，称为**特征向量**。每个向量概括了一小段音频（通常约25毫秒）的重要声学特性。这些特征旨在突出与语音相关的特性，例如能量在不同频率上的分布，同时丢弃不相关的信息。

ASR中一种非常常见的特征类型是**梅尔频率倒谱系数 (MFCC)**。我们将在第2章“处理音频信号”中详细说明如何创建MFCC和其他特征。目前，只需了解这第一步是清理和准备音频，以便进入流程的下一阶段。

### 声学模型

一旦我们有了一系列特征向量 (vector)，它们就会被传递给**声学模型 (AM)**。声学模型是系统中连接音频和语言单位的部分。它的主要任务是查看特征向量并确定它最可能代表哪种语言声音。

在语言学中，能够区分不同词语的最小声音单位称为**音素**。例如，“cat”中的/k/、/æ/和/t/声音就是音素。声学模型本质上充当语音转录器。它获取一段音频特征，并计算其属于每个可能音素的概率。

AM的输出不是单个音素，而是一组概率。对于给定的一小段音频，它可能会报告：

- 属于/s/的概率：0.7
- 属于/f/的概率：0.2
- 属于/z/的概率：0.1

该模型在大量转录音频上进行训练，学习声学特征与产生它们的声音之间的关联。在第3章“声学建模”中，我们将考察这些模型的构建方式。

### 语言模型

声学模型擅长识别单个声音，但它对语言或语境没有理解。它可能会混淆两个声学上相似的短语，例如“recognize speech”和“wreck a nice beach”。对AM而言，两种声音序列都看似合理。

这就是**语言模型 (LM)** 发挥作用的地方。语言模型为系统添加语言知识。它完全处理文本，无法访问音频。它的任务是计算给定词语序列的概率。它能告诉我们，短语“recognize speech”在英语中出现的可能性远大于“wreck a nice beach”。

通过分析大量文本，例如书籍和网页，LM学习哪些词语组合常见，哪些不常见。它回答了这个问题：“给定前一个词语，下一个词语的概率是多少？”这有助于ASR系统选择构成连贯句子的词语。我们将在第4章“语言建模”中考察语言模型如何工作。

### 解码器

最后一个组成部分是**解码器**，它充当决策者。解码器的任务是获取来自声学模型和语言模型的信息，并找到最可能生成原始音频的唯一词语序列。

这是一个复杂的搜索问题。解码器审查一个庞大的可能性之树。在每个时间点，声学模型提供可能的音素列表。这些音素形成可能的词语，词语形成可能的句子。解码器的任务是找到通过这个可能性空间中具有最佳综合得分的唯一路径。

任何给定句子假设的最终得分是以下两者的结合：

1. **声学得分：** 提议的句子与音频特征的匹配程度如何？（来自声学模型）
2. **语言得分：** 提议的句子是真实句子的可能性有多大？（来自语言模型）

解码器使用精密的搜索算法，例如维特比算法，以有效地找到使该综合得分最大化的句子。解码器的输出是最终的文本转录，即系统对原始语音的最佳猜测。我们将在第5章“解码与整体运用”中进一步讨论此过程。

## 参考资料

- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  涵盖语音识别基础知识的综合性教材，包括特征提取、声学模型、语言模型和解码算法的详细解释。
- [Automatic Speech Recognition: A Deep Learning Approach](https://link.springer.com/book/10.1007/978-3-319-95914-9) — Li Deng, Dong Yu (2014)
  Publisher: Springer; DOI: [10.1007/978-3-319-95914-9](https://doi.org/10.1007/978-3-319-95914-9)
  从现代视角概述ASR管道，将传统组件与深度学习进展相结合。
- [Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Sentences](https://ieeexplore.ieee.org/document/1163420) — Steven B. Davis and Paul Mermelstein (1980)
  Journal: IEEE Transactions on Acoustics, Speech, and Signal Processing; Publisher: IEEE; Volume: 28; Pages: 357-366; DOI: [10.1109/TASSP.1980.1163420](https://doi.org/10.1109/TASSP.1980.1163420)
  提出梅尔频率倒谱系数（MFCCs）的奠基性论文，MFCCs是语音识别中广泛使用的特征提取技术。
