---
course: "speech-recognition-synthesis-asr-tts"
sourceUrl: "https://apxml.com/zh/courses/speech-recognition-synthesis-asr-tts/chapter-2-advanced-acoustic-modeling-asr"
sourceId: 684
chapter: "advanced-acoustic-modeling-asr"
title: "高级ASR声学建模"
order: 2
description: "运用深度神经网络和端到端架构，构建语音识别中精密的声学模型。"
hasQuiz: false
---

声学建模是自动语音识别（ASR）系统的核心部分，它负责将输入的音频特征映射成音素或字符等语言单元。承接上一章内容，本章将介绍支撑当前先进ASR的复杂深度学习架构。

本章将详细介绍现代声学模型的架构和训练方法。你将学习：

*   **混合HMM-DNN系统：** 了解深度神经网络如何与隐马尔可夫模型结合。
*   **端到端模型：** 实现和分析直接将语音映射到文本的架构，包括连接时序分类（CTC）、基于注意力的编码器-解码器、RNN转导器（RNN-T）和基于Transformer的模型。
*   **高级训练策略：** 诸如SpecAugment和多任务学习等技术，用于提高模型泛化能力。
*   **解码算法：** 比较这些不同模型类型中使用的各种搜索策略，例如波束搜索。

我们将分析每种方法相关的数学公式、实现细节和权衡。本章包含一个实践环节，你将构建并训练一个端到端ASR模型，将这些思想付诸实践。
