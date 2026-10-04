---
course: "introduction-to-speech-recognition"
sourceUrl: "https://apxml.com/zh/courses/introduction-to-speech-recognition/chapter-3-acoustic-modeling"
sourceId: 1240
chapter: "acoustic-modeling"
title: "声学模型"
order: 3
description: "了解声学模型如何将音频特征映射到音素等语言单位。内容包括GMM-HMM以及神经网络模型简介。"
hasQuiz: false
---

将原始音频转换为特征向量序列后，下一步是将这些特征映射到语言的基本语音单位。这是声学模型的主要作用。它回答了这样一个问题：给定一小段音频，其中特定音素（如 /k/、/æ/ 或 /t/）被说出的概率是多少？

声学模型提供了音频信号与其对应音素单位之间的统计关联。它计算似然度 $P(\text{audio_features} | \text{phoneme})$，这个概率是最终转录过程的重要输入。

在本章中，您将学习到：

*   声学模型在ASR（自动语音识别）流程中的作用。
*   传统的使用高斯混合模型（GMMs）来表示每个音素的音频特征的方法。
*   隐马尔可夫模型（HMMs）如何用于处理语音中的序列模式。
*   神经网络如何用于现代声学模型搭建的介绍。

最后，您将清楚了解这个组件如何将处理过的声音与语音的基本构成单位联系起来。
