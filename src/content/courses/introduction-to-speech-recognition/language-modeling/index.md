---
course: "introduction-to-speech-recognition"
sourceUrl: "https://apxml.com/zh/courses/introduction-to-speech-recognition/chapter-4-language-modeling"
sourceId: 1242
chapter: "language-modeling"
title: "语言模型"
order: 4
description: "了解语言模型如何为ASR添加语言上下文，帮助区分听起来相似的短语。"
hasQuiz: false
---

声学模型可以根据一段音频确定音素的概率，但仅凭这一点不足以实现准确转录。例如，“recognize speech”和“wreck a nice beach”这两个短语听起来可能非常相似。声学模型可能会为这两种解释分配较高的概率。为了解决这种歧义，系统需要明白在给定语言中哪个词序更有可能出现。

本章介绍语言模型，它是负责为识别过程添加语言上下文的组成部分。通过为词序分配概率，语言模型帮助ASR系统从一组听起来相似的候选中选择最合理的转录。

您将了解以下内容：
*   语言模型在计算词序概率（即$P(\text{word sequence})$）方面的作用。
*   经典的N-gram模型，它们根据前N-1个词来估计一个词的概率，例如使用$P(w_n | w_{n-1})$的双词模型。
*   如何从大规模文本语料库计算概率。
*   困惑度，一个评估语言模型预测文本有效性的标准衡量指标。
*   简要介绍神经网络如何用于构建更先进的语言模型。
