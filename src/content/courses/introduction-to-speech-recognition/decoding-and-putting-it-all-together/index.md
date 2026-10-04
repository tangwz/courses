---
course: "introduction-to-speech-recognition"
sourceUrl: "https://apxml.com/zh/courses/introduction-to-speech-recognition/chapter-5-decoding-and-putting-it-all-together"
sourceId: 1244
chapter: "decoding-and-putting-it-all-together"
title: "解码与系统集成"
order: 5
description: "学习解码器如何结合声学模型和语言模型的信息，以找到最有可能的文本转录结果。"
hasQuiz: false
---

在前面的章节中，你学习了如何构建声学模型（将音频特征映射到音素）和语言模型（用于确定词语序列的可能性）。本章将介绍最后一步：将这两种信息来源结合起来，以生成最有可能的文本转录结果。

语音识别的主要任务是找到在给定观测音频特征（$O$）下，最有可能的词语序列（$W$）。这通常表示为找到使两个概率乘积最大化的序列：

$$
\hat{W} = \underset{W}{\mathrm{argmax}} \, P(O|W) \times P(W)
$$

这里，$P(O|W)$ 是声学模型给出的概率，而 $P(W)$ 是语言模型提供的概率。负责此计算和搜索的组件是解码器。

你将学习解码器如何作为一个搜索算法发挥作用，在庞大的可能句子空间中寻找最优解。我们将介绍维特比算法等搜索策略背后的原理。之后，我们将回顾 ASR 的完整流程。本章最后，你将学习如何使用词错率（WER）评估系统性能，并找出影响转录准确性的常见难题。
