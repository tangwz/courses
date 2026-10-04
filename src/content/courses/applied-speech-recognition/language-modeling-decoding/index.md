---
course: "applied-speech-recognition"
sourceUrl: "https://apxml.com/zh/courses/applied-speech-recognition/chapter-5-language-modeling-decoding"
sourceId: 1247
chapter: "language-modeling-decoding"
title: "语言模型与解码"
order: 5
description: "整合语言模型以提高转录准确性，并了解贪心搜索和束搜索等解码算法。"
hasQuiz: false
---

在前面的章节中，我们构建了将音频特征映射到字符概率序列的声学模型。虽然这些模型在识别语音内容方面很有效，但它们的输出可能在声学上听起来合理，但在语言上不正确。例如，模型可能会将“recognize speech”转录为发音相似的“wreck a nice beach”。

这就是引入语言模型 (LM) 的地方。语言模型根据其语法结构和出现可能性来评估一个词语序列，帮助系统区分合理与不合理的转录。使用语言模型指导从声学模型预测中选择最终文本的过程称为**解码**。解码器的目标是找到使组合得分最大化的词语序列 $W$，这通常是声学和语言模型概率的加权和：

$$
\text{score}(W) = \log P_{\text{Acoustic}}(X|W) + \alpha \log P_{\text{Language Model}}(W)
$$

这里，$P_{\text{Acoustic}}(X|W)$ 是声学模型在给定词语序列 $W$ 时分配给音频特征 $X$ 的概率。$P_{\text{Language Model}}(W)$ 项是词语序列本身的概率，而 $\alpha$ 是一个平衡两种模型影响的权重。

本章讲述了将语言模型整合到 ASR 系统中的理论与实践。您将学习：

*   阐述语言模型在 ASR 流程中的作用。
*   使用 KenLM 工具包从文本语料库构建统计 N-gram 语言模型。
*   比较简单的贪心搜索解码与更有效的束搜索算法。
*   实现一个束搜索解码器，该解码器引入外部语言模型的评分以提高转录准确性。
