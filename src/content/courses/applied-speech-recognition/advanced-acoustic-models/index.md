---
course: "applied-speech-recognition"
sourceUrl: "https://apxml.com/zh/courses/applied-speech-recognition/chapter-4-advanced-acoustic-models"
sourceId: 1245
chapter: "advanced-acoustic-models"
title: "高级声学模型与架构"
order: 4
description: "介绍高级ASR架构，包括带注意力机制的序列到序列模型、Transformer，以及如何使用Wav2Vec 2.0等预训练模型。"
hasQuiz: false
---

上一章中，我们使用LSTM和连接时序分类（CTC）损失函数构建了声学模型。尽管CTC是一种可行的方法，但它做了一个强假设，即每个时间步的输出预测是条件独立的。本章将介绍现代架构，这些架构直接建模输出字符或词语间的依赖关系，从而带来更准确、考虑语境的转录。

您将从注意力机制开始学习，它允许模型在生成转录本的每个部分时，选择性地关注输入音频中的相关片段。在此基础上，您将了解序列到序列（Seq2Seq）架构，例如Listen, Attend, and Spell (LAS) 模型，这些模型能将输入音频序列直接映射到输出文本序列。

随后，我们将介绍Transformer架构及其在音频处理中自注意力机制的使用。您将了解到这种设计如何捕捉整个音频输入中的依赖关系。我们还将考察Conformer模型，它是一种混合架构，结合了用于局部特征提取的卷积和Transformer的全局语境建模能力。

本章最后将概述Wav2Vec 2.0等大型预训练ASR模型。您将学习如何在特定数据集上微调这些模型，这是一种获得高表现的常用且有效方法。实践部分将指导您完成从Hugging Face库微调预训练模型的过程，让您直接体验领先的ASR工作流程。
