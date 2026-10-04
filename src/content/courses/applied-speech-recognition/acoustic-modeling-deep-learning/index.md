---
course: "applied-speech-recognition"
sourceUrl: "https://apxml.com/zh/courses/applied-speech-recognition/chapter-3-acoustic-modeling-deep-learning"
sourceId: 1243
chapter: "acoustic-modeling-deep-learning"
title: "基于深度神经网络的声学建模"
order: 3
description: "学习使用循环神经网络 (RNN)、长短期记忆 (LSTM) 网络和连接主义时间分类 (CTC) 损失函数来构建 ASR 声学模型。"
hasQuiz: false
---

将原始音频转换为结构化特征向量后，下一步是构建一个模型，将这些特征映射到它们对应的文本转录。这是声学模型的主要作用，它是学习语音与字符等语言单元之间关联的组成部分。

本章介绍深度神经网络在声学建模中的使用。我们将从分析循环神经网络 (RNN) 及其更强大的变体——长短期记忆 (LSTM) 网络和门控循环单元 (GRU) 网络——如何构建以处理序列数据开始。

此过程中的一个主要难点在于输入特征序列和输出文本序列的长度不同，并且没有明确的对齐关系。我们将使用连接主义时间分类 (CTC) 损失函数来解决这个问题。CTC 是一种机制，它允许网络自动学习对齐，从而简化训练过程。目标是训练一个模型，将输入特征序列 $X = (x_1, x_2, ..., x_T)$ 映射到目标字符序列 $Y = (y_1, y_2, ..., y_N)$，其中输入长度 $T$ 通常不等于输出长度 $N$。

本章结束时，你将学习到如何：
*   使用 LSTM 等循环架构构建声学模型。
*   应用 CTC 损失函数在未对齐数据上训练模型。
*   实现基于 CTC 模型的完整训练流程。

本章以一个动手实践环节作结，在此环节中，你将构建并训练一个基本的 LSTM-CTC 声学模型。
