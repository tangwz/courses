---
course: "rnns-and-sequence-modeling"
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-8-preparing-sequence-data"
sourceId: 617
chapter: "preparing-sequence-data"
title: "为循环神经网络准备序列数据"
order: 8
description: "学习对文本和时间序列数据进行预处理的重要方法，以有效应用于 RNN、LSTM 和 GRU 模型。"
hasQuiz: true
---

循环神经网络、LSTM 和 GRU 模型需要数值张量格式的输入数据，通常具有像 $(batch, time\_steps, features)$ 这样的特定形状。然而，实际中的序列数据，例如原始文本或时间序列测量值，很少以这种即用型格式存在。本章侧重于弥补这一差距的必要步骤。

您将学到将序列数据转换为适用于循环模型的结构的常用方法。对于文本数据，这包括以下步骤：

*   **分词**: 将文本拆分为独立的单元（词或字符）。
*   **词汇表构建**: 建立从词元到唯一整数的映射。
*   **整数编码**: 将词元序列转换为数值序列。
*   **嵌入**: 使用嵌入层将整数表示为密集向量（在此处介绍，通常作为模型的一部分进行训练）。
*   **填充与掩码**: 处理批次中长度不同的序列。

对于时间序列数据，我们将介绍标准化和创建观测值滑动窗口等方法。最后，我们将讨论如何将准备好的数据组织成批次，以便进行高效的模型训练。学完本章，您将能够构建数据管道，对常见序列数据集进行预处理，以便输入到循环神经网络中。
