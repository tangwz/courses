---
course: "introduction-to-transformer-models"
sourceUrl: "https://apxml.com/zh/courses/introduction-to-transformer-models/chapter-3-transformer-encoder-decoder-architecture"
sourceId: 666
chapter: "transformer-encoder-decoder-architecture"
title: "Transformer 编码器-解码器架构"
order: 3
description: "了解 Transformer 的完整架构，包括编码器/解码器堆栈、位置编码和残差连接。"
hasQuiz: true
---

在上一章了解了核心注意力机制后，我们将这些部件组合起来，以构建完整的 Transformer 架构。本章将详细介绍模型的结构，解释编码器和解码器堆栈如何在序列到序列任务中配合工作。

您将了解以下内容：

*   整体编码器-解码器布局。
*   输入处理，包括词元嵌入以及使用正弦和余弦函数等位置编码技术加入位置信息。
*   编码器层的结构，包含多头自注意力模块和逐位置前馈网络。
*   解码器层的结构，包括遮蔽多头自注意力、编码器-解码器注意力和前馈网络。
*   残差连接和层归一化（`Add & Norm`）在稳定网络和改善梯度流动中的作用。
*   解码器输出如何通过线性层和 Softmax 函数转换为最终词元概率。
*   一个实践实现部分，重点在于构建单个编码器层。

在本章结束时，您将理解这些不同部分如何组合形成完整的 Transformer 模型，以及每个组件设计背后的原理。
