---
course: "foundations-transformers-architecture"
sourceUrl: "https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-5-encoder-decoder-stacks"
sourceId: 539
chapter: "encoder-decoder-stacks"
title: "编码器和解码器堆栈"
order: 5
description: "剖析完整的Transformer架构，包括编码器层、解码器层、层归一化和前馈网络。"
hasQuiz: false
---

确立了自注意力机制和位置编码的要点后，我们现在着重关注这些组成部分如何整合到完整的Transformer模型中。本章考察编码器和解码器堆栈的架构，它们是模型的基本组成部分。

您将学习如何：

*   识别标准Transformer编码器层的组成部分，包括多头自注意力机制和逐位置前馈网络。
*   分析解码器层的结构，着重于遮蔽自注意力机制、编码器-解码器交叉注意力机制和前馈子层。
*   理解遮蔽自注意力机制在序列生成过程中阻止未来标记信息流动的作用。
*   掌握交叉注意力机制如何使解码器通过编码器的输出来关注输入序列的相关部分。
*   认识到残差连接（常表示为 $Add$）和层归一化（$Norm$）在促进信息传播和稳定深度网络训练中的作用和实现方式。
*   明白堆叠多个相同层如何形成深层的编码器和解码器结构。
*   追溯数据流如何从输入嵌入经过编码器和解码器堆栈到达最终输出概率。
