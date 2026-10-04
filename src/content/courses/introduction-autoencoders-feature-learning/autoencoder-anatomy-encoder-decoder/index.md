---
course: "introduction-autoencoders-feature-learning"
sourceUrl: "https://apxml.com/zh/courses/introduction-autoencoders-feature-learning/chapter-2-autoencoder-anatomy-encoder-decoder"
sourceId: 1148
chapter: "autoencoder-anatomy-encoder-decoder"
title: "自动编码器的构成：编码器与解码器"
order: 2
description: "详细阐述自动编码器中的编码器和解码器、它们如何压缩/重建数据以及典型的激活函数。"
hasQuiz: true
---

在介绍了自动编码器及其目标之后，本章将侧重于它们的结构构成。我们将把这种架构拆解成两个主要部分：编码器和解码器。您将看到编码器如何处理输入数据以生成压缩的低维概括表示，以及解码器如何使用此概括表示尝试重建原始输入。

具体来说，我们将介绍：
*   编码器的作用：降低数据维度，以及涉及的典型层。
*   瓶颈层（或潜在空间）：它作为存储压缩数据表示的层的功能，通常表示为 $z$。
*   解码器的作用：从 $z$ 重建数据，以及它与编码器通常对称的结构。
*   激活函数，例如用于隐藏层的 $ReLU$ 和用于输出层的 $Sigmoid$，以及它们的具体职能。
*   数据的完整路径，从输入 $X$ 经过编码器到 $z$，再经过解码器到重建输出 $X'$。

通过理解这些核心架构组成，您将能更好地明白自动编码器如何学习有用的数据表示并完成诸如降维等任务。
