---
course: "how-to-build-a-large-language-model"
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-15-distributed-training-strategies"
sourceId: 1093
chapter: "distributed-training-strategies"
title: "分布式训练策略"
order: 15
description: "阐述用于在多个设备上训练大型模型的基本并行化技术：数据并行、张量并行和流水线并行。"
hasQuiz: false
---

训练大型语言模型对现有硬件构成严峻挑战。对于拥有数十亿甚至数万亿参数的模型，存储参数、激活值和梯度经常超出单个加速器（如 GPU 或 TPU）的内存容量。同样地，计算成本，尤其是自注意力机制对于序列长度 $N$ 的二次方复杂度 $O(N^2)$，使得在单个设备上训练时间过长，难以接受。

为克服这些限制，我们必须将训练过程分布到多个计算设备上。本章介绍用于大型语言模型训练的基本并行化方法。

您将了解到：

*   **数据并行 (DP)：** 在不同设备上复制模型并分割数据批次。
*   **张量并行 (TP)：** 将单个模型层或张量拆分到不同设备上（层内并行）。
*   **流水线并行 (PP)：** 将模型层划分为不同阶段并将这些阶段分配给不同设备。
*   **混合方法：** 组合这些策略（例如，DP + TP，DP + PP）以获得最佳性能。
*   **通信开销：** 分析与不同并行方法相关的通信额外负担。

理解这些策略对于有效训练先进的语言模型必不可少。我们将考查每种方法的运作方式、优势和权衡。
