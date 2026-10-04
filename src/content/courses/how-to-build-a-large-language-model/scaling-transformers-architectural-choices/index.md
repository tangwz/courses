---
course: "how-to-build-a-large-language-model"
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-11-scaling-transformers-architectural-choices"
sourceId: 1089
chapter: "scaling-transformers-architectural-choices"
title: "Transformer模型规模化：架构选择"
order: 11
description: "讨论Transformer模型规模化时的设计考量，包含深度与宽度、激活函数及稀疏注意力等。"
hasQuiz: false
---

在构建了一个Transformer模型之后，下一步是要弄明白如何有效地增大其规模。单纯地增大模型规模并非总是最佳途径；随着模型增长，特定的架构选择对性能、训练稳定性和计算需求有很大影响。

本章侧重讨论Transformer模型规模化过程中的设计考量。我们将讨论：

*   将模型大小、数据集大小和计算量与性能关联起来的经验性缩放法则，常表示为 $Performance \propto Compute^{\alpha} Data^{\beta} Size^{\gamma}$。
*   增加模型深度（层数）与宽度（隐藏维度大小）所带来的影响。
*   GeLU或SwiGLU等不同激活函数在前馈网络中的表现。
*   将层归一化置于残差连接之前或之后（Pre-LN对比Post-LN）所带来的影响。
*   对稀疏注意力机制的介绍，这些机制旨在处理自注意力在极长序列中的二次方复杂度问题。

在本章结束时，你将对设计更大、能力更强的Transformer模型时可用的架构调整手段，以及每种选择所关联的权衡取舍，有更清楚的认识。
