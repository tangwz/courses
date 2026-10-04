---
course: "how-to-build-a-large-language-model"
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-17-optimization-algorithms-llms"
sourceId: 1095
chapter: "optimization-algorithms-llms"
title: "LLMs的优化算法"
order: 17
description: "讨论AdamW等优化器、学习率调度（预热、衰减）以及适合LLMs的梯度裁剪方法。"
hasQuiz: false
---

训练大型语言模型面临着特有的优化难题，因为它们规模庞大且计算需求高。标准的优化方法通常需要进行调整，以便有效地训练LLMs。本章将讲解常用到的优化算法和策略。

我们会先简要回顾梯度下降方法，然后把重点放在自适应优化器上，例如Adam和AdamW，并解释解耦权重衰减的原理。您将学习如何实现包含预热和衰减阶段的常见学习率调度策略 (例如：$$ \eta_{t} = \text{schedule}(\text{step}) $$)。我们还会介绍梯度裁剪，这是一种常用来防止梯度爆炸并提升训练稳定性的方法，通常通过重新缩放那些范数超出阈值$c$的梯度来实现：$$ \mathbf{g} \leftarrow \frac{c}{\|\mathbf{g}\|} \mathbf{g} \text{ if } \|\mathbf{g}\| > c $$ 最后，我们将谈到选择优化器重要超参数的实用指导，例如学习率$\eta$、Adam的动量项($\beta_1, \beta_2$)、数值稳定性项$\epsilon$以及权重衰减系数$\lambda$。
