---
course: "fine-tuning-small-language-model"
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-small-language-model/chapter-1-principles-of-small-language-models"
sourceId: 1501
chapter: "principles-of-small-language-models"
title: "小语言模型原理"
order: 1
description: "了解小语言模型的架构以及监督微调背后的运作机制。"
hasQuiz: false
---

小语言模型 (SLM) 为拥有数千亿参数的庞大网络提供了另一种方案。它们可以在本地运行，并在严格的硬件限制下操作。在编写微调模型的代码之前，需要建立对这些架构运作方式以及监督学习如何更新其内部权重的基本认识。

本章将讲解小语言模型的具体运行机制。我们首先会明确 SLM 的定义，并将微调与检索增强生成 (RAG) 等其他方式进行对比。

你将分析监督学习的数学逻辑。以梯度下降中基本的权重更新规则为例，其表示为：

$$w_{t+1} = w_t - \alpha \nabla L(w_t)$$

这里，$w_t$ 代表模型在特定步骤的权重，$\alpha$ 是学习率，$\nabla L(w_t)$ 是损失函数的梯度。理解这些机制能让你弄清楚将自定义数据输入预训练网络时的实际变化。

我们还将计算具体的硬件配置需求。你将学会如何预估加载和训练这些模型所需的显存 (VRAM)，以防出现显存不足的错误。最后，你将编写一个 Python 脚本，将预训练的 SLM 加载至内存并进行基础的文本生成。
