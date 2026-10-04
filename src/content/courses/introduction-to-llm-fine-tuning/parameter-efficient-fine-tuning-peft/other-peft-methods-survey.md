---
course: "introduction-to-llm-fine-tuning"
chapter: "parameter-efficient-fine-tuning-peft"
lesson: "other-peft-methods-survey"
sourceId: 7285
sourceUrl: "https://apxml.com/zh/courses/introduction-to-llm-fine-tuning/chapter-4-parameter-efficient-fine-tuning-peft/other-peft-methods-survey"
title: "其他PEFT方法：概述"
description: "简要介绍替代PEFT方法，包括基于适配器的方法和前缀调整。"
order: 5
plots: []
sourceHash: "ad5c0ea0ffb3804b1adefa35386e66fbc28109bd35bd68930f5daefdd68665ed"
sourceCorrections: []
---

参数 (parameter)高效微调 (fine-tuning)（PEFT）提供了多种创新的方法，用于高效的模型调整。低秩适配（LoRA）是一种高效且广受欢迎的PEFT技术，但还有其他几种方法。这些技术在修改策略上有所不同：有些（如LoRA）改变现有权重 (weight)矩阵，而另一些则向模型添加新组件或直接操作输入。理解这些替代方案能扩展您用于使大型语言模型（LLMs）适应特定限制和任务的工具范围。

本节简要介绍了两种主要的替代PEFT系列：基于适配器的方法以及学习连续提示的方法，例如Prefix-Tuning和Prompt-Tuning。

### 适配器调整：注入新层

最早且最直接的PEFT方法之一是**适配器调整**。其核心思路很简单：将所有预训练 (pre-training)的模型权重 (weight)完全冻结，并在每个transformer块内部注入新的、小的、可训练的神经网络 (neural network)模块。这些模块被称为“适配器”。

一个适配器通常包含一个瓶颈结构：

1. 一个前馈下投影层，它将transformer子层（如注意力或前馈网络）的高维输出降低到小得多的维度。
2. 一个非线性激活函数 (activation function)，例如GELU或ReLU。
3. 一个前馈上投影层，它将特征恢复到其原始维度。
4. 一个残差连接，将适配器的输出添加回transformer子层的原始、未修改的输出中。

在微调 (fine-tuning)期间，只有这些新添加的适配器模块的参数 (parameter)会被更新。由于瓶颈维度（例如32或64）与模型的隐藏维度（例如4096）相比非常小，因此可训练参数的数量只占总数的一小部分。

> 一个适配器模块，插入在transformer块的前馈网络之后。原始模型权重（蓝色）是冻结的，而只有适配器（橙色）被训练。

适配器的主要优点是它们的模块化。您可以为不同的任务训练独立的适配器，并在推理 (inference)时简单地“插入”相应的适配器，而无需修改底层模型。这对于提供多个定制模型来说非常高效。然而，这种方法可能会引入少量的推理延迟，因为它增加了必须遍历的额外网络层。

### 提示和前缀调整：修改输入

与其修改模型的架构，另一类方法则侧重于操作模型的输入。**提示调整**（Prompt-Tuning）和**前缀调整**（Prefix-Tuning）是两个主要例子，它们通过学习一种“软提示”来引导模型的行为。

与您在使用ChatGPT等模型时使用的离散文本提示不同，软提示是连续数值向量 (vector)的序列。这些向量被前置到输入嵌入 (embedding)中，并通过反向传播 (backpropagation)进行训练，以优化模型针对特定任务的输出。整个语言模型保持冻结。

- **提示调整**是最简单的形式。它仅向输入嵌入层添加一系列可训练向量。这些向量被视为虚拟的token，提供特定任务的上下文 (context)。它的参数 (parameter)效率极高，有时即使对于数十亿参数的模型也只需几千个可训练参数。
- **前缀调整**是更具表现力的变体。它不仅仅是在输入嵌入中添加一个可训练前缀，而是在transformer的*每个*注意力层中，在key和value向量的开头插入一个独特的、可训练的前缀。这使得模型能够更直接地、逐层控制其内部表示和生成过程。

> 输入修改方法的比较。提示调整向输入嵌入添加一个可训练的软提示，而前缀调整则将可训练向量插入到每个注意力块中。

这些方法非常轻量。由于您只为每个任务训练和存储小的软提示向量，因此您可以为数百个不同任务定制一个单一的冻结底层模型，存储开销极小。主要的挑战在于，它们的性能有时会落后于LoRA或完全微调 (fine-tuning)等更具侵入性的方法，因为它们对模型的内部计算影响较小。

### 选择合适的PEFT方法

最适合的PEFT方法取决于您的具体目标，需要平衡性能要求与计算和存储限制。LoRA通常能在性能和效率之间提供良好平衡，这也促成了它的普及。然而，适配器为多任务部署提供了更强的模块化特性，而基于提示的方法在参数 (parameter)效率方面是无与伦比的。

以下是讨论过的技术比较汇总：

| 方法 | 核心思想 | 可训练参数 | 推理 (inference)延迟 | 主要优点 |
| --- | --- | --- | --- | --- |
| **LoRA** | 将权重 (weight)更新矩阵分解为低秩因子。 | 非常低 | 无（可合并） | 性能卓越，参数效率高。 |
| **适配器调整** | 在冻结层之间注入小的可训练模块。 | 非常低 | 少量增加 | 模块化程度高；易于为不同任务切换适配器。 |
| **提示调整** | 学习一个前置到输入的连续“软提示”。 | 极低 | 可忽略 | 存储需求极小；非常适合将一个模型用于多种任务。 |
| **前缀调整** | 为每个注意力块中的key/value学习连续前缀。 | 极低 | 可忽略 | 比提示调整更具表现力，同时保持高效率。 |

## 参考资料

- [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751) — Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, Sylvain Gelly (2019)
  Journal: Proceedings of the 36th International Conference on Machine Learning (ICML); DOI: [10.48550/arXiv.1902.00751](https://doi.org/10.48550/arXiv.1902.00751)
  介绍了用于高效迁移学习的适配器模块，通过冻结预训练权重并插入小型可训练层来展示其有效性。
- [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://aclanthology.org/2021.acl-long.353/) — Li, Xiang Lisa and Liang, Percy (2021)
  Journal: Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers); Publisher: Association for Computational Linguistics; Pages: 4582-4597; DOI: [10.18653/v1/2021.acl-long.353](https://doi.org/10.18653/v1/2021.acl-long.353)
  提出了一种参数高效的微调方法--Prefix-Tuning，该方法在每个Transformer层注意力机制的输入前添加可训练的连续前缀。
