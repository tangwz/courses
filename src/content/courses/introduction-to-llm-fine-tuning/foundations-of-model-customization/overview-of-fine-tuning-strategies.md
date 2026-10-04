---
course: "introduction-to-llm-fine-tuning"
chapter: "foundations-of-model-customization"
lesson: "overview-of-fine-tuning-strategies"
sourceId: 7265
sourceUrl: "https://apxml.com/zh/courses/introduction-to-llm-fine-tuning/chapter-1-foundations-of-model-customization/overview-of-fine-tuning-strategies"
title: "微调策略概述"
description: "对不同微调方法（包括全参数微调和参数高效方法）的概述。"
order: 4
plots: []
sourceHash: "93f5e5fd9eb24ee55ecf5aa5f59b9cdd0aa0001784e767173776aae55f901bb7"
sourceCorrections: []
---

在决定调整预训练 (pre-training)模型后，您必须选择一种微调 (fine-tuning)策略。这一决定直接影响您所需的计算资源、模型训练所需时间以及最终产物的特性。两种主要方法位于资源消耗和参数 (parameter)修改数量所界定的范围两端。

### 全参数 (parameter)微调 (fine-tuning)

全参数微调，通常简称为“微调”，是最直接的方法。在此方法中，您加载一个预训练 (pre-training)模型，并在自定义数据集上继续训练过程，更新模型中的每个权重 (weight)和偏差。可以将其看作是调整整个神经网络 (neural network)的所有连接，使其更好地适应您的特定任务。

这种策略很有效，因为它给予模型最大的灵活性来适应新数据。如果您的任务数据分布与预训练数据显著不同，允许所有参数变化可以带来更高的性能。然而，这伴随着高昂的代价：

- **高计算需求：** 更新数十亿参数需要大量的GPU内存。模型的权重、它们的梯度以及优化器状态都必须存放在内存中，通常需要多块高端GPU。
- **存储要求：** 由于每个参数都被修改，生成的微调模型与原始模型大小相同。如果您微调一个70亿参数的模型，您必须保存一个新的70亿参数模型（`bfloat16` 权重约14 GB）。
- **灾难性遗忘的风险：** 当您在一个小型、狭窄的数据集上训练整个模型时，它可能会失去预训练期间获得的一些通用知识。这种现象被称为灾难性遗忘，即模型变得高度专业化，但在该专业领域之外的任务上表现不佳。

当您拥有充足的计算资源且目标是在单一、明确的任务上获得尽可能高的性能时，全参数微调是最适合的。

### 参数 (parameter)高效微调 (fine-tuning) (PEFT)

参数高效微调（PEFT）方法为全参数微调的资源密集型需求提供了一种实用替代方案。PEFT的核心理念是冻结预训练 (pre-training)模型的大部分参数，并只训练极少量新增或现有参数。这显著减少了训练过程的内存和计算占用。

您不是修改整个模型，而是在原始架构中注入小型、可训练的模块或“适配器”。只有这些适配器（可能只占总参数量的0.1%以下）在训练期间进行更新。基础模型的原始权重 (weight)保持不变。

这种方法具有多项优势：

- **低计算需求：** 由于您只更新极小部分的参数，梯度和优化器状态的内存需求显著降低。这使得在单个消费级GPU上微调大型模型成为可能。
- **高效存储：** PEFT过程的输出只是一小组训练好的适配器权重，通常只有几兆字节大小。您可以维护一个大型基础模型的副本，并为各种任务应用不同的轻量级适配器。
- **减轻灾难性遗忘：** 通过保持基础模型冻结，PEFT有助于保留模型原有的通用能力，从而降低灾难性遗忘的风险。

下图说明了这两种策略之间的根本区别。

> 微调方法的对比。全参数微调修改所有权重，形成一个新的大型模型。PEFT仅修改小型新增模块，并保持基础模型冻结。

### 策略对比

全参数 (parameter)微调 (fine-tuning)和PEFT之间的选择涉及一系列权衡。下表概括了主要差异，以协助您做出决定。

| 特性 | 全参数微调 | 参数高效微调 (PEFT) |
| --- | --- | --- |
| **更新参数** | 全部 (100%) | 小部分 (< 1%) |
| **GPU内存需求** | 非常高 | 低 |
| **存储成本** | 高（模型完整副本） | 低（仅小型适配器权重 (weight)） |
| **训练速度** | 较慢 | 较快 |
| **灾难性遗忘** | 风险较高 | 风险较低 |
| **任务便携性** | 每个任务一个模型 | 一个基础模型，多个轻量级适配器 |

PEFT不是单一技术，而是一系列方法。其中最常用的是低秩适应（LoRA），它涉及将可训练的低秩矩阵注入到Transformer层中。其他方法包括添加新瓶颈层的适配器微调（Adapter Tuning），以及向输入序列添加可训练前缀的前缀微调（Prefix-Tuning）。我们将在第四章详细实现LoRA。

最终，您对微调策略的选择将取决于您项目的限制和目标。如果您硬件有限且需要支持多项任务，PEFT是一个很好的选择。如果您需要单一应用的最高性能并拥有充足的计算能力，全参数微调可能是更好的途径。本课程的后续章节将为您提供实践技能来实现这两种方法。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍了低秩适配（LoRA），一种重要的参数高效微调方法。
- [PEFT: State-of-the-art Parameter-Efficient Fine-Tuning](https://huggingface.co/docs/peft/en/index) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face PEFT库的官方文档，提供了实践指导和实现细节。
- [Overcoming catastrophic forgetting in neural networks](https://www.pnas.org/doi/10.1073/pnas.1611835114) — James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, Raia Hadsell (2017)
  Journal: Proceedings of the National Academy of Sciences; Publisher: United States National Academy of Sciences; Volume: 114; Pages: 3521-3526; DOI: [10.1073/pnas.1611835114](https://doi.org/10.1073/pnas.1611835114)
  一篇关于灾难性遗忘问题及其神经网络中缓解技术的奠基性论文。
