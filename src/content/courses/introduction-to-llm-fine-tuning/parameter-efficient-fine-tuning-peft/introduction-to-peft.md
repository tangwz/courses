---
course: "introduction-to-llm-fine-tuning"
chapter: "parameter-efficient-fine-tuning-peft"
lesson: "introduction-to-peft"
sourceId: 7281
sourceUrl: "https://apxml.com/zh/courses/introduction-to-llm-fine-tuning/chapter-4-parameter-efficient-fine-tuning-peft/introduction-to-peft"
title: "参数高效微调概述"
description: "了解PEFT方法在消费级硬件上微调超大型模型的理由和优势。"
order: 1
plots: []
sourceHash: "b44ff3a3d4c4dee17673709e96f4a640ec5f2d5cf9072c68e76c1765e2df9ece"
sourceCorrections: []
---

虽然全参数 (parameter)微调 (fine-tuning)是实现模型专业化的直接途径，但其资源需求是一大阻碍。更新数十亿参数需要大量GPU内存和计算时间，这使得该过程在许多开发环境中不切实际。例如，完全微调一个70亿参数的模型，仅存储模型权重 (weight)、梯度和优化器状态就需要超过80 GB的显存 (VRAM)，这一要求超出了大多数市售GPU的容量。

参数高效微调（PEFT）提供了一系列方法来解决这个计算瓶颈。PEFT的核心思路是冻结大部分预训练 (pre-training)模型的参数，并引入少量可管理的新可训练参数。这些新参数旨在有效引导模型行为以适应特定任务，同时不改变其权重中编码的原始知识。这种方法减少了微调过程中的内存和计算占用。

下方的图表说明了这两种微调理念的根本区别。在全微调中，基础模型的每个权重都是更新的候选对象。而在PEFT中，庞大的基础模型保持不变，只训练轻量级的补充组件。

> 训练方法的比较。全微调会修改所有模型权重，而PEFT则冻结基础模型，仅训练一小组适配器参数。

采用PEFT的理由不只在于管理资源限制。这类技术提供了多项重要优势，使得模型定制更为灵活和可扩展。

### 降低计算和内存成本

通过仅训练总参数 (parameter)的一小部分（通常小于模型大小的1%），PEFT显著降低了微调 (fine-tuning)的门槛。存储梯度和优化器状态所需的内存（这是全微调中高显存 (VRAM)占用的主要原因）按比例减少。这种效率使得在单个高端消费级或专业级GPU上微调超大型模型（例如700亿参数或更多）成为可能。因此，训练时间也大大缩短。

### 高效且便携的模型检查点

由于全微调 (fine-tuning)会修改整个模型，保存一个微调版本意味着存储其所有权重 (weight)的完整副本，这可能达到数十甚至数百千兆字节。使用PEFT，您只需保存一小组训练过的适配器权重。这些检查点通常只有几兆字节大小。这种便携性带来极大的操作便利。它使您能够维护基础模型的单个副本，并为不同任务应用不同的轻量级适配器，例如一个用于摘要，另一个用于代码生成，第三个用于客户支持对话。这种模块化方法简化了模型管理和部署流程。

### 减轻灾难性遗忘

当您在一个狭窄、针对特定任务的数据集上微调 (fine-tuning)模型时，它存在“遗忘”其在大量预训练 (pre-training)期间学到的通用知识的风险。这种现象被称为灾难性遗忘。由于PEFT方法保持原始模型权重 (weight)冻结，它们本身就能防止这种退化。模型的核心推理 (inference)和语言理解能力保持不变，而小型可训练模块则指导其输出与新任务匹配。这使得模型更加稳定和可靠，它在获得专业技能的同时，保留了通用能力。

PEFT并非单一技术，而是一系列方法的集合。在接下来的章节中，我们将介绍一些主要方法，主要关注低秩适应（LoRA）。我们还将简要介绍适配器微调（Adapter Tuning）和前缀微调（Prefix-Tuning）等其他策略，以便您对现有选择有一个更全面的了解。每种方法都以独特的方式引入可训练参数 (parameter)，但它们都有一个共同的目的，即以最小的计算代价实现高表现。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍了LoRA，一种通过将低秩矩阵注入Transformer架构来大幅减少可训练参数数量的方法。
- [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751) — Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, Sylvain Gelly (2019)
  Journal: International Conference on Machine Learning (ICML); DOI: [10.48550/arXiv.1902.00751](https://doi.org/10.48550/arXiv.1902.00751)
  提出了Adapter Tuning，这是一种早期的PEFT技术，其中将小的、特定于任务的神经网络模块（适配器）插入到预训练模型中。
- [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190) — Xiang Lisa Li, Percy Liang (2021)
  Journal: Annual Meeting of the Association for Computational Linguistics (ACL); DOI: [10.48550/arXiv.2101.00190](https://doi.org/10.48550/arXiv.2101.00190)
  描述了Prefix-Tuning，这是一种PEFT方法，它优化了一小段连续的、特定于任务的向量（前缀），并将其添加到输入之前。
- [Hugging Face PEFT Library Documentation](https://huggingface.co/docs/peft/en/index) — Hugging Face team ()
  Publisher: Hugging Face
  Hugging Face PEFT库的官方文档，提供了各种参数高效微调方法的实用指南和说明。
