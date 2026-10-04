---
course: "fine-tuning-adapting-large-language-models"
chapter: "llm-adaptation-foundations"
lesson: "overview-fine-tuning-approaches"
sourceId: 2968
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-adapting-large-language-models/chapter-1-llm-adaptation-foundations/overview-fine-tuning-approaches"
title: "微调方法概述"
description: "介绍微调方法的各种方式，从全参数更新到高效技术。"
order: 5
plots: []
sourceHash: "996cb99eb8b19047f37e55ad239e18d64eb27ba34dfeb148e3c28f79611a42c5"
sourceCorrections: []
---

调整预训练 (pre-training)大型语言模型 (LLM) 的实际方法是定制其能力的核心。核心思想是获取一个通用预训练模型，并使用特定任务或特定领域的数据对其进行调整。然而，这种调整的具体执行方式差异很大，从而形成了一系列微调 (fine-tuning)方法，它们在计算成本、内存需求和可能结果上各有不同。

在这个范围的一端是**全参数 (parameter)微调 (Full FT)**。顾名思义，这种方法涉及在调整过程中更新预训练LLM的*所有*参数（权重 (weight)和偏差）。

- **工作方式：** 您从预训练模型权重开始，使用您的特定数据集继续训练过程。标准的反向传播 (backpropagation)和优化算法（如 AdamW）用于计算每个参数的梯度并进行更新。
- **比喻：** 可以想象，这就像让一位博学专家（预训练LLM）将其全部精力和学习能力投入到掌握一个新专业领域（您的特定任务）中。他们会调整所有现有的知识联系。
- **优点：** 能够在该目标任务上获得最高的表现，因为模型的全部能力都用于调整。
- **缺点：** 计算成本极高，需要大量GPU内存（通常是多个高端GPU）和训练时间，尤其是对于拥有数十亿参数的模型。为每个微调任务存储一个完整的模型副本变得不切实际。它还容易出现“灾难性遗忘”，即模型失去在预训练期间获得的一些通用能力。

由于全参数微调 (Full FT) 巨大的资源需求，特别是随着模型规模的不断增大，**参数高效微调 (PEFT)** 方法受到了广泛关注。这些方法旨在通过仅修改总参数的一小部分，或引入少量*新的*可训练参数来调整LLM，同时保持绝大多数原始预训练权重不变。

- **工作方式：** PEFT方法不更新所有$N$个参数（其中$N$可以是数十亿个），而是侧重于更新或添加$\Delta N$个参数，其中$\Delta N \ll N$。这些$\Delta N$参数的具体选择或引入方式定义了特定的PEFT方法。
- **比喻：** 与其完全重新训练这位博学专家，不如给他们提供专门的工具或一小组集中的指令（高效参数）来调整其专业知识以适应新任务，使其核心知识基本保持不变。
- **示例（稍后会详细介绍）：**
  - **适配器模块 (Adapter Modules)：** 在冻结的Transformer模块中插入小的、可训练的前馈层。
  - **低秩适应 (LoRA)：** 将可训练的低秩矩阵注入现有层中，有效地学习对原始权重矩阵的低秩*更新*。
  - **提示微调 (Prompt Tuning)：** 保持整个模型冻结，只学习一个小的连续“提示”嵌入 (embedding)，该嵌入被添加到输入前面。
  - **前缀微调 (Prefix Tuning)：** 与提示微调类似，但学习自注意力 (self-attention)机制 (attention mechanism)中键和值的前缀。
- **优点：** 大幅降低计算成本和内存需求（可训练参数可以少几个数量级）。便于存储和部署，因为您只需存储少量修改/添加的参数以及单个基础模型。通常有助于减轻灾难性遗忘。
- **缺点：** 表现有时可能会略低于全参数微调 (Full FT)，尽管考虑到效率提升，这种差距通常很小且可以接受。其有效性可能取决于所选的PEFT方法和具体任务。

> 根据计算资源、潜在表现和存储需求，对全参数微调和参数高效微调方法进行比较。

全参数微调 (Full FT) 和各种PEFT方法之间的选择在很大程度上取决于可用的计算资源、特定任务需求、期望的表现水平以及部署限制。本课程将介绍全参数微调 (Full FT)（第3章）和几种重要的PEFT技术（第4章）的实际实施和权衡，为您提供所需知识，以便选择和应用最适合您需求的调整策略。在介绍后续章节中涵盖的数据准备和实施细节之前，理解这个范围是很重要的。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  这篇有影响力的论文介绍了低秩适应（LoRA），这是一种高效且广泛采用的参数高效微调技术，通过学习现有权重矩阵的低秩更新来实现。
