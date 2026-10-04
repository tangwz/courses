---
course: "introduction-to-llm-fine-tuning"
chapter: "parameter-efficient-fine-tuning-peft"
lesson: "comparing-peft-full-fine-tuning"
sourceId: 7286
sourceUrl: "https://apxml.com/zh/courses/introduction-to-llm-fine-tuning/chapter-4-parameter-efficient-fine-tuning-peft/comparing-peft-full-fine-tuning"
title: "PEFT与完全微调的优劣对比"
description: "关于PEFT与完全微调在性能、成本及训练时间上的优劣分析。"
order: 6
plots: ["plots/7286-0.json"]
sourceHash: "99efd6538488ef7f90bd776a715fe87c8eff43adff41eb0f89efd05dedc9e25e"
sourceCorrections: []
---

完全参数 (parameter)微调 (fine-tuning)与LoRA等参数高效方法之间的选择，需要考量多方面的取舍。这不是寻找一个普适性更优的方法，而是根据你的具体目标、现有硬件和运行限制来选择合适的工具。通过直接对比几个主要方面，有助于你为项目做出明智的选择。

### 计算与内存开销

最直接且突出的区别在于计算资源的消耗。顾名思义，完全微调 (fine-tuning)会更新模型中的所有参数 (parameter)。对于一个70亿参数的模型，这涉及到计算、存储和应用所有70亿权重 (weight)的梯度。AdamW等优化器的状态会进一步增加内存需求，通常需要模型参数本身内存的2到4倍。这使得大型模型的完全微调成为一项需要配备大量显存 (VRAM)的高端数据中心级GPU的操作。

相比之下，LoRA等PEFT方法大幅减轻了此负担。通过冻结原始权重，仅训练小型、低秩的适配器矩阵，你操作的是总参数的极小一部分。对于一个70亿参数的模型，LoRA配置可能只引入几百万个可训练参数。这种减少直接转化为更低的GPU内存占用，使得在单个消费级或专业级GPU上微调大型模型变为可能。QLoRA等技术通过量化 (quantization)基础模型，以低精度格式（例如4位）加载到内存中，然后附加可训练适配器，进一步减小了占用空间。



![70亿参数模型微调的GPU显存估算](plots/7286-0.json)



> 70亿参数模型微调的显存需求示意图。实际内存使用量取决于批处理大小、序列长度和具体的模型架构。

### 模型性能与任务适应性

一个常见问题是PEFT的效率是否以牺牲模型性能为代价。对于指令遵循或风格迁移等许多常见适配任务，LoRA及其他PEFT方法可以达到与完全微调 (fine-tuning)相媲美的效果。底层的预训练 (pre-training)模型已包含丰富的知识储备，PEFT在引导模型行为适应新任务方面非常有效，无需改变其核心。

然而，完全微调在特定情况下保留性能优势。如果你的目标是为模型注入大量新的专业知识，或从根本上改变其核心功能，那么更新所有权重 (weight)可能更有效。完全微调赋予模型最大的适应灵活性，而PEFT本质上受限于冻结的预训练权重。

### 存储与部署效率

资源需求上的差异不仅体现在训练阶段，也延伸到存储和部署。当你完全微调 (fine-tuning)一个模型时，输出的是一个完整、独立的模型。如果你将一个70亿参数 (parameter)模型适配十个不同任务，你必须存储十个独立的70亿参数模型，每个占用14GB或更多的存储空间（`bfloat16`精度）。

PEFT提供一个更为巧妙的方案。基础模型保持不变，每个微调任务生成一个小型适配器文件，通常只有几兆字节大小。这允许你维护一个基础模型的副本，并动态应用不同的适配器以适应不同任务。这种“一个基础模型，多个适配器”的方法在存储方面非常高效，并简化MLOps，因为你只需管理和提供一个大型模型以及一系列轻量级适配器文件。

> 存储模型对比。完全微调会创建多个大型模型副本，而PEFT则使用一个基础模型配合小型、特定任务适配器。

### 灾难性遗忘

灾难性遗忘是指模型在窄范围数据集上微调 (fine-tuning)后，丧失其通用能力的倾向。由于完全微调会修改所有模型权重 (weight)以针对新任务进行优化，因此有更高风险覆盖预训练 (pre-training)期间习得的知识。

PEFT方法对这个问题明显更具抵御能力。由于大型语言模型的原始权重被冻结，其核心推理 (inference)和语言能力得以保留。适配器温和地引导模型的输出以适应特定任务，而不干扰其底层结构。这使得PEFT在需要确保模型在专项化后仍能保持其广泛的通用能力时，成为一个更安全的选择。

### 优劣点总结

下表总结了在完全微调 (fine-tuning)和PEFT之间选择时的主要考量点：

| 特性维度 | 完全参数 (parameter)微调 | 参数高效微调 (PEFT) |
| --- | --- | --- |
| **性能** | 潜力最高，最适合深度知识转移。 | 对特定任务通常不相上下；非常有效。 |
| **GPU显存 (VRAM)** | 很高 | 低到中等 (QLoRA使其非常低)。 |
| **训练时间** | 高 | 低 |
| **存储开销** | 高 (每个任务一个完整模型)。 | 很低 (一个基础模型，每个任务配小型适配器)。 |
| **灾难性遗忘** | 风险较高。 | 风险低得多。 |
| **便携性与模块化** | 低；单一巨型模型文件。 | 高；轻量级适配器易于共享和管理。 |
| **主要应用场景** | 为广泛的知识范围创建新的基座模型。 | 为专业化、窄范围任务适配模型。 |

最终，选择取决于你的项目目标。如果你拥有资源并需要在新的、大型知识范围上从根本上重新训练模型，那么完全微调是一个有力的选择。对于大多数其他适配场景，例如构建专业聊天机器人、摘要工具或代码生成器，PEFT在性能和效率之间提供了出色的平衡。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍LoRA的原始论文，LoRA是一种核心参数高效微调方法。
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2305.14314](https://doi.org/10.48550/arXiv.2305.14314)
  介绍QLoRA，它通过量化大幅减少了大型模型微调的内存需求。
- [Parameter-Efficient Fine-Tuning (PEFT)](https://huggingface.co/docs/peft/index) — Hugging Face (2024)
  Hugging Face PEFT库的官方文档，提供了实用指导和PEFT方法的概览。
