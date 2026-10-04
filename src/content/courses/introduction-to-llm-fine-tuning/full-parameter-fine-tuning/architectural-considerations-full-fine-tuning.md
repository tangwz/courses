---
course: "introduction-to-llm-fine-tuning"
chapter: "full-parameter-fine-tuning"
lesson: "architectural-considerations-full-fine-tuning"
sourceId: 7275
sourceUrl: "https://apxml.com/zh/courses/introduction-to-llm-fine-tuning/chapter-3-full-parameter-fine-tuning/architectural-considerations-full-fine-tuning"
title: "全参数微调的架构考量"
description: "选择适合全参数微调的基础模型架构时的考量。"
order: 2
plots: []
sourceHash: "23d7de48590e898d6fd94486056fb0114fb44ea8052b08fd088503ebe6fd80a5"
sourceCorrections: []
---

为全参数 (parameter)微调 (fine-tuning)选择一个基础模型不仅仅是挑选基准测试得分最高的那个。模型的底层架构决定了它是否适合你的具体任务，并且更实际地，还决定了你是否有计算资源来更新它的所有权重 (weight)。由于全参数微调会修改每个参数，最初的架构选择对内存使用、训练时间以及你专用模型的最终表现有直接且明显的影响。

### 架构与任务适配

语言模型并非千篇一律。它们的设计通常针对特定类型的问题进行优化。你会遇到的三种主要架构类别是：仅编码器、仅解码器和编码器-解码器。

- **仅解码器模型（例如 GPT、Llama、Mistral）：** 这类模型是生成任务的标准选择。它们按顺序处理文本，旨在预测序列中的下一个token，因此非常适合文本生成、聊天机器人和指令遵循任务。它们的自回归 (autoregressive)特性意味着它们擅长开放式内容创作。当你的目标是让模型习得一种新的对话风格或特定的生成技能时，仅解码器模型几乎总是正确的选择。
- **仅编码器模型（例如 BERT、RoBERTa）：** 这类模型旨在通过同时考虑左右上下文 (context)来对整个文本输入形成充分的理解。这使得它们不适合文本生成，但在自然语言理解（NLU）任务中非常有效。如果你的目标是分类、情感分析或命名实体识别，微调 (fine-tuning)仅编码器模型将比微调大得多的仅解码器模型以更少的资源获得更好的结果。
- **编码器-解码器模型（例如 T5、BART）：** 这类模型结合了编码器和解码器，使其成为序列到序列任务的有力选择。编码器处理源文本以生成丰富的表示，解码器运用该表示来生成新的目标文本。这种架构非常适合机器翻译、文本摘要和问答等任务，其中输入需要转换为新的输出格式。

总结一下，何时选择哪种架构：

- **生成或聊天：** 使用**仅解码器**模型。
- **分类或信息提取：** 使用**仅编码器**模型。
- **翻译或摘要：** 使用**编码器-解码器**模型。

### 模型大小和近期改进的影响

模型中的参数 (parameter)数量是直接影响全参数微调 (fine-tuning)硬件要求的主要因素。一个70亿参数的模型所需的GPU内存远少于一个700亿参数的模型。然而，大小并非唯一因素。现代架构包含了能够大幅减少计算开销的特性。

一个突出的进展在于注意力机制 (attention mechanism)。最初的Transformer架构使用多头注意力 (multi-head attention)（MHA），其中每个注意力头都有自己独立的查询（Query）、键（Key）和值（Value）投影矩阵。较新的模型常采用更高效的变体：

- **多查询注意力（MQA）：** 所有注意力头共享单个键和值投影。这大幅减少了推理 (inference)期间存储KV缓存所需的内存，并降低了训练期间的内存带宽要求。
- **分组查询注意力（GQA）：** 这是MHA和MQA之间的一种折中方案。它将多个头分组，共享单个键和值投影，在MHA的性能和MQA的效率之间取得了平衡。

Llama 2和Mistral等模型采用了这些优化的注意力机制。选择带有GQA或MQA的模型，可以在现有硬件上使全参数微调更易于管理，因为它降低了模型中最占用资源的组件的内存占用。

> 注意力机制示意图。MHA 为每个头使用独有的键（K）和值（V）投影，而 GQA 将多个头分组以共享投影，MQA 则为所有头使用单个 K/V 投影，从而降低了内存占用。

另一个需要了解的架构模式是**专家混合（MoE）**。在 Mixtral 8x7B 这样的 MoE 模型中，模型包含多个“专家”子网络。对于任何给定的输入token，一个门控网络会将计算路由到这些专家中的一小部分。尽管总参数数量很高（例如 Mixtral 约为 470 亿），但每次前向传播只使用其中一小部分参数（例如约 130 亿）。这使得推理速度非常快。然而，对于全参数微调，你必须将所有专家加载到内存中，这意味着显存 (VRAM)需求与总参数数量相对应，而非活跃参数数量。

### 基础模型选择清单

在确定用于全参数 (parameter)微调 (fine-tuning)的基础模型之前，请考量以下几点：

1. **任务与架构匹配：** 模型的架构（解码器、编码器等）是否与你的最终目标一致？使用错误类型的模型将导致性能不佳，无论你如何调整。
2. **计算资源预算：** 全参数微调对内存要求很高。粗略估计，使用标准AdamW优化器进行全精度（32位）训练大约需要模型参数大小四倍的GPU显存 (VRAM)（例如，一个70亿参数模型需要约28GB显存）。混合精度训练可以降低这一要求，但这仍然是一个主要限制。选择一个符合你硬件限制的模型。
3. **模型许可：** 始终检查模型的许可，以确保它允许你预期的使用场景，特别是对于商业应用。像Llama 2这样的模型有特定的使用限制，而Falcon和Mistral等模型则使用更宽松的许可，例如Apache 2.0。
4. **社区和生态系统支持：** 该模型是否在Hugging Face `transformers` 和 `datasets` 等库中得到良好支持？一个活跃的社区提供宝贵的资源、现有的微调脚本，以及在项目期间可能出现问题时的支持网络。选择一个流行且支持良好的模型可以为你节省大量时间和精力。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍了Transformer架构、多头注意力机制以及基础的编码器-解码器设计。
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed (2024)
  Publisher: arXiv; DOI: [10.48550/arXiv.2401.04088](https://doi.org/10.48550/arXiv.2401.04088)
  介绍了Mixtral模型，这是一种专家混合（MoE）架构的示例，可实现大型模型的有效推理。
