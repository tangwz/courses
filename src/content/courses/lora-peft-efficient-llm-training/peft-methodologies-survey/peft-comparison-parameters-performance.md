---
course: "lora-peft-efficient-llm-training"
chapter: "peft-methodologies-survey"
lesson: "peft-comparison-parameters-performance"
sourceId: 4138
sourceUrl: "https://apxml.com/zh/courses/lora-peft-efficient-llm-training/chapter-3-peft-methodologies-survey/peft-comparison-parameters-performance"
title: "比较分析：参数与性能的权衡"
description: "分析不同PEFT方法在可调参数数量和任务表现方面的权衡。"
order: 5
plots: ["plots/4138-0.json"]
sourceHash: "ea2ba473a3eb1c7f38f6749f64c13085183cf94c8a341c970be3fa640d2361d0"
sourceCorrections: []
---

选择合适的参数 (parameter)高效微调 (fine-tuning)（PEFT）方法需要仔细平衡。您需要衡量计算节省（主要反映在可训练参数的数量上）与对模型在特定下游任务表现的潜在影响。本分析将评估适配器微调、前缀微调和提示微调，以及LoRA（在第2章中提到），以比较它们在此核心权衡中的表现。

### 理解范围：从少量到大量参数 (parameter)

PEFT 方法的原理是只修改大型预训练 (pre-training)模型总参数的一小部分。然而，这部分的大小在不同技术间差异很大：

1. **完全微调 (fine-tuning)：** 代表了上限，修改模型100%的参数。它提供了最高性能的可能，但会产生最大的计算成本。
2. **LoRA（低秩适应）：** 引入低秩矩阵（A和B）以适应特定权重 (weight)矩阵（通常在注意力或前馈层中）。可训练参数的数量直接取决于所选秩$r$、调整层($d_{in}, d_{out}$)的维度以及调整的层数。它通常涉及比完全微调少得多的参数，通常不到总参数的1%甚至0.1%。参数数量大致按 2 \times (\text{# 调整层数}) \times r \times d 缩放，其中$d$是相关层维度。
3. **适配器微调：** 在Transformer块内插入小的瓶颈层（适配器模块）。参数数量取决于瓶颈维度（通常远小于Transformer的隐藏维度）以及适配器的放置位置。与LoRA类似，这通常只占模型总参数的一小部分。参数数量按 (\text{# 适配器位置}) \times 2 \times d\_{hidden} \times d\_{bottleneck} 缩放，其中 $d_{bottleneck} \ll d_{hidden}$。
4. **前缀微调：** 在注意力层中向键和值添加一系列可训练的连续向量 (vector)（前缀）。参数数量取决于前缀长度$L$和模型的隐藏维度$d_{hidden}$，按 (\text{# 层数}) \times 2 \times L \times d\_{hidden} 缩放。这种方法在参数方面非常高效，特别是对于较短的前缀。
5. **提示微调：** 这些方法中参数效率最高的方法。它只学习添加到输入前面的一小段虚拟提示标记 (token)的连续嵌入 (embedding)。参数数量仅取决于提示长度$P$和模型的嵌入维度$d_{embed}$，按 $P \times d_{embed}$ 缩放。这通常只占总参数的极小部分（例如，0.001%）。

### 性能考量

虽然减少参数 (parameter)是为了效率而理想的，但最终目标是有效的任务适应。这些方法在实现接近完全微调 (fine-tuning)的性能方面如何比较？

- **LoRA 和适配器：** 在许多标准基准测试中，通常能达到与完全微调非常接近的性能，特别是在适当配置（例如，为LoRA选择合适的秩$r$或为适配器选择瓶颈维度）并应用于相关层（如注意力机制 (attention mechanism)）时。它们修改模型内部表示的能力似乎对各种任务都有益处。LoRA尤其因其经验成功和直接的集成而获得了广泛关注。
- **前缀微调：** 通常表现良好，常优于提示微调，特别是在生成任务或模型需要更详细条件作用时。它平衡了提示微调的最小干预和LoRA/适配器更直接的权重 (weight)修改。
- **提示微调：** 由于其参数数量最少，提示微调有时可能落后于其他方法，特别是在复杂任务或需要对模型内部推理 (inference)过程进行重大改变的任务（例如，结构化预测、复杂算术）上。然而，其极高的效率使其在计算资源严重受限或任务相对简单时具有吸引力。性能往往随着基础模型尺寸的增加而显著提高。
- **完全微调：** 仍然是性能潜力的黄金标准，特别是在与预训练 (pre-training)目标差异很大或需要大量知识更新的任务上。然而，其资源需求通常使其不切实际。

### 权衡的可视化

参数 (parameter)数量与性能之间的关系并非总是线性的。增加更多参数并不能保证更好的结果，不同方法在效率-有效性范围上占据着不同的位置。



![PEFT 权衡](plots/4138-0.json)



> 图示了各种PEFT方法在可训练参数百分比（对数尺度）与相对于完全微调 (fine-tuning)的相对任务表现之间的权衡。位置是示意性的，可能因模型、任务和实现细节而异。

### 进行选择

如图表所示：

- 如果**最大参数 (parameter)效率**是主要限制，**提示微调 (fine-tuning)**是明确的选择，接受在复杂任务上可能出现的性能下降。
- 如果目标是实现**最接近完全微调**的性能，同时仍能显著节省计算资源，**LoRA**和**适配器微调**是强有力的竞争者。两者的选择可能取决于具体的库支持、实现的简易性，或针对您特定任务的验证集上的实证结果。QLoRA等技术进一步提高了LoRA的内存效率。
- **前缀微调**提供了一个中间选项，通常以仅适度增加参数为代价，提供优于提示微调的性能。

最终，最佳的PEFT方法在很大程度上取决于具体的应用场景：

- **任务复杂度：** 简单的分类任务可能适合提示微调，而复杂的推理 (inference)或生成任务则可能从LoRA或适配器中受益。
- **计算预算：** QLoRA在内存受限环境中表现出色。提示微调总体上所需的内存最少。
- **基础模型：** 随着基础模型尺寸的增加，方法间的性能差距可能会缩小。
- **部署场景：** 如果您需要使用相同的基础模型在多个任务之间快速切换，存储大量小的LoRA或适配器权重 (weight)比存储完全微调的模型更可行。

对您的目标任务和数据集进行实证评估仍然是确定适合您特定需求的最佳参数-性能权衡的最可靠方式。以下关于内存和计算占用方面的部分将提供关于每种方法实际资源影响的更多信息。

## 参考资料

- [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190) — Xiang Lisa Li, Percy Liang (2021)
  Journal: Annual Meeting of the Association for Computational Linguistics (ACL); DOI: [10.48550/arXiv.2101.00190](https://doi.org/10.48550/arXiv.2101.00190)
  本文介绍了前缀微调（Prefix-Tuning），一种通过优化预置到输入前的连续前缀进行微调的方法，展示了其在文本生成任务中的效果。
- [Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691) — Brian Lester, Rami Al-Rfou, Noah Constant (2021)
  Journal: Conference on Empirical Methods in Natural Language Processing (EMNLP); DOI: [10.48550/arXiv.2104.08691](https://doi.org/10.48550/arXiv.2104.08691)
  这项工作提出了提示微调（Prompt Tuning），一种学习软提示嵌入以进行任务适应的参数高效方法，在大模型上实现了强大的性能。
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023)
  Journal: Conference on Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2305.14314](https://doi.org/10.48550/arXiv.2305.14314)
  本文描述了 QLoRA，一种对 LoRA 的优化，它将基础模型量化到 4 位，使得在消费级 GPU 上微调大型模型成为可能，这与内存效率高度相关。
