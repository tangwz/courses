---
course: "llm-compression-acceleration"
chapter: "parameter-efficient-fine-tuning-peft"
lesson: "motivation-peft"
sourceId: 4381
sourceUrl: "https://apxml.com/zh/courses/llm-compression-acceleration/chapter-5-parameter-efficient-fine-tuning-peft/motivation-peft"
title: "PEFT 的缘由"
description: "理解 PEFT 的必要性：降低微调成本、存储开销和减轻灾难性遗忘。"
order: 1
plots: ["plots/4381-0.json"]
sourceHash: "8e7dbbe2ddb015530547fe32605080d169bf0fe5b67eb0f4e07b3c560f6be8c4"
sourceCorrections: []
---

对大型语言模型（LLM）进行专业化处理的传统方法是完全微调 (fine-tuning)（FFT），即使用任务特定数据更新所有模型参数 (parameter)。尽管此方法有效，但它带来了巨大的运行难题，尤其是在模型规模持续增长至数千亿甚至数万亿参数时。这些难题构成了研究参数高效微调（PEFT）技术的主要原因。

### 完全微调 (fine-tuning)的负担

完全微调在多个方面带来了显著负担：

1. **计算开销：** 更新大型 LLM 中的每个参数 (parameter)需要大量的计算资源。训练时需要计算所有权重 (weight)的梯度，这在反向传播 (backpropagation)过程中涉及大量的矩阵乘法。为数十亿参数存储和更新优化器状态（如 Adam 的动量）会进一步增加计算负荷。这直接导致高 GPU/TPU 使用率、漫长的训练时间和巨大的能耗。例如，微调像 GPT-3（1750 亿参数）这样的模型需要庞大的分布式训练基础设施。
2. **内存需求：** FFT 期间的内存占用通常是难以承受的。不仅是模型权重需要适应加速器内存（VRAM）；前向传播期间计算的激活值（梯度计算所需）、梯度本身以及优化器状态也都需要。对于大型模型，即使采用混合精度训练（例如 FP16 或 BF16），这些需求也容易超出商用 GPU 的容量，从而需要复杂的模型并行策略，这会进一步增加工程的复杂性。训练期间内存消耗的简化视图可能如下所示：

   
   $$
   \text{内存} \approx \text{模型参数} + \text{优化器状态} + \text{梯度} + \text{激活}
   $$
   

   对于 AdamW 等优化器，仅优化器状态通常就需要模型参数两倍的内存（用于存储一阶和二阶动量）。
3. **存储成本：** 最直接的问题可能出现在为多个下游任务调整单个预训练 (pre-training) LLM 时。使用 FFT，每个任务特定模型都是原始 LLM 的完整副本，尽管权重略有修改。如果您需要为 10 个不同任务部署微调版本，则必须存储 10 个独立的数十亿参数模型实例。一个以 BF16 精度存储的 700 亿参数模型大约需要 140GB。管理 10 个这样的模型将需要 1.4TB 的存储空间，这是一个巨大的运行开销。



![存储对比：完全微调 vs. PEFT (10 个任务)](plots/4381-0.json)



> 估计 10 个任务特定版本的 700 亿参数模型（BF16 精度下约 140GB）所需的存储空间，对比完全微调与 PEFT（假设 PEFT 模块约占总参数的 0.1%，每个约 100MB）。注意 Y 轴的对数刻度突出显示了巨大的差异。完全微调需要存储 10 个完整模型（1400 GB），而 PEFT 需要存储一个基础模型加上 10 个小型模块（140 GB + 10 \* 0.1 GB ≈ 141 GB）。

4. **灾难性遗忘：** 当预训练模型的所有参数在一个狭窄的任务特定数据集上更新时，模型可能会失去其在大量预训练阶段获得的部分通用知识和能力。这种现象被称为灾难性遗忘，它会降低模型在特定微调分布之外的任务上的性能。虽然在 FFT 期间存在减轻这种现象的方法，但这仍然是一个重要的顾虑。

### PEFT 作为一种高效替代方案

参数 (parameter)高效微调 (fine-tuning)方法通过从根本上改变适应方式，直接解决了这些局限。PEFT 技术通常不修改所有参数，而是包括：

- **冻结预训练 (pre-training)权重 (weight)：** 原始 LLM 绝大多数参数被冻结，在微调期间不进行更新。
- **定向更新：** 只有少量参数被训练。这些参数可以是：
  - 原始参数的一个小部分。
  - 添加到模型架构中的一小组*新*参数（例如，适配器层、提示嵌入 (embedding)）。

这种定向方法带来了几个令人信服的优点：

- **降低计算和内存开销：** 由于梯度和优化器状态仅需要总参数的极小一部分（通常小于 1%），微调的计算成本和内存占用都大幅降低。这使得在要求较低的硬件（例如，单个 GPU）上微调大型模型成为可能，并显著缩短了训练时间。
- **最小存储开销：** 核心预训练模型保持不变。对于每个新任务，只需要存储少量更新的或新增的参数。这使得存储问题从管理多个 TB 级别模型转变为管理一个基础模型加上众多 MB 级别的“增量”或适配器权重，如上图所示。
- **减轻灾难性遗忘：** 通过冻结模型的大部分，PEFT 自然保留了预训练期间学习到的通用知识。调整侧重于通过可训练参数学习任务特定偏差，降低了覆盖基础能力的风险。
- **简化部署：** 通过将单个基础模型加载到内存中，并根据传入请求动态加载或替换小的 PEFT 模块，可以实现为多个任务提供服务。与托管大量独立的超大模型相比，这极大地简化了模型管理和部署基础设施。

> 适应策略的对比。完全微调为每个任务创建完整、独立的模型副本。PEFT 则维护一个基础模型，并添加小的任务特定模块，从而显著减少存储和潜在的计算需求。

PEFT 本质上提供了一种实用且高效的途径，用于为各种应用定制基础模型，同时避免了与完全微调相关的过高成本。接下来的部分将审视 Adapters、基于提示的方法、LoRA 和 QLoRA 等流行 PEFT 方法的具体机制，详细说明它们如何在保持下游任务高性能的同时实现这些效率。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  本文介绍了低秩适配（LoRA），这是一种著名的PEFT技术。它详细阐述了添加小的、可训练的低秩矩阵如何显著减少微调参数的数量，从而解决了本节所述的计算、内存和存储挑战。
- [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751) — Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, Sylvain Gelly (2019)
  Journal: International Conference on Machine Learning (ICML); DOI: [10.48550/arXiv.1902.00751](https://doi.org/10.48550/arXiv.1902.00751)
  这篇基础论文介绍了适配器模块，展示了在预训练模型中添加少量新参数，如何在大幅减少可训练参数的同时，达到与全量微调相当的性能。
- [PEFT: Parameter-Efficient Fine-tuning of Foundation Models](https://huggingface.co/docs/peft/en/index) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face PEFT库的官方文档，提供了各种PEFT方法的实践见解和实现。这是理解所讨论的实际应用和优势的资源。
