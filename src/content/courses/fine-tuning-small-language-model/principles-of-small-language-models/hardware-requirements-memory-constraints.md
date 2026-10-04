---
course: "fine-tuning-small-language-model"
chapter: "principles-of-small-language-models"
lesson: "hardware-requirements-memory-constraints"
sourceId: 8500
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-small-language-model/chapter-1-principles-of-small-language-models/hardware-requirements-memory-constraints"
title: "硬件要求与显存限制"
description: "计算显存需求并了解训练小语言模型时的硬件限制。"
order: 4
plots: ["plots/8500-0.json"]
sourceHash: "1e9de5af57ca60d6d3deeb682d15883dfd622d14864587901ae8b2e75814d691"
sourceCorrections: []
---

应用数学权重 (weight)更新需要物理硬件显存。具体来说，这些操作依赖于图形处理器（GPU）的显存（VRAM）。明确训练期间显存的消耗方式，可以防止显存不足（OOM）错误，并帮助你确定在现有硬件上实际可以训练哪些模型。

语言模型训练期间的显存消耗分为四个主要部分：模型权重、优化器状态、梯度和激活值。在推理 (inference)过程中，你只需要足够的显存来容纳模型权重和少量上下文 (context)数据。然而，训练需要同时维持这四个部分。

为了估算模型权重所需的显存，你需要将参数 (parameter)数量乘以存储它们的数据类型大小。大多数小语言模型使用 16 位精度进行训练，包括 FP16 或 BF16 等格式。在 16 位格式中，每个参数占用 2 字节显存。如果 $P$ 代表参数总数，则权重所需的显存计算如下：

$M_{	ext{权重}} = P \times 2 \text{ 字节}$

在梯度下降 (gradient descent)的反向传播 (backpropagation)过程中，网络会为每个权重计算梯度。因为这些梯度也以 16 位精度存储，它们的显存占用与模型权重相同。

$M_{	ext{梯度}} = P \times 2 \text{ 字节}$

在全量微调 (fine-tuning)期间，优化器状态是显存消耗最大的部分。训练语言模型的标准优化器是 AdamW。AdamW 为每个参数维护两个移动平均值：动量和方差。为了确保数值稳定性，这些状态以 32 位精度（FP32）存储，每个数值占用 4 字节。由于每个参数有两个状态，优化器每个参数需要 8 字节。

$M_{	ext{优化器}} = P \times 8 \text{ 字节}$

我们可以将这些静态显存需求整合为一个全量微调公式。在将任何数据传入模型之前，所需的总固定显存为：

$M_{	ext{固定}} = M_{	ext{权重}} + M_{	ext{梯度}} + M_{	ext{优化器}} = P \times 12 \text{ 字节}$



![10亿参数模型显存分配（全量微调）](plots/8500-0.json)



> 使用全量微调和 AdamW 优化器的 10 亿参数模型在不同训练组件中分配的显存比例。

例如，根据上述公式，一个拥有 15 亿参数的小语言模型仅固定显存就需要 18 GB。

这 18 GB 的计算还不包括显存消耗的第四个部分：激活值。激活值是指在前向传播过程中创建的中间张量。这些张量必须保留在显存中，因为在反向传播过程中计算梯度时需要用到它们。

激活值消耗的显存变化很大。它根据你的批次大小（batch size）和最大序列长度进行缩放。在标准的 Transformer 架构中，自注意力 (self-attention)机制 (attention mechanism)所需的显存随序列长度呈平方增长。如果你将文本输入的长度增加一倍，注意力机制的显存需求会增加到原来的四倍。对于一个 15 亿参数的模型，在适中的批次大小下，激活值可以轻松额外消耗 4 到 6 GB 的显存。

将固定状态和激活值加在一起，一个 15 亿参数模型的总显存需求将超过 22 GB。标准的消费级高端 GPU 通常提供 24 GB 显存。这意味着训练一个相对较小的 15 亿参数模型已经达到了硬件的极限。如果你尝试对一个 70 亿参数的模型进行全量微调，仅固定状态就需要超过 84 GB 的显存。这使得在单块消费级 GPU 上进行训练变得不再可能。

了解这些硬件限制说明了为什么全量微调对语言模型来说通常是不切实际的。这也构成了使用参数高效微调（PEFT）方法的技术动机。通过有选择地仅更新极小比例的参数，我们可以大幅减少梯度和优化器状态所需的显存。

在我们介绍那些高效的训练技术之前，你需要一个可运行的基础。下一节将指导你编写 Python 代码，以便在这些硬件限制内将预训练 (pre-training)模型加载到显存中。

## 参考资料

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://doi.org/10.1109/SC41405.2020.00024) — Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He (2020)
  Journal: SC20: International Conference for High Performance Computing, Networking, Storage and Analysis; Publisher: IEEE; Pages: 1-16; DOI: [10.1109/SC41405.2020.00024](https://doi.org/10.1109/SC41405.2020.00024)
  该基础论文将内存消耗细分为模型状态（优化器状态、梯度和参数）和残余状态。
- [Decoupled Weight Decay Regularization](https://openreview.net/forum?id=Bkg6RiCqY7) — Ilya Loshchilov, Frank Hutter (2019)
  Journal: International Conference on Learning Representations; DOI: [10.48550/arXiv.1711.05101](https://doi.org/10.48550/arXiv.1711.05101)
  定义 AdamW 优化器的原始论文，该优化器是本节中讨论的每参数 8 字节内存需求的来源。
- [Methods and tools for efficient training on a single GPU](https://huggingface.co/docs/transformers/perf_train_gpu_one) — Hugging Face (2023)
  该技术文档提供了在标准硬件上训练 Transformer 模型时显存占用的数学明细。
- [LoRA: Low-Rank Adaptation of Large Language Models](https://openreview.net/forum?id=n77K0kG369) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2022)
  Journal: International Conference on Learning Representations (ICLR)
  为参数高效微调（PEFT）提供了理论基础，作为解决文中提到的内存限制的方案。
