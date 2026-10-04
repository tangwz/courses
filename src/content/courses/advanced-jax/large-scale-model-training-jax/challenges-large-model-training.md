---
course: "advanced-jax"
chapter: "large-scale-model-training-jax"
lesson: "challenges-large-model-training"
sourceId: 2927
sourceUrl: "https://apxml.com/zh/courses/advanced-jax/chapter-6-large-scale-model-training-jax/challenges-large-model-training"
title: "大型模型训练中的挑战概述"
description: "讨论与大型机器学习模型相关的内存限制、计算成本和通信开销。"
order: 1
plots: ["plots/2927-0.json"]
sourceHash: "fd44583173d4a05d7896ec4c648d53418f303aa9f4633c91ed896638ae374519"
sourceCorrections: []
---

随着机器学习 (machine learning)模型在复杂度和参数 (parameter)数量上持续增长，特别是在自然语言处理和计算机视觉等领域，有效训练它们带来了重大障碍。尽管JAX提供了强大的加速和微分工具，但扩展这些模型会遇到根本性的硬件限制。概述了训练大型模型时遇到的主要难题，为解决这些问题的JAX特定技术提供了背景。

### 内存限制：主要的瓶颈

最直接的难题通常是训练期间保存模型及其相关数据所需的庞大内存。单个现代加速器（GPU或TPU核心）拥有有限的高带宽内存（HBM），通常范围在16GB到80GB。大型模型很容易超出此容量。内存的主要消耗者包括：

1. **模型参数 (parameter)：** 这些是训练过程中学习到的权重 (weight)和偏差。对于大型模型，这很容易达到数十亿个参数。一个具有100亿参数的模型，以标准32位浮点精度（FP32）存储时，仅参数本身就需要40GB（$10 \times 10^9 \text{ 参数} \times 4 \text{ 字节/参数}$）。这个数字随着模型大小的增加而迅速增长。
2. **优化器状态：** Adam或AdamW等优化器会为每个参数维护内部状态（例如，一阶矩和二阶矩）。这些状态通常需要相当于模型参数两到三倍的存储空间。对于我们100亿参数的例子，Adam可能会额外增加80GB（$10 \times 10^9 \times 2 \times 4$ 字节），使总的静态内存需求达到120GB，这已经超出了大多数单设备容量。
3. **激活值：** 在前向传播期间，每一层的中间结果（激活值）必须存储起来，以便在反向传播 (backpropagation)（梯度计算）中使用。激活值所需的内存随批处理大小、序列长度（对于序列模型）以及模型深度/宽度而变化。对于深度网络和长序列，激活值内存可以远超过参数内存。
4. **梯度：** 在反向传播期间计算的梯度通常与模型参数具有相同的维度，需要等量的存储空间（对于我们100亿参数的FP32例子，额外需要40GB）。
5. **工作区内存：** XLA等编译器和cuDNN等库通常需要额外的临时工作区内存以实现高效的核函数执行。



![大型模型训练期间的典型内存分配](plots/2927-0.json)



> 用于训练具有Adam优化器和适中批处理大小/序列长度的100亿参数模型的内存估计分解。实际激活内存可能会有很大差异。

超出设备内存会导致内存不足（OOM）错误，从而中止训练。梯度检查点和混合精度等策略（将在后面讨论）直接旨在减少激活值和参数/梯度内存占用。

### 计算成本和训练时间

除了内存，计算成本（以浮点运算数FLOPs衡量）随模型大小显着增加。训练大型模型涉及对海量数据集进行潜在数万亿次的矩阵乘法和卷积运算。

- **FLOPs扩展：** 计算成本通常与模型维度呈非线性增长。例如，在Transformer模型中，自注意力 (self-attention)机制 (attention mechanism)通常随序列长度呈二次方增长。训练一个深度或宽度增加一倍的模型可能需要远超过两倍的计算量。
- **训练时长：** 即使使用强大的加速器，庞大的运算量也意味着训练可能需要数天、数周甚至数月。如此长的持续时间增加了硬件成本、能源消耗以及研发迭代时间。
- **硬件要求：** 有效训练这些模型需要高性能GPU集群或大型TPU Pods，这代表着巨大的资本或运营支出。

减少总FLOPs通常涉及算法变更或模型架构修改，但混合精度等技术有时可以通过使用更快、精度更低的计算单元来提供加速。

### 分布式训练中的通信开销

当模型或所需批处理大小超出单个加速器的能力时，跨多个设备的分布式训练变得必不可少。尽管JAX的`pmap`简化了分布式代码的编写（如第三章所述），但设备间的通信引入了新的瓶颈。

- **数据并行：** 最常用的策略是在每个设备上复制模型，并并行处理数据批次的不同分片。在每个设备上完成局部反向传播 (backpropagation)后，梯度必须在所有设备之间同步，然后进行优化器步骤。这种同步通常使用All-Reduce集合操作。
- **通信成本：** 梯度同步所需的时间取决于模型大小（要传输的数据总量）以及设备间的互连带宽和延迟。对于非常大型的模型，这个通信步骤可能成为总步骤时间的重要组成部分，从而限制扩展效率。慢速互连（例如，以太网与NVLink或TPU互连相比）会加剧此问题。
- **其他并行策略：** 更复杂的策略，如模型并行（在设备之间分割单个层）或流水线并行（在设备之间分阶段处理层），会引入不同且通常更复杂的通信模式，涉及特定设备子集之间的激活值和梯度。

> 使用`pmap`进行数据并行时的通信模式。在优化器同步更新所有副本的模型权重 (weight)之前，每个设备本地计算的梯度必须通过All-Reduce等集合通信操作进行聚合（例如求和）。

最小化通信通常涉及优化集合通信算法，在可能的情况下将通信与计算重叠，以及选择平衡计算负载和通信需求的并行策略。解决这些相互关联的难题需要结合高效的编程模型、算法技术和硬件感知。本章的后续部分将介绍JAX及其生态系统如何提供工具来管理内存、优化计算并使用分布式硬件训练真正的大型模型。

## 参考资料

- [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 5998-6008; DOI: [10.55988/neurips-2017-578](https://doi.org/10.55988/neurips-2017-578)
  提出 Transformer 架构的奠基性论文，该架构极大地推动了大型语言模型的增长及其相关的训练挑战。
- [Mixed-Precision Training](https://doi.org/10.48550/arXiv.1710.03740) — Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, Hao Wu (2017)
  Journal: ICLR 2018; DOI: [10.48550/arXiv.1710.03740](https://doi.org/10.48550/arXiv.1710.03740)
  描述了混合精度训练技术，通过使用 FP16 等低精度浮点格式，减少了内存占用并加速了计算。
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) — Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro (2019)
  Journal: arXiv preprint arXiv:1909.08053; DOI: [10.48550/arXiv.1909.08053](https://doi.org/10.48550/arXiv.1909.08053)
  提出了一个用于训练超大型语言模型的框架，该框架利用模型并行和流水线并行，解决了分布式训练中的内存和通信挑战。
