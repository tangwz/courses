---
course: "mixture-of-experts-advanced-implementation"
chapter: "integrating-moe-into-architectures"
lesson: "placement-of-moe-layers"
sourceId: 7074
sourceUrl: "https://apxml.com/zh/courses/mixture-of-experts-advanced-implementation/chapter-5-integrating-moe-into-architectures/placement-of-moe-layers"
title: "MoE 层的位置：频率与部位"
description: "分析关于 MoE 层应以何种频率和在何种深度放置的架构考量，以获得最佳性能。"
order: 2
plots: ["plots/7074-0.json"]
sourceHash: "61b55ff8ee10d35cf2dd9790bff5178a320ec042b63f54455bb2f080360483e3"
sourceCorrections: []
---

专家混合层（Mixture of Experts, MoE）是神经网络 (neural network)架构中密集反馈网络（Feed-Forward Network, FFN）的替代方案。在使用MoE层时，一个主要的架构设计考量是如何确定这些稀疏层应放在何处以及其出现的频率。简单地将每个FFN都替换为MoE是一种可行但通常不是最优的策略。这种方法会大幅增加模型的参数 (parameter)量，并在分布式环境中引入显著的通信开销，而性能提升却不成比例。因此，MoE 层的放置是一个重要的设计选项，它平衡着模型容量、计算成本和训练过程。

这个决定涉及两个主要方面：MoE 层的出现频率（例如，在每个块中，或每隔一个块）以及它们在网络深度中的位置（例如，集中在早期、中期或后期）。

### MoE 层的频率

最常见且经验上成功的方法是以交替模式替换FFN。例如，你可以在每第二个或每第四个 Transformer 块中使用一个 MoE 层，而其他块则保留其标准的密集 FFN。这种方式是一种经验法则，它在增加模型容量和管理计算与通信成本之间提供了良好的平衡。

Google 的 Switch Transformer 和 GLaM 等架构采用了这种“每隔一层”的模式。主要的考虑是减轻专家并行所需的 `all-to-all` 通信开销。训练期间，每个设备会将分配给特定专家的令牌发送到保存该专家权重 (weight)的设备。这是一种网络密集型操作。通过让 MoE 层与不需要此类通信的标准 FFN 交替出现，整体的通信与计算比率得以保持在可控范围。

下图展示了 6 层 Transformer 的三种可能的频率策略。交替模式通常是最实用的起点。

> 6 层模型中三种不同的 MoE 放置策略。交替模式（策略 2）是常用且有效的基准。

### 位置：早期层与后期层

MoE 层所处的深度影响它们学习的专门化类型。在标准 Transformer 中，早期层通常捕获更普遍的、语法或低级别的特征，而后期层则学习更抽象、语义和特定于任务的表示。

这引出了一个有说服力的假设：**MoE 层在网络的后期阶段可能更有效。** 原因是令牌表示在更深层变得更具区分度且语义更丰富，这使得路由决定更有意义。例如，在后期层中，路由器可以更可靠地区分与“物理”和“金融”相关的令牌，并将它们路由到专门处理这些领域的专家。在早期层，这些区分可能尚未完全从原始嵌入 (embedding)中出现，这使得门控网络难以学习有用的路由策略。

相反，也有观点认为应将 MoE 层放置在早期，以鼓励从一开始就形成多样化的特征路径。然而，经验证据通常表明，在中后期层的专门化能带来更大的好处。

归根结底，最佳放置位置是一个经验性问题。目标是找到一个最佳点，在给定计算预算下最大限度地提升模型性能。下图显示了基于放置策略的性能曲线，表明均匀或略偏后期的放置通常会产生最佳结果。



![模型性能与 MoE 层放置位置](plots/7074-0.json)



> MoE 层位置与模型性能之间的关系。均匀或交替放置通常能提供最佳平衡。

### 实用指南与建议

在设计 MoE 架构时，请考虑以下几点：

1. **从交替模式开始：** 对于大多数应用，在每隔一个 Transformer 块中替换 FFN 是一个基准做法。它能有效增加模型容量，同时控制通信成本。
2. **分析任务：** 如果你的任务需要大量高级推理 (inference)和专门化（例如，多领域问答系统），将 MoE 放置在语义表示更丰富的中后期层可能更有益。对于可能受益于早期特征区分的任务，尝试更早的放置位置是值得的。
3. **考虑模型深度：** 对于非常深的模型（例如，100+ 层），每隔一层的策略可能仍然过于频繁。在这种情况下，更稀疏的放置，例如每三或四个块放置一个 MoE 层，可能更合适，以避免过多的通信开销。
4. **微调 (fine-tuning)考量：** 在微调期间，模型的后期层通常对于适应新任务最为重要。如果你正在微调一个预训练 (pre-training)的 MoE 模型，你可能会发现只训练 MoE 层，特别是网络后半部分的那些层，是一种高效且有效的方法。

最佳的放置策略并非普遍适用。它取决于具体的模型架构、任务性质以及你的硬件环境限制。此处讨论的原则和模式为做出明智的架构决定提供了有力的起点，后续应通过实验进行优化。

## 参考资料

- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://www.jmlr.org/papers/v23/21-0998.html) — William Fedus, Barret Zoph, and Noam Shazeer (2022)
  Journal: Journal of Machine Learning Research; Volume: 23; Pages: 1-39; DOI: [10.5555/3544548.3544723](https://doi.org/10.5555/3544548.3544723)
  本文介绍了Switch Transformer架构，该架构采用交替的MoE层放置策略，以有效管理大规模模型中的计算和通信成本。
