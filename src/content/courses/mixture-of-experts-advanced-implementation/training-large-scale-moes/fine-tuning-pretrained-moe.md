---
course: "mixture-of-experts-advanced-implementation"
chapter: "training-large-scale-moes"
lesson: "fine-tuning-pretrained-moe"
sourceId: 7064
sourceUrl: "https://apxml.com/zh/courses/mixture-of-experts-advanced-implementation/chapter-3-training-large-scale-moes/fine-tuning-pretrained-moe"
title: "预训练MoE模型的微调策略"
description: "用于在下游任务上有效微调大型预训练专家混合模型的先进策略。"
order: 6
plots: []
sourceHash: "61433d5718f598eb8e8a0ddbaa1e7f5a6892fc3f68a246515a85075fa8a4d6b2"
sourceCorrections: []
---

将预训练 (pre-training)的专家混合模型调整到新任务，与微调 (fine-tuning)密集模型相比，带来一组不同的挑战。虽然预训练已赋予专家专业知识，但微调的目标是将这些知识引向新的应用方向或功能，同时避免从头训练的全部成本或导致灾难性遗忘。MoE模型的规模很大，这使得简单的全量微调方法对多数应用而言不实用。因此，需要一系列更精确的策略。

这些策略平衡三个相互竞争的因素：计算成本、内存占用和最终任务性能。策略的选择将取决于您的可用资源和性能要求。

### 全量微调 (fine-tuning)：蛮力法

最直接的方法是使用新任务的数据集，更新所有模型参数 (parameter)，包括门控网络和每个专家。这类似于密集模型的标准微调。

- **流程：** 训练过程与预训练 (pre-training)相似，需要相同的分布式设置，并采用专家并行来管理模型规模。学习率通常设置远低于预训练时的值。
- **优点：** 这种方法有可能在目标任务上获得最高性能，因为每个参数都可以自由调整。
- **缺点：** 资源需求是主要障碍。它需要大型GPU集群，并且通常成本过高。此外，它存在破坏预训练中学到的宝贵专业化的风险，这种现象称为灾难性遗忘。模型可能对新的、通常较小的数据集过拟合 (overfitting)，并失去其通用能力。

全量微调通常适用于微调数据集非常大且质量高，并且最高性能是首要目标（不考虑成本）的情况。

### 选择性参数 (parameter)更新

一套更节省资源且常用的技术涉及冻结模型的大部分，仅更新特定组件。这隔离了更改，保留了通用知识，并显著减少了计算负担。

下图展示了针对单个MoE层的三种不同更新策略。

> 微调 (fine-tuning)策略比较。绿色部分表示正在更新的参数，而灰色部分表示已冻结的参数。

#### 仅微调门控网络

这是最有效的方法之一。所有专家网络都被冻结，仅更新门控网络（路由器）的参数。

- **机制：** 模型学习如何将令牌从新任务的数据分布中路由到最合适、预先存在的专家。它不会在专家内部学习新的专业知识，而是学习使用其已有知识的新*策略*。
- **用例：** 这在适应特定应用场景时非常有效，其中目标任务（例如，总结法律文档）需要通用预训练 (pre-training)模型（例如，在网页上训练的模型）中已存在的部分知识。
- **优点：** 内存和计算需求很小，因为您只更新小型路由器网络。这通常可以在单台机器上完成。

#### 微调部分专家

合乎逻辑的下一步是，除了路由器之外，解冻少量专家。这在效率和适应性之间提供了一个平衡。有两种常见方法：

1. **微调最活跃专家：** 首先，在您的微调数据集上运行一次推理 (inference)，并记录哪些专家被路由器最频繁地激活。然后，冻结所有其他专家，仅微调这个“热门”子集以及路由器。这使得计算集中在与您的任务最相关的模型部分。
2. **添加并训练新专家：** 冻结所有原始专家。向MoE层添加少量新的、随机初始化的专家。在微调期间，您只训练这些新专家和路由器。这种方法将任务特定知识隔离在新专家中，最大程度地保留了原始模型的通用能力。

### 参数 (parameter)高效微调 (fine-tuning)（PEFT）方法

低秩适应（LoRA）等技术在模型中引入小的、可训练的“适配器”矩阵，同时保持大量的预训练 (pre-training)权重 (weight)冻结。这个原理自然地适用于MoE模型。

LoRA不是更新原始权重矩阵$W$，而是通过低秩分解来近似更新，即$W' = W + BA$，其中$B$和$A$是小的、可训练的矩阵。

对于MoE模型，可以通过几种方式应用LoRA：

- **在非MoE层上应用LoRA：** 将LoRA应用于自注意力 (self-attention)层和其他非MoE前馈层。这可以调整模型的通用处理方式，而不触及专家权重。
- **在路由器上应用LoRA：** 将LoRA应用于门控网络。这是一种非常高效的路由策略调整方式，类似于仅微调路由器，但可训练参数更少。
- **在专家上应用LoRA：** 可以将LoRA应用于专家本身。将其应用于*所有*专家可能仍然要求很高。更实际的策略是仅将其应用于部分专家，例如对您的任务最常激活的那些。

这些方法的组合，例如将LoRA应用于注意力层和路由器，通常是一个非常有效且高效的起点。

### 微调 (fine-tuning)中的辅助损失

对MoE模型而言，一个重要考量是负载均衡辅助损失$L_{aux}$在微调期间的命运。舍弃它是一个错误。如果没有分配令牌的压力，路由器会迅速出现“表示崩溃”，学习将所有令牌发送到一个专家，该专家变得高度针对微调任务过度专业化。这破坏了MoE架构的优势。

因此，辅助损失在微调期间应保持活跃。然而，微调数据集通常比预训练 (pre-training)语料库窄得多。强制专家之间完美均匀的分布可能适得其反。常见的做法是，通过减小辅助损失的系数$\alpha$来降低其权重 (weight)。


$$
L_{总} = L_{交叉熵} + \alpha_{微调} \cdot L_{辅助}
$$


$\alpha_{微调}$的典型值可能比预训练时使用的值小一个数量级。这保持足够的压力以防止专家崩溃，同时给予路由器足够的灵活性，使其偏好某些确实与下游任务更相关的专家。在微调期间，请始终监控专家利用率，以确保健康的分布。

## 参考资料

- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean (2017)
  Journal: arXiv preprint arXiv:1701.06538; DOI: [10.48550/arXiv.1701.06538](https://doi.org/10.48550/arXiv.1701.06538)
  介绍了稀疏门控混合专家（MoE）层和用于负载均衡的辅助损失，这是MoE架构的基础。
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — William Fedus, Barret Zoph, Noam Shazeer (2021)
  Journal: Journal of Machine Learning Research; DOI: [10.48550/arXiv.2101.03961](https://doi.org/10.48550/arXiv.2101.03961)
  介绍了Switch Transformers，一种高度可扩展的MoE架构，详细说明了训练方法和辅助损失在大规模稀疏模型中的作用。
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: arXiv preprint arXiv:2106.09685; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  提出了低秩适应（LoRA），一种参数高效的微调方法，减少了适应大型模型所需训练的参数量。
