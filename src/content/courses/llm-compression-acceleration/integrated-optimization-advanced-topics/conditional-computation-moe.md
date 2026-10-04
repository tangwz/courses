---
course: "llm-compression-acceleration"
chapter: "integrated-optimization-advanced-topics"
lesson: "conditional-computation-moe"
sourceId: 4424
sourceUrl: "https://apxml.com/zh/courses/llm-compression-acceleration/chapter-7-integrated-optimization-advanced-topics/conditional-computation-moe"
title: "条件计算与专家混合（MoE）"
description: "分析MoE架构及其他条件计算方法以提升LLM效率。"
order: 3
plots: ["plots/4424-0.json"]
sourceHash: "b596d8095843addc99d8c4818acb1af7fedd60d619fde3a342880312e18a887d"
sourceCorrections: []
---

量化 (quantization)和剪枝等其他技术可以降低*现有*模型参数 (parameter)的计算或内存成本。相比之下，条件计算则从根本上改变了给定输入使用*哪些*参数。其目的是仅激活模型参数的一个子集，使其适应特定输入，从而减少每次推理 (inference)步骤所需的浮点运算（FLOPs），即使总参数数量保持庞大甚至增加也是如此。这种方法与标准密集模型形成鲜明对比，在标准密集模型中，几乎所有参数都参与处理每个输入标记 (token)。

专家混合（MoE）是体现条件计算的最突出和成功的架构，尤其是在Transformer模型的馈线网络（FFN）层中。

### 专家混合架构

MoE层用以下几个组成部分取代了标准的密集FFN层：

1. **专家：** 它们是多个独立的FFN（或其他网络块）。每个专家可以专注于处理不同类型的信息或执行不同的子任务。在Transformer MoE层中，你可能有例如8、64甚至更多的并行FFN专家，而不是只有一个。
2. **门控网络（路由器）：** 这通常是一个小型网络（通常是一个线性层，后跟softmax或类似的选择机制）。对于每个输入标记 (token)，门控网络动态决定哪些专家应处理该标记。
3. **组合机制：** 给定标记的所选专家输出会进行组合，通常是通过基于门控网络得分的加权和。

> MoE层的一个视图。门控网络将输入标记表示路由到专家子集（例如，专家1和专家M）。它们的输出根据门控分数进行组合，以生成最终输出表示。

最常见的路由策略是**Top-k路由**，门控网络为每个专家输出一个分数，并选择分数最高的$k$个专家来处理标记。通常，$k$很小（例如，Switch Transformers中的$k=1$，或$k=2$）。最终输出是这些Top-k专家输出的加权和，权重 (weight)从归一化 (normalization)的门控分数中得出。

### 稀疏性带来的效率提升

MoE通过*稀疏激活*实现计算效率。虽然所有专家的总参数 (parameter)数量可能很多（导致高模型容量），但对于任何给定标记 (token)，只有所选的$k$个专家和门控网络中的参数被激活。

考虑一个维度为$d_{model}$、隐藏维度为$d_{ff}$的FFN层。其FLOPs大约是$2 \times d_{model} \times d_{ff}$。在一个具有$M$个专家和top-k路由（$k \ll M$）的MoE层中，每个标记的FLOPs大约是：


$$
\text{FLOPs}_{MoE} \approx \text{FLOPs}_{路由器} + k \times (2 \times d_{model} \times d_{ff})
$$


由于路由器很小，$k$通常也小，因此每个标记的计算成本显著低于具有等效*总*参数数量（$M \times (2 \times d_{model} \times d_{ff})$）的密集模型。MoE有效地将参数数量与每个标记的计算成本分离。



![参数数量与推理FLOPs](plots/4424-0.json)



> 比较密集模型和MoE模型中，每个标记的推理 (inference)FLOPs如何随总模型参数的数量而变化。MoE允许更大的参数数量，而每个标记的FLOPs增长较慢。（数值为示例）。

### 训练与实施挑战

训练MoE模型面临独特的挑战：

1. **负载均衡：** 如果门控网络持续将大多数标记 (token)路由到少数“热门”专家，这些专家就会成为瓶颈，而其他专家则训练不足。这种不平衡会降低性能和硬件效率。为了解决这个问题，训练通常会引入一个辅助的**负载均衡损失**。这种损失促使路由器将标记更均匀地分配给专家，通常通过惩罚分配给每个专家的标记比例不平衡来达到。
2. **训练稳定性：** 离散的路由决策和不活跃专家导致梯度消失的可能性，有时会引发训练不稳定。添加噪声到门控分数或使用专家容量因子（限制每个专家每批次可以处理的标记数量）等技术有助于稳定训练。
3. **内存占用大：** 尽管每个标记的FLOPs较低，但MoE模型（所有专家合并）的*总*参数 (parameter)数量非常庞大。这在训练和推理 (inference)过程中都需要大量内存（通常还有模型并行）进行存储。
4. **通信开销：** 在专家位于不同设备的分布式设置中（由于模型规模大，这是常见情况），将标记路由到正确的专家并收集其输出会产生大量通信开销，如果底层系统未能妥善管理（例如，使用优化集合操作或拓扑感知放置），这可能成为瓶颈。

### MoE在优化中的应用

MoE从根本上说是一种提高效率的*架构*方法。它可以与其他技术结合使用：

- **量化 (quantization)/剪枝：** 这些方法可以应用于每个专家内部甚至门控网络，以进一步减少内存和计算。一个拥有8个专家的MoE层，即使每个专家都量化到INT8，仍受益于稀疏激活原理。
- **知识蒸馏 (knowledge distillation)：** 理论上可以将一个大型、强大的MoE模型蒸馏成一个较小的密集模型，甚至是一个较小的MoE模型。
- **PEFT：** 高效调整大型MoE模型可能仅涉及微调 (fine-tuning)门控网络、一部分专家，或将LoRA等方法应用于专家本身。

然而，MoE主要通过增加参数 (parameter)数量同时稀疏激活来解决每个标记 (token)的FLOPs问题。这使其与减少现有密集计算成本的方法有所区别。

### 变体与考量

虽然FFN的top-k路由是最常见形式，但条件计算的思路进一步拓展。Switch Transformers推广了$k=1$的极端情况，将每个标记 (token)只路由到一个专家。研究也关注Transformer其他部分的条件计算，例如注意力机制 (attention mechanism)，尽管FFN层的MoE仍然是最广泛应用的技术。

选择MoE涉及权衡：

- **优点：** 模型容量非常大，每个标记的推理 (inference)成本可控，专家专业化。
- **缺点：** 总内存需求显著增加，训练动态复杂需要仔细的负载均衡，分布式推理中潜在的通信瓶颈，与密集模型相比实现复杂性增加。

总之，MoE代表了向LLM高效扩展的重要架构转变。通过每个输入标记仅激活一小部分参数 (parameter)，它允许模型在参数数量上大幅增长，而每个标记的计算成本不会按比例增加，为在受限推理预算下获得更强大的模型提供了途径，尽管面临大量的系统级和训练挑战。MoE是几个最先进大型语言模型的支柱，理解它变得越来越重要。

## 参考资料

- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean (2017)
  Journal: arXiv preprint arXiv:1701.06538; DOI: [10.48550/arXiv.1701.06538](https://doi.org/10.48550/arXiv.1701.06538)
  介绍了稀疏门控的混合专家 (MoE) 层，这是神经网络中条件计算的基础架构。
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://www.jmlr.org/papers/v23/21-0990.html) — William Fedus, Barret Zoph, Noam Shazeer (2022)
  Journal: Journal of Machine Learning Research; Publisher: Microtome Publishing; Volume: 23; Pages: 1-39
  介绍了Switch Transformers，一种简化的MoE架构（k=1路由），可扩展到超大型模型并解决了训练稳定性和负载均衡问题。
