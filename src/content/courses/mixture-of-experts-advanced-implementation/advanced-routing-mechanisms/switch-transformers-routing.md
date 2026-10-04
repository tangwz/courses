---
course: "mixture-of-experts-advanced-implementation"
chapter: "advanced-routing-mechanisms"
lesson: "switch-transformers-routing"
sourceId: 7055
sourceUrl: "https://apxml.com/zh/courses/mixture-of-experts-advanced-implementation/chapter-2-advanced-routing-mechanisms/switch-transformers-routing"
title: "Switch Transformer：简化路由"
description: "了解Switch Transformer架构，它通过为每个令牌只选择一个专家来简化路由。"
order: 4
plots: ["plots/7055-0.json"]
sourceHash: "7af91bcdbfa2ecc95e9810d73305dccf15c7f31d2b4b5a3bfa6ea4cdb2e09a6e"
sourceCorrections: []
---

将令牌路由到多个专家（k>1）是整合专门知识的常见方法，但它带来了显著的计算和通信开销。这种方法要求每个令牌由多个专家处理，其输出需加权并合并。由 Fedus、Zoph 和 Shazeer 提出的 Switch Transformer 架构，为这一过程提供了一项根本性简化：如果每个令牌只路由到一个专家，会怎样？

这种 top-1 路由（k=1）极大简化了 MoE 层。无需计算多个专家输出的加权和，该层对给定令牌的输出只是单个最适合的专家的输出，再通过路由器的门控分数进行缩放。

### Switch 层架构

Switch Transformer 的核心是其路由机制。对于输入令牌表示 $x$，门控网络计算所有 $N$ 个专家的分布。但它并非选择排名前 k 的专家，而只选择得分最高的单个专家。


$$
y = G(x) \cdot E_{\text{argmax}_i(G(x)_i)}(x)
$$


这里，$G(x)$ 是门控分数向量 (vector)，而 $E_i$ 是第 $i$ 个专家网络。这种设计选择带来了直接的益处：

- **计算量减少：** 令牌的前向传递只激活一个专家，而非 k 个专家。
- **通信简化：** 无需聚合多个专家的输出，减少了分布式环境中的通信带宽。
- **架构简化：** 相比于为每个令牌管理多个分配，其实现更直接。

下方图示展现了标准 MoE (k=2) 与 Switch 层 (k=1) 之间的差异。在标准模型中，每个令牌由两个专家处理。在 Switch 模型中，每个令牌精确地分派给一个专家。

> 两种令牌路由策略的比较。标准 MoE 将每个令牌发送到其排名前二的专家，而 Switch Transformer 只将每个令牌发送到其排名第一的专家。

### Switch Transformer 中的负载均衡

top-1 路由策略似乎会加剧负载不均衡的问题。如果路由器有强烈的偏好，它可能会持续选择同一专家，导致其他专家闲置。为缓解此问题，Switch Transformer 采用了一种改进的辅助负载均衡损失。

目标保持不变：即鼓励路由器将令牌均匀地分发给所有可用专家。损失是在一批 $T$ 个令牌和 $N$ 个专家上计算的。它是两个向量 (vector)的点积：

1. **分派给每个专家的令牌比例 ($f_i$)：** 该向量衡量每个专家接收到的批次令牌的比例。
2. **分配给每个专家的路由器概率比例 ($P_i$)：** 该向量衡量路由器在批次中为所有令牌分配给每个专家的平均概率。

辅助损失 $L_{aux}$ 定义为：


$$
L_{aux} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot P_i
$$


我们分解一下这些组成部分：

- $f_i = \frac{1}{T} \sum_{x \in \text{Batch}} \mathbb{I}(\text{argmax } G(x) = i)$: 批次中路由到专家 $i$ 的令牌比例。
- $P_i = \frac{1}{T} \sum_{x \in \text{Batch}} p_i(x)$: 批次中专家 $i$ 的平均路由器概率，其中 $p_i(x)$ 是路由器对令牌 $x$ 的专家 $i$ 的 softmax 输出。
- $\alpha$: 一个可调超参数 (parameter) (hyperparameter)，用于控制此辅助损失的权重 (weight)。原论文发现约 `0.01` 的值有效。

将 $f_i$ 和 $P_i$ 相乘，鼓励路由器的概率分布 $P$ 与实际令牌分发分布 $f$ 对齐 (alignment)。这促使路由器更均匀地分布其概率质量，从而带来更均衡的负载。

### 容量因子与丢弃令牌

k=1 路由的简化引入了一个新的实现细节：专家容量。在分布式设置中，每个专家（驻留在特定设备上）都分配有一个静态缓冲区，用于处理每批次一定数量的令牌。此缓冲区的大小由 `capacity_factor` (C) 决定。


$$
\text{专家容量} = \left( \frac{\text{每批次令牌数}}{N_{\text{experts}}} \right) \cdot C
$$


一个理想的、完美均衡的路由器将精确地向每个专家发送 $\frac{T}{N}$ 个令牌。容量因子为统计方差提供了缓冲区。例如，`capacity_factor` 为 1.25 意味着每个专家可以处理比平均值多 25% 的令牌。

如果专家的容量超限会怎样？Switch 架构做了另一个简化选择：令牌被丢弃。它不被任何专家处理。相反，其来自残差连接的表示直接传递到下一层。这等同于令牌在 MoE 层内通过一个恒等函数。虽然丢弃令牌似乎不利，但实验表明，通过良好调整的容量因子和辅助损失，丢弃令牌的百分比通常很低（<1%），并且不会显著影响整体模型性能。

### 性能与实现要点

Switch Transformer 的设计实现了引人注目的权衡，平衡了模型大小和计算成本。通过使用大量专家，模型的总参数 (parameter)量可以非常庞大，但由于每个令牌只激活一个专家，训练和推理 (inference)的浮点运算次数仍与小得多、紧密型的模型相当。



![模型参数与计算成本对比](plots/7055-0.json)



> 模型规模与计算的示意性比较。两种 MoE 模型都拥有紧密型模型 8 倍的参数量。标准 MoE (k=2) 使 FLOPs 增加了一倍以上，而 Switch Transformer (k=1) 只增加了少量的计算开销。

Switch Transformer 论文的作者也提到训练稳定性可能是一个难题。他们发现使用 BFloat16 等低精度格式很重要，特别是对于路由器的 softmax 计算，以防止由大 logit 值引起的不稳定。

总之，Switch Transformer 呈现出一种高效且可扩展的架构。其主要权衡点包括：

- **优点：** 计算效率高，路由逻辑简化，通信需求减少。
- **缺点：** 容量超限时可能丢弃令牌，对训练稳定性敏感度增加，需要仔细调整辅助损失和容量因子的超参数 (hyperparameter)。

## 参考资料

- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://jmlr.org/papers/v23/21-0990.html) — William Fedus, Barret Zoph, Noam Shazeer (2022)
  Journal: Journal of Machine Learning Research; Publisher: JMLR, Inc.; Volume: 23; Pages: 1-39
  介绍了Switch Transformer架构及其简化的top-1路由机制，以实现高效扩展。
- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1701.06538](https://doi.org/10.48550/arXiv.1701.06538)
  混合专家层的基础论文，概述了稀疏激活模型的普遍概念。
- [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841) — Shaden Smith, Brandon Norick, Sam Ade Jacobs, Jonathan Frankle, Jeremy Gray, Elias Frantar, Tal Ben-Nun, Dan Alistarh (2023)
  Journal: International Conference on Machine Learning (ICML); Volume: 202; Pages: 32363-32378
  侧重于大型稀疏混合专家模型内存高效且计算优化的训练。
