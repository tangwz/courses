---
course: "mixture-of-experts-advanced-implementation"
chapter: "advanced-routing-mechanisms"
lesson: "soft-moe-differentiable-routing"
sourceId: 7056
sourceUrl: "https://apxml.com/zh/courses/mixture-of-experts-advanced-implementation/chapter-2-advanced-routing-mechanisms/soft-moe-differentiable-routing"
title: "软MoE：可微分路由"
description: "了解软MoE，一种使用所有专家加权组合来创建完全可微分路由系统的方法。"
order: 5
plots: []
sourceHash: "7aaa9536626de0f70ef2222ef410b6f5c88801eaae7e5e3b133ac1a0866a1ae5"
sourceCorrections: []
---

路由机制，例如top-k和Switch门控，执行“硬性”分配。一个令牌被路由到一个离散的、少数专家集合，而所有其他专家在该特定令牌的计算中被忽略。这种硬性选择是稀疏模型计算节省的原因，但也引入了不可微分性和需要辅助负载均衡损失等难题。

软MoE提供了一种不同方法，用所有专家的“软性”加权组合来替代这种离散选择。门控网络不再是选择使用哪些专家，而是确定每个专家的权重 (weight)，最终输出是所有专家输出的加权和。这使整个MoE层完全可微分，并巧妙地避开了与硬性门控相关的训练不稳定性。

### 软路由的机制

在软MoE层中，门控网络通过为每个专家生成对数来运作，方式与标准路由器相似。然而，我们不是用这些对数来选择top-k专家，而是对它们应用`softmax`函数。这会将对数转换为一组和为一的正权重 (weight)，从而有效地形成专家上的概率分布。

输入令牌 $x$ 的最终输出不是少数被选专家的输出，而是所有 $N$ 个专家输出的线性组合。每个专家 $E_i(x)$ 的贡献由其对应的softmax权重 $w_i$ 加权。

数学公式是直接的。给定输入 $x$，门控网络 $G$ 计算一个对数向量 (vector) $h(x)$。然后权重 $w$ 计算如下：


$$
w = \text{softmax}(h(x))
$$


软MoE层的最终输出 $y$ 是加权和：


$$
y = \sum_{i=1}^{N} w_i \cdot E_i(x)
$$


这个公式可能看起来很熟悉。它与注意力机制 (attention mechanism)非常相似，查询关注一组键以产生权重，然后这些权重用于计算值的加权和。在软MoE中，可以将令牌的表示视为查询，并将专家视为键和值。

下图说明了硬门控MoE和软MoE中数据流的区别。

> 在硬路由中，门控网络选择一个离散专家（专家1），所有计算都通过它进行。其他专家对该令牌保持不活跃。

> 在软MoE中，门控网络计算每个专家的权重。最终输出是所有专家输出的加权组合。

### 权衡：可微分性与计算量

软MoE的主要优势在于它为稀疏模型的训练难题提供了一个精妙的解决方案。

- **完全可微分性：** 整个层，包括路由决策，是一个连续函数。这使得梯度在反向传播 (backpropagation)期间能够平稳地流过整个网络。
- **隐式负载均衡：** 由于每个专家都获得非零权重 (weight)（无论多小），因此每个专家都参与每个令牌的前向和反向传播。这自然地避免了“专家崩溃”现象，即某些专家没有接收到训练信号。无需显式辅助损失来促进负载均衡。

然而，这种精妙性伴随着巨大的且通常令人望而却步的代价。

- **计算开销：** 稀疏MoE的根本益处在于，对于任何给定输入，它只激活模型参数 (parameter)的一小部分，从而将参数数量与计算成本（FLOPs）分离。软MoE完全消除了这一优势。由于每个令牌都由*所有专家*处理，因此计算成本等同于运行 $N$ 个模型的集成，然后对其输出进行平均。这使得它像参数数量等于所有专家总和的密集模型一样昂贵。
- **内存占用：** 在稀疏模型中，只有活跃专家的权重需要随时在高速内存（例如，GPU HBM）中可用。在软MoE中，所有专家权重必须同时加载，这导致巨大的内存占用，对于拥有大量专家的模型而言不切实际。

### 何时选用软MoE

考虑到其计算需求，在将计算效率作为主要设计目标的大规模语言模型中，“纯”软MoE很少使用。它的公式更多地是作为理论基准和分析工具。

然而，软性、可微分分配的核心思想影响了更实用的混合系统的设计。例如，一些方法可能会使用top-k路由器选择一小部分专家，然后*在该子集内*计算软性加权组合。这可以提供软路由的一些训练稳定性，同时保留稀疏性的大部分计算优势。

理解软MoE很重要，因为它清楚地界定了训练中的数学简洁性与扩展所需的计算稀疏性之间的权衡。它代表了MoE设计谱系的一端，在此训练稳定性以牺牲推理 (inference)效率为代价最大化。这与像Switch Transformers这样的机制形成有价值的对比，后者通过将计算效率置于首位而占据了谱系的另一端。

## 参考资料

- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean (2017)
  Journal: arXiv preprint arXiv:1701.06538; DOI: [10.48550/arXiv.1701.06538](https://doi.org/10.48550/arXiv.1701.06538)
  这篇基础论文介绍了稀疏门控专家混合（MoE）层，通过与密集（软）门控的对比，为理解其背景提供了依据，并讨论了稀疏性的计算优势。
- [Learning to Route: A Differentiable Approach to Mixture of Experts](https://proceedings.mlr.press/v80/rosenbaum18a.html) — Clemens Rosenbaum, Chetan Sanan, Charith Gunasekara, Josh Trani, Andrew Gordon Wilson, Kyunghyun Cho (2018)
  Journal: Proceedings of the 35th International Conference on Machine Learning (ICML); Publisher: PMLR; Volume: 80; Pages: 4392-4401; DOI: [10.5555/3326938.3326955](https://doi.org/10.5555/3326938.3326955)
  本文提出了一种在专家混合模型中实现可微分路由的方法，与本节中对软路由的讨论直接相关。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS) 30; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍了Transformer架构及其自注意力机制，这为软路由中的加权求和计算提供了很好的类比。
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — William Fedus, Barret Zoph, Noam Shazeer (2021)
  Journal: arXiv; Pages: 1-39; DOI: [10.48550/arXiv.2101.03961](https://doi.org/10.48550/arXiv.2101.03961)
  介绍了Switch Transformers，这是硬门控专家混合的一个代表性例子，展示了稀疏性的计算效率，而软专家混合则与之不同。
