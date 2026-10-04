---
course: "mixture-of-experts"
chapter: "scaling-moe-distributed-training"
lesson: "pipeline-parallelism-moe"
sourceId: 4702
sourceUrl: "https://apxml.com/zh/courses/mixture-of-experts/chapter-4-scaling-moe-distributed-training/pipeline-parallelism-moe"
title: "MoE 模型的流水线并行"
description: "将流水线并行应用于 MoE 架构，以进一步缓解内存限制。"
order: 5
plots: []
sourceHash: "5e498d9cd63f642ecf15506197ceec6e51f053cf5ee031d38ef5d7d2d858703d"
sourceCorrections: []
---

虽然专家并行（EP）直接处理 MoE 层内专家的分布，数据并行（DP）复制整个模型，但流水线并行（PP）提供了一种补充方法来管理大型模型（包括包含 MoE 层的模型）可观的内存占用和计算图深度。流水线并行将模型的层（而非数据或专家本身）划分为顺序阶段，并将每个阶段分配给一组不同的处理设备。

设想一个深度 Transformer 模型。与将所有层放置到每个设备上（如纯 DP）或仅将 MoE 专家拆分到不同设备上（如纯 EP）不同，PP 划分层序列。例如，第 1-8 层可能在设备组 A 上构成阶段 1，第 9-16 层在设备组 B 上构成阶段 2，依此类推。

### 微批处理与流水线执行

PP 的简单实现会通过阶段 1 处理完整数据批次，将激活数据传递给阶段 2，在那里处理，然后顺序继续。这会导致设备大量空闲时间，因为任何时刻只有一个阶段处于活跃状态。为提高利用率，PP 采用微批处理。完整数据批次被拆分为更小的微批次（$m_1, m_2, ..., m_k$）。

执行以交错方式进行：

1. 阶段 1 处理 $m_1$。
2. 阶段 1 处理 $m_2$，同时阶段 2 处理来自阶段 1 的 $m_1$ 输出。
3. 阶段 1 处理 $m_3$，阶段 2 处理 $m_2$ 的输出，阶段 3 处理 $m_1$ 的输出，等等。

这产生了一个“流水线”效应，使多个阶段同时保持忙碌。前向传播将微批次传递通过各个阶段，反向传播 (backpropagation)则以相反顺序将梯度传回。

> 包含 3 个阶段和微批处理的流水线并行可视化。F 表示前向传播，B 表示反向传播，m 表示微批次索引。空闲期（气泡）发生在调度开始和结束时。

### 整合流水线并行与 MoE 层

MoE 层如何与流水线阶段配合？通常，整个 MoE 层（门控网络和相关专家）位于单个流水线阶段内。由于将 token 路由到专家涉及复杂的依赖关系和通信，通常避免将 MoE 层*本身*跨越阶段边界进行拆分。

因此，包含 MoE 层的流水线阶段内部结构可能如下：

1. 输入激活数据来自前一阶段（或初始嵌入 (embedding)层）。
2. 这些激活数据由当前阶段*内部*的层处理，可能包括标准 Transformer 块。
3. 如果存在 MoE 层：
   - 门控网络计算每个 token 的专家分配。
   - All-to-All 通信*在该阶段分配的设备组内*发生，以将 token 路由到其指定专家（假设该阶段内也使用了 EP）。
   - 专家处理其分配的 token。
   - 输出被合并，并可能传递给同一阶段内的后续层。
4. 阶段的最终激活数据传递给下一个流水线阶段。

### 混合并行策略：PP + DP + EP

对于真正大型的 MoE 模型，流水线并行与数据并行和专家并行结合使用时最有效。常见配置是：

1. **流水线并行（PP）：** 模型的层被划分为跨不同节点或 GPU 组的阶段。这解决了激活内存瓶颈，并允许模型规模突破单节点内存容量的限制。
2. **数据并行（DP）：** 在每个流水线阶段内，分配给该阶段的层被复制到多个 GPU。每个 GPU 处理不同的微批次（或其中一部分）。这扩展了阶段内的计算。
3. **专家并行（EP）：** 对于流水线阶段内的任何 MoE 层，专家分布在该阶段的数据并行副本上。token 路由的 All-to-All 通信发生在该特定阶段中参与的 GPU 之间。

> 结合了 PP、DP 和 EP 的混合并行。阶段通过 PP 形成。在每个阶段内，DP 复制阶段逻辑，EP 将专家分布在 DP 秩上。EP 的 All-to-All 发生在一个阶段*内部*，而 PP 的激活/梯度则在阶段之间流动。

### MoE 模型中 PP 的注意事项

- **流水线气泡：** 尽管微批处理有帮助，但流水线启动和结束阶段仍然会引入空闲时间（气泡）。高级调度技术，如交错流水线调度（例如 1F1B 调度），可以通过更有效地重叠前向和后向传播来进一步缓解此问题。
- **阶段负载均衡：** 实现最佳吞吐量 (throughput)需要平衡流水线阶段之间的计算负载。MoE 层通常比标准密集层计算密集得多。放置 MoE 层需要仔细考虑，避免使某个阶段成为主要瓶颈。专家数量和专家大小影响阶段持续时间。
- **内存均衡：** 类似地，内存需求（参数 (parameter)和激活）应在阶段之间均衡。PP 在分配激活内存方面表现优异，激活内存会随着序列长度和批次大小扩展。MoE 层增加参数内存，在阶段内通过 EP 分配。
- **通信：** PP 引入了阶段间通信开销，用于激活（前向）和梯度（反向）。这通常是相邻阶段设备组之间的点对点或集体通信。必须权衡此通信成本与包含 MoE 层的阶段内部的 All-to-All 成本。优化两者非常重要。
- **框架复杂性：** 实现混合并行策略需要复杂的分布式训练框架，如 DeepSpeed、Megatron-LM 或 Tutel，这些框架提供抽象来管理 PP、DP 和 EP/TP（张量并行）之间复杂的交互。

流水线并行，特别是当它整合到混合策略中时，提供了一种强有力的机制，使得 MoE 模型能够突破单节点内存和计算的限制，通过有效地在层和设备间划分工作负载，使得训练具有万亿参数的模型成为可能。

## 参考资料

- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://proceedings.neurips.cc/paper/2019/file/0c9a5s501e4a60dfc0f08969799c5123-Paper.pdf) — Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Dehao Chen, Mia Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V Le, Yonghui Wu, Zhifeng Chen (2019)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 32; DOI: [10.5555/3454287.3455171](https://doi.org/10.5555/3454287.3455171)
  介绍了用于训练大型神经网络的带有微批处理的流水线并行技术，是该领域的奠基性工作。
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) — Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro (2019)
  Journal: arXiv preprint arXiv:1909.08053; DOI: [10.48550/arXiv.1909.08053](https://doi.org/10.48550/arXiv.1909.08053)
  描述了用于高效训练数十亿参数语言模型的模型并行技术，其中包括流水线并行。
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/pdf/2101.03961.pdf) — William Fedus, Barret Zoph, and Noam Shazeer (2022)
  Journal: Journal of Machine Learning Research; Publisher: Journal of Machine Learning Research; Volume: 23; Pages: 1-39; DOI: [10.48550/arXiv.2101.03961](https://doi.org/10.48550/arXiv.2101.03961)
  详细介绍了Switch Transformers的架构和分布式训练策略，这是一种扩展到万亿参数的稀疏专家混合模型。
