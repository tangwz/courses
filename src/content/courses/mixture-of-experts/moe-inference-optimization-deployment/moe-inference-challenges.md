---
course: "mixture-of-experts"
chapter: "moe-inference-optimization-deployment"
lesson: "moe-inference-challenges"
sourceId: 4706
sourceUrl: "https://apxml.com/zh/courses/mixture-of-experts/chapter-5-moe-inference-optimization-deployment/moe-inference-challenges"
title: "稀疏模型的推理难题"
description: "讨论 MoE 模型推理期间的延迟、吞吐量和内存问题。"
order: 1
plots: ["plots/4706-0.json"]
sourceHash: "e224b4c5da668caf5b822943d80b3b4f2f2fe33c6a5145ab159dc32d9a96223d"
sourceCorrections: []
---

虽然专家混合（MoE）模型在训练时仅激活每输入参数 (parameter)的一小部分，从而带来可观的计算节省，但正是这种稀疏性在优化推理 (inference)性能时带来了特有的难题。要实现低延迟、高吞吐量 (throughput)和高效内存利用，需要仔细考虑稀疏条件计算的特点。这些难题通常要求采用专门的技术来进行模型部署。

### 延迟考量

推理 (inference)期间的一个主要目标是最小化延迟，即处理单个输入或小批次所需的时间。对于MoE模型，与具有相似计算预算（FLOPs）的稠密模型相比，有几个因素可能导致延迟增加：

1. **路由开销：** 在专家计算开始前，门控网络必须处理每个token的表示，以确定应将其路由到哪个或哪些专家。这种路由计算通常包含一个线性层和softmax，为每个MoE层的主要流程增加了一点但不容忽视的延迟。虽然相对于专家计算本身来说，这种开销通常较小，但在深度网络中，它会随着多个MoE层而逐层叠加。
2. **条件计算流程：** 路由是随数据变化的，其执行流程无法提前完全预知。这会阻碍某些编译器优化和硬件预取机制，因为它们依赖于稠密模型中常见的、可预期的计算模式。
3. **通信延迟（分布式推理）：** 大型MoE模型由于其庞大的总参数 (parameter)量，通常需要将专家分散到多个加速器（GPU/TPU）上。在推理期间，在一个设备上处理的token可能需要路由到位于另一个设备上的专家。这使得需要高效的全连接（All-to-All）通信原语来在设备间传输token表示。此通信步骤的延迟可能成为一个重要的瓶颈，特别是当设备数量增加时。网络带宽和通信集合体的效率直接关联此延迟部分。

因此，简单地比较MoE模型的活跃FLOPs与稠密模型，并不能完整展现推理延迟的情况。门控机制和潜在的通信开销必须纳入性能分析中。

### 吞吐量 (throughput)难题

吞吐量，通常以每秒处理的token数量衡量，是另一个重要的推理 (inference)指标，特别是对于同时服务众多用户的应用。稀疏性在最大化吞吐量方面带来特定难题：

1. **负载不均衡：** 门控网络根据学习到的专业化来路由token。在推理期间，由于没有训练时使用的辅助负载均衡损失，token在专家间的分布可能变得非常不均衡。某些专家可能会收到过多数量的token，而其他专家则处于空闲或未充分利用状态。这种不均衡妨碍了所有可用专家间的有效并行处理，从而限制了整个系统的吞吐量。一个过载的专家会成为瓶颈，即使其他专家有可用容量，也会使流水线停滞。

> 推理负载不均衡的示意图，其中门控网络将大多数token路由到专家3，造成了瓶颈。

2. **专家容量限制：** MoE实现通常定义一个容量系数，限制单个专家在一个批次中可处理的token数量，以确保可预期的内存分配和计算形状。如果负载不均衡导致路由到某个专家的token数量超出其容量允许，这些多余的token可能会被丢弃或需要复杂的缓冲机制，这两者都会对吞吐量和潜在的模型质量产生负面影响。
3. **硬件利用效率低下：** 现代加速器针对稠密矩阵乘法进行了很好地优化。稀疏计算，涉及为选定专家收集权重 (weight)并执行较小、可能不规则的计算，可能导致可用计算单元（例如NVIDIA GPU上的Tensor Cores）未能充分利用。获得最佳硬件性能通常需要专门的计算核和细致的批处理策略，这些会在后续章节中讨论。

### 内存占用与带宽

影响MoE部署的最显著特点可能是内存需求。

1. **总参数 (parameter)量庞大：** 尽管每个token只有少数专家活跃，但通常需要将*所有*专家参数加载到加速器的高带宽内存（HBM）中，才能使模型正常运行。MoE模型的总参数量可能是每前向传播计算成本相近的稠密模型的5-10倍（甚至更多）。这种庞大的参数占用通常会超过单个GPU或TPU的内存容量，即使在推理 (inference)期间也需要进行模型分发（专家并行）。

   
   
   ![推理时的参数规模](plots/4706-0.json)
   
   

   > 稠密模型与一个设计为每token计算成本相近、但规模大得多的MoE模型在内存中存储的总参数量与每token活跃使用的参数量对比。MoE的总内存需求大得多。
2. **内存带宽瓶颈：** 即使总参数量能容纳在多个加速器的聚合内存中，性能仍可能受限于内存带宽。对于每个token（或微批次），选定专家的权重 (weight)必须从HBM获取到计算单元。如果路由模式变化很大或专家权重很大，这些权重加载的速度可能成为推理速度的限制因素，甚至超过了计算成本本身。

### 实现复杂性

与稠密模型相比，高效部署MoE模型通常需要更精密的架构。标准的推理 (inference)服务器和库可能欠缺对以下方面的优化支持：

- **动态路由核：** 高效执行门控逻辑并选择合适的专家权重 (weight)。
- **优化过的全连接（All-to-All）集合通信：** 处理分布式专家并行所需的token混洗，同时保持很小的开销。
- **处理稀疏性：** 专门的计算核，借助稀疏性避免不必要的计算或内存访问。
- **集成批处理策略：** 实现适应MoE工作负载的动态或容量感知批处理。

像DeepSpeed和Tutel这样的框架，在训练背景下已经讨论过，它们也提供用于优化MoE推理的功能，但它们的集成和调优使部署流程变得复杂。

解决这些延迟、吞吐量 (throughput)、内存和实现方面的难题，是高效部署MoE模型的必要前提。接下来的章节将阐述特定的优化方案，包括高级批处理、模型压缩、硬件适应性调整以及旨在减轻这些固有问题的部署模式。

## 参考资料

- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1701.06538](https://doi.org/10.48550/arXiv.1701.06538)
  奠基性工作，介绍了混合专家架构，讨论了稀疏性在模型扩展中的必要性，并暗示了其计算和内存方面的影响。
- [DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training for Large Language Models](https://proceedings.mlr.press/v162/rajbhandari22a.html) — Samyam Rajbhandari, Conglong Li, Zhewei Yao, Minjia Zhang, Reza Yazdani Aminabadi, Ammar Ahmad Awan, Jeff Rasley, Yuxiong He (2022)
  Journal: Proceedings of the 39th International Conference on Machine Learning (ICML 2022); Publisher: PMLR; Volume: 162; Pages: 18332-18346; DOI: [10.48550/arXiv.2201.05596](https://doi.org/10.48550/arXiv.2201.05596)
  描述了MoE模型的具体推理挑战，并提出了优化延迟、吞吐量和内存利用率的解决方案，包括负载均衡和通信技术。
- [Tutel: Adaptive Mixture-of-Experts Training for Faster Convergence and Better Quality](https://ieeexplore.ieee.org/document/10137336) — Zhengda Chen, Xiaoxia Wu, Minjia Zhang, Yuxiong He (2023)
  Journal: IEEE Transactions on Parallel and Distributed Systems; Publisher: IEEE; Volume: 34; Pages: 2623-2634; DOI: [10.1109/TPDS.2023.3283259](https://doi.org/10.1109/TPDS.2023.3283259)
  详细介绍了一个MoE系统，侧重于优化执行，包括在推理过程中管理通信开销和负载不均衡的技术。
- [GLaM: Efficient Scaling of Language Models with MoE](https://arxiv.org/abs/2112.06905) — Nan Du, Yanping Huang, Andrew M. Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, Barret Zoph, Liam Fedus, Maarten Bosma, Zongwei Zhou, Tao Wang, Yu Emma Wang, Kellie Webster, Marie Pellat, Kevin Robinson, Kathleen Meier-Hellstern, Toju Duke, Lucas Dixon, Kun Zhang, Quoc V Le, Yonghui Wu, Zhifeng Chen, Claire Cui (2022)
  Journal: International Conference on Machine Learning (ICML) 2022; DOI: [10.48550/arXiv.2112.06905](https://doi.org/10.48550/arXiv.2112.06905)
  提出了一种大规模MoE语言模型，讨论了其在推理过程中的效率的实际考虑，包括大规模内存和计算方面的因素。
