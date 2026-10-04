---
course: "mixture-of-experts"
chapter: "scaling-moe-distributed-training"
lesson: "frameworks-distributed-moe"
sourceId: 4704
sourceUrl: "https://apxml.com/zh/courses/mixture-of-experts/chapter-4-scaling-moe-distributed-training/frameworks-distributed-moe"
title: "用于分布式MoE的框架和库 (例如：DeepSpeed, Tutel)"
description: "概述有助于高效实现分布式MoE的库。"
order: 7
plots: []
sourceHash: "8cf02b25c4731611b6d38782bb7e672ca5eea9f6c329f8f38601b37c4ffe38f7"
sourceCorrections: []
---

实施大型专家混合（MoE）模型所需的复杂分布式训练策略，特别是管理专家并行（Expert Parallelism）和相关的All-to-All通信模式，带来了显著的工程挑战。手动协调数据移动、同步梯度以及整合不同并行维度（数据、专家、流水线、张量）既复杂又容易出错。幸运的是，一些专用框架和库已经出现，它们抽象了大部分这种复杂性，使得研究人员和工程师能够更高效地扩展MoE模型。

本节考察重要的软件工具，旨在促进分布式MoE训练，侧重于它们的架构、功能和典型使用场景。

### DeepSpeed: 集成式MoE扩展

DeepSpeed 由微软开发，是一个全面的深度学习 (deep learning)优化库，旨在处理各种维度的大模型训练。它在其现有并行策略中集成了MoE支持，其中最显著的是ZeRO（零冗余优化器）系列。

**MoE的特点:**

1. **集成并行:** DeepSpeed 允许将专家并行与数据并行（由ZeRO阶段1、2和3管理）以及流水线并行结合。这种统一的方法简化了MoE所需的复杂混合并行策略的配置。用户通常只需很少的代码修改，主要是通过配置文件即可启用MoE支持。
2. **高效的All-to-All:** DeepSpeed 包含了All-to-All集体通信的优化实现，这对于专家并行中设备间令牌的路由是必需的。它旨在通过使用高效的底层通信库（如NCCL）来减少通信开销，并尽可能地将通信与计算重叠。
3. **配置驱动:** 在DeepSpeed中设置MoE并行通常涉及修改JSON配置文件。参数 (parameter)通常包括启用MoE、指定每个设备（或总共）的专家数量，以及配置相关的ZeRO和流水线设置。这种声明式方法相比手动实现降低了使用门槛。

```json
{
  "train_batch_size": 1024,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1e-5
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 1e-5,
      "warmup_num_steps": 100
    }
  },
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 1
  },
  "pipeline": {
      "stages": "auto",
      "pipe_partition_method": "parameters",
      "gradient_accumulation_steps": 1
  },
  "moe": {
    "enabled": true,
    "ep_size": 8,
    "num_experts": 64,
    "loss_coef": 0.1
  }
}
```

> DeepSpeed JSON配置的一个简化示例，在跨设备启用MoE时，专家并行大小（`ep_size`）为8，总专家数为64的模型。

DeepSpeed 的优势在于为大模型训练提供了一个整体系统，其中MoE是更广泛优化技术套件中的一个组成部分。当您已经将DeepSpeed用于其他扩展方面或寻求集成式解决方案时，它是一个合适的选择。

### Tutel: 优化的MoE计算核和通信

Tutel 也源自微软研究院，是一个更为专用的库，专注于优化分布式环境中的MoE层。尽管DeepSpeed提供系统级集成，但Tutel侧重于最大化MoE计算和通信本身的性能。

**MoE的特点:**

1. **高度优化的All-to-All:** Tutel 的主要贡献在于其高度优化的All-to-All实现，该实现专为MoE路由中出现的稀疏、不规则通信模式而设计。它通常通过采用自适应路由算法和拓扑感知通信调度等技术，比通用All-to-All原语实现更优的通信性能。
2. **融合计算核:** Tutel 通常包含定制的CUDA计算核，它们融合了MoE层内的操作（例如，门控计算、数据分发、专家计算、数据组合），从而减少了计算核启动开销并改善了GPU上的内存局部性。
3. **灵活性和模块化:** Tutel 被设计为可以集成到各种训练框架中。尽管它与Fairscale或自定义PyTorch设置等框架有紧密集成示例，但其专注于特定任务的特性允许开发人员将其优化的MoE层作为组件集成到更大、可能自定义的训练基础设施中。

> 令牌流经由Tutel或DeepSpeed等库在双设备专家并行设置中管理的优化过的All-to-All机制。令牌（蓝色）从其源设备分派到持有其指定专家（绿色）的设备。

当All-to-All通信被认为是主要瓶颈且希望MoE层本身达到最高性能时，Tutel尤其有利。它可能比DeepSpeed需要更多的集成工作，但能为MoE特有的计算部分提供显著的加速。

### 其他框架和考虑

尽管DeepSpeed和Tutel是重要示例，其他库和框架也对MoE扩展有所贡献:

- **Megatron-LM:** 由NVIDIA开发，Megatron-LM 侧重于使用张量并行和流水线并行来扩展Transformer。尽管它最初的核心不是MoE，但其先进的并行方法和优化过的计算核已经影响并有时与MoE实现结合。结合其张量切分方法与专家并行的研究仍在进行中。
- **Fairscale:** Fairscale 最初来自Facebook AI (Meta)，提供了各种并行技术的实现，包括早期的MoE支持以及Tutel等库可以使用的集成点。它的开发已放缓，但其贡献仍然重要。
- **自定义实现:** 对于特定的研究目标或硬件配置（例如，专用互连、TPU），团队可能会开发定制的MoE层和通信策略，以满足其确切需求，可能会基于PyTorch的`distributed`模块或JAX的`pmap`/`shmap`等库中的基本功能进行构建。

**选择库:**

这些库的选择取决于具体需求:

- 对于一个能处理多种并行形式、MoE性能适中且配置更简便的集成系统，**DeepSpeed**通常是一个不错的选择。
- 当MoE通信是主要的瓶颈，并需要顶尖的优化时，这可能需要更多的集成工作，**Tutel**提供专业的、高性能的解决方案。
- 考虑现有生态：如果一个项目已经大量依赖Megatron-LM或需要由底层功能提供的细粒度控制，其他方法可能更适合。

使用这些框架显著降低了实现分布式MoE训练的复杂性。然而，理解专家并行、All-to-All通信的底层原理以及潜在瓶颈对于在这些强大的抽象中进行有效配置、性能调优和调试仍然很重要。有效使用通常需要仔细分析性能，以确定在特定设置中是通信、专家计算还是内存限制是制约因素。

## 参考资料

- [Switch Transformers: Scaling to Trillion Parameter Models with Sparsely Activated Expert Layers](https://arxiv.org/abs/2101.03961) — William Fedus, Barret Zoph, Noam Shazeer (2021)
  Journal: Proceedings of Machine Learning Research (PMLR); Volume: 139; Pages: 3964-3976; DOI: [10.48550/arXiv.2101.03961](https://doi.org/10.48550/arXiv.2101.03961)
  介绍了Switch Transformers，一个大规模的MoE模型，展示了稀疏MoE架构的应用及其带来的分布式训练挑战。
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) — Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro (2019)
  Journal: arXiv preprint arXiv:1909.08053; DOI: [10.48550/arXiv.1909.08053](https://doi.org/10.48550/arXiv.1909.08053)
  介绍了Megatron-LM使用模型并行技术扩展大型语言模型的方法，为通常与MoE结合的通用分布式训练策略提供了背景。
