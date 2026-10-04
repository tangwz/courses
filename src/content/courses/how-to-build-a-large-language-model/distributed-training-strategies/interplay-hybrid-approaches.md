---
course: "how-to-build-a-large-language-model"
chapter: "distributed-training-strategies"
lesson: "interplay-hybrid-approaches"
sourceId: 6025
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-15-distributed-training-strategies/interplay-hybrid-approaches"
title: "混合并行策略 (DP+TP, DP+PP等)"
description: "讨论如何有效地组合不同的并行技术。"
order: 5
plots: []
sourceHash: "d1d9b81f10682d432ff0c921c7aab5832e6c11d8430f1849a254b76896593914"
sourceCorrections: []
---

数据并行（DP）、张量并行（TP）和流水线并行（PP）等方法各自在分布式训练中具备独特的优点。为了提升模型规模上限，通常需要结合使用这些技术。没有单一策略是普遍最佳的；适合的方法取决于具体的模型架构、硬件限制（GPU内存、互联带宽/延迟）和期望的吞吐量 (throughput)。混合方法让工程师能够更好地安排更精细的工作分配，平衡内存节省、计算效率和通信开销。

### 组合并行策略

混合方法的主要思路是同时运用多种并行方式的优点。一种常见做法是使用TP或PP来让模型能够 *容纳* 在一组设备上（处理内存限制），然后使用DP在这些集合的副本之间扩展训练吞吐量 (throughput)。

#### 数据并行 + 张量并行 (DP+TP)

这是一种常用的组合，尤其适合具有非常宽层级的模型，TP在这种情况下能大幅节省内存。

- **工作原理：** 模型在某些层中的参数 (parameter)（例如注意力或MLP块中的大型权重 (weight)矩阵）通过TP分割到一组设备上（例如2向、4向或8向TP）。整个TP组，共同拥有模型的一个分片实例，表现为一个逻辑设备。然后数据并行应用于此TP组的多个副本。每个副本处理全局数据批次中的不同分片。
- **通讯：** 这涉及两类通讯：
  1. **组内（TP）：** 在每个TP组内进行频繁、低延迟通讯（AllReduce、Scatter、Gather），以处理前向和后向传播期间的分片张量操作。这通常需要在节点内部使用NVLink等高带宽、低延迟互联。
  2. **组间（DP）：** 频率较低的AllReduce通讯在DP副本间进行（具体是指不同TP组中对应设备间的通讯），以在反向传播 (backpropagation)后同步梯度。这通常可以接受节点之间（如InfiniBand或以太网）更高延迟的互联。

> 一个2向TP x 2向DP配置。GPU 0和1构成一个TP组（DP 0级），GPU 2和3构成另一个（DP 1级）。TP通讯发生在组内（蓝色虚线），DP梯度同步发生在对应的TP等级间（红色实线）。

像NVIDIA的Megatron-LM这样的框架尤其适合用于实现TP，并提供将其与用于DP部分的标准PyTorch DistributedDataParallel (DDP) 集成的方法。

#### 数据并行 + 流水线并行 (DP+PP)

这种组合对深度模型很有效，其中PP用于降低激活所需的峰值内存，而DP则提升吞吐量。

- **工作原理：** 模型被划分为多个阶段，每个阶段分配给单个或一组设备（PP）。该完整流水线的多个实例被创建，构成DP副本。每个流水线副本处理全局数据批次中的不同组微批次。
- **通讯：**
  1. **流水线内（PP）：** 相邻流水线阶段之间的点对点通讯，用于前向传递激活，后向传递梯度。这通常涉及激活张量的发送/接收。
  2. **流水线间（DP）：** 跨DP副本的AllReduce通讯（具体是指不同副本中持有相同流水线阶段的设备之间），以同步每个阶段内各层的梯度。
- **气泡消除：** DP有助于减少PP固有的流水线气泡（空闲时间）。当一个流水线副本可能部分空闲等待依赖时，其他副本可以积极计算各自的微批次，从而提升整体硬件利用率。

> 一个2阶段PP x 2向DP配置。GPU 0和1构成一个流水线（DP 0级），GPU 2和3构成另一个（DP 1级）。PP通讯发生在阶段之间（绿线）。DP梯度同步发生在对应的阶段之间（红色虚线）。

DeepSpeed之类的库提供复杂的流水线并行实现，可以轻松地与其基于ZeRO的数据并行结合使用。

#### 张量并行 + 流水线并行 (TP+PP) 和 “3D” 并行 (DP+TP+PP)

对于最大的模型，参数量超过数千亿或万亿时，可能需要将所有三种主要策略结合起来。这通常称之为“3D”并行。

- **工作原理：**
  - **TP+PP：** 流水线中的每个阶段可能对单个设备来说过大，因此使用TP在多个设备上并行化该阶段 *内部* 的层。这构成一个“阶段组”。通讯通过TP在阶段组 *内部* 进行，并通过PP在阶段组 *之间* 进行。
  - **DP+TP+PP：** 在此之上添加数据并行。创建TP+PP配置的多个副本，每个副本处理不同的数据分片。
- **复杂程度：** 这种配置在管理设备映射、通讯调度和潜在的负载均衡问题时带来显著复杂性。通讯模式变得复杂，涉及阶段组内的TP集合操作、阶段组间的PP点对点通讯，以及跨副本的DP AllReduce。
- **应用场景：** 适用于同时满足以下条件下的模型：单个层很大（需要TP）、模型深度很深（需要PP），并且需要高吞吐量（需要DP）。

#### ZeRO (Zero Redundancy Optimizer) 优化

ZeRO，特别是ZeRO Stage 3，并非与DP、TP和PP相同的并行维度，而是一种优化数据并行内存使用的技术。它将优化器状态、梯度以及（可选地）参数本身分布到数据并行等级上。ZeRO几乎总是与其他策略 *配合使用*：

- **ZeRO-DP + TP：** ZeRO降低DP的内存负担，使TP能够专注于在必要时分割模型参数和激活。这种组合使得每个节点能够容纳更大的模型，并扩展更宽的模型。
- **ZeRO-DP + PP：** ZeRO降低由DP组管理的每个流水线阶段内权重和优化器状态的内存占用，补充了PP带来的激活内存节省。这对深度模型有效。

DeepSpeed是用于实现ZeRO的具有代表性的框架，并提供集成功能，能有效结合TP和PP（通常借助Megatron-LM的TP实现）。

### 实现考量

选择并应用合适的混合策略需要仔细分析：

- **模型结构：** 宽模型更适合TP；深度模型更适合PP。
- **硬件：** TP需要低延迟的节点内连接（NVLink）。PP性能很大程度上取决于节点间带宽和流水线调度隐藏通讯延迟的能力。DP的扩展性取决于AllReduce集合操作的性能。
- **内存、计算与通讯：** 每种策略都会改变瓶颈。TP节省内存但增加层内通讯。PP节省激活内存但引入气泡和层间通讯。DP提高计算吞吐量 (throughput)但需要副本的内存（由ZeRO减轻），并增加梯度通讯。
- **框架支持：** 从零开始实现这些复杂策略非常困难。依赖DeepSpeed和Megatron-LM等框架几乎总是必须的。这些框架提供抽象和优化的通讯集合操作。

一个PyTorch代码片段，展示如何组合这些（使用类似于DeepSpeed或Megatron-LM中的高级API），可能如下所示：

```python
import torch
import torch.distributed as dist
from some_framework import (
    initialize_parallelism,
    get_data_parallel_group,
    get_tensor_parallel_group,
    get_pipeline_parallel_group,
    PipelineModule,
    TensorParallelLinear, # 示例TP层
    ZeROOptimizer # 示例ZeRO集成
)

# 假设环境变量或配置文件已设置等级/组
# 示例：2向DP，4向TP，2阶段PP（总计2*4*2 = 16个GPU）
initialize_parallelism(
    data_parallel_size=2,
    tensor_parallel_size=4,
    pipeline_parallel_size=2
)

# 根据需要使用TP层定义模型部分
class Stage0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 输入嵌入层可能使用张量并行
        self.embedding = TPInputEmbedding(...)
        # 一些Transformer层，其中可能使用TP
        self.layer1 = TPLayer(...)
        self.layer2 = TPLayer(...)

    def forward(self, x):
        # ... 阶段0的前向传播 ...
        return self.layer2(self.layer1(self.embedding(x)))

class Stage1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 更多层
        self.layer3 = TPLayer(...)
        # 输出层可能使用TP
        self.output = TPOutputLayer(...)

    def forward(self, x):
        # ... 阶段1的前向传播 ...
        return self.output(self.layer3(x))

# 创建流水线模型
model = PipelineModule(
    stages=[Stage0(), Stage1()],
    num_microbatches=8 # 示例微批次配置
)

# 使用ZeRO包装优化器（ZeRO理解DP组）
optimizer = ZeROOptimizer(
    model.parameters(),
    lr=1e-4,
    # ZeRO配置选项...
)

# 训练循环（简化版）
for data in dataloader:
    optimizer.zero_grad()
    # PipelineModule处理跨阶段的前向/后向传播
    # 并在内部处理微批次
    loss = model(data)
    optimizer.step() # ZeRO处理DP组的梯度平均
```

> PyTorch代码，呈现如何使用张量并行层（`TPLayer`、`TPInputEmbedding`）定义模块并组合成一个`PipelineModule`。`ZeROOptimizer`隐式处理数据并行维度上的梯度同步。

成功训练大型模型通常需要对不同的混合配置（例如，改变TP大小、PP阶段、微批次大小）进行迭代实验，以找到最大化硬件利用率和最小化给定模型和集群架构训练时间的最佳平衡点。因此，理解这些策略之间的关联对于任何从事大规模模型训练的工程师都非常重要。

## 参考资料

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) — Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He (2020)
  Journal: SC20: International Conference for High Performance Computing, Networking, Storage and Analysis; DOI: [10.48550/arXiv.1910.02054](https://doi.org/10.48550/arXiv.1910.02054)
  引入了ZeRO优化器，一种面向大规模模型的内存高效数据并行方法。
- [DeepSpeed Features: Model Parallelism](https://www.deepspeed.ai/) — Microsoft DeepSpeed Team (2023)
  Publisher: Microsoft DeepSpeed Team
  提供了ZeRO、流水线并行以及DeepSpeed-Megatron-LM集成实现混合并行策略的概述。
- [Efficient Large-Scale Language Model Training: A Case Study with Megatron-LM](https://developer.nvidia.com/blog/efficient-large-scale-language-model-training-a-case-study-with-megatron-lm/) — NVIDIA Developer (2021)
  Journal: NVIDIA Developer Blog; Publisher: NVIDIA
  NVIDIA提供的一份实用指南，演示了如何使用Megatron-LM的并行策略训练大型模型。
