---
course: "how-to-build-a-large-language-model"
chapter: "implementing-distributed-training-frameworks"
lesson: "combining-frameworks-strategies"
sourceId: 6032
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-16-implementing-distributed-training-frameworks/combining-frameworks-strategies"
title: "结合框架与策略"
description: "讨论结合使用多个库功能的方法。"
order: 6
plots: []
sourceHash: "427b66bb15e93435deed43e58f805678a7feb140d47a5c9f3e6e28a8d8db99c3"
sourceCorrections: []
---

虽然DeepSpeed和Megatron-LM等框架为实现特定并行策略提供了强大的工具，但训练非常大的模型通常需要同时结合多种技术。当模型规模和硬件能力的界限被推向极致时，单独使用数据并行（DP）、张量并行（TP）或流水线并行（PP）可能不足或并非最优。仅依靠DP即使有ZeRO等优化也可能达到内存限制。仅依靠TP可能导致大量设备间的通信开销过高。单独使用PP会引入流水线气泡，降低利用率。因此，复杂的训练设置常会混合这些策略，利用可协同工作的框架或提供集成方案的框架。

### 混合并行化的必要性

设想训练一个拥有万亿参数 (parameter)的模型。

- **仅DP：** 即使ZeRO-3优化了内存，在每个设备上复制完整的正向/反向计算过程可能太慢，或者对于非常大的模型或批量大小，每个设备仍需要过多的激活内存。
- **仅TP：** 分割张量需要张量并行组内大量的通信（AllReduce，点对点）。将TP扩展到非常多的GPU（例如数百个）可能使通信成为主要瓶颈。
- **仅PP：** 虽然PP将层分割到不同设备上，减少了每个设备的内存，但它会受到流水线气泡的影响，特别是当阶段数量很多时，这会导致GPU空闲时间。

将这些策略结合起来可以减轻各自的缺点。一种常见的方法通常被称为“3D并行化”：

1. **流水线并行（PP）：** 将模型的层分割到多个阶段，分布在多个节点或GPU上。这主要减少了每个GPU所需的激活和参数内存。
2. **张量并行（TP）：** 将每个层*内部*的操作（如大型权重 (weight)矩阵）分割到多个GPU上，通常在一个节点内或通过高速互连（如NVLink）连接的一组节点内。这进一步减少了每个GPU的内存需求，并允许适应更大的层。
3. **数据并行（DP）：** 将TP/PP模型配置复制到多组设备上。每个副本处理不同的微批数据。这可以扩大整体批量大小和吞吐量 (throughput)。DeepSpeed的ZeRO优化常在此处应用，以管理数据并行副本间的优化器状态、梯度以及可能的参数。

### 整合DeepSpeed和Megatron-LM

一种流行且有效的方式是使用Megatron-LM实现其高效的TP和PP，并结合DeepSpeed的高级DP优化（ZeRO）以及可能包括激活检查点或高效优化器等其他功能。

它们通常的协同方式如下：

- **Megatron-LM处理TP和PP：** 它提供函数和模块来定义具有张量并行层（例如`ColumnParallelLinear`，`RowParallelLinear`）的模型结构，并管理跨阶段的流水线调度。您首先初始化Megatron-LM以设置TP和PP rank所需的进程组。
- **DeepSpeed封装Megatron-LM模型：** DeepSpeed在Megatron-LM模型设置*之后*初始化。它将Megatron定义的模型（其中已包含TP/PP逻辑）作为输入。然后，DeepSpeed通常使用ZeRO将其DP逻辑应用于数据并行维度。DeepSpeed引擎在DP组*内*管理优化器、梯度累积和通信，同时遵循Megatron-LM建立的底层TP/PP结构。

> 一个简化的2阶段流水线并行（PP）、2路张量并行（TP）设置视图。带ZeRO的数据并行（DP）将复制此整个结构，并管理这些副本间的状态。

### 配置与初始化

设置此类混合系统需要细致的配置。您通常需要：

1. **初始化进程组：** 使用`torch.distributed.init_process_group`，然后根据进程的rank定义数据并行、张量并行和流水线并行的特定进程组。Megatron-LM通常提供实用函数来帮助管理这些组。
2. **配置Megatron-LM：** 设置与张量模型并行大小（`--tensor-model-parallel-size`）、流水线模型并行大小（`--pipeline-model-parallel-size`）、虚拟流水线阶段（`--num-layers-per-virtual-pipeline-stage`）等相关的参数 (parameter)。
3. **配置DeepSpeed：** 创建一个DeepSpeed配置JSON文件，指定ZeRO优化阶段（`zero_optimization.stage`）、学习率、批量大小、梯度裁剪、AMP设置以及可能的激活检查点详情。值得注意的是，DeepSpeed需要了解DP组，但其操作应独立于Megatron-LM管理的TP/PP组。

以下是使用PyTorch进行Python初始化的一个示意：

```python
import torch
import deepspeed
from megatron.initialize import initialize_megatron
from megatron.model import GPTModel # 模型定义
from megatron.training import get_args # 解析Megatron/项目参数的函数

# 1. 初始化基础分布式环境
torch.distributed.init_process_group(backend='nccl')

# 2. 初始化Megatron以设置TP/PP进程组和参数
# 这会解析TP/PP大小等的命令行参数，
# 并设置Megatron的内部状态，包括进程组。
initialize_megatron(args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})

# 3. 使用Megatron的TP/PP模块定义模型
# 参数将包含由initialize_megatron解析的TP/PP配置
args = get_args()
model = GPTModel(
    num_tokentypes=0, # 示例参数
    parallel_output=True, # TP通常需要此项
    # ... 基于args的其他模型配置 ...
)

# 4. 准备模型、优化器等（可能使用Megatron辅助函数）
# （优化器定义、学习率调度器等将在此处）
# optimizer = ...
# lr_scheduler = ...

# 5. 初始化DeepSpeed，传入Megatron模型
# DeepSpeed配置来自JSON文件或字典 (args.deepspeed_config)
# DeepSpeed使用其自己的DP组（通常是最初的默认组，
# 但如果设置正确，会尊重Megatron的TP/PP组）
model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
    args=args,
    model=model,
    # optimizer=optimizer,
    # # DeepSpeed可以创建自己的优化器（例如AdamW）
    # lr_scheduler=lr_scheduler,
    config_params=args.deepspeed_config # DeepSpeed JSON配置文件的路径
)

# 现在，model_engine已准备好用于训练循环
# model_engine.forward(...)
# model_engine.backward(...)
# model_engine.step(...)
```

> PyTorch代码展示了初始化`torch.distributed`、Megatron-LM（用于TP/PP设置）、使用Megatron组件定义模型以及最后初始化DeepSpeed来封装模型并处理DP/ZeRO的顺序。实际实现涉及更多细节，尤其是在参数解析和进程组管理方面。

### 注意事项与挑战

结合框架会增加复杂性：

- **配置：** 管理DeepSpeed和Megatron-LM的配置文件和命令行参数 (parameter)需要细致考量，以确保兼容性和正确性。了解哪个框架控制哪个方面（例如，优化器状态分片与张量分片）很重要。
- **调试：** 在混合设置中调试问题可能具有挑战性，因为问题可能源于DP、TP、PP、ZeRO、激活检查点或底层硬件/通信库（NCCL）之间的相互作用。
- **通信：** 不同通信模式（DP梯度的AllReduce，TP的AllReduce/点对点，PP的点对点）之间的协同需要高效的网络基础设施（例如，节点内TP的NVLink，节点间DP/PP的InfiniBand/RoCE）。
- **兼容性：** 确保DeepSpeed、Megatron-LM、PyTorch以及CUDA/NCCL库的版本兼容。这些框架快速迭代。

尽管存在复杂性，但结合DeepSpeed和Megatron-LM等策略与框架通常是训练先进大型语言模型最实用的方法，它有效地平衡了大型GPU集群中的计算、内存和通信限制。理解如何组织这些组件是当今构建和扩展LLM的一个重要部分。

## 参考资料

- [ZeRO: Memory Optimizations Towards Training Trillion Parameter Models](https://dl.acm.org/doi/10.1145/3458817.3476202) — Samyam Anand, Olatunji Ruwase, Jeff Rasley, Shaden Smith, Deepthi Karkada, Reza Yazdani Aminabadi, Ronald Pope, Sam Ade Jacobs, Yuxiong He (2021)
  Journal: SC '21: Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis; Publisher: ACM; Pages: 80–91; DOI: [10.1145/3458817.3476202](https://doi.org/10.1145/3458817.3476202)
  介绍了ZeRO（零冗余优化器）内存优化策略系列，它们是DeepSpeed中数据并行的基础，对训练大型模型至关重要。
- [DeepSpeed Documentation](https://www.deepspeed.ai/) — DeepSpeed Team (2025)
  Publisher: Microsoft
  DeepSpeed官方在线文档，提供全面的指南、API参考和教程，涵盖DeepSpeed的使用，包括其高级并行功能和ZeRO优化。
- [Megatron-LM Documentation](https://github.com/NVIDIA/Megatron-LM) — NVIDIA (2024)
  NVIDIA Megatron-LM框架官方文档、示例和持续开发的主要来源，详细介绍了其张量并行和管道并行实现。
- [Megatron-DeepSpeed: A Deep Learning Training System for Extreme Scale Model Training](https://www.microsoft.com/en-us/research/blog/using-deepspeed-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model/) — Olatunji Ruwase, Samyam Anand, Shaden Smith, Jeff Rasley, Reza Yazdani Aminabadi, Yuxiong He (2021)
  Publisher: Microsoft Research Blog
  这篇微软研究院的博文明确地详细阐述了Megatron-LM（用于模型并行，即TP/PP）和DeepSpeed（用于数据并行，即ZeRO）的协同集成，以实现LLM的超大规模训练。
