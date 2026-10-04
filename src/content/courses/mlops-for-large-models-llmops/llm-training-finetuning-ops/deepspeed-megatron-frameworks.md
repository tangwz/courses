---
course: "mlops-for-large-models-llmops"
chapter: "llm-training-finetuning-ops"
lesson: "deepspeed-megatron-frameworks"
sourceId: 3145
sourceUrl: "https://apxml.com/zh/courses/mlops-for-large-models-llmops/chapter-3-llm-training-finetuning-ops/deepspeed-megatron-frameworks"
title: "使用 DeepSpeed 和 Megatron-LM 等框架"
description: "借助专用框架进行高效的大规模模型训练。"
order: 4
plots: []
sourceHash: "782d04020af704fe7aecc1a0f9745aad5b146c6a715860548455c686c544a2a5"
sourceCorrections: []
---

训练包含数十亿参数 (parameter)的模型会带来计算和内存方面的难题，仅凭标准数据并行无法解决。当模型的参数、梯度、优化器状态或中间激活超出单个加速器（GPU/TPU）的内存容量时，需要更精巧的技术。DeepSpeed 和 Megatron-LM 等框架提供工程化方案，它们结合了多种并行策略和内存优化技术，使得训练这些庞大模型成为可能。有效运用这些框架是 LLMOps 在模型训练方面的一个重要考量。

### DeepSpeed：优化内存和规模

DeepSpeed 由微软开发，是一个旨在让大规模模型训练更高效、更易用的库。其最受认可的贡献是零冗余优化器（ZeRO）。ZeRO 解决了标准数据并行中固有的内存冗余问题，即每个工作进程通常都保留了优化器状态、梯度，有时甚至参数 (parameter)的完整副本。

#### ZeRO 冗余优化器

ZeRO 逐步将这些状态在数据并行工作进程间进行分区，显著降低了每个 GPU 的内存占用。它包含多个阶段：

1. **ZeRO 阶段 1（优化器状态分区）：** 仅分区优化器状态（例如 Adam 优化器的动量和方差向量 (vector)）。每个 GPU 保留参数和梯度的完整副本，但只管理优化器状态的一个分片。它计算其负责的权重 (weight)更新分片，然后使用 all-gather 操作通信更新后的权重。
2. **ZeRO 阶段 2（优化器状态与梯度分区）：** 分区优化器状态和梯度。在反向传播 (backpropagation)过程中，梯度被归约到负责该参数梯度分片的进程，与阶段 1 相比，这大大降低了内存占用。参数仍是复制的。
3. **ZeRO 阶段 3（优化器状态、梯度与参数分区）：** 分区所有三者：优化器状态、梯度以及模型参数本身。每个 GPU 只持有一个模型分区。在前向和反向传播过程中，必要的参数在需要时会动态地从其他 GPU 收集，并在之后丢弃。这提供了最大的内存节省，使得训练真正庞大的模型成为可能。

#### ZeRO-Offload

DeepSpeed 还引入了 ZeRO-Offload，它将分区后的优化器状态和/或参数移动到 CPU 内存甚至 NVMe 存储中。虽然这比将所有内容保留在 GPU 内存中要慢，但这进一步增加了可训练模型的最大规模，用计算时间换取了内存容量。当 GPU 内存是主要瓶颈时，这尤其有用。

#### 集成与使用

DeepSpeed 设计为相对直接地与 PyTorch 训练脚本集成。通常，您会修改脚本，使用 `deepspeed.initialize` 封装模型和优化器。配置通过 JSON 文件管理，您可以在其中指定 ZeRO 阶段、批处理大小、梯度累积步数、混合精度设置以及其他选项。

```python
# DeepSpeed 集成示例
import deepspeed
import torch

# ... 模型、优化器、数据加载器设置 ...

# 从 deepspeed_config.json 加载配置
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    model_parameters=model.parameters(),
    config_params='deepspeed_config.json'
)

# 训练循环使用 model_engine
for step, batch in enumerate(dataloader):
    loss = model_engine(batch) # 前向传播
    model_engine.backward(loss) # 反向传播
    model_engine.step() # 优化器步进
```

从 LLMOps 的角度看，管理 DeepSpeed 涉及：

- **配置管理：** 对 `deepspeed_config.json` 文件进行版本控制和管理，使其与您的代码和实验保持一致。
- **检查点管理：** DeepSpeed 处理分布式检查点，正确保存分区状态。确保这些检查点得到管理、版本控制，并可用于恢复训练，这一点很重要。
- **资源分配：** 理解不同 ZeRO 阶段对内存和网络带宽的影响，以便提供合适的集群资源。

### Megatron-LM：高级并行策略

Megatron-LM 最初由英伟达开发，主要侧重于实现张量并行和流水线并行，以训练即便 ZeRO 阶段 3 也无法处理的过大模型，或当性能要求分发计算本身而非仅仅状态时。

#### 张量并行（层内模型并行）

张量并行将单个层（特别是权重 (weight)矩阵）的计算拆分到多个 GPU 上。例如，Transformer 层内的大型矩阵乘法可以被分解，由不同的 GPU 计算部分结果，然后将它们组合起来。这需要在层执行内部采用专门的通信模式（例如 all-reduce、all-gather）。它在减少每个 GPU 参数 (parameter)和激活所需的内存方面是有效的，但会引入通信开销。

#### 流水线并行（层间模型并行）

流水线并行将模型的层按顺序分区到多个 GPU 上。GPU 1 可能处理第 1-8 层，GPU 2 处理第 9-16 层，以此类推。数据流经这些阶段，就像生产线一样。

简单的实现会导致显著的空闲时间（“气泡”），因为后面的阶段需要等待前面的阶段。Megatron-LM 采用**交错流水线调度**（例如 GPipe 或 PipeDream 风格的调度）等技术，将迷你批次划分为更小的微批次。这使得各阶段可以同时处理不同的微批次，大大提高了 GPU 利用率。

> 微批次（MB0、MB1、MB2）在不同 GPU 上，随时间步（T0、T1 等）流经三个流水线阶段的示意图。交错调度使得 GPU 1 可以在 GPU 0 处理 MB1 的同时开始处理 MB0。

#### 集成与使用

使用 Megatron-LM 通常需要修改模型定义本身，以使用 Megatron 纳入张量并行逻辑的专用层。它还需要仔细配置流水线阶段和并行度。虽然功能强大，但这通常比 DeepSpeed 的 ZeRO 需要更深入的集成工作。

操作考量包括：

- **复杂配置：** 定义张量并行大小、流水线并行大小和数据并行大小，以及它们如何映射到可用硬件。
- **模型代码适配：** 维护与 Megatron 并行原语紧密关联的模型代码。
- **性能调优：** 在微批次大小、流水线阶段和张量并行度之间找到最佳平衡通常需要大量实验。

### 框架结合与操作影响

现代大规模训练通常结合了两种框架的技术。例如，DeepSpeed 集成了受 Megatron-LM 启发的特性，允许用户通过其配置在 ZeRO 的基础上运用张量和流水线并行。

使用这些框架会显著影响 LLMOps 生命周期：

- **复杂性增加：** 虽然它们抽象了底层细节，但这些框架引入了必须管理的新配置参数 (parameter)和操作模式。对框架配置（如 `deepspeed_config.json`）进行版本控制与对代码进行版本控制同样重要。
- **资源管理：** 有效使用需要理解不同并行策略和 ZeRO 阶段在 GPU 内存使用、计算效率和网络带宽消耗方面的权衡。这会指导基础设施配置和作业调度。
- **实验追踪：** 实验不仅要追踪超参数 (hyperparameter)，还要追踪所使用的具体框架配置（ZeRO 阶段、并行度、卸载设置），因为这些配置会严重影响性能和资源消耗。
- **调试：** 调试使用复杂并行策略、跨数百个 GPU 的分布式运行中的故障，需要专用工具和专业知识。框架日志和性能分析变得不可或缺。
- **检查点管理：** 管理这些框架创建的分布式检查点对于容错和恢复长时间训练作业是必要的。确保检查点在框架版本或配置更改之间兼容可能具有挑战性。

总而言之，DeepSpeed 和 Megatron-LM（以及它们所包含的技术）是实现最先进大型语言模型训练的基本工具。它们通过精巧的内存优化和分布式计算策略，提供了克服单 GPU 限制所需的机制。熟练掌握它们的配置和操作管理是高级 LLMOps 中的一项核心能力。

## 参考资料

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://www.microsoft.com/en-us/research/publication/zero-memory-optimizations-toward-training-trillion-parameter-models/) — Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He (2020)
  Journal: SC20: International Conference for High Performance Computing, Networking, Storage and Analysis; DOI: [10.1109/SC41405.2020.00072](https://doi.org/10.1109/SC41405.2020.00072)
  介绍零冗余优化器（ZeRO）的基础论文，它是DeepSpeed内存效率技术的重心。
- [DeepSpeed Documentation](https://deepspeed.readthedocs.io/) — Microsoft DeepSpeed Team (2024)
  官方文档，提供DeepSpeed特性（包括ZeRO阶段和卸载）的实用指南、API参考和配置示例。
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) — Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro (2019)
  Journal: arXiv preprint arXiv:1909.08053; DOI: [10.48550/arXiv.1909.08053](https://doi.org/10.48550/arXiv.1909.08053)
  描述Megatron-LM用于训练超大型语言模型的张量和流水线并行方法的原始论文。
- [NVIDIA/Megatron-LM GitHub Repository](https://github.com/NVIDIA/Megatron-LM) — NVIDIA (2024)
  Megatron-LM的官方源代码库，提供实现细节、使用示例和高级并行策略的配置模式。
