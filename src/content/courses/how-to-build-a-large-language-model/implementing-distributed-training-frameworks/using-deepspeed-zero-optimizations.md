---
course: "how-to-build-a-large-language-model"
chapter: "implementing-distributed-training-frameworks"
lesson: "using-deepspeed-zero-optimizations"
sourceId: 6029
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-16-implementing-distributed-training-frameworks/using-deepspeed-zero-optimizations"
title: "使用 DeepSpeed ZeRO 优化"
description: "实现 ZeRO 阶段 (1, 2, 3) 以优化数据并行中的内存使用。"
order: 3
plots: ["plots/6029-0.json"]
sourceHash: "58226b8fd00b8e6b52b3320d8d6395ae3addcef61ef569cab8f7496f31e3dff5"
sourceCorrections: []
---

虽然标准数据并行 (DP) 通过在每个设备上复制模型和处理不同的数据分片来在多个 GPU 上扩展训练，但它很快就会遇到内存限制。每个 GPU 仍需保存模型参数 (parameter)、梯度和优化器状态的完整副本。对于拥有数十亿参数的模型，这种重复的状态会消耗大量 GPU 内存，经常超出高端加速器的容量。DeepSpeed 提供的零冗余优化器 (ZeRO) 在这方面显示出很大优势。ZeRO 旨在消除标准 DP 中固有的内存冗余，让您可以在相同的硬件限制下训练更大的模型或使用更大的批次大小。

ZeRO 通过将模型状态（优化器状态、梯度和参数）分配到数据并行进程中而不是复制它们来实现这一点。它分为三个渐进阶段，每个阶段都能节省更多内存，但可能会增加通信开销。

### 了解数据并行中的内存冗余

在分析 ZeRO 之前，让我们直观地看看标准 DP 中每个 GPU 的内存消耗。对于一个拥有 $\Psi$ 参数 (parameter)的模型，使用像 Adam 这样的标准优化器（它通常存储参数、梯度、一阶动量和二阶方差），每个 GPU 大致所需的内存是：

- **参数:** $\Psi \times (\text{参数类型的大小，例如 FP32 为 4 字节})$
- **梯度:** $\Psi \times (\text{梯度类型的大小，例如 FP32 为 4 字节})$
- **优化器状态:** 对于 Adam（动量和方差），通常是 $2 \times \Psi \times (\text{优化器状态类型的大小})$。如果使用混合精度，优化器可能还会存储参数的 FP32 副本。

激活值也消耗内存，但模型状态通常是大型模型的主要影响因素。在标准 DP 中，*所有*这些状态都会在*每个* GPU 上复制。ZeRO 系统地减少了这些复制。

> 标准数据并行中，每个 GPU 上复制的内存组件。

### ZeRO 阶段 1: 优化器状态划分

ZeRO 阶段 1 处理第一层冗余：优化器状态。例如，Adam 维护动量和方差缓冲区，它们的大小通常与模型参数 (parameter)本身相同（如果为混合精度训练存储 FP32 副本，甚至更大）。阶段 1 将这些优化器状态分配到可用的数据并行 GPU 上。每个 GPU 只保存对应其数据并行等级的优化器状态总量的部分。

在优化器步进 (`optimizer.step()`) 期间，梯度仍然需要在所有 GPU 之间归约（像标准 DP 一样），但每个 GPU 只更新其持有相应优化器状态分区的参数部分。

**优点:**

- 内存消耗明显减少，特别是使用 Adam/AdamW 等优化器时。
- 通信开销与标准 DP 相似（梯度归约）。

**配置示例 (DeepSpeed JSON):**

```json
{
  "zero_optimization": {
    "stage": 1
  },
  "fp16": {
    "enabled": true
  },
  "train_batch_size": 32,
  "gradient_accumulation_steps": 1
}
```

### ZeRO 阶段 2: 优化器状态和梯度划分

ZeRO 阶段 2 更进一步，将优化器状态*和*梯度都划分到数据并行 GPU 上。在反向传播 (backpropagation)期间，阶段 2 使用 ReduceScatter 操作，而不是使用 AllReduce 操作来汇总所有 GPU 上的梯度。此操作计算总和，但立即分散结果，因此每个 GPU 只接收对应其优化器状态分区的梯度分区。

**优点:**

- 通过消除重复的梯度，内存消耗比阶段 1 进一步减少。
- 梯度归约通信量与标准 DP 或阶段 1 相比减半（ReduceScatter 对比 AllReduce）。

**配置示例 (DeepSpeed JSON):**

```json
{
  "zero_optimization": {
    "stage": 2
  },
  "fp16": {
    "enabled": true
  },
  "train_batch_size": 32,
  "gradient_accumulation_steps": 1
}
```

### ZeRO 阶段 3: 优化器状态、梯度和参数 (parameter)划分

ZeRO 阶段 3 是最具侵略性的优化，它划分所有三个主要模型状态：优化器状态、梯度*以及*模型参数本身。每个 GPU 在任何给定时间只保存参数的一个分片。

这需要更精密的管理。在前向和反向传播 (backpropagation)期间，每个 GPU 都需要访问给定层计算的完整参数。ZeRO 阶段 3 通过在计算需要时动态地从其他 GPU 收集所需参数分片，并在计算后立即丢弃它们以释放内存来处理此问题。

**优点:**

- 内存节省最大化，允许在给定 GPU 内存预算内训练最大的模型。模型状态所需的内存变得与模型大小无关，仅随 GPU 数量 ($N$) 扩展。每个 GPU 的内存大致与 $\Psi / N$ 成正比。
- 有效消除了模型状态的内存冗余。

**缺点:**

- 与阶段 1 和 2 相比，通信量增加，这是由于在前向和反向传播期间频繁收集参数分片。性能高度依赖于 GPU 之间的互连速度（例如 NVLink、InfiniBand）。

**配置示例 (DeepSpeed JSON):**

```json
{
  "zero_optimization": {
    "stage": 3
  },
  "fp16": {
    "enabled": true
  },
  "train_batch_size": 32,
  "gradient_accumulation_steps": 1
}
```



![每个 GPU 的相对模型状态内存](plots/6029-0.json)



> 相对于标准数据并行，ZeRO 各阶段每个 GPU 模型状态内存的近似减少量。实际节省取决于使用的优化器和精度。

### 在 PyTorch 中集成 ZeRO 与 DeepSpeed

使用 ZeRO 需要用 DeepSpeed 的 `initialize` 函数包装您的模型、优化器以及可能的数据加载器。您将配置详细信息（通常通过 JSON 文件）提供给此函数。

```python
import torch
import deepspeed

# 假设 model, optimizer, dataloader 和 args (包含 deepspeed_config 路径) 已定义

# 初始化 DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    model_parameters=model.parameters(),
    config_params=args.deepspeed_config # JSON 配置文件的路径
)

# 训练循环示例片段
for step, batch in enumerate(dataloader):
    # 将批次数据移动到 DeepSpeed 管理的设备
    batch = {
        k: v.to(model_engine.local_rank) for k, v in batch.items()
    }

    # 前向传播
    outputs = model_engine(**batch)
    loss = outputs.loss

    # DeepSpeed 管理的反向传播
    model_engine.backward(loss)

    # DeepSpeed 管理的优化器步进
    model_engine.step()

    # 日志记录、检查点等
```

DeepSpeed 根据配置文件中指定的 ZeRO 阶段，处理划分、通信（梯度归约、参数 (parameter)收集）和优化器步进。

### 选择合适的阶段

- **阶段 1:** 如果标准 DP 因优化器状态而内存不足，这是一个不错的起点。提供了显著的节省，同时对通信模式的改变最小。
- **阶段 2:** 比阶段 1 提供更好的内存节省，当梯度消耗大量内存时尤其明显。梯度归约的通信量也更低。通常在内存节省和通信开销之间取得了良好平衡。
- **阶段 3:** 提供最大的内存节省，对于即使使用阶段 2 也无法容纳的超大型模型来说是不可或缺的。然而，它引入了参数 (parameter)收集的显著通信开销，使得高带宽互连（如 NVLink 或 NVSwitch）对于性能来说非常重要。还可以考虑阶段 3 中的 `ZeRO-Offload` 变体，它可以将分区卸载到 CPU RAM 或 NVMe 存储，以支持更大的模型，尽管代价是较慢的访问时间。

通常需要进行实验才能找到适合您的特定模型、硬件配置和性能要求的最佳阶段。从阶段 1 或 2 开始，如果内存限制要求，则转向阶段 3，并仔细监控训练吞吐量 (throughput)以评估通信增加的影响。ZeRO 提供了一套强大的工具来克服大规模模型训练中的内存障碍，使其成为构建最新 LLM 的一项基本技术。

## 参考资料

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://dl.acm.org/doi/10.1145/3418579.3441319) — Samyam Rajbhandari, Cong Li, Zhewei Yao, Minjia Zhang, Reza Yazdani Aminabadi, Ammar Ahmad Awan, Jeff Rasley, Yuxiong He (2020)
  Journal: SC '20: Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis; Publisher: ACM; Pages: 1–16; DOI: [10.1145/3418579.3441319](https://doi.org/10.1145/3418579.3441319)
  阐述了ZeRO概念及其三个阶段，解释了内存优化机制。
- [DeepSpeed ZeRO-powered Data Parallelism](https://deepspeed.readthedocs.io/en/stable/zero.html) — DeepSpeed Team (2024)
  在DeepSpeed框架中配置和使用ZeRO优化的官方资源。
- [DeepSpeed: Extreme-Scale Model Training for Everyone](https://www.microsoft.com/en-us/research/publication/deepspeed-extreme-scale-model-training-for-everyone/) — Jeff Rasley, Samyam Rajbhandari, Kazem Cheshmi, Chris Ping, Yuxiong He (2020)
  Journal: KDD '20: Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining; Publisher: Association for Computing Machinery (ACM); Pages: 3491–3499; DOI: [10.1145/3394486.3403154](https://doi.org/10.1145/3394486.3403154)
  介绍了DeepSpeed框架，ZeRO是其核心组件之一，并讨论了其训练大规模模型的能力。
