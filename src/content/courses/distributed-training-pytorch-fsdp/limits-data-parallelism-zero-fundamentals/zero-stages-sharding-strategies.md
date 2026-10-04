---
course: "distributed-training-pytorch-fsdp"
chapter: "limits-data-parallelism-zero-fundamentals"
lesson: "zero-stages-sharding-strategies"
sourceId: 7885
sourceUrl: "https://apxml.com/zh/courses/distributed-training-pytorch-fsdp/chapter-1-limits-data-parallelism-zero-fundamentals/zero-stages-sharding-strategies"
title: "ZeRO 阶段与分片策略"
description: "对 ZeRO 优化阶段的技术分析，说明如何划分优化器状态、梯度和参数。"
order: 2
plots: ["plots/7885-0.json"]
sourceHash: "50009a6577f3f0a24cb87fbcb4727bdfa9b68df07191580272d18c0da743da03"
sourceCorrections: []
---

分布式数据并行 (DDP) 的局限在于其根本的冗余。在 $N$ 个 GPU 的 DDP 设置中，系统会维护 $N$ 份相同的模型参数 (parameter)、梯度和优化器状态副本。尽管这允许反向传播 (backpropagation)的并行计算，但它制造了一个内存瓶颈：最大模型大小严格受限于单个 GPU 的显存 (VRAM)，与集群的总容量无关。

为突破此限制，我们采用零冗余优化器 (ZeRO) 算法策略。ZeRO 通过在数据并行进程中划分模型状态来消除这种冗余。每个设备拥有数据的一个不同分片，而不是复制全部状态。优化分三个渐进阶段进行，每个阶段都通过增加通信开销来换取大量的内存节省。

### 训练的内存组成

在划分之前，我们必须量化 (quantization) GPU 内存的消耗。对于一个包含 $\Psi$ 个参数 (parameter)、使用混合精度 (FP16/BF16) 和 Adam 优化器训练的模型，内存占用主要由三个部分构成：

1. **优化器状态 ($O$):** 内存主要占用者。Adam 维护参数的 FP32 副本（主权重 (weight)），以及动量和方差缓冲区。这大约占用每个参数 12 字节 ($4+4+4$ )。
2. **梯度 ($G$):** 在反向传播 (backpropagation)期间以 FP16/BF16 存储。占用每个参数 2 字节。
3. **参数 ($P$):** 模型权重本身，用于正向传播和反向传播。占用每个参数 2 字节。

在标准 DDP 配置中，每个 GPU 都持有全部 $16\Psi$ 字节。ZeRO 依序处理这些组件。

### ZeRO 阶段 1：优化器状态分片

阶段 1 ($P_{os}$) 针对最大的内存占用者：优化器状态。在此配置中，参数 (parameter) ($P$) 和梯度 ($G$) 在所有设备上保持复制，从而保持 DDP 在正向和反向传播 (backpropagation)中的通信模式。然而，优化器步骤是分片的。

如果您有 $N_d$ 个设备，优化器状态会被分成 $N_d$ 个相等的分区。每个第 $i$ 个设备仅更新其特定的参数分片。在步骤结束时，`AllGather` 操作会在所有设备间同步更新后的参数。

每个设备的内存消耗从 $2\Psi + 2\Psi + 12\Psi$ 降至大约：


$$
\text{阶段1内存} = 2\Psi + 2\Psi + \frac{12\Psi}{N_d}
$$


对于大型集群，与 DDP 相比，这能将内存使用量减少近 75%，因为优化器状态项趋近于零。

### ZeRO 阶段 2：梯度分片

阶段 2 ($P_{os+g}$) 将分片扩展到梯度。在标准 DDP 中，梯度在本地计算，然后使用 `AllReduce` 操作进行同步。`AllReduce` 逻辑上等同于先进行 `ReduceScatter`，再进行 `AllGather`。

ZeRO 阶段 2 修改了此流程。反向传播 (backpropagation)后，系统执行 `ReduceScatter` 操作。每个 GPU 只接收并聚合与其负责更新的参数 (parameter)分区对应的梯度。然后它丢弃其余部分。

由于优化器状态已经分片（来自阶段 1），每个 GPU 现在都拥有更新其特定参数分片所需的一切：特定的优化器状态和特定的累积梯度。

内存消耗变为：


$$
\text{阶段2内存} = 2\Psi + \frac{2\Psi}{N_d} + \frac{12\Psi}{N_d}
$$


此阶段带来了显著收益，且通信开销极小，因为 `ReduceScatter` 是 DDP 所用 `AllReduce` 操作中固有的基本操作。

### ZeRO 阶段 3：参数 (parameter)分片

阶段 3 ($P_{os+g+p}$) 是通俗地称为“完整”FSDP 的核心机制。在此阶段，模型参数本身被分片。没有单个 GPU 在静止状态下持有完整的模型权重 (weight)。

这带来了一个新挑战：计算正向和反向传播 (backpropagation)需要处理特定层的完整权重。ZeRO-3 通过临时实例化来解决此问题。

1. **正向传播：** 在层计算其输出之前，FSDP 触发 `AllGather` 以从其他 GPU 获取缺失的参数分片。层完成计算后，参数会立即被释放（丢弃）以节省内存。
2. **反向传播：** 系统再次 `AllGather` 完整的参数以计算梯度，然后丢弃它们。

此方法无法训练大小超过整个集群总内存的模型，但可以训练大小等同于所有 GPU 内存总和减去激活开销的模型。

每个设备的内存被减少到理论最小值：


$$
\text{阶段3内存} = \frac{2\Psi + 2\Psi + 12\Psi}{N_d} = \frac{16\Psi}{N_d}
$$


下面的图表说明了在不同策略下，内存如何在 4 个设备之间分配。注意阶段 3 (FSDP) 如何均匀分配所有组件。

> 4 GPU 集群上 DDP 与 ZeRO-3 (FSDP) 状态分配的比较。DDP 复制完整状态；FSDP 分片所有组件。

### 对内存的量化 (quantization)影响

阶段的选择会显著改变最大可训练模型的大小。虽然阶段 1 和 2 提供了大量减少，但阶段 3 实现了与 GPU 数量的线性扩展。

下表显示了在 8 GPU 集群上，一个理论上的百亿参数 (parameter)模型（总状态约需 160GB）每个 GPU 的内存消耗。



![每个 GPU 的内存占用 (百亿参数模型，8 个 GPU)](plots/7885-0.json)



> 每个 GPU 的内存使用细分。请注意，ZeRO-1 通过分片优化器状态实现了内存使用量的最大单次降幅，而 ZeRO-3 将所有组件的占用最小化。

### 通信权衡

这些内存优势并非没有代价；代价是网络带宽。

- **DDP：** 需要对梯度进行 `AllReduce`。每步通信量为 $2\Psi$（发送 + 接收）。
- **ZeRO-3：** 需要对参数 (parameter)进行 `AllGather`（正向传播），对参数进行 `AllGather`（反向传播 (backpropagation)），以及对梯度进行 `ReduceScatter`。总通信量增加到大约 $3\Psi$。

在带宽受限的环境（如标准以太网）中，阶段 3 中按需获取参数的额外延迟会限制计算吞吐量 (throughput)。这使得配置 NVLink 或 InfiniBand 等高速互连对于阶段 3 训练不可或缺，我们将在多节点网络章节中讨论此话题。

### 对 FSDP 实现的影响

在 PyTorch FSDP 中，这些策略并非总是互斥的固定模式，而是通过 `sharding_strategy` 参数 (parameter)进行配置的。

- `ShardingStrategy.FULL_SHARD` 对应 ZeRO-3。
- `ShardingStrategy.SHARD_GRAD_OP` 对应 ZeRO-2。
- `ShardingStrategy.NO_SHARD` 的行为类似于 DDP。

理解这些阶段可以帮助您根据硬件限制选择合适的策略。如果您的模型使用 ZeRO-2 即可适应内存，通常会因为它较低的通信开销而优于 ZeRO-3。然而，对于定义现代 AI 的 TB 级模型，ZeRO-3 通常是唯一可行的途径。
