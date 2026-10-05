# CPU 卸载的实现

来源：[原文](https://apxml.com/zh/courses/distributed-training-pytorch-fsdp/chapter-3-mixed-precision-memory-optimization/cpu-offloading-implementation)

[返回章节目录](README.md) · [返回课程目录](../README.md)

现代加速器上的高带宽内存 (HBM) 是扩展模型大小时的主要限制。尽管分片和量化 (quantization)等技术可以减少参数 (parameter)和优化器状态的内存占用，但它们并不能增加 GPU 的物理容量。CPU 卸载通过将主机 RAM 视为 GPU 内存的分层扩展来解决这一物理限制。通过将参数、梯度和优化器状态迁移到 CPU，您可以训练比集群总 HBM 大得多的模型，但这会带来 PCIe 总线引入的延迟开销。

### 卸载的架构机制

在没有卸载的标准 FSDP 配置中，分片参数 (parameter)位于 GPU 上。在前向传播期间，FSDP 从其他 GPU 收集特定层的完整参数，执行计算，然后移除非本地分片。启用 CPU 卸载后，分片参数的静止状态会移至系统 RAM。

数据流会改变标准执行生命周期：

1. **主机到设备 (H2D)：** 在模块执行之前，FSDP 会启动将分片参数从 CPU RAM 异步传输到 GPU 显存 (VRAM)。
2. **AllGather：** 一旦参数到达 GPU，它们会参与标准的集合通信 (AllGather)，以实现完整的权重 (weight)用于计算。
3. **计算：** GPU 执行前向或反向传播 (backpropagation)。
4. **设备到主机 (D2H)：** 在反向传播期间，梯度在 GPU 上计算。FSDP 会立即对这些梯度进行归约（ReduceScatter），并将生成的分片梯度卸载回 CPU。
5. **优化器步骤：** 优化器步骤完全在 CPU 上执行，更新位于系统 RAM 中的主权重和优化器状态。

这种架构依赖于 CPU 和 GPU 之间的互连带宽。一个标准的 PCIe Gen4 x16 通道提供大约 32 GB/s 的理论带宽。如果层的计算强度（算术强度）较低，训练过程就会受限于这种传输速度，而不是 GPU 的浮点运算能力。

> 数据流图，说明了张量在主机 RAM 和设备 HBM 之间循环移动。优化器步骤完全在 CPU 上发生，以节省显存。

### 配置 CPU 卸载

PyTorch FSDP 通过 `CPUOffload` 数据类来控制卸载行为。此配置将参数 (parameter)卸载与梯度卸载分开，尽管它们通常配合使用以最大程度地节省内存。

要实现这一点，您需要实例化配置对象并将其传递给 FSDP 包装器。

```python
import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision
)

# 标准混合精度策略
bf16_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

# 配置 CPU 卸载
# offload_params=True 会将参数和梯度都移至 CPU
offload_policy = CPUOffload(offload_params=True)

model = MyLargeTransformer()

# 应用带有卸载功能的 FSDP
fsdp_model = FSDP(
    model,
    cpu_offload=offload_policy,
    mixed_precision=bf16_policy,
    device_id=torch.cuda.current_device()
)
```

当设置 `offload_params=True` 时，FSDP 会管理 `fsdp_model.parameters()` 的驻留位置。请注意，这会为初始化和检查点引入不同的行为。参数将驻留在 CPU 设备上，这意味着如果代码预期 CUDA 张量，则在没有上下文 (context)管理器的情况下直接对 `model.parameters()` 进行操作可能会失败。

### 锁页内存与异步传输

为了使 CPU 卸载具有良好性能，计算与通信的重叠是必不可少的。如果没有重叠，GPU 在等待权重 (weight)从 CPU 到达时会一直空闲。要启用异步传输（非阻塞复制），主机内存缓冲区必须是锁页（pinned）的。

标准操作系统内存是可分页的，这意味着操作系统可以将其交换到磁盘。CUDA 驱动程序无法通过直接内存访问 (DMA) 安全地访问可分页内存，因为物理地址可能会改变，或者数据可能不在 RAM 中。锁页内存保证数据驻留，允许 DMA 引擎在 CPU 执行其他指令的同时并发地将数据复制到 GPU。

在 FSDP 中，设置 `offload_params=True` 会自动尝试为参数 (parameter)锁定内存。但是，您必须确保数据加载器也使用锁页内存，以防止 PCIe 总线因数据加载和参数卸载争夺带宽而变得拥堵。

```python
# 确保数据加载器使用锁页内存，以与 FSDP 卸载共存
train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,
    pin_memory=True  # 对吞吐量很重要
)
```

### 性能分析与吞吐量 (throughput)

实现 CPU 卸载是一个权衡的决定。您是用训练吞吐量（每秒 token 数）来换取模型容量（参数 (parameter)数量）。性能损失与参数数量和计算量的比率有很大关联。

1. **计算受限层：** Transformer 中像大型线性投影（$O(N^2)$）这样的层通常具有足够的算术强度，可以隐藏从 CPU 获取下一层权重 (weight)的延迟。
2. **带宽受限层：** 像 LayerNorm 或逐元素激活这样的操作具有较低的算术强度。GPU 很可能会停顿，等待 PCIe 传输完成。

您可以使用时间线追踪来查看这种重叠效率。在理想情况下，H2D 复制流（传输下一层）与当前层的计算流完美对齐 (alignment)。



[交互图表：时间线分析：计算与 PCIe 传输重叠](https://apxml.com/zh/courses/distributed-training-pytorch-fsdp/chapter-3-mixed-precision-memory-optimization/cpu-offloading-implementation#plot-eje9bb)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "时间线分析：计算与 PCIe 传输重叠",
    "xaxis": {
      "title": "时间（微秒）",
      "showgrid": false,
      "zeroline": false
    },
    "yaxis": {
      "showgrid": false,
      "zeroline": false,
      "showticklabels": false
    },
    "showlegend": true,
    "height": 300,
    "margin": {
      "t": 40,
      "b": 40,
      "l": 20,
      "r": 20
    },
    "plot_bgcolor": "#f8f9fa"
  },
  "data": [
    {
      "type": "bar",
      "y": [
        "GPU 计算"
      ],
      "x": [
        100
      ],
      "base": [
        0
      ],
      "orientation": "h",
      "name": "前向传播层 N",
      "marker": {
        "color": "#4dabf7"
      },
      "width": 0.4
    },
    {
      "type": "bar",
      "y": [
        "PCIe 总线"
      ],
      "x": [
        80
      ],
      "base": [
        10
      ],
      "orientation": "h",
      "name": "H2D 复制层 N+1",
      "marker": {
        "color": "#ff8787"
      },
      "width": 0.4
    },
    {
      "type": "bar",
      "y": [
        "GPU 计算"
      ],
      "x": [
        100
      ],
      "base": [
        110
      ],
      "orientation": "h",
      "name": "前向传播层 N+1",
      "marker": {
        "color": "#4dabf7"
      },
      "width": 0.4
    },
    {
      "type": "bar",
      "y": [
        "PCIe 总线"
      ],
      "x": [
        30
      ],
      "base": [
        120
      ],
      "orientation": "h",
      "name": "停滞（等待数据）",
      "marker": {
        "color": "#868e96",
        "pattern": {
          "shape": "/"
        }
      },
      "width": 0.4
    }
  ]
}
```

</details>



> 风格化的时间线，显示执行重叠。红色块代表通过 PCIe 的数据传输。如果传输时间长于计算（层 N），GPU 在开始层 N+1 之前必须停顿（灰色块），从而降低模型浮点运算利用率 (MFU)。

### 优化器状态的优化

CPU 卸载最显著的内存优势来自于移动优化器状态。在采用混合精度的标准 Adam 优化器设置中，内存消耗主要由 FP32 主权重 (weight)和两个优化器状态（动量和方差，它们也都是 FP32）所占据。

对于一个具有 $\Psi$ 个参数 (parameter)的模型，内存分配通常如下：

- **FP16 参数：** $2\Psi$ 字节
- **FP16 梯度：** $2\Psi$ 字节
- **FP32 主权重：** $4\Psi$ 字节
- **优化器状态（动量）：** $4\Psi$ 字节
- **优化器状态（方差）：** $4\Psi$ 字节

每个参数的总静态内存为 $16\Psi$ 字节。通过启用 `CPUOffload`，FP32 主权重和优化器状态（$12\Psi$ 字节）永久驻留在主机 RAM 中。GPU 只需保存临时的 FP16 参数和梯度（$4\Psi$）以及激活。这有效地将模型数据的显存 (VRAM)需求减少了 75%，从而实现了与 DeepSpeed 等 ZeRO Stage 3 实现相似的训练扩展。

启用卸载时，强烈建议使用 `torch.optim.AdamW` 或类似的标准优化器。FSDP 会包装优化器步骤，以确保计算发生在参数所在的设备（CPU）上。如果您尝试在使用 CPU 卸载参数时使用融合的 CUDA 优化器（如 `Apex FusedAdam`），训练将失败或静默回退到慢速实现，因为 CUDA 内核无法访问这些张量。

### 何时使用 CPU 卸载

CPU 卸载并非所有情况下的默认设置。它是一种特定优化，适用于即使应用了分片和激活检查点后，模型规模仍然超出可用显存 (VRAM)的情况。

在以下情况下使用 CPU 卸载：

1. **批次大小较小：** 您被迫每个 GPU 使用批次大小为 1，并且仍然遇到内存不足 (OOM) 错误。
2. **硬件受限：** 您正在消费级 GPU（24GB 显存）或小型集群上训练大型模型（7B+）。
3. **吞吐量 (throughput)容忍度：** 您可以接受迭代速度降低 20-40%，以换取能够适配模型的能力。

在以下情况下避免使用 CPU 卸载：

1. **网络受限：** 您的训练已经受限于节点间通信 (NCCL)。增加 PCIe 传输会加剧延迟。
2. **小型模型：** 如果模型适合显存，卸载会因不必要的数据移动而必然降低性能。

通过将 CPU 卸载与上一节中提到的激活检查点技术结合使用，您可以最大化硬件的参数 (parameter)容量，突破单个节点可训练模型的界限。

---

[上一节](02-%E6%BF%80%E6%B4%BB%E6%A3%80%E6%9F%A5%E7%82%B9%E5%B7%A5%E4%BD%9C%E5%8E%9F%E7%90%86.md) · [下一节](04-%E5%88%86%E7%89%87%E4%B8%8B%E7%9A%84%E6%A2%AF%E5%BA%A6%E7%B4%AF%E7%A7%AF.md)
