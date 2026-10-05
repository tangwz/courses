# 延迟初始化与meta设备

来源：[原文](https://apxml.com/zh/courses/distributed-training-pytorch-fsdp/chapter-2-model-wrapping-initialization-policies/delayed-initialization-meta-device)

[返回章节目录](README.md) · [返回课程目录](../README.md)

为模型初始化分配内存是分布式训练流程中的一个主要瓶颈。实例化标准PyTorch `nn.Module`时，框架会立即在CPU上分配连续内存以存储参数 (parameter)和缓冲区。对于一个使用混合精度训练的具有700亿参数的模型，仅权重 (weight)所需的初始CPU内存就接近使用 `float16` 时的140 GB，或使用 `float32` 时的280 GB。此计算不包括优化器状态和梯度。在多节点集群中，高性能计算（HPC）节点通常优先考虑GPU内存而非系统RAM，尝试在主机处理器上加载完整的模型定义，在FSDP封装过程开始前经常会立即触发内存不足（OOM）错误。

为规避此硬件限制，PyTorch引入了`meta`设备。这种抽象允许创建只存储形状和数据类型信息而不分配实际数据存储空间的张量。通过将meta设备与延迟初始化策略结合，工程师可以在标准CPU上实例化任意大小的定义架构，只要系统有足够的内存来存储Python对象开销和张量元数据。

### meta设备的工作方式

位于`meta`设备上的张量在结构上与标准张量行为相同，但不包含数据块。对这些张量执行的操作会传播形状和数据类型信息，但跳过实际的核函数执行。此特性对FSDP来说很重要，FSDP需要了解模型架构，特别是参数 (parameter)形状和层级结构，才能计算分片策略和封装策略。

当模型在`meta`设备环境下初始化时，PyTorch会递归构建模块树。权重 (weight)被注册为meta张量。

```python
import torch
import torch.nn as nn

# 用于meta初始化的上下文管理器
with torch.device("meta"):
    # 此分配几乎是即时的，并且占用可忽略不计的RAM
    model = nn.Sequential(
        nn.Linear(8192, 8192),
        nn.ReLU(),
        nn.Linear(8192, 8192)
    )

print(f"Device: {model[0].weight.device}")  # 输出: meta
print(f"Storage: {model[0].weight.element_size() * model[0].weight.numel()}") 
# 注意：存储查询基于数据类型工作，但未占用实际RAM。
```

生成的模型是一个“壳”。它无法执行前向传播，因为它不包含数值。但是，它包含足够的信息供FSDP分析结构，并决定如何在可用进程中划分参数。

### 延迟初始化与实体化

将meta初始化模型与FSDP一起使用时，核心问题是从元数据到实体化权重 (weight)的转换。FSDP必须用分配在特定GPU设备（例如`cuda:0`）上的实际张量替换meta张量。此外，由于模型是在没有数据的情况下初始化的，权重实际上是随机的或未初始化的。我们必须在GPU上分配存储空间 *之后* 才重新应用初始化逻辑（例如Xavier或Kaiming初始化），以避免CPU内存激增。

FSDP通过`param_init_fn`参数 (parameter)管理此过程。当FSDP封装包含meta张量的模块时，它执行以下步骤：

1. **分析：** 扫描模块结构以确定分片组（单元）。
2. **分配：** 仅为本地分片在GPU上分配一个`FlatParameter`。
3. **实体化：** 调用提供的`param_init_fn`，将权重直接初始化到已分配的GPU内存中。

此过程确保完整模型不会存在于CPU内存中。每个GPU只实体化其负责的模型部分（$1/N$，其中$N$是大小）。

标准初始化和延迟初始化之间的内存消耗情况存在显著差异。

> 内存生命周期比较。标准初始化在开始时会导致CPU内存大幅激增。延迟初始化在主机上保持平稳的内存曲线，将分配直接转移到分片的GPU内存。

### 实现 `param_init_fn`

为了实现这一操作，你必须定义一个函数来初始化模块的参数 (parameter)。此函数被传递给FSDP构造器。FSDP会遍历模块并在存储就绪后调用此函数。

The initialization function must handle the distinction between the `meta` device and the materialized device. 请注意，标准PyTorch初始化方法（如`nn.init.uniform_`）不能在meta张量上操作。因此，该函数仅在FSDP使用实际存储支持张量后才会被调用。

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import math

def rigorous_init_fn(module: nn.Module) -> None:
    """
    延迟实体化的自定义初始化逻辑。
    这将在存储分配后在特定设备上运行。
    """
    # 仅初始化具有参数的叶模块
    for name, param in module.named_parameters(recurse=False):
        # 如果不知何故仍为meta（安全检查），则跳过
        if param.device.type == "meta":
            continue
            
        # 根据层类型或名称应用特定的初始化逻辑
        if "weight" in name and param.dim() > 1:
            nn.init.kaiming_uniform_(param, a=math.sqrt(5))
        elif "bias" in name:
            nn.init.zeros_(param)

    # 如果需要，处理缓冲区（例如，BatchNorm运行统计数据）
    for name, buffer in module.named_buffers(recurse=False):
         if buffer.device.type != "meta":
             # 如果需要，重新应用缓冲区默认值
             pass

# 1. 在Meta设备上初始化
with torch.device("meta"):
    large_model = TransformerArchitecture_70B()

# 2. 使用FSDP封装并传递初始化函数
# device_id 必须设置为本地进程的GPU
local_device = torch.device(f"cuda:{local_rank}")

sharded_model = FSDP(
    large_model,
    device_id=local_device,
    param_init_fn=rigorous_init_fn,
    sync_module_states=True # 对于确保所有进程初始化一致很重要
)
```

### 同步与随机数生成器种子

分布式初始化中一个重要问题是确保参数 (parameter)在各个进程之间同步。在标准DDP中，进程0初始化模型并将权重 (weight)广播到所有其他进程。在使用延迟初始化的FSDP中，每个进程都在本地初始化自己的分片以节省内存。

如果`param_init_fn`使用随机数生成器（RNG），就像`kaiming_uniform_`那样，每个进程理论上必须生成相同的初始权重，以确保模型在训练开始前数学上一致。然而，由于每个进程只持有一个分片，它们只需要就全局模型中 *本应* 存在的值达成一致。

有两种处理方式：

1. **设置种子：** 在初始化前，在所有进程上设置相同的全局随机种子。这确保了如果每个进程生成完整模型，它们将获得相同数值。由于它们只生成分片，数学上仍保持一致。
2. **同步模块状态：** FSDP构造器中的`sync_module_states=True`标志是解决此问题的方法。启用后，FSDP允许每个进程初始化参数（可能使用不同的随机种子），然后从进程0执行广播以同步分片。尽管这会在启动时引入通信开销，但它确保了状态一致性，无需手动管理种子。

### 对实体化吞吐量 (throughput)的影响

使用meta设备显著改变了启动吞吐量。初始化时间成为网络带宽（用于同步）和本地GPU内存带宽的函数，而不是CPU内存带宽的函数。

当使用meta设备初始化时，我们可以将内存效率增益 $E_{mem}$ 定义为：

$E_{mem} = 1 - \frac{M_{meta}}{M_{full}} \approx 1$

其中，$M_{meta}$是元数据大小（可忽略不计），$M_{full}$是完整模型参数 (parameter)大小。然而，启动时间 $T_{start}$ 的依赖关系发生变化：

$T_{start} \propto \max(T_{compute\_init}, T_{network\_sync})$

在涉及超大型集群（例如512+ GPU）的情况下，`sync_module_states`广播可能成为一个临时瓶颈。在这种高级配置中，工程师通常倾向于精确的RNG种子设置（方式1）以避免全局广播，从而实现分片的纯本地初始化。该技术相对于节点数量有效地实现了$O(1)$的初始化时间，前提是并行执行。

---

[上一节](02-%E8%87%AA%E5%AE%9A%E4%B9%89%E5%8C%85%E8%A3%85%E7%AD%96%E7%95%A5.md) · [下一节](04-%E5%A4%84%E7%90%86%E5%85%B1%E4%BA%AB%E5%8F%82%E6%95%B0.md)
