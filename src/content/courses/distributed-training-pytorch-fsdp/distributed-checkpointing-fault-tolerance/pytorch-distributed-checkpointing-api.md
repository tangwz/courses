---
course: "distributed-training-pytorch-fsdp"
chapter: "distributed-checkpointing-fault-tolerance"
lesson: "pytorch-distributed-checkpointing-api"
sourceId: 7904
sourceUrl: "https://apxml.com/zh/courses/distributed-training-pytorch-fsdp/chapter-5-distributed-checkpointing-fault-tolerance/pytorch-distributed-checkpointing-api"
title: "PyTorch 分布式检查点 API"
description: "使用现代分布式检查点 API 实现高效的并行 I/O。"
order: 2
plots: ["plots/7904-0.json"]
sourceHash: "9131a0c9931412da19dfcff626e3825e4b447973e4b025f0a71686e7cd9b6d3e"
sourceCorrections: []
---

使用 `torch.save` 的传统序列化在单个进程上运行，通过一个瓶颈实际上序列化了整个集群的内存。随着模型参数 (parameter) ($P$) 和优化器状态 ($O$) 的增长，这种方式在数学上变得不可行。如果 $P + O$ 超过了进程 0 的宿主内存，训练任务就会崩溃。即使内存足够，序列化延迟也会与模型大小呈线性增长，导致 GPU 长时间空闲。

`torch.distributed.checkpoint` (DCP) API 从根本上改变了这种操作，从集中收集转变为分布式并行写入。DCP 允许 FSDP 组中的每个进程将其状态字典的本地分片直接流式传输到持久存储。这将有效写入时间从 $T \propto M$（总模型大小）缩短到 $T \propto \frac{M}{N}$（其中 $N$ 是进程数），前提是存储后端支持足够的 IOPS。

### 分布式检查点的架构

DCP 的运行方式与标准 PyTorch 序列化不同。它不是对单个 Python 对象进行序列化，而是调度一个规划好的张量写入图。当触发保存时，DCP 会创建描述全局张量结构的元数据文件以及一系列包含实际数据负载的二进制分片。

存储表示与运行时拓扑分离。这种分离实现了一个主要功能：**拓扑无关加载**。在 128 个 GPU 上训练的模型可以进行检查点，随后在 64 个 GPU 上恢复，前提是总内存容量足够。DCP 加载器读取元数据并重新分片张量以匹配当前的 `process_group` 配置。

> 分布式检查点操作中的数据流，其中本地分片与 DCP 规划器交互以生成并行写入流。

### 实现分片保存

为了结合 FSDP 使用 DCP，我们必须明确配置模型以生成分片状态字典。默认情况下，在 FSDP 模块上调用 `.state_dict()` 可能会尝试收集完整权重 (weight)，这违背了使用 DCP 的目的。我们使用 `FSDP.state_dict_type` 上下文 (context)管理器来强制使用 `SHARDED_STATE_DICT`。

以下实现说明了如何保存模型权重和优化器状态。注意，优化器状态也必须通过 FSDP API 处理，以确保它与分片参数 (parameter)正确对应。

```python
import torch.distributed.checkpoint as dcp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

def save_checkpoint(model, optimizer, step, checkpoint_path):
    # 确保路径在进程 0 上存在，或让写入器处理
    # 对于分布式保存，StateDictType 必须是 SHARDED
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        # 1. 创建状态字典
        # 这不会将数据移动到 CPU 或将其收集到单个进程
        state_dict = {
            "model": model.state_dict(),
            # 优化器必须感知 FSDP
            "optimizer": FSDP.optim_state_dict(model, optimizer),
            "step": step
        }

        # 2. 执行分布式保存
        # 与 torch.save 不同，dcp.save 处理布局
        dcp.save(
            state_dict=state_dict,
            checkpoint_id=checkpoint_path,
        )

# 在训练循环中的使用
# save_checkpoint(fsdp_model, optimizer, current_step, "checkpoints/step_1000")
```

在此实现中，`dcp.save` 默认使用 `FileSystemWriter`。每个进程将其部分数据写入 `checkpoint_path` 内的特定文件结构。开销很小，因为不需要跨节点通信来聚合张量。

### 异步检查点

对于参数 (parameter)超过 1000 亿的模型，即使是并行 IO 也可能需要数秒或数分钟，具体取决于存储带宽。在此期间暂停计算会降低模型浮点运算利用率 (MFU)。DCP 支持异步保存，允许训练循环立即继续，而 IO 操作在后台线程中进行。

为此，我们通常依赖 `async_save=True`（在较新的 PyTorch 版本中或通过快照扩展提供）。然而，这样做会引入竞争条件：如果训练循环在保存线程仍在读取权重 (weight)时更新权重，检查点就会损坏。

解决方法包括在宿主内存中捕获权重的快照，或确保 IO 在下一次反向传播 (backpropagation)修改梯度之前完成。FSDP 创建写时复制机制或分片引用的快速克隆来缓解这种情况，但在自定义循环中，明确的同步通常更安全。

```python
def async_checkpoint_handler(model, optimizer, path):
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        state_dict = {
            "model": model.state_dict(),
            "optimizer": FSDP.optim_state_dict(model, optimizer)
        }
        
        # 开始保存操作
        # 注意：确保您的存储后端高效支持并发写入
        future = dcp.async_save(
            state_dict=state_dict,
            checkpoint_id=path
        )
        return future

# 在训练循环中
# future = async_checkpoint_handler(model, opt, "ckpt/step_N")
# ... 执行前向传播 ...
# future.result() # 如有需要，确保在关键区域之前完成保存
```

### 加载与重分片

恢复检查点不像将文件名映射到进程那么简单。由于集群大小可能已改变，DCP 使用元数据文件来确定全局张量的哪些部分属于当前进程的分片。

当加载到 FSDP 模块时，该模块必须已经初始化并分片。`dcp.load` 函数在原地读取数据。这是内存高效的，因为我们从不实体化完整模型；我们只读取本地 GPU 所需的特定字节。

```python
def load_checkpoint(model, optimizer, checkpoint_path):
    # 我们必须使用相同的 StateDictType 上下文
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        # 创建一个具有正确结构的占位状态字典
        # 此字典中的值用作加载的“计划”
        state_dict = {
            "model": model.state_dict(),
            # 优化器加载需要已知模型结构
            "optimizer": FSDP.optim_state_dict(model, optimizer),
            "step": 0 # 占位符
        }

        # 直接加载到占位符中
        dcp.load(
            state_dict=state_dict,
            checkpoint_id=checkpoint_path,
        )
        
        # 将加载的优化器状态应用回优化器引擎
        # 这是经常被忽略的一个必要步骤
        FSDP.optim_load_state_dict(model, optimizer, state_dict["optimizer"])
        
        return state_dict["step"]
```

`FSDP.optim_load_state_dict` 函数是必需的。标准 PyTorch 优化器本身不理解分片状态。FSDP 充当翻译器，将加载的优化器分区分散到每个设备上正确的参数 (parameter)组。

### 性能影响

转向分片检查点显著改变了训练集群的 IO 特性。在传统设置中，进程 0 上的网络带宽是限制因素。使用 DCP，限制转移到存储系统的总写入带宽。

配置分布式文件系统（例如 Lustre、GPFS）或对象存储（S3、Azure Blob）时，确保后端能够处理 $N$ 个并发连接有其必要。



![检查点保存延迟：传统与分布式对比](plots/7904-0.json)



> 随着模型大小增加，传统收集和分布式检查点之间的保存延迟比较。

如所示，传统序列化产生了指数级瓶颈，而 DCP 相对于每个 GPU 内存分片大小而非总模型大小，保持了近线性性能特征。对于万亿字节规模的模型，DCP 不仅仅是一种优化；它是唯一可行的持久化机制。
