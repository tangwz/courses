---
course: "distributed-training-pytorch-fsdp"
chapter: "distributed-checkpointing-fault-tolerance"
lesson: "elastic-training-integration"
sourceId: 7905
sourceUrl: "https://apxml.com/zh/courses/distributed-training-pytorch-fsdp/chapter-5-distributed-checkpointing-fault-tolerance/elastic-training-integration"
title: "弹性训练集成"
description: "配置TorchElastic以处理动态节点扩展和故障恢复。"
order: 3
plots: ["plots/7905-0.json"]
sourceHash: "a3659f9fdf546601aa737f5cb90edcb44317de8dda4cffc2485b47acd751708a"
sourceCorrections: []
---

对于太字节级模型而言，硬件可靠性从一个次要问题转变为主要的工程限制。随着GPU数量 ($N$) 的增加，训练顺利完成且不中断的概率呈指数级下降。如果单个GPU在特定时间范围内发生故障的概率为 $p$，那么整个集群保持稳定的概率为 $(1-p)^N$。在一个 $N=1024$ 且 $p=0.001$ (0.1%) 的集群中，在该时间段内不发生故障而完成的几率约为36%。

对于大型语言模型而言，一旦单个等级丢失就完全崩溃的训练作业是不可持续的。PyTorch通过TorchElastic解决了这一问题，TorchElastic是一个现在已集成到核心库中、管理工作进程生命周期的组件。分布式数据并行 (DDP) 或 FSDP 处理梯度同步，而TorchElastic处理进程编排。它提供了检测工作进程故障、暂停其余正常工作进程、重新组织进程组以及重新启动失败进程以恢复训练的能力。

### 弹性执行层

标准分布式训练依赖于静态定义。每个等级在初始化时都确切知道有多少对等节点及其地址。如果等级5失败，等级0将无限期地等待一个永远不会到达的信号，导致超时（挂起）。

TorchElastic在集群管理器（如Slurm或Kubernetes）与PyTorch训练脚本之间引入了一个间接层。该层由每个节点上运行的**弹性代理**组成。这些代理通过一个**Rendezvous**后端进行协调，以建立 `group_world`。

故障事件期间的操作流程遵循特定的状态转换：

1. **观察：** 本地代理监控工作进程（训练脚本）。
2. **故障检测：** 工作进程崩溃（SIGSEGV、OOM或硬件错误）。
3. **关闭：** 受影响节点上的代理终止任何剩余的本地工作进程，并通知Rendezvous后端。
4. **通知：** Rendezvous后端将当前 `run_id` 标记 (token)为无效。
5. **重新Rendezvous：** 存活的代理进入等待状态。当资源管理器重启故障节点（或如果作业以更少的节点继续）时，代理重新协商 `world_size` 和等级分配。
6. **重启：** 代理使用更新的环境变量（`RANK`、`WORLD_SIZE`、`MASTER_ADDR`）生成新的工作进程。

这种架构要求训练脚本在初始化方面是幂等的。由于脚本在故障后实际上会从头开始运行，它必须能够检测到现有的检查点并恢复，而不是覆盖它们。

> 本地弹性代理与全局Rendezvous后端在故障事件中的互动。节点2上的代理报告故障，促使Rendezvous系统指示节点1进行关闭并准备重新初始化。

### 通过Torchrun调用弹性训练

弹性训练的入口点是 `torchrun`（以前是 `python -m torch.distributed.launch`）。这个CLI工具设置了FSDP正确初始化进程组所需的e环境变量。

在非弹性设置中，你可能需要手动定义 `MASTER_ADDR` 和 `MASTER_PORT`。使用 `torchrun` 时，你依赖Rendezvous后端。对于高性能集群， `c10d` 后端优于 `etcd`，因为它直接在训练节点上运行，无需外部依赖。

多节点FSDP作业的典型命令如下：

```bash
torchrun \
    --nnodes=4 \
    --nproc_per_node=8 \
    --rdzv_id=job_101 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=node-01.internal:29500 \
    train_fsdp.py
```

`rdzv_id` 作为唯一的会话标识符。如果一个节点失败并重启，它必须使用相同的 `rdzv_id` 重新加入正在进行的训练集群。 `nnodes` 参数 (parameter)也可以指定一个范围（例如 `3:4`），即使一个节点永久丢失，也能让作业继续运行，前提是批量大小和梯度累积步数动态调整。

### 容错脚本结构

为了支持上述重启机制，你的训练代码需要特定的结构模式。FSDP不会自动持久化状态；你必须实现保存/加载逻辑。

当发生故障时， `torchrun` 会终止所有进程并从头开始重启脚本。因此，脚本初始化阶段必须检查检查点是否存在。

#### 快照管理

我们使用术语**快照**来指代恢复训练所需的完整状态，这包括模型权重 (weight)、优化器状态、调度器状态以及当前周期/步数计数器。

在上一节中讨论过的分布式检查点（DCP）API的使用是这里的核心点。标准的 `torch.save` 通常需要将所有权重收集到等级0，这会导致内存峰值，可能导致你正在尝试实现的恢复过程崩溃。DCP保存分片状态，允许每个等级并行写入。

以下是 `main` 函数中所需的逻辑流程：

```python
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
import torch.distributed as dist
import os

def load_snapshot(model, optimizer, path):
    # 检查快照是否存在于路径中
    if not os.path.exists(path):
        return 0  # 从周期 0 开始

    # 使用 DCP 加载分片状态
    # 模型和优化器必须已经初始化（分片）
    state_dict = {
        "model": model,
        "optimizer": optimizer
    }
    
    # DCP 处理磁盘上分片权重的映射 
    # 到内存中当前的 sharding 策略
    dist.checkpoint.load(
        state_dict=state_dict,
        checkpoint_id=path
    )
    
    # 独立或包含在 state_dict 中获取元数据（步数/周期）
    # ... 实现细节 ...
    print(f"从快照恢复: {path}")
    return loaded_epoch

def train(model, optimizer):
    # 初始化 FSDP 进程组
    dist.init_process_group(backend="nccl")
    
    # FSDP 封装和初始化
    # ...
    
    # 尝试加载快照
    start_epoch = load_snapshot(model, optimizer, "checkpoints/latest")
    
    for epoch in range(start_epoch, TOTAL_EPOCHS):
        # 训练循环
        # ...
        
        # 在周期结束或每 N 步保存快照
        if dist.get_rank() == 0 or snapshot_all_ranks:
            save_snapshot(model, optimizer, "checkpoints/latest")
```

### 处理拓扑变化

弹性训练中一个复杂的边界情况出现在集群大小发生变化时。假设你在4个节点（32个GPU）上开始训练，其中一个节点遭遇灾难性故障。你可能决定只在3个节点（24个GPU）上恢复训练，而不是等待硬件更换。

在标准FSDP设置中，模型参数 (parameter)在整个大小上分片。32个GPU设置中的等级0持有 $\frac{1}{32}$ 的参数。在24个GPU设置中，等级0必须持有 $\frac{1}{24}$。

如果你使用 `torch.save(model.state_dict())`（它保存未分片的完整权重 (weight)），恢复直接但内存效率低下。如果你保存了分片检查点（例如 `ShardedStateDict`），磁盘上的分片数量与之前的集群大小相对应。

`torch.distributed.checkpoint` (DCP) 模块通过将存储的数据结构与运行时分片策略解耦来解决此问题。加载DCP检查点时：

1. **元数据读取：** 系统读取描述已保存张量分片的元数据文件。
2. **重新分片：** 它计算存储的分片与当前FSDP实例请求的分片之间的交集。
3. **重新分发：** 它执行必要的散布/收集操作来填充当前模型的内存，即使 `world_size` 已经改变。

这项能力将FSDP从僵硬的并行化方案转变为一个灵活的分布式系统，能够适应不稳定的基础设施。

### 检查点频率优化

确定检查点频率涉及I/O开销和故障时浪费的计算时间之间的权衡。我们可以将故障成本 $C_{故障}$ 模型化为：

$C_{故障} = T_{重启} + T_{重计算}$

其中 $T_{重启}$ 是重新加载模型所需的时间，$T_{重计算}$ 是自上次检查点以来损失的时间。为了最小化预期浪费时间，最佳检查点间隔 $\tau$ 可以使用Young近似法（针对分布式系统进行了修改）来近似：

$\tau \approx \sqrt{2 \times \delta \times \text{平均无故障时间}}$

其中 $\delta$ 是写入检查点所需的时间。由于带有DCP的FSDP允许并行写入，$\delta$ 明显低于仅限等级0的序列化方式。这允许更频繁地进行检查点（例如，每30分钟一次而不是每4小时一次），大幅减少长时间运行作业中不可避免的硬件故障期间浪费的计算量。

以下图表显示了并行分布式检查点对I/O开销的影响，从而实现更高的频率。



![检查点写入延迟：等级0聚合 vs. 分布式 (DCP)](plots/7905-0.json)



> 检查点写入延迟的比较。随着模型大小的增长，聚合到单个等级成为瓶颈，而DCP使用整个集群的聚合带宽。

通过将 `torchrun` 与适当的快照逻辑和DCP API结合，你确保你的训练运行具有弹性。这种弹性不仅仅是一种便利；对于需要数月GPU时间的模型，这是在不完善的物理环境中保证收敛的唯一方法。
