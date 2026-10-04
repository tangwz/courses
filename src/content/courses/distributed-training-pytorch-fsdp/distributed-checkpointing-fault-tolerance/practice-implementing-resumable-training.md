---
course: "distributed-training-pytorch-fsdp"
chapter: "distributed-checkpointing-fault-tolerance"
lesson: "practice-implementing-resumable-training"
sourceId: 7906
sourceUrl: "https://apxml.com/zh/courses/distributed-training-pytorch-fsdp/chapter-5-distributed-checkpointing-fault-tolerance/practice-implementing-resumable-training"
title: "实践：实现可恢复训练"
description: "动手操作，创建可持久化分片状态并平滑恢复的训练循环。"
order: 4
plots: ["plots/7906-0.json"]
sourceHash: "e2c430d99438df6baafbfcb73bd427087a27699b1b74e4b69fee842e27c21e26"
sourceCorrections: []
---

在训练TB级模型时，硬件故障是统计上的必然，而非偶然。当扩展到数百个GPU时，平均故障间隔时间（MTBF）急剧下降，需要训练循环不仅高效，而且有韧性。依赖手动重启或简单的基于周期的保存，对于可能运行数周的任务来说是不足够的。

本节实现了一个生产级别的训练循环，使用PyTorch的分布式检查点（DCP）API和TorchElastic。我们侧重于持久化`SHARDED_STATE_DICT`，这使得每个GPU能够将其参数 (parameter)和优化器状态的本地部分直接保存到存储中。这种方法避免了在单个进程上聚合完整模型所产生的内存瓶颈。

### 可恢复训练生命周期

为实现容错，训练脚本必须作为幂等状态机运行。启动时，它会检查是否存在快照。如果存在快照，系统会将模型、优化器和学习率调度器恢复到上次记录步骤时的准确状态。如果未找到快照，训练将从头开始。

以下状态图展示了当工作进程失败时，TorchElastic管理的恢复流程。

> 该流程说明了TorchElastic如何检测故障（八边形）并自动重新执行入口点，立即触发检查点检查逻辑。

### 配置分片状态字典

在实现保存/加载逻辑之前，我们必须配置FSDP模型以生成分片张量。默认情况下，在FSDP模块上调用`.state_dict()`可能会尝试收集全部权重 (weight)，导致内存不足（OOM）。我们使用`FSDP.state_dict_type`上下文 (context)管理器来强制实现分片持久化。

我们定义一个`CheckpointManager`类，以封装分布式I/O的复杂操作。这个类处理路径管理并与`torch.distributed.checkpoint`交互。

```python
import os
import shutil
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

class CheckpointManager:
    def __init__(self, checkpoint_folder):
        self.checkpoint_folder = checkpoint_folder

    def save(self, model, optimizer, scheduler, step):
        # 创建一个包含所有必要组件的状态载荷
        # 模型和优化器必须作为引用传入，以捕获它们的分片状态
        state_payload = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "metadata": {"step": step}
        }

        # 配置FSDP只生成本地分片
        # 这避免了将完整权重收集到CPU
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            # dcp.save自动处理所有进程的并行写入
            dcp.save(
                state_dict=state_payload,
                storage_writer=dcp.FileSystemWriter(self.checkpoint_folder)
            )

    def load(self, model, optimizer, scheduler):
        # 在加载之前，我们必须确保载荷结构匹配
        state_payload = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "metadata": {"step": 0} # 默认值，将被覆盖
        }

        # 检查检查点是否存在
        if not os.path.exists(self.checkpoint_folder):
            return 0 # 从步骤0开始

        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            dcp.load(
                state_dict=state_payload,
                storage_reader=dcp.FileSystemReader(self.checkpoint_folder)
            )
        
        return state_payload["metadata"]["step"]
```

### 原子检查点

分布式系统中一个常见问题是由于写入操作*期间*发生崩溃而导致的数据损坏。如果一个任务在写入`checkpoint_1000`时被中断，该文件夹可能包含部分数据。重启后，加载这个损坏的检查点将再次导致训练循环崩溃，形成一个故障循环。

为解决此问题，我们实现了一个原子保存策略。我们首先写入一个临时目录，并在验证成功后将其重命名为永久检查点路径。由于目录重命名在POSIX文件系统上是原子的，检查点要么完全存在，要么不存在。

以下是集成了`CheckpointManager`和原子逻辑的修改后的训练循环：

```python
def train_loop(rank, model, optimizer, scheduler, train_loader):
    # 初始化管理器
    ckpt_dir = "checkpoints/latest"
    manager = CheckpointManager(ckpt_dir)
    
    # 尝试恢复
    # 这必须在模型封装和优化器创建之后进行
    start_step = manager.load(model, optimizer, scheduler)
    
    if start_step > 0 and rank == 0:
        print(f"Resuming training from step {start_step}")

    # 如有必要，快进数据加载器（为简洁省略）
    # 实际中，使用StatefulDataLoader来恢复迭代器位置
    
    model.train()
    
    for step, batch in enumerate(train_loader, start=start_step):
        inputs, targets = batch[0].to(rank), batch[1].to(rank)
        
        optimizer.zero_grad()
        output = model(inputs)
        loss = torch.nn.functional.cross_entropy(output, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # 每500步保存一次检查点
        if step > 0 and step % 500 == 0:
            # 使用临时路径实现原子性
            tmp_path = f"checkpoints/tmp_{step}"
            
            # 1. 写入临时位置
            # 所有进程参与写入
            with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
                dcp.save(
                    state_dict={
                        "model": model,
                        "optimizer": optimizer, 
                        "scheduler": scheduler,
                        "metadata": {"step": step}
                    },
                    storage_writer=dcp.FileSystemWriter(tmp_path)
                )
            
            # 2. 原子交换（仅协调器）
            # 等待所有进程完成写入后再重命名
            torch.distributed.barrier()
            if rank == 0:
                if os.path.exists(ckpt_dir):
                    shutil.rmtree(ckpt_dir)
                os.rename(tmp_path, ckpt_dir)
                print(f"Checkpoint saved atomically at step {step}")
            
            # 确保所有进程在继续之前看到新的目录结构
            torch.distributed.barrier()
```

### 处理拓扑变化

当TorchElastic重启一个任务时，集群拓扑结构可能会改变。例如，如果一个节点永久性故障，任务可能会以更少的节点重启（如果配置了弹性扩展）或等待替换节点。

FSDP分片依赖于`world_size`。如果你在8个GPU上训练，模型将按8种方式分片。如果你在16个GPU上重启，分片模式会改变。标准的`torch.load`在这里会失败，因为张量形状不匹配。

然而，`torch.distributed.checkpoint`会自动处理这种重新分片。它以与布局无关的格式保存张量。加载时，DCP根据*当前*的`world_size`和FSDP配置重新分配权重 (weight)。这种功能允许你在64个GPU上训练、保存检查点，并在8个GPU上恢复调试，无需手动转换脚本。

### I/O的性能影响

写入数TB的分片数据可能会使存储带宽饱和。下图分析了随着模型规模增加，检查点保存时间相对于计算时间的开销。



![检查点开销与模型规模的对比 (HDD存储)](plots/7906-0.json)



> 随着模型规模的增长，I/O延迟（蓝色条）变得显著。虽然计算时间（红线）通过高效并行化线性扩展，但存储吞吐量 (throughput)常常达到上限。

为尽量减少这种阻塞时间，请使用**异步检查点**。PyTorch在DCP模块中提供了`async_save`。这会启动一个后台线程来处理I/O写入，使GPU能够立即返回训练循环。

```python
# 启用异步保存以隐藏I/O延迟
# 注意：需要细致的内存管理，因为CPU内存会暂时增加
dcp.async_save(
    state_dict=state_payload,
    storage_writer=dcp.FileSystemWriter(ckpt_dir, thread_count=4)
)
```

在实现异步保存时，请监控CPU RAM使用情况。状态字典会一直固定在主机内存中，直到写入完成。在CPU RAM有限的节点上，如果上一个检查点在下一个检查点开始写入之前尚未完成，那么重叠的复制到主机和写入磁盘阶段可能会触发OOM错误。
