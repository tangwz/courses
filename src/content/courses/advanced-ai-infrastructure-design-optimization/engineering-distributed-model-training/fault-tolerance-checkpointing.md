---
course: "advanced-ai-infrastructure-design-optimization"
chapter: "engineering-distributed-model-training"
lesson: "fault-tolerance-checkpointing"
sourceId: 6984
sourceUrl: "https://apxml.com/zh/courses/advanced-ai-infrastructure-design-optimization/chapter-2-engineering-distributed-model-training/fault-tolerance-checkpointing"
title: "长时间运行任务中的容错与检查点技术"
description: "设计能够从硬件故障中恢复的弹性训练任务的策略。"
order: 5
plots: []
sourceHash: "3f181fb825cfdf01598c800f68724c674db6b5dcd7d72354baf6d7b0e246511d"
sourceCorrections: []
---

大型机器集群上数天或数周的模型训练，使得硬件或软件故障成为运营中不可避免的情况。单个节点故障、网络分区或抢占式实例被收回，都可能导致数千小时的GPU计算量付诸东流。因此，将容错功能直接构建到训练流程中，并非可有可无的改进，而是大规模人工智能的核心架构要求。实现这种恢复能力的主要技术是系统化的检查点。

检查点是训练任务状态的一个完整、持久的快照，它使得任务可以从故障发生的确切位置恢复。仅保存模型权重 (weight)不足以实现平滑恢复。一个全面的检查点必须包含：

- **模型参数 (parameter)：** 所有层的当前权重和偏置 (bias)。
- **优化器状态：** 优化器的内部状态，例如Adam的动量和方差缓冲区，或SGD的速度。未能恢复此状态可能导致训练不稳定，并抵消收敛进度。
- **学习率调度器状态：** 调度器的当前学习率和步数。
- **迭代状态：** 当前的周期（epoch）和步数，以确保训练循环正确恢复。
- **数据管道状态：** 数据加载器的状态，包括随机数生成器（RNG）的状态，以保持数据洗牌和增强顺序的可重复性。

### 设计检查点策略

一个有效的检查点策略需要在保存状态的开销与潜在的计算损失之间取得平衡。

#### 频率和触发条件

最主要的权衡点是频率。过于频繁地进行检查点会引入大量的I/O开销，因为将数GB数据写入远程存储可能导致训练停滞。检查点频率过低则会增加故障发生时丢失的工作量。一种平衡的方法通常涉及根据固定时间间隔（例如，每60分钟）或设定的训练步数来触发检查点。最佳频率取决于您的基础设施的稳定性和计算资源的成本。

#### 存储后端

存储后端的选择对性能和可靠性有重要影响。尽管第一章详细介绍了各种存储系统，但它们在检查点使用方面有其独特的权衡：

- **本地磁盘：** 提供最快的写入速度，但不可持久。如果节点发生故障，其本地检查点将丢失，因此不适用于多节点任务。
- **云对象存储（S3、GCS、Azure Blob）：** 容错检查点的行业标准。它具有高持久性、可扩展性，并可从集群中的任何节点访问，这对于在不同机器组上恢复任务非常重要。主要缺点是较高的延迟，可以通过异步检查点等技术缓解。
- **并行文件系统（Lustre、BeeGFS）：** 在高性能计算（HPC）环境中常见，它们为大型分布式写入提供高吞吐量 (throughput)，但维护起来比对象存储更复杂且成本更高。

### 在分布式框架中的实现

现代分布式训练框架提供内置支持，用于管理保存分片状态的复杂性。

#### Horovod

Horovod与框架无关，并将检查点逻辑委托给用户。标准的实现模式是指定单个工作进程（通常是`rank == 0`）来处理保存操作。这可以防止“雷鸣般的羊群”问题，即所有工作进程尝试写入同一位置，从而导致写入竞争和潜在的文件损坏。

```python
# Horovod训练脚本中常见的检查点模式
import torch
import horovod.torch as hvd
import os

# 初始化Horovod
hvd.init()
# ... 模型、优化器及其他初始化

def save_checkpoint(model, optimizer, step):
    # 只有主工作进程 (rank 0) 保存检查点。
    if hvd.rank() == 0:
        state = {
            'step': step,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            # ... 包含其他状态，例如调度器 ...
        }

        # 最佳实践：保存到临时文件，然后执行原子重命名。
        # 这可以防止从部分写入、损坏的检查点恢复。
        tmp_path = "/path/to/durable/storage/checkpoint_step_{}.tmp".format(step)
        final_path = "/path/to/durable/storage/checkpoint_step_{}.pt".format(step)

        torch.save(state, tmp_path)
        os.rename(tmp_path, final_path)
        print(f"检查点已保存到 {final_path}")

# 在训练循环中：
# if step % config.save_interval == 0:
#     save_checkpoint(model, optimizer, step)
```

#### DeepSpeed 和 PyTorch FSDP

DeepSpeed和PyTorch的完全分片数据并行（FSDP）等框架了解模型和优化器状态如何在设备间分片。它们提供高级API，抽象化了收集和保存这种分布式状态的复杂性。

- **DeepSpeed：** 提供简单、统一的API `model_engine.save_checkpoint()`。它自动处理分片模型参数 (parameter)、优化器状态和其他训练组件到指定目录的序列化。
- **PyTorch FSDP：** 在 `torch.distributed.checkpoint` 模块中提供高级API。这些API旨在将分片张量直接保存和加载到存储中，而无需先将整个模型收集到单个GPU内存上，这对于训练过大而无法容纳在一个设备上的模型来说是一种非常重要的能力。

### 自动化故障恢复

保存检查点只是解决方案的一半。生产级别的系统必须自动化恢复过程。这项责任落在工作负载协调器身上，例如Kubernetes Job Controller或Slurm调度器。

自动化恢复流程包含以下几个步骤：

1. **故障检测：** 协调器检测到训练Pod或任务已失败。
2. **检查点查找：** 协调器逻辑查询持久存储（例如S3存储桶）以找到最新、有效的检查点。
3. **任务重新启动：** 它启动一个新的训练任务，将找到的检查点路径作为命令行参数 (parameter)或环境变量传递。
4. **状态加载：** 训练脚本设计为在启动时检查此检查点路径。如果提供了路径，它会在开始训练循环之前从检查点加载状态，从而恢复计算。

> 自动化恢复循环。当Pod失败时，协调器会找到对象存储中最后一次成功的检查点，并启动一个新的Pod，指示其从该状态恢复。

### 高级技术

对于高度优化的环境，更高级的模式很常见。

#### 优雅处理抢占式实例回收

云服务提供商通常会在终止抢占式实例前提供短暂警告（例如，30-120秒）。设计良好的训练应用程序可以捕获此信号。后台进程可以轮询实例元数据服务以获取终止通知。收到通知后，它会触发紧急检查点，确保丢失的工作量最少。这使得易变的低成本抢占式实例成为长时间运行训练任务的一个非常可行的选择。

#### 异步检查点

为尽量减少将大型检查点写入远程存储引起的训练停滞，可以使用异步模式。训练过程将快速保存到本地SSD，使计算几乎可以立即恢复。独立的后台线程或进程随后负责将检查点从本地磁盘上传到持久对象存储。这将训练循环与高延迟的网络I/O分离，提高了计算效率。

## 参考资料

- [Checkpointing Strategies for Fault Tolerance in Deep Learning Systems: A Survey](https://ieeexplore.ieee.org/document/9253503) — Yu Liang, Cong Jiang, Yong Zhang (2020)
  Journal: IEEE Access; Publisher: IEEE; Volume: 8; Pages: 206775-206788; DOI: [10.1109/ACCESS.2020.3039750](https://doi.org/10.1109/ACCESS.2020.3039750)
  全面概述深度学习系统中各种检查点策略、其权衡和故障容忍应用。
- [Horovod: Checkpointing](https://horovod.readthedocs.io/en/stable/getting_started.html) — The Linux Foundation (2024)
  Publisher: The Linux Foundation
  Horovod中实施检查点策略的官方指南，包括在分布式环境中保存和恢复模型和优化器状态的最佳实践。
- [torch.distributed.checkpoint documentation](https://pytorch.org/docs/stable/distributed.checkpoint.html) — PyTorch Authors (2022)
  Publisher: PyTorch Foundation
  官方文档，详细介绍了分布式PyTorch中保存和加载分片模型和优化器状态（包括FSDP）的API。
