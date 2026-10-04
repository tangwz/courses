---
course: "advanced-pytorch"
chapter: "distributed-training-parallelism"
lesson: "distributed-data-parallel"
sourceId: 2763
sourceUrl: "https://apxml.com/zh/courses/advanced-pytorch/chapter-5-distributed-training-parallelism/distributed-data-parallel"
title: "使用 DistributedDataParallel (DDP) 进行数据并行"
description: "使用 DDP 实现高效的多 GPU/多节点数据并行。"
order: 2
plots: []
sourceHash: "23f15c312b8752b00c03c1fcd7a00d6900d4235793d7057047fa4d1269ac8edc"
sourceCorrections: []
---

训练大型模型或使用大量数据集时，在单个 GPU 上顺序处理所有数据会很快成为性能瓶颈。数据并行是一种策略，将相同的模型复制到多个处理单元（通常是 GPU）上，每个单元处理输入数据批次的不同子集。尽管 PyTorch 提供了直接的 `torch.nn.DataParallel` (DP) 模块，但由于 Python 全局解释器锁 (GIL) 的限制以及其梯度聚合的集中式方法，它在性能上常常不足。

为了高效、可扩展的数据并行，特别是在多 GPU 和多节点环境中，`torch.nn.parallel.DistributedDataParallel` (DDP) 是推荐的方案。DDP 使用多进程，为每个 GPU 分配一个独立的 Python 进程。这绕过了 GIL，实现了真正的并行执行。此外，它采用高效的集合通信操作（如 `all-reduce`），由 NCCL（适用于 NVIDIA GPU）或 Gloo（适用于 CPU 或 NCCL 不可用时）等后端管理，以直接在 GPU 之间同步梯度，在反向传播 (backpropagation)期间将通信与计算重叠，以提高性能。

### 理解 DDP 工作流程

DDP 的核心思想巧妙而强大：

1. **初始化：** 使用 `torch.distributed.init_process_group` 设置分布式环境。每个参与的进程被分配一个唯一的 `rank`（从 0 到 `world_size - 1`），它们协调通信，通常通过指定的后端（如 NCCL）。`world_size` 指的是训练中涉及的进程总数。
2. **模型复制：** 模型架构在每个进程（GPU）上完全相同地复制。
3. **数据划分：** 输入数据批次被分成更小的分片。每个进程接收一个分片。这通常通过 `torch.utils.data.distributed.DistributedSampler` 来管理，它确保每个进程在每个 epoch 中看到数据集的独特且不重叠的部分。
4. **前向传播：** 每个模型副本独立地对其本地数据分片执行前向传播。
5. **梯度计算：** 在反向传播 (backpropagation) (`loss.backward()`) 期间，梯度在每个副本上进行本地计算。
6. **梯度平均：** 这是 DDP 的亮点所在。在梯度计算时，DDP 自动介入自动求导引擎。它在后台启动一个 `all-reduce` 集合操作。此操作汇总所有副本中每个参数 (parameter)的梯度，然后除以 `world_size`，从而有效地进行平均。结果会分发回所有副本。重要的是，DDP 将通信与梯度计算重叠，从而隐藏了通信延迟。
7. **参数更新：** 每个进程上的优化器 (`optimizer.step()`) 使用*相同*的平均梯度更新其本地模型副本的参数。因为所有副本都以相同的权重 (weight)开始并接收相同的平均梯度，它们的参数在整个训练过程中保持同步，更新后无需显式参数广播。

> 工作流程说明了通过 `DistributedSampler` 进行数据分片、模型副本上的独立前向/反向传播，以及在每个进程的优化器步骤之前，用于梯度平均的 `all-reduce` 核心操作。

### 在实际中实现 DDP

将 DDP 集成到标准 PyTorch 训练脚本中需要进行一些修改：

1. **环境配置：** 您需要一种方式来启动多个 Python 进程，每个 GPU 一个。像 `torchrun`（推荐）或旧的 `torch.distributed.launch` 这样的标准工具可以处理此任务。它们负责设置 `init_process_group` 所需的环境变量，例如 `MASTER_ADDR`、`MASTER_PORT`、`RANK` 和 `WORLD_SIZE`。您还需要确定 `local_rank`，它通常对应于当前进程应使用的 GPU 索引。
2. **初始化进程组：** 在脚本早期，初始化分布式后端：

   ```python
   import torch
   import torch.distributed as dist
   import os

   # 假设环境变量 RANK, WORLD_SIZE, LOCAL_RANK 已由启动器设置
   rank = int(os.environ['RANK'])
   world_size = int(os.environ['WORLD_SIZE'])
   local_rank = int(os.environ['LOCAL_RANK'])

   # 初始化进程组
   dist.init_process_group(backend='nccl', # 'nccl' 用于 GPU, 'gloo' 用于 CPU
                           rank=rank,
                           world_size=world_size)

   # 为当前进程设置设备
   torch.cuda.set_device(local_rank)
   device = torch.device(f"cuda:{local_rank}")
   ```

   强烈建议在 NVIDIA GPU 训练中使用 `nccl`，因为它性能优越。
3. **准备分布式数据加载器：** 修改您的数据加载以使用 `DistributedSampler`。这个采样器确保每个进程获得数据的一个不同部分，且不重叠。

   ```python
   from torch.utils.data import DataLoader, Dataset
   from torch.utils.data.distributed import DistributedSampler

   # 假设 'train_dataset' 是您的 torch.utils.data.Dataset 实例
   train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)

   # 重要：shuffle=False 因为 DistributedSampler 处理了洗牌
   # 重要：pin_memory=True 可以加快主机到设备的传输速度
   train_loader = DataLoader(train_dataset,
                             batch_size=per_device_batch_size,
                             sampler=train_sampler,
                             num_workers=num_workers_per_process,
                             pin_memory=True,
                             shuffle=False) # 采样器处理洗牌
   ```

   请注意，`DataLoader` 中的 `batch_size` 现在指的是*每个进程*的批次大小。所有 GPU 上的总有效批次大小是 `per_device_batch_size * world_size`。请记住在 `DataLoader` 中设置 `shuffle=False`，因为 `DistributedSampler` 会负责在每个 epoch 中适当地打乱数据。
4. **包装模型：** 实例化您的模型并将其移动到当前进程的指定设备上，*然后*再用 `DDP` 包装。

   ```python
   import torch.nn as nn
   from torch.nn.parallel import DistributedDataParallel as DDP

   # 实例化您的模型
   model = YourModel().to(device) # 首先将模型移动到正确的 GPU

   # 使用 DDP 包装模型
   # device_ids 应包含此进程的单个 GPU ID
   # output_device 应与 device_ids[0] 相同
   model = DDP(model, device_ids=[local_rank], output_device=local_rank)
   ```

   `device_ids` 告知 DDP 此进程管理哪个或哪些 GPU（通常只有一个，即 `local_rank`），`output_device` 指定模型输出应放置的位置（通常是同一设备）。
5. **训练循环调整：** 核心训练循环基本保持不变。主要区别在于 `loss.backward()` 现在隐式地触发所有进程间的梯度同步。

   ```python
   optimizer = YourOptimizer(model.parameters(), lr=learning_rate)

   for epoch in range(num_epochs):
       # 为采样器设置 epoch，以确保每个 epoch 之间数据正确洗牌
       train_sampler.set_epoch(epoch)

       model.train()
       for batch_idx, (data, target) in enumerate(train_loader):
           data, target = data.to(device), target.to(device) # 将数据移动到进程的 GPU

           optimizer.zero_grad()
           output = model(data)
           loss = loss_fn(output, target)

           loss.backward() # 触发梯度同步

           optimizer.step() # 使用平均梯度更新本地副本

           if rank == 0 and batch_idx % log_interval == 0: # 只在 rank 0 上记录日志
               print(f"Epoch: {epoch} | Batch: {batch_idx} | Loss: {loss.item()}")

       # 验证循环（通常只在 rank 0 上执行或使用分布式采样器）
       # ...

   # 清理
   dist.destroy_process_group()
   ```

   通常的做法是，日志记录、保存检查点或验证等操作主要在一个进程（通常是 `rank == 0`）上执行，以避免冗余操作和混乱的输出。请记住在每个 epoch 开始时调用 `train_sampler.set_epoch(epoch)`，以确保在使用 `DistributedSampler` 时每个 epoch 的数据洗牌不同。最后，在训练结束时调用 `dist.destroy_process_group()` 来清理资源。

### DDP 的注意事项

- **保存与加载：** 保存 DDP 模型时，通常只需要一个进程（例如 `rank == 0`）来保存状态字典。DDP 包装了原始模型，因此保存的状态字典的键会带有 `module.` 前缀。将状态字典加载回非 DDP 模型时，您需要处理这个前缀，或者通过 `model.module.state_dict()` 访问底层模型。

  ```python
  # 保存 (仅在 rank 0 上)
  if rank == 0:
      torch.save(model.module.state_dict(), "model_checkpoint.pt")

  # 加载 (在所有 rank 上)
  map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank} # 映射到当前设备
  checkpoint = torch.load("model_checkpoint.pt", map_location=map_location)
  # 首先创建模型实例
  model_instance = YourModel().to(device)
  # 将状态字典加载到原始模型中
  model_instance.load_state_dict(checkpoint)
  # 然后用 DDP 包装
  ddp_model = DDP(model_instance, device_ids=[local_rank], output_device=local_rank)
  ```
- **批量归一化 (normalization)：** DDP 默认正确处理了跨进程的批量归一化统计信息同步。与 `DataParallel` 不同，您通常不需要 `torch.nn.SyncBatchNorm`，尽管在需要时也可以使用。
- **混合精度：** DDP 与 `torch.cuda.amp`（自动混合精度）兼容。在实例化 `GradScaler` *之后*用 DDP 包装模型，但要在训练循环中遵循标准的 AMP 模式。
- **`find_unused_parameters`：** 如果您的模型存在在反向传播 (backpropagation)期间未接收到梯度的参数 (parameter)（例如，由于 `forward` 方法中的条件逻辑），DDP 的反向传播同步可能会停滞，等待永远不会到达的梯度。在 `DDP` 构造函数中设置 `find_unused_parameters=True` 可以解决此问题，但这会增加一些开销。通常最好确保所有需要梯度的参数都参与损失计算（如果可能）。

`DistributedDataParallel` 提供了一种高性能机制，用于在多个 GPU 和节点上扩展训练。通过了解其多进程架构、梯度平均对集合通信的依赖以及对数据加载和模型包装的必要调整，您可以有效地训练更大、更复杂的模型，比以往任何时候都快。

## 参考资料

- [Distributed Data Parallel](https://pytorch.org/docs/stable/notes/ddp.html) — PyTorch Documentation (2024)
  PyTorch DistributedDataParallel的官方指南，涵盖架构、用法和实践。
- [torch.distributed package - PyTorch Documentation](https://pytorch.org/docs/stable/distributed.html) — PyTorch Documentation (2024)
  Publisher: PyTorch Foundation
  提供torch.distributed包的信息，包括DDP使用的init_process_group和集体操作。
- [NVIDIA Collective Communications Library (NCCL)](https://developer.nvidia.com/nccl) — NVIDIA (2024)
  Publisher: NVIDIA
  NCCL的官方资源，NCCL是NVIDIA GPU上高性能分布式通信的关键后端。
- [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) — Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, Kaiming He (2017)
  Publisher: Facebook AI Research; DOI: [10.48550/arXiv.1706.02677](https://doi.org/10.48550/arXiv.1706.02677)
  一篇讨论使用大批量大小训练深度神经网络的挑战和解决方案的论文，侧重于高效分布式梯度聚合。
