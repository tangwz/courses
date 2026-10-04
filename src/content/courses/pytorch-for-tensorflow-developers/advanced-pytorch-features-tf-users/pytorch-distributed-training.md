---
course: "pytorch-for-tensorflow-developers"
chapter: "advanced-pytorch-features-tf-users"
lesson: "pytorch-distributed-training"
sourceId: 5775
sourceUrl: "https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-6-advanced-pytorch-features-tf-users/pytorch-distributed-training"
title: "分布式训练方法"
description: "PyTorch分布式训练（例如，`DistributedDataParallel`）概述，与TensorFlow的`tf.distribute.Strategy`进行原理对比。"
order: 2
plots: []
sourceHash: "14914a0fef5cc3695628a657ba0fc0324b01d808ca5f9da717f20002843c01a5"
sourceCorrections: []
---

随着模型和数据集的复杂性和规模增加，在单个GPU或CPU上训练它们可能变得非常慢。分布式训练可以让你将此工作负载并行分配到多个处理单元，无论是单机上的GPU还是多台机器上的GPU。这不仅能加快训练速度，还使你能够处理那些原本会超出单个设备内存容量的模型或批次大小。PyTorch提供了自己有效的工具，主要在`torch.distributed`包中，以实现类似结果。

分发训练工作负载主要有两种策略：数据并行和模型并行。

### 数据并行

数据并行是最常用的策略。在此方法中，模型在每个可用设备（如GPU）上复制。每个副本处理输入数据的不同子集（一个分片或小批次）。每个副本计算的梯度随后被汇总，模型权重 (weight)在所有副本间同步更新。

> 图示数据并行方法。模型在每个GPU上复制，处理独特的数据分片，并在更新权重前汇总梯度。

在PyTorch中，数据并行主要通过`torch.nn.parallel.DistributedDataParallel` (DDP) 实现。这个模块封装你现有模型，并处理数据分发、梯度同步和跨多个进程（通常每个GPU一个进程）的模型更新的复杂性。DDP比旧的`torch.nn.DataParallel` (DP) 更受青睐，因为DDP使用多进程，这避免了Python全局解释器锁（GIL）的限制，并且通常提供更好的性能，特别是对于Python开销显著或使用多节点时。

DDP的工作流程通常包括：

1. 初始化进程组（稍后会详细介绍）。
2. 在每个进程上创建模型实例。
3. 用`DistributedDataParallel`包装模型。
4. 将`DistributedSampler`与你的`DataLoader`一起使用，以确保每个进程接收到数据集的独特部分。
5. 执行标准训练循环。DDP在反向传播 (backpropagation)期间自动处理梯度平均。

这种方法类似于TensorFlow中用于单节点、多GPU训练的`tf.distribute.MirroredStrategy`或用于多节点情况的`tf.distribute.MultiWorkerMirroredStrategy`。主要思想是每个工作进程都拥有模型的完整副本，并处理数据的一部分。

### 模型并行

当模型过大无法适应单个GPU的内存时，会采用模型并行。不是在每个设备上复制整个模型，而是将模型的不同部分（例如，层或层块）放置在不同的设备上。数据在正向和反向传播 (backpropagation)期间，按顺序流经这些跨设备的部分。

> 图示模型并行。模型的不同部分放置在不同的GPU上，数据在它们之间流动。

实现模型并行可能比数据并行更复杂，因为你需要手动管理模型组件的放置以及设备间中间激活和梯度的传输。PyTorch允许你使用`.to(device)`将模型的不同部分分配到不同的设备。例如，你可以将大型NLP模型的嵌入 (embedding)层放在一个GPU上，随后的Transformer块放在其他GPU上。

尽管PyTorch提供了手动模型并行的基本工具，但更复杂的类型，例如流水线并行（设备同时处理不同微批次的流水线不同阶段），通常会受益于像FairScale或DeepSpeed这样的专用库，这些库是基于PyTorch的基础构建的。`torch.distributed.rpc`模块也提供了一个用于更通用分布式计算模式的框架，可用于实现自定义模型并行策略。

TensorFlow用户可能会在手动将`tf.Variable`或层计算放置在特定设备上找到相似之处。两种框架都需要仔细考虑通信开销，因为GPU之间的数据移动可能成为瓶颈。

### PyTorch分布式训练的核心组件

`torch.distributed`包是PyTorch中分布式训练的根本。以下是它的一些核心组件：

- **进程组（`torch.distributed.init_process_group`）**：在进行任何分布式操作之前，进程必须加入一个组。此函数初始化分布式环境。你需要指定：

  - `backend`：要使用的通信后端（例如，`gloo`、用于GPU的`nccl`或`mpi`）。`nccl`因其高性能通常推荐用于基于GPU的训练。
  - `init_method`：进程如何相互发现（例如，`env://`用于环境变量设置，或`tcp://<master_addr>:<master_port>`）。
  - `world_size`：参与作业的进程总数。
  - `rank`：当前进程的唯一标识符，从0到`world_size - 1`。
- **通信原语**：`torch.distributed`提供了几个用于进程间集体通信的函数：

  - `all_reduce(tensor, op=ReduceOp.SUM)`：在所有机器上归约张量数据。每个进程最终获得相同的最终结果（例如，所有张量的和）。这在DDP中对于梯度平均是根本的。
  - `broadcast(tensor, src)`：将张量从rank为`src`的进程复制到组中的所有其他进程。
  - `scatter(tensor, scatter_list, src)`：将张量列表分散到组中的所有进程。
  - `gather(tensor, gather_list, dst)`：将张量列表从组中的所有进程收集到一个目标进程。
- **`torch.nn.parallel.DistributedDataParallel` (DDP)**：如前所述，这是数据并行的主力。它封装你的模型并处理：

  - 从rank 0向所有其他进程广播初始模型状态以确保一致性。
  - 跨进程分片输入数据（尽管数据加载本身由带有`DistributedSampler`的`DataLoader`管理）。
  - 在反向传播 (backpropagation)期间对梯度执行`all_reduce`操作。
  - 同步批量归一化 (normalization)层。
- **启动工具**：

  - `torch.multiprocessing.spawn(fn, args=(), nprocs=None, ...)`：一个用于生成`nprocs`个进程的工具，这些进程将运行目标函数`fn`。常用于单节点多GPU训练。
  - `torchrun`（以前是`python -m torch.distributed.launch`）：PyTorch提供的一个命令行工具，用于启动分布式训练作业，在多节点设置中特别有用。它负责为每个进程设置环境变量，例如`MASTER_ADDR`、`MASTER_PORT`、`WORLD_SIZE`和`RANK`。

### 设置分布式训练环境

#### 单节点、多GPU

这是你有一台机器带有多个GPU的常见情况。

1. **环境**：如果你使用NVIDIA GPU，请确保CUDA和NCCL (NVIDIA Collective Communications Library) 已正确安装。
2. **脚本**：你的训练脚本需要修改：
   - 导入`torch.distributed`和`torch.multiprocessing`。
   - 定义一个以`rank`和`world_size`为参数 (parameter)的主训练函数。
   - 在此函数内部，使用适当的后端（`nccl`）、rank和大小调用`dist.init_process_group()`。
   - 为当前进程设置设备：`torch.cuda.set_device(rank)`。
   - 创建你的模型并将其移动到设备：`model.to(rank)`。
   - 用`DistributedDataParallel`包装模型：`model = DDP(model, device_ids=[rank])`。
   - 将`torch.utils.data.distributed.DistributedSampler`与你的`DataLoader`一起使用，以确保每个进程获取数据的独特部分。
   - 运行你的训练循环。
   - 最后调用`dist.destroy_process_group()`。
   - 在你的主执行块（`if __name__ == '__main__':`）中使用`mp.spawn()`来跨多个进程启动训练函数。

这里是一个简化结构：

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# 假设MyModel和MyDataset在其他地方定义

def setup(rank, world_size):
    # 对于TCP初始化
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    # dist.init_process_group("nccl", init_method='env://', rank=rank, world_size=world_size)

    # 单节点使用文件进行更简单的初始化（替代环境变量）
    # 确保文件路径可访问且每个作业唯一
    init_file = "file:///tmp/my_shared_file_for_dist_init" 
    dist.init_process_group(backend="nccl", init_method=init_file,
                            world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_fn(rank, world_size, epochs):
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)

    # Create model and move it to GPU with id rank
    model = MyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank], output_device=rank) # output_device可能很有用

    # 用于说明的虚拟数据集
    dataset = MyDataset(...) 
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=2) # 每个进程的工作线程数

    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)

    for epoch in range(epochs):
        sampler.set_epoch(epoch) # 对于使用DistributedSampler进行洗牌很重要
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(rank), target.to(rank)
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = torch.nn.functional.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0 and rank == 0: # 从rank 0打印日志
                print(f"Epoch: {epoch} | Batch: {batch_idx} | Loss: {loss.item()}")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count() # GPU数量
    epochs = 10
    # mp.spawn(train_fn, args=(world_size, epochs), nprocs=world_size, join=True)
    # 注意：对于实际执行，请用真实实现替换MyModel和MyDataset
    # 并取消对mp.spawn的注释。
    print(f"为{world_size}个GPU设置的示例。要运行，请实现MyModel、MyDataset并取消对mp.spawn的注释。")
```

你可以使用`CUDA_VISIBLE_DEVICES`环境变量控制哪些GPU对PyTorch可见。例如，`CUDA_VISIBLE_DEVICES=0,1`将仅使GPU 0和GPU 1可用。

#### 多节点、多GPU

跨多台机器进行训练会引入更多的设置复杂性，主要与网络通信和进程发现有关。`torchrun`是对此的推荐工具。

你通常会在每个节点上使用`torchrun`启动你的训练脚本。`torchrun`的重要参数包括：

- `--nnodes`：节点总数。
- `--nproc_per_node`：每个节点的进程（通常是GPU）数量。
- `--rdzv_id`：唯一的作业ID。
- `--rdzv_backend`：汇合后端（例如，基于TCP的`c10d`）。
- `--rdzv_endpoint`：汇合服务器的端点（例如，`MASTER_NODE_IP:PORT`）。一个节点充当协调的主节点。

PyTorch脚本本身（如上面的`train_fn`）基本保持不变。`torchrun`设置`init_process_group(backend="nccl", init_method="env://")`用来建立通信的环境变量（`MASTER_ADDR`、`MASTER_PORT`、`WORLD_SIZE`、`RANK`）。像Slurm或Kubernetes这样的集群管理系统通常有集成或工具来简化跨节点启动`torchrun`。

### 与TensorFlow的`tf.distribute.Strategy`的对应关系

如果你使用过TensorFlow的`tf.distribute.Strategy`，你会发现相似之处：

- **`tf.distribute.MirroredStrategy`**：这与PyTorch在单个节点上使用多个GPU的`DistributedDataParallel` (DDP) 非常相似。它们都在每个GPU上复制模型，并使用AllReduce进行梯度同步。TensorFlow API可能会抽象化一些显式的进程组设置，将其更直接地集成到`Strategy`的范围内。
- **`tf.distribute.MultiWorkerMirroredStrategy`**：这对应于多节点设置中的DDP。两者都需要协调跨机器边界的进程。TensorFlow的策略依赖`TF_CONFIG`环境变量进行配置，而PyTorch通常使用`torchrun`或手动设置类似的环境变量（`MASTER_ADDR`等）。
- **`tf.distribute.ParameterServerStrategy`**：这涉及专用的参数 (parameter)服务器存储变量，而工作进程计算梯度。尽管PyTorch的DDP更类似于AllReduce架构，但`torch.distributed.rpc`可用于构建参数服务器风格的训练，尽管与DDP相比，这对于典型的深度学习 (deep learning)工作负载来说不太常见。
- **模型并行（`tf.distribute.experimental.TPUStrategy`或手动放置）**：TensorFlow对模型并行的支持，尤其是在TPU上通过`TPUStrategy`，可能涉及复杂的模型分片。手动情况下，使用`tf.device`作用域，类似于PyTorch的`.to(device)`。

主要区别通常在于设置的明确性。PyTorch的`torch.distributed`和DDP给你细粒度控制，但在初始化和进程启动方面需要更多的样板代码，特别是与`tf.distribute.Strategy`的上下文 (context)管理器风格相比。然而，分发数据和同步梯度的基本原理是相似的。

### 分布式训练的实践考量

- **批次大小和学习率**：当使用带有$N$个GPU的数据并行时，你的全局批次大小实际上变为$N \times (\text{每个GPU的批次大小})$。常见做法是缩放学习率，通常是线性缩放（例如，将基本学习率乘以$N$），但这可能需要调整。
- **使用`DistributedSampler`加载数据**：必须使用`torch.utils.data.distributed.DistributedSampler`。这个采样器确保每个进程加载数据集的独特、不重叠的子集。如果你希望洗牌在不同epoch之间正常工作，请记住在每个epoch开始时调用`sampler.set_epoch(epoch)`。
- **保存和加载检查点**：
  - **保存**：通常，只有一个进程（例如，`rank == 0`）应该保存模型检查点，以避免竞争条件或多次写入。你可以使用`ddp_model.module.state_dict()`从DDP访问底层模型。
  - **加载**：所有进程都需要加载模型权重 (weight)，以从一致状态开始。首先将`state_dict`加载到CPU上，然后将其映射到每个rank的正确GPU，以避免在rank 0上出现GPU内存问题（如果模型很大）：

    ```python
    # 在你的训练函数中，在setup(rank, world_size)之后
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank} # 映射到当前rank的GPU
    checkpoint = torch.load(PATH_TO_CHECKPOINT, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    # 可能加载优化器状态、epoch等
    ```

    或者，rank 0可以加载然后将状态字典广播给其他进程。
- **批量归一化 (normalization)**：DDP正确处理批量归一化统计信息的同步。每个DDP进程在其本地数据批次上计算BN统计信息。`SyncBatchNorm` (`torch.nn.SyncBatchNorm`) 可用于同步所有进程的统计信息，如果每个GPU的批次大小非常小，这会很有益。如果你要求或DDP检测到需要，它会自动将`BatchNorm`层转换为`SyncBatchNorm`。
- **日志记录和调试**：分布式训练调试起来可能更难。
  - 打印语句通常应由`if rank == 0:`保护，以避免输出混乱。
  - 对于错误，PyTorch通常提供来自所有rank的堆栈跟踪，这可能令人不知所措但有必要。
  - `torch.distributed.barrier()`等工具可用于在某些点同步进程以进行调试。
- **可复现性**：在分布式设置中实现完美可复现性可能具有挑战性，因为某些CUDA操作和通信模式的非确定性。在所有进程中一致地设置随机种子是必要的第一步：`torch.manual_seed(seed)`。

通过理解这些方法和组件，你可以有效地扩展你的PyTorch训练工作流，就像你在TensorFlow中使用`tf.distribute.Strategy`一样，使你能够处理更大、更复杂的机器学习 (machine learning)问题。

## 参考资料

- [Distributed communication package - torch.distributed](https://pytorch.org/docs/stable/distributed.html) — PyTorch Core Team (2024)
  PyTorch 分布式通信原语和进程组管理的官方文档。
- [\`torch.nn.parallel.DistributedDataParallel\`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) — PyTorch Core Team (2024)
  PyTorch 主要数据并行模块的官方 API 参考和使用指南。
- [Accurate, Large Minibatch SGD: A Case Study of ImageNet Training](https://arxiv.org/abs/1706.02677) — Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, Kaiming He (2017)
  Journal: arXiv preprint arXiv:1706.02677; DOI: [10.48550/arXiv.1706.02677](https://doi.org/10.48550/arXiv.1706.02677)
  本文展示了如何通过线性学习率缩放和高效的梯度聚合（AllReduce）将深度学习训练有效扩展到多个 GPU 上的大批量数据。
- [DeepSpeed: Extreme-scale distributed training for DL models](https://www.usenix.org/conference/atc21/presentation/yao-zhewei) — Zhewei Yao, Minjia Zhang, Conglong Li, Yuxiong He (2021)
  Journal: In 2021 USENIX Annual Technical Conference (USENIX ATC 21); Publisher: USENIX; Pages: 1729-1744; DOI: [10.5555/3472061.3472251](https://doi.org/10.5555/3472061.3472251)
  该系统用于训练深度学习模型，涵盖了基于 PyTorch 原语构建的流水线并行和 ZeRO 等技术。
