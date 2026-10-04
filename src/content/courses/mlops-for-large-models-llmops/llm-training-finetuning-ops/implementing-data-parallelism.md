---
course: "mlops-for-large-models-llmops"
chapter: "llm-training-finetuning-ops"
lesson: "implementing-data-parallelism"
sourceId: 3140
sourceUrl: "https://apxml.com/zh/courses/mlops-for-large-models-llmops/chapter-3-llm-training-finetuning-ops/implementing-data-parallelism"
title: "实现数据并行策略"
description: "应用数据并行技术来加速大型语言模型训练。"
order: 2
plots: []
sourceHash: "c594df8fd3cbf7673044403157425a41f49996c476bb65dea5107e015542923a"
sourceCorrections: []
---

在训练大型语言模型时，在单个加速器上处理所需的**海量**数据很快就会变得**不可行**。数据并行提供了一种**基本策略**，可将此工作负载**分发**到**多个处理单元**（通常是 GPU 或 TPU），从而更快地训练模型并处理更大的有效批处理大小。

数据并行的**核心思想**很简单：在每个**可用工作器（设备）**上**复制整个**模型，向每个工作器提供输入数据批次的**不同片段（分片）**，然后在每一步之后**合并结果**。我们来**分析一下**典型的工作流程：

1. **模型复制：** 在训练开始时（或参数 (parameter)更新后），确保每个工作器都拥有模型的参数（$\theta$）的相同副本。
2. **数据分片：** 将当前训练数据的全局微批次（$B$）划分为更小、**不同的微批次**（$b_i$），其中 $i$ 是工作器的索引。每个工作器 $i$ 接收其特定的微批次 $b_i$。
3. **本地计算：** 每个工作器 $i$ 使用其模型副本（$\theta$）和其本地数据分片（$b_i$）执行一次前向传播以**计算损失**（$L_i$）。随后，它执行一次反向传播 (backpropagation)以计算损失相对于其模型参数的梯度（$\nabla L_i(\theta)$）。这在所有工作器上**独立并同时**进行。
4. **梯度同步：** 这是协调步骤。每个工作器上本地计算的梯度（$\nabla L_i(\theta)$）需要在所有工作器之间**聚合**，以获得基于全局微批次的梯度（$\nabla L(\theta)$）。在现代大型语言模型训练中，最常用的方法是 **AllReduce** 集体通信操作。

- **AllReduce：** AllReduce 会对所有工作器的梯度求和，并将结果（和，或通常是平均值）**分发**回所有工作器。每个工作器 $i$ 发送其 $\nabla L_i(\theta)$ 并接收最终的 $\nabla L(\theta) = \frac{1}{N} \sum_{j=1}^{N} \nabla L_j(\theta)$，其中 $N$ 是工作器的数量。这**确保**所有工作器在更新步骤中具有相同的梯度。高效的实现（如环形 AllReduce）可**最大程度地减少**通信瓶颈，但通信成本通常仍随模型参数的数量而扩展。

5. **参数更新：** 每个工作器使用同步梯度（$\nabla L(\theta)$）通过所选优化器（例如 AdamW）更新其模型参数（$\theta$）的本地副本。由于所有工作器都从相同的 $\theta$ 开始并接收相同的 $\nabla L(\theta)$，它们的参数副本在更新后保持同步。

> 数据并行工作流程：全局批次被**拆分**，每个工作器使用模型副本在其分片上计算梯度。梯度通过 AllReduce 同步，并且每个工作器以相同方式更新其模型副本。

### 实现框架和考量

PyTorch (`DistributedDataParallel` 或 DDP)、TensorFlow (`tf.distribute.Strategy`) 和 Horovod 等框架**抽离了**大部分实现数据并行的复杂性。DeepSpeed 也建立在这些**思想**之上，并添加了额外的优化。

一个使用 PyTorch DDP 的典型结构可能如下所示（简化版）：

```python
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    # 初始化进程组（例如，使用 NCCL 后端用于 GPU）
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_step(rank, world_size, model, data_loader, optimizer, criterion):
    model.train()
    # 如果 DataLoader 使用 DistributedSampler，DDP 会自动处理数据分片
    for data, target in data_loader:
        data, target = data.to(rank), target.to(rank) # 将数据移动到工作器的设备

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # loss.backward() 计算本地梯度
        loss.backward()

        # DDP 会在反向传播期间自动触发 AllReduce
        # 梯度在此处在工作器之间同步

        # optimizer.step() 使用同步梯度更新本地模型参数
        optimizer.step()

def main_worker(rank, world_size, model_definition, dataset):
    setup(rank, world_size)

    model = model_definition().to(rank)
    # 使用 DDP 包装模型
    ddp_model = DDP(model, device_ids=[rank])

    # 为 DataLoader 使用 DistributedSampler
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=local_batch_size, sampler=sampler)

    optimizer = optim.AdamW(ddp_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss() # 示例损失函数

    for epoch in range(num_epochs):
        train_step(rank, world_size, ddp_model, data_loader, optimizer, criterion)
        # 添加验证、检查点等

    cleanup()

# --- 主执行逻辑以启动进程 ---
# if __name__ == "__main__":
#     world_size = torch.cuda.device_count()
#     mp.spawn(main_worker, args=(world_size, model_def, dataset), nprocs=world_size, join=True)
```

在使用大规模数据并行时，会出现一些**重要的操作点**：

- **通信开销：** AllReduce 步骤引入了通信依赖。其成本随模型大小（需要同步更多参数 (parameter)）增加，并可能受限于节点间的互连带宽（例如，NVLink、InfiniBand、以太网）。对于数十亿参数的模型，这种同步会成为步骤时间的很大一部分。
- **批次大小影响：** 数据并行有效增加了全局批次大小：$\text{全局批次大小} = \text{每个工作器的本地批次大小} \times \text{工作器数量}$。训练动态会随非常大的批次大小而改变。通常，学习率需要进行缩放（例如，线性缩放规则），收敛可能需要调整优化器超参数 (hyperparameter)（如 Adam 或动量中的 beta 值）。
- **梯度累积：** 当模型适合单个工作器，但所需的全局批次大小对 `local_batch_size` 来说需要比每个工作器可用内存更多的内存时，会使用梯度累积。工作器会**按顺序处理**多个更小的“微批次”，在执行 AllReduce 和优化器步骤之前在本地累积梯度。这**模拟了**更大的本地批次大小，同时不增加内存需求，通过计算时间来换取内存。AllReduce 同步仅在每 N 个微批次发生一次，其中 N 是累积步数。
- **内存限制：** 纯数据并行的**主要限制**是整个模型必须适应每个工作器的内存。这对于最大的大型语言模型（例如，在典型的当前 GPU 上超过 1000 亿参数）来说**限制性很强**。

数据并行因其相对简单和有效，通常是分布式训练的**起点**，尤其当每个数据点的计算量很大时。然而，当模型对单个设备的内存来说变得太大时，您必须将数据并行与模型并行技术**结合**起来，我们接下来会**讨论**这些技术。

## 参考资料

- [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) — PyTorch Documentation (2023)
  PyTorch数据并行主要API的官方文档，详细说明了其使用方法和同步训练的内部机制。
- [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) — Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, Kaiming He (2017)
  Journal: arXiv preprint arXiv:1706.02677; DOI: [10.48550/arXiv.1706.02677](https://doi.org/10.48550/arXiv.1706.02677)
  一篇基础性论文，讨论了使用大批量进行训练的技术，包括学习率的线性缩放规则，这在数据并行设置中很重要。
- [NVIDIA Collective Communications Library (NCCL)](https://developer.nvidia.com/nccl) — NVIDIA Corporation (2023)
  用于高度优化的集合通信库的官方资源，该库提供AllReduce等原语，对于NVIDIA GPU上的高效数据并行至关重要。
