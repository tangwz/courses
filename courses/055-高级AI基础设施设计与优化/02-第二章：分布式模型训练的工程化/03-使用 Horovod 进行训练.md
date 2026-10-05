# 使用 Horovod 进行训练

来源：[原文](https://apxml.com/zh/courses/advanced-ai-infrastructure-design-optimization/chapter-2-engineering-distributed-model-training/implementing-training-horovod)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管原生的深度学习 (deep learning)框架提供了数据并行的基本功能，但它们的设置可能很冗长，需要大量的样板代码来管理进程组和通信。Horovod 最初由 Uber 开发，旨在简化此过程。它提供了一个高级的、不依赖特定框架的 API，构建在 MPI（消息传递接口）或 Gloo 等通信后端之上，使得将单 GPU 训练脚本转换为多 GPU 分布式脚本变得简单。

Horovod 采用单程序多数据 (SPMD) 模型运行。您编写一个脚本，`horovodrun` 会以独立进程的形式启动其多个副本。每个进程通过其独特的“rank”（等级）标识，运行相同的代码，但处理训练数据的不同部分。

### 核心 API 和初始化

要使脚本适应 Horovod，您只需插入几个重要的函数调用。Horovod API (`hvd`) 的主要组成部分是：

- `hvd.init()`: 初始化 Horovod。这是第一个 Horovod 操作。它建立通信后端并使进程相互发现。
- `hvd.rank()`: 返回当前进程的唯一 ID 或等级，范围从 0 到 `hvd.size() - 1`。等级为 0 的进程通常处理特殊任务，例如日志记录或保存检查点。
- `hvd.local_rank()`: 返回单个服务器节点内进程的唯一等级。这对于将进程绑定到特定 GPU 非常重要。
- `hvd.size()`: 返回参与分布式任务的进程总数。

数据并行中最重要的任务是在反向传播 (backpropagation)后同步所有工作进程的模型梯度。一种简单方法是指定一个工作进程（例如，rank 0）作为参数 (parameter)服务器来收集梯度，平均它们，然后发回。这会造成通信瓶颈，因为 rank 0 的带宽限制了整个集群的性能。

Horovod 通过高效的去中心化算法——环形Allreduce——解决了此问题。不同于中心服务器，进程以逻辑环形排列。每个 GPU 上的梯度张量被分成块。在第一阶段，散布-归约，每个进程向环中的邻居发送一个块，同时从其另一个邻居接收一个不同的块，并在接收时进行求和。在 `N-1` 步之后（其中 `N` 是 GPU 的数量），每个进程持有特定块的完整总和。在第二阶段，全收集，进程在环中交换这些求和后的块，直到所有进程都拥有完整且完全平均的梯度张量的副本。此方法确保网络带宽使用达到最佳，并且独立于工作进程数量。

> 环形Allreduce算法通过将工作进程组织成逻辑环，避免了中心瓶颈。梯度在两个阶段高效地平均，而不会使单个节点过载。

### 修改训练脚本以使用 Horovod

我们将介绍将标准 PyTorch 训练脚本转换为使用 Horovod 进行分布式训练所需的五项修改。

#### 1. 初始化 Horovod

在脚本开头添加 `hvd.init()` 以设置通信后端。

#### 2. 为每个进程分配 GPU

为防止同一机器上的进程争抢 GPU，每个进程必须绑定到独占的 GPU。为此，请使用 `hvd.local_rank()`。

```python
# 将 GPU 绑定到用于处理本地等级的进程（每个进程一个 GPU）
torch.cuda.set_device(hvd.local_rank())
```

#### 3. 缩放学习率

使用数据并行时，全局批处理大小变为 `local_batch_size * hvd.size()`。更大的批处理大小通常需要更高的学习率以保持收敛速度。一种常见做法是根据工作进程数量来缩放基础学习率。

```python
# 一种常见做法是根据工作进程数量来缩放学习率。
optimizer = torch.optim.Adam(model.parameters(), lr=base_lr * hvd.size())
```

#### 4. 封装优化器

这是实现梯度同步的核心步骤。Horovod 提供了一个 `DistributedOptimizer` 封装器，它会拦截 `optimizer.step()` 调用。在更新权重 (weight)之前，它会在后台自动对梯度执行 All-Reduce 操作。

```python
import horovod.torch as hvd

# ... (定义模型和优化器)

# 使用 Horovod 的 DistributedOptimizer 封装优化器。
optimizer = hvd.DistributedOptimizer(optimizer,
                                   named_parameters=model.named_parameters())
```

`named_parameters` 参数 (parameter)帮助 Horovod 将梯度映射到模型参数，这对于某些高级功能（如梯度压缩）很重要。

#### 5. 广播初始状态并划分数据

为确保一致的起始点，来自 rank 0 的初始模型权重必须广播到所有其他进程。这防止工作进程因不同的随机初始化而发散。Horovod 在优化器封装后通过一个函数调用处理此问题。

```python
# 将参数和优化器状态从 rank 0 广播到所有其他进程。
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)
```

此外，您必须确保每个进程在每个 epoch 接收到数据的唯一且不重叠的子集。PyTorch 的 `DistributedSampler` 为此可直接与 Horovod 集成。

```python
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=..., sampler=train_sampler)
```

### 示例：完整的 Horovod 脚本

这是一个精简的 PyTorch 脚本，体现了这五项修改在实际应用中的情况。

```python
import torch
import horovod.torch as hvd

# 1. 初始化 Horovod
hvd.init()

# 2. 将 GPU 绑定到本地等级
torch.cuda.set_device(hvd.local_rank())

# 定义一个简单模型
model = torch.nn.Sequential(torch.nn.Linear(784, 128),
                            torch.nn.ReLU(),
                            torch.nn.Linear(128, 10))
model.cuda()

# 定义损失函数和优化器
base_lr = 0.001
criterion = torch.nn.CrossEntropyLoss()
# 3. 根据工作进程数量缩放学习率
optimizer = torch.optim.Adam(model.parameters(), lr=base_lr * hvd.size())

# 4. 使用 DistributedOptimizer 封装优化器
optimizer = hvd.DistributedOptimizer(
    optimizer,
    named_parameters=model.named_parameters(),
)

# 5. 从 rank 0 广播初始状态
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# 创建一个用于演示的虚拟数据集和采样器
# 5. 使用 DistributedSampler 划分数据
data = torch.randn(1000, 784)
labels = torch.randint(0, 10, (1000,))
dataset = torch.utils.data.TensorDataset(data, labels)
sampler = torch.utils.data.distributed.DistributedSampler(
    dataset, num_replicas=hvd.size(), rank=hvd.rank())
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=sampler)

# 训练循环
model.train()
for epoch in range(5):
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step() # All-reduce 由此处的封装器处理

    if hvd.rank() == 0:
        print(f"轮次 {epoch}: 损失={loss.item()}")
```

### 启动 Horovod 任务

要运行此脚本，您使用 `horovodrun` 命令行工具。它管理 MPI 进程的启动。要在具有四个 GPU 的单台机器上运行此脚本，命令如下：

```bash
# -np 指定进程（工作进程）总数
# -H 指定主机。'localhost:4' 表示在本地机器上运行 4 个进程。
horovodrun -np 4 -H localhost:4 python train_horovod.py
```

Horovod 的优势在于其简单性和效率，适用于纯数据并行工作负载。通过添加几行代码，您可以将单 GPU 脚本分发到多个加速器上。然而，其模型假定整个模型副本及其优化器状态和梯度可以适应单个加速器的内存。接下来我们将看到，对于超出此限制的模型，我们需要更先进的内存优化和模型分片技术，这些技术由 Microsoft DeepSpeed 等框架提供。

## 参考资料

- [Horovod documentation](https://horovod.readthedocs.io/en/stable/) — Horovod contributors (2024)
  官方文档，介绍 Horovod 的 API、安装和分布式深度学习的最佳实践。
- [Horovod: Fast and Easy Distributed Deep Learning](https://arxiv.org/abs/1802.05799) — Alexander Sergeev, Mike Del Balso (2018)
  Journal: arXiv:1802.05799 [cs.LG]; DOI: [10.48550/arXiv.1802.05799](https://doi.org/10.48550/arXiv.1802.05799)
  介绍 Horovod 的基础论文，阐述其设计原则、性能优化和高效的 Ring-Allreduce 算法。
- [torch.distributed - PyTorch 2.3 documentation](https://pytorch.org/docs/stable/distributed.html) — PyTorch Contributors (2017)
  PyTorch 分布式通信原语和实用工具的官方文档，包括用于 Horovod 集成脚本中数据分区的 `DistributedSampler`。

---

[上一节](02-%E5%A4%A7%E5%9E%8B%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%B5%81%E6%B0%B4%E7%BA%BF%E5%B9%B6%E8%A1%8C.md) · [下一节](04-%E4%BD%BF%E7%94%A8%20Microsoft%20DeepSpeed%20%E5%AE%9E%E7%8E%B0%20ZeRO%20%E5%92%8C%E5%8D%B8%E8%BD%BD.md)
