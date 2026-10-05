# 使用 torch.distributed 通信原语

来源：[原文](https://apxml.com/zh/courses/advanced-pytorch/chapter-5-distributed-training-parallelism/torch-distributed-primitives)

[返回章节目录](README.md) · [返回课程目录](../README.md)

高级抽象，例如 `DistributedDataParallel` (DDP)，会自动处理分布式训练的许多细节。然而，了解 `torch.distributed` 包提供的底层通信原语可以提供全面的理解，并能够实现自定义的并行化策略。这些原语是用于在分布式环境中协调不同进程之间通信的基本组成部分。

在使用任何通信原语之前，必须先初始化分布式环境，通常使用 `torch.distributed.init_process_group` 函数。这会建立通信后端（例如 NCCL 或 Gloo），并为总 `world_size` 中的每个进程分配一个唯一的 `rank`。一旦初始化完成，默认组（或自定义创建的组）中的进程就可以使用集体和点对点操作进行协调。

### 集体通信操作

集体操作涉及组内所有进程之间的通信。它们对于同步梯度或分发模型参数 (parameter)等任务非常重要。以下是一些最常用的集体操作：

#### 广播 (`dist.broadcast`)

此操作将张量从一个源进程 (`src`) 发送到组中的所有其他进程。它通常用于确保所有进程都使用相同的初始模型参数开始。

```python
import torch
import torch.distributed as dist
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # 初始化进程组
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def run_broadcast(rank, world_size):
    setup(rank, world_size)
    tensor = torch.zeros(1)
    if rank == 0:
        # 源进程创建数据
        tensor += 1

    # Rank 0 将 'tensor' 广播到所有其他进程
    dist.broadcast(tensor=tensor, src=0)

    print(f"Rank {rank} has data: {tensor[0]}")
    dist.destroy_process_group()

# 假设 world_size = 4 进行演示
# 在实际脚本中，这将通过 torchrun 或类似方式启动
# run_broadcast(0, 4)
# run_broadcast(1, 4)
# run_broadcast(2, 4)
# run_broadcast(3, 4)
```

此操作后，Rank 1、2 和 3 上的 `tensor` 将从 `0` 更新为 `1`。

> `dist.broadcast` 操作中数据流的示意图，从 Rank 0 到 4 进程组中的所有其他 Rank。

#### 全局归约 (`dist.all_reduce`)

此操作使用指定的归约操作（`op`，例如 `dist.ReduceOp.SUM`、`dist.ReduceOp.AVG`）合并所有进程的张量，并将最终结果分发回所有进程。这是 DDP 中梯度同步的根本。每个进程贡献其局部梯度，这些梯度在所有进程中求和（或求平均），然后每个进程都收到合并后的梯度。

```python
import torch
import torch.distributed as dist
import os

# 假设 setup 函数已如上定义

def run_all_reduce(rank, world_size):
    setup(rank, world_size)
    # 每个 Rank 根据其 Rank 创建数据
    tensor = torch.tensor([rank + 1], dtype=torch.float32) 

    print(f"Rank {rank} initial tensor: {tensor[0]}")

    # 执行 SUM 操作的全归约
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # 结果 (1+2+3+4 = 10 的和) 在所有 Rank 上都可用
    print(f"Rank {rank} final tensor: {tensor[0]}") 
    dist.destroy_process_group()

# world_size = 4 的执行示例
# run_all_reduce(0, 4) # 初始值: 1, 最终值: 10
# run_all_reduce(1, 4) # 初始值: 2, 最终值: 10
# run_all_reduce(2, 4) # 初始值: 3, 最终值: 10
# run_all_reduce(3, 4) # 初始值: 4, 最终值: 10
```

> `dist.all_reduce` 进行求和操作的数据流示意图。所有 Rank 贡献数据，数据被聚合，结果再分发回所有 Rank。

#### 归约 (`dist.reduce`)

与 `all_reduce` 类似，`reduce` 使用归约操作合并所有进程的张量。但其结果只存储在目标进程 (`dst`) 上。其他进程不会收到结果。

#### 分散 (`dist.scatter`)

此操作获取单个源进程 (`src`) 上的一个张量列表 (`scatter_list`)，并将列表中的一个张量分发给组中的每个进程，包括其自身。`scatter_list` 中第 $i$ 个张量会发送给 Rank 为 $i$ 的进程。这有助于在进程间分发数据批次。

```python
import torch
import torch.distributed as dist
import os

# 假设 setup 函数已如上定义

def run_scatter(rank, world_size):
    setup(rank, world_size)
    my_tensor = torch.zeros(1)
    scatter_list = None

    if rank == 0:
        # 源 Rank 准备要分散的张量列表
        scatter_list = [torch.tensor([i + 1.0]) for i in range(world_size)]
        print(f"Rank 0 scatter list: {[t.item() for t in scatter_list]}")

    # Rank 0 分散列表。每个 Rank 都会收到一个张量到 my_tensor 中。
    dist.scatter(tensor=my_tensor, scatter_list=scatter_list, src=0)

    print(f"Rank {rank} received tensor: {my_tensor.item()}")
    dist.destroy_process_group()

# world_size = 4 的执行示例
# run_scatter(0, 4) # 收到: 1.0
# run_scatter(1, 4) # 收到: 2.0
# run_scatter(2, 4) # 收到: 3.0
# run_scatter(3, 4) # 收到: 4.0
```

> `dist.scatter` 的数据流示意图。Rank 0 持有张量列表 [A, B, C, D]，并将 A 发送给 Rank 0，B 发送给 Rank 1，C 发送给 Rank 2，D 发送给 Rank 3。

#### 收集 (`dist.gather`)

`scatter` 的反向操作。每个进程将其张量发送到一个目标进程 (`dst`)。目标进程接收这些张量并将它们存储在一个列表 (`gather_list`) 中。`gather_list` 中的顺序与发送进程的 Rank 对应。

```python
import torch
import torch.distributed as dist
import os

# 假设 setup 函数已如上定义

def run_gather(rank, world_size):
    setup(rank, world_size)
    # 每个 Rank 创建自己的张量
    my_tensor = torch.tensor([rank + 1.0]) 
    gather_list = None

    if rank == 0:
        # 目标 Rank 准备一个列表来存储收集到的张量
        gather_list = [torch.zeros(1) for _ in range(world_size)]

    # 所有 Rank 将其张量发送给 Rank 0
    dist.gather(tensor=my_tensor, gather_list=gather_list, dst=0)

    if rank == 0:
        print(f"Rank 0 gathered list: {[t.item() for t in gather_list]}")
    else:
        print(f"Rank {rank} sent tensor: {my_tensor.item()}")

    dist.destroy_process_group()

# world_size = 4 的执行示例
# run_gather(0, 4) # 收集到: [1.0, 2.0, 3.0, 4.0]
# run_gather(1, 4) # 已发送: 2.0
# run_gather(2, 4) # 已发送: 3.0
# run_gather(3, 4) # 已发送: 4.0
```

> `dist.gather` 的数据流示意图。Rank 0、1、2、3 分别将其张量 A、B、C、D 发送给 Rank 0，Rank 0 将它们收集到列表 [A, B, C, D] 中。

#### 全局收集 (`dist.all_gather`)

与 `gather` 类似，但从所有进程收集到的张量列表结果会分发回组中的*所有*进程。每个进程都会收到相同的最终列表。

### 点对点通信操作

这些操作涉及两个特定进程之间的通信，通过它们的 Rank 进行识别。

- `dist.send(tensor, dst)`: 将张量从当前进程发送到目标进程 (`dst`)。这是发送方的阻塞操作。
- `dist.recv(tensor, src)`: 从源进程 (`src`) 接收张量到提供的 `tensor` 缓冲区中。这是接收方的阻塞操作，直到张量被接收。

虽然功能强大，但点对点操作需要仔细管理以避免死锁（例如，两个进程在发送之前互相等待接收）。与集体操作相比，它们在标准数据并行训练中不那么常用，但对于模型并行或自定义算法等更复杂的通信模式很重要。

### 阻塞与非阻塞操作

大多数集体操作（`broadcast`、`all_reduce`、`scatter`、`gather` 等）默认是*阻塞*（同步）的。这意味着进程上的程序执行会暂停，直到该进程完成了其在集体通信中的部分。

PyTorch 还提供了许多操作的*非阻塞*（异步）版本，通常以 `i` 作为前缀（例如 `dist.isend`、`dist.irecv`、`dist.all_reduce(..., async_op=True)`）。这些调用会启动通信并立即返回一个 `Work` 对象（或类似的句柄）。在通信在后台进行的同时，程序可以继续执行其他任务。您可以稍后使用返回句柄上的 `wait()` 等方法检查完成情况或等待操作结束。

```python
# 非阻塞全归约示例
tensor = torch.ones(1) * rank
# ... 其他设置 ...

# 启动非阻塞全归约
work_handle = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)

# 在通信进行时执行其他计算...
# result = compute_something_else() 

# 等待 all_reduce 操作完成
work_handle.wait() 

# 现在 'tensor' 包含了归约后的结果
print(f"Rank {rank} async all_reduce result: {tensor[0]}")
```

使用非阻塞操作可以将计算与通信重叠，从而显著提升性能，尤其是在具有快速互连的系统上。但是，这需要仔细管理依赖关系和同步点。

了解这些 `torch.distributed` 原语为实现复杂的分布式训练流程奠定了根基。它们允许对进程间通信进行细粒度控制，这对于流水线并行、自定义梯度聚合方案或与专用硬件通信库交互等技术是必要的。

## 参考资料

- [Distributed Communication Package - torch.distributed](https://pytorch.org/docs/stable/distributed.html) — PyTorch Authors (2017)
  Publisher: PyTorch Foundation
  提供 `torch.distributed` 原语的官方API参考、使用示例和参数详情。
- [MPI: A Message-Passing Interface Standard, Version 4.0](https://www.mpi-forum.org/docs/mpi-4.0/mpi40-report.pdf) — Message Passing Interface Forum (2021)
  Journal: MPI Forum
  介绍了消息传递通信原语的理论基础和架构原理，适用于 `torch.distributed`。
- [NVIDIA Collective Communications Library (NCCL) Documentation](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html) — NVIDIA Corporation (2024)
  Publisher: NVIDIA Corporation
  详细说明了用于集合操作的优化后端，这对于GPU加速分布式训练的性能至关重要。
- [Hands-On Distributed PyTorch: Training models with multiple GPUs and multiple machines](https://learning.oreilly.com/library/view/hands-on-distributed-pytorch/9781098111974/) — Robert Yu, Luke Melas-Kyriazi (2022)
  Publisher: O'Reilly Media
  提供PyTorch分布式训练的实用指导和解释，包括通信原语的使用。

---

[上一节](05-%E5%85%A8%E5%88%86%E7%89%87%E6%95%B0%E6%8D%AE%E5%B9%B6%E8%A1%8C%EF%BC%88FSDP%EF%BC%89.md) · [下一节](07-%E8%AE%BE%E7%BD%AE%E5%88%86%E5%B8%83%E5%BC%8F%E7%8E%AF%E5%A2%83.md)
