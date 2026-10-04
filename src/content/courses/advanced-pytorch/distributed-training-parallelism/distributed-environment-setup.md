---
course: "advanced-pytorch"
chapter: "distributed-training-parallelism"
lesson: "distributed-environment-setup"
sourceId: 2774
sourceUrl: "https://apxml.com/zh/courses/advanced-pytorch/chapter-5-distributed-training-parallelism/distributed-environment-setup"
title: "设置分布式环境"
description: "配置分布式训练任务，使用不同的通信后端（NCCL, Gloo）和进程组初始化。"
order: 7
plots: []
sourceHash: "bd1802d8d917aea316427fc7daf7c22c4778345785ca331193da8848af988374"
sourceCorrections: []
---

建立进程间的连接是分布式训练任务中重要第一步。PyTorch 的 `torch.distributed` 包提供管理这种通信设置所需的工具。主要思路是创建一个 *进程组*，它包含所有参与任务的进程。每个进程被分配一个唯一 *rank*，它们共同了解进程总数，称为 *size*。

### 初始化进程组

设置分布式环境的主要函数是 `torch.distributed.init_process_group`。此函数初始化默认进程组，且必须由每个参与分布式任务的进程调用。

```python
import torch
import torch.distributed as dist
import os

def setup_distributed(backend='nccl'):
    """初始化分布式环境。"""
    if not dist.is_available():
        print("分布式训练不可用。")
        return

    if not dist.is_initialized():
        # 这些环境变量通常由启动工具设置
        # (例如，torchrun, Slurm)
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "29500") # 默认端口

        print(f"正在初始化进程组: Rank {rank}/{world_size}")
        dist.init_process_group(
            backend=backend,
            init_method=f'tcp://{master_addr}:{master_port}',
            rank=rank,
            world_size=world_size
        )
        print(f"进程组已初始化 ({backend})。")

    # 为当前进程设置设备。这很重要！
    # 假设每个GPU一个进程。
    if backend == 'nccl' and torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        print(f"Rank {dist.get_rank()} 正在使用 GPU {local_rank}")

# 示例用法（通常在脚本开头调用）
# setup_distributed(backend='nccl') # 或 'gloo'
```

我们来分析一下 `init_process_group` 的参数 (parameter)：

- `backend`: 指定使用的通信库。此选择取决于您的硬件和要求。
- `init_method`: 定义进程如何互相发现。最常用方法是使用 TCP，并需要 rank 0 进程（主进程）的地址和端口。基于环境变量的初始化 (`'env://'`) 也常使用，它依赖于预设的特定环境变量。
- `rank`: 当前进程的唯一标识符，范围从 0 到 `world_size - 1`。
- `world_size`: 参与训练任务的进程总数。

### 通信后端

PyTorch 支持多种后端来处理进程间的底层通信：

1. **NCCL (NVIDIA Collective Communication Library):** 这是 NVIDIA 硬件上基于 GPU 的分布式训练的推荐后端。它为 CUDA 张量提供了集体通信操作（如 `all_reduce`、`broadcast`）的优化实现，提供高带宽和低延迟。当您的所有进程都在配备 NVIDIA GPU 并通过 NVLink 或 InfiniBand 等高速互连连接的机器上运行时，请使用 NCCL。
2. **Gloo:** Gloo 是一个更通用的后端，适用于 CPU 和 GPU 通信。对于异构环境、仅限 CPU 的训练，或者当 NCCL 不可用或不适用时（例如，某些云环境或旧硬件），它是一个不错的选择。虽然在 GPU 到 GPU 通信方面通常比 NCCL 慢，但它提供更广泛的兼容性。
3. **MPI (Message Passing Interface):** MPI 是一个高性能计算标准。如果您的集群环境已配置 MPI 实现（如 Open MPI），您可以使用 MPI 后端。PyTorch 需要编译时包含 MPI 支持才能使用此选项。它常用于学术或研究集群。

后端选择会显著影响性能。对于 NVIDIA 硬件上的典型多 GPU 训练，NCCL 几乎总是最佳选择。

> 使用通信后端和发现方法（如带主地址/端口的 TCP）进行进程组初始化。每个进程都调用 `init_process_group`。

### 环境变量

分布式训练设置通常依赖环境变量进行配置，尤其是在使用启动工具时。最重要的包括：

- `MASTER_ADDR`: 托管 rank 0 进程的机器的 IP 地址或主机名。所有其他进程都需要连接到此地址。
- `MASTER_PORT`: rank 0 机器上的一个开放网络端口，供进程协调初始化。选择一个不太可能被占用的端口（例如，10000 以上）。
- `RANK`: 当前进程的唯一 rank。
- `WORLD_SIZE`: 参与任务的进程总数。
- `LOCAL_RANK` (常用): 当在同一节点上运行多个进程时（例如，每个 GPU 一个进程），`LOCAL_RANK` 通常标识进程在该节点内的索引（通常从 0 到 `num_gpus_on_node - 1`）。这常用于通过 `torch.cuda.set_device(local_rank)` 为每个进程分配一个特定 GPU。

这些变量通常由启动脚本或集群调度器自动设置。

### 启动分布式脚本

手动设置环境变量并在每台机器上启动 Python 脚本可能繁琐且易错。PyTorch 提供 `torchrun`（以前是 `torch.distributed.launch`）工具来简化此过程。

`torchrun` 负责设置必要的环境变量（`RANK`、`WORLD_SIZE`、`MASTER_ADDR`、`MASTER_PORT`、`LOCAL_RANK`）并启动每个节点指定数量的进程。

**在单机 2 GPU 上使用 `torchrun` 的示例：**

```bash
# 假设您的训练脚本名为 train.py
# 并且它包含前面所示的 setup_distributed() 函数。

# --nproc_per_node 指定在此机器上启动多少个进程。
# 通常设置为可用 GPU 的数量。
# --nnodes=1 表示我们在单机上运行。
# train.py 是您的脚本，后面跟着它的参数。

torchrun --nproc_per_node=2 --nnodes=1 train.py --arg1 value1 --arg2 value2
```

对于多节点训练，您通常在每个节点上运行 `torchrun`，指定节点总数（`--nnodes`）、当前节点的 rank（`--node_rank`）以及主节点的地址和端口（`--rdzv_endpoint`）：

```bash
# 在主节点 (Rank 0):
torchrun \
    --nproc_per_node=<gpus_on_master> \
    --nnodes=<total_nodes> \
    --node_rank=0 \
    --rdzv_id=<job_id> \
    --rdzv_backend=c10d \
    --rdzv_endpoint="<master_node_ip>:<port>" \
    train.py --args...

# 在工作节点 1 (Rank 1):
torchrun \
    --nproc_per_node=<gpus_on_worker1> \
    --nnodes=<total_nodes> \
    --node_rank=1 \
    --rdzv_id=<job_id> \
    --rdzv_backend=c10d \
    --rdzv_endpoint="<master_node_ip>:<port>" \
    train.py --args...

# ... 其他工作节点以此类推
```

这里，`rdzv` 是 rendezvous 的缩写。`torchrun` 使用此机制（通常构建在 `torch.distributed` 的 `c10d` 后端之上）来协调跨节点启动。`<job_id>` 对于此特定训练运行应是唯一的。

### 清理

训练结束后，清理分布式环境资源是良好实践：

```python
def cleanup_distributed():
    """销毁默认进程组。"""
    if dist.is_initialized():
        dist.destroy_process_group()
        print("进程组已销毁。")

# 示例用法（通常在脚本末尾或 finally 块中）
# try:
#     # setup_distributed(...)
#     # ... training loop ...
# finally:
#     cleanup_distributed()
```

调用 `dist.destroy_process_group()` 会释放与进程组关联的资源。虽然 Python 的退出通常会处理此问题，但明确清理更安全，尤其是在复杂应用或长期运行的服务中。

正确配置分布式环境是构建所有并行训练策略的根本。了解进程组、rank、size、通信后端以及 `torchrun` 等启动工具对于有效扩展您的 PyTorch 训练任务非常重要。

## 参考资料

- [torch.distributed (Distributed communications)](https://pytorch.org/docs/stable/distributed.html) — PyTorch Developers (2024)
  提供 PyTorch 分布式通信包的官方API参考，详细介绍了 `init_process_group`、通信后端和分布式编程概念。
- [TorchRun (Elastic Launch)](https://pytorch.org/docs/stable/elastic/run.html) — PyTorch Developers (2021)
  `torchrun`（以前是 `torch.distributed.launch`）的官方文档，描述了其用于启动多进程和多节点分布式训练作业以及管理环境变量的用法。
- [NVIDIA Collective Communications Library (NCCL) User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html) — NVIDIA Corporation (2024)
  Publisher: NVIDIA
  NVIDIA Collective Communications Library 的官方文档，详细介绍了其架构、功能以及针对GPU加速分布式计算的性能优化。
- [Getting Started with Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) — Shen Li (2019)
  Publisher: PyTorch Foundation
  一个关于设置和使用 PyTorch `DistributedDataParallel` 进行高效分布式训练的教程，说明了 `torch.distributed` 组件的实际应用。
