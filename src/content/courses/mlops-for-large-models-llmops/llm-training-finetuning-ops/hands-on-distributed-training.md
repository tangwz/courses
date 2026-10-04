---
course: "mlops-for-large-models-llmops"
chapter: "llm-training-finetuning-ops"
lesson: "hands-on-distributed-training"
sourceId: 3160
sourceUrl: "https://apxml.com/zh/courses/mlops-for-large-models-llmops/chapter-3-llm-training-finetuning-ops/hands-on-distributed-training"
title: "实践操作：分布式训练配置"
description: "使用常用框架配置并运行一个基本分布式训练任务。"
order: 8
plots: []
sourceHash: "a102f69edecd92b6256a7cf82f0275652b025da1bd816a54961c6e53fa9d3771"
sourceCorrections: []
---

配置分布式训练任务需要协调多个进程，这些进程通常运行在不同设备（GPU）上，共同完成一个模型的训练。可以搭建一个使用 PyTorch 原生 `DistributedDataParallel` (DDP) 模块的分布式训练环境，该模块实现了数据并行策略。尽管 DeepSpeed 等框架提供了更高级的功能，但通过 DDP 理解核心机制能打下一个良好的基础。

我们将模拟一个场景，在多块 GPU 上训练一个简单模型（即使您只有一块物理 GPU，PyTorch 也允许在其上模拟多个进程，但性能优势只在使用实际多设备时才会显现）。

### 前提条件

请确保已安装 PyTorch (`pip install torch torchvision torchaudio`)。对于实际的多 GPU 训练，您需要配置相应的 CUDA 环境和 NVIDIA 驱动。

### PyTorch DDP 配置的核心组成部分

1. **初始化**：每个进程都需要加入一个通信组。这通过 `torch.distributed.init_process_group()` 完成。此函数要求指定一个后端（例如用于 NVIDIA GPU 的 `nccl` 或用于 CPU/混合环境的 `gloo`），它通常依赖于启动工具设置的环境变量 (`MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`)。
2. **模型封装**：标准的 PyTorch 模型需要使用 `torch.nn.parallel.DistributedDataParallel` 进行封装。这个封装器在反向传播 (backpropagation)期间处理跨进程的梯度同步。它还确保模型被放置在每个进程的正确设备上。
3. **数据划分**：每个进程在每个 epoch 都应处理数据的独特子集。`torch.utils.data.distributed.DistributedSampler` 与 DataLoader 一同使用，以自动实现这种划分。
4. **启动工具**：使用 `torchrun` 这样的工具（推荐的工具，取代了 `torch.distributed.launch`）来生成多个工作进程并设置初始化所需的相应环境变量。

### 示例实现

我们来创建一个 Python 脚本 (`basic_ddp_train.py`)，它演示了这些组件。我们将使用一个非常简单的模型和合成数据，以便专注于分布式配置本身。

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

import os
import argparse

# 1. 定义一个简单模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(10, 5)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# 2. 分布式环境的设置函数
def setup(rank, world_size):
    """初始化分布式环境。"""
    os.environ['MASTER_ADDR'] = 'localhost' # 主节点的地址
    os.environ['MASTER_PORT'] = '12355'     # 通信端口
    # 初始化进程组
    # 'nccl' 更适合 NVIDIA GPU
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank) # 将进程绑定到特定的 GPU

# 3. 清理函数
def cleanup():
    """销毁进程组。"""
    dist.destroy_process_group()

# 4. 单个进程的主要训练函数
def train_process(rank, world_size, epochs=3):
    """每个进程执行的核心训练逻辑。"""
    print(f"正在初始化进程 {rank} 共 {world_size} 个...")
    setup(rank, world_size)

    # 创建模型并将其移动到此 rank 对应的 GPU
    model = SimpleModel().to(rank)
    # 使用 DDP 封装模型
    ddp_model = DDP(model, device_ids=[rank])

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    # 创建合成数据
    # 如果数据生成需要跨进程重现性，请使用固定种子
    # torch.manual_seed(42) # 可选：用于一致的随机数据生成
    inputs = torch.randn(128, 10) # 128 个样本，10 个特征
    labels = torch.randint(0, 5, (128,)) # 128 个标签，用于 5 个类别

    dataset = TensorDataset(inputs, labels)
    # 使用 DistributedSampler 划分数据
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    # 设置 shuffle=False 因为 DistributedSampler 处理混洗
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler, shuffle=False)

    print(f"Rank {rank} 开始训练...")
    for epoch in range(epochs):
        sampler.set_epoch(epoch) # 确保每个 epoch 正确混洗
        epoch_loss = 0.0
        num_batches = 0
        for batch_inputs, batch_labels in dataloader:
            # 将数据移动到进程分配的 GPU
            batch_inputs = batch_inputs.to(rank)
            batch_labels = batch_labels.to(rank)

            # 前向传播
            outputs = ddp_model(batch_inputs)
            loss = criterion(outputs, batch_labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward() # DDP 在此处处理梯度平均
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        # 只从 rank 0 打印损失，避免输出混乱
        if rank == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    print(f"Rank {rank} 完成训练。")
    cleanup()

# 5. 启动器代码 (使用 torchrun 约定)
if __name__ == "__main__":
    # torchrun 自动设置 RANK, LOCAL_RANK, WORLD_SIZE
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"]) # 所有节点上的进程总数

    print(f"在 rank {local_rank} (总大小 {world_size}) 上开始 DDP 示例。")
    train_process(rank=local_rank, world_size=world_size, epochs=5)
    print("DDP 示例完成。")
```

### 运行示例

要在一台机器上（即使只有一块 GPU，仅用于演示目的）模拟 2 个 GPU 进程来运行此脚本，请将其保存为 `basic_ddp_train.py`，然后从终端使用 `torchrun` 执行：

```bash
# 在当前机器上使用 2 个进程运行（模拟 2 个 GPU）
torchrun --standalone --nnodes=1 --nproc_per_node=2 basic_ddp_train.py
```

- `--standalone`：表示在没有 Slurm 等外部集群管理器的情况下运行。
- `--nnodes=1`：指定在单台机器（节点）上运行。
- `--nproc_per_node=2`：设置在此节点上启动的进程数量（以及隐式使用的 GPU 数量，每个进程一个）。

您应该会看到来自两个进程（Rank 0 和 Rank 1）的初始化输出，随后是仅由 Rank 0 打印的 epoch 损失，最后是来自两者的清理消息。

### 解释说明

- **`setup(rank, world_size)`**：初始化进程之间的连接。`MASTER_ADDR` 和 `MASTER_PORT` 告知进程协调点（进程 Rank 0）的位置。`init_process_group` 建立通信后端（`nccl` 对 NVIDIA GPU 有很高的优化）。`torch.cuda.set_device(rank)` 确保每个进程控制一个特定的 GPU，避免冲突。
- **`SimpleModel`**：一个标准的 PyTorch 模型。DDP 本身无需在此处进行更改。
- **`DDP(model, device_ids=[rank])`**：这是一个重要的封装器。它接收本地模型副本和模型应运行的 GPU ID。DDP 会自动添加钩子，在 `loss.backward()` 期间同步组中所有进程的梯度。它还确保模型参数 (parameter)保持同步。
- **`DistributedSampler`**：对于数据并行很重要。它确保每个进程在每个 epoch 接收数据集的一个不同且不重叠的部分。如果没有它，所有进程将在相同数据上进行训练，这会失去意义。`sampler.set_epoch(epoch)` 是必要的，以确保每个 epoch 的混洗正确进行。
- **`DataLoader`**：采样器被传递给 DataLoader。请注意，DataLoader 中的 `shuffle=True` 应该设为 `False`，因为 `DistributedSampler` 以分布式方式处理混洗逻辑。
- **训练循环**：核心循环与单 GPU 训练非常相似。主要区别：
  - 数据被移动到特定的 `rank` 设备 (`batch_inputs.to(rank)`)。
  - `loss.backward()` 通过 DDP 封装器隐式触发梯度同步。
  - 日志记录/保存：通常，您只从一个进程（通常是 `rank == 0`）执行打印日志或保存检查点等操作，以避免冗余和潜在的竞争条件。
- **`torchrun`**：这个工具负责启动 `nproc_per_node` 个脚本副本，并注入所需的环境变量 (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`)。`LOCAL_RANK` 指的是*当前节点内的*进程排序，而 `RANK` 则是所有节点上的全局排序。对于我们的单节点示例 (`--nnodes=1`)，`RANK` 和 `LOCAL_RANK` 是相同的。

### 与高级概念的关联

本示例使用了基本的数据并行。正如本章前面所讨论的：

- **模型大小限制**：DDP 会在每块 GPU 上复制*整个*模型。这在模型能够适应单块 GPU 内存时效果良好，但如果模型过大则会受限。
- **DeepSpeed 等框架**：DeepSpeed 在这些理念基础上构建，集成了 ZeRO（零冗余优化器）等技术，将优化器状态、梯度乃至模型参数 (parameter)分布到多块 GPU 上（这是一种更高级的数据并行形式，与模型并行元素相结合），显著减少了每块 GPU 的内存占用。它还集成了对张量并行和流水线并行的支持，适用于即使是 ZeRO 也无法处理的过大模型。
- **协调调度**：在多节点场景中，`torchrun` 将与集群管理器（如 Slurm、Kubernetes）一同使用，以便在不同机器上启动进程，协调 `MASTER_ADDR` 和网络配置，这与基础设施部分内容相关联。

本次实践操作提供了分布式训练的基本代码结构。构建复杂的 LLM 训练流水线涉及扩展这种模式，集成专用框架，稳妥地管理检查点（`torch.distributed.save_checkpoint` / 框架专用工具），以及处理这些长时间运行任务中可能出现的硬件故障。

## 参考资料

- [Distributed communication package - torch.distributed](https://pytorch.org/docs/stable/distributed.html) — PyTorch Core Team (2024)
  PyTorch分布式包的全面概述，包括`init_process_group`和通信后端。
- [\`torch.nn.parallel.DistributedDataParallel\`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) — PyTorch Core Team (2024)
  Publisher: PyTorch Foundation
  DDP模块的官方文档，解释了其用法和数据并行的内部机制。
- [Distributed Data Parallel Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) — Shen Li (2024)
  Publisher: PyTorch Foundation
  一个官方PyTorch教程，提供实现DDP的分步指南，涵盖了相关概念和实践方面。
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://doi.org/10.1109/SC41405.2020.00024) — Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He (2020)
  Journal: SC20: International Conference for High Performance Computing, Networking, Storage and Analysis; Publisher: IEEE/ACM; Pages: 1-16; DOI: [10.1109/SC41405.2020.00024](https://doi.org/10.1109/SC41405.2020.00024)
  关于ZeRO优化器的基础论文，它是DeepSpeed的核心组件，解决了基本数据并行之外的大模型训练内存限制问题。
