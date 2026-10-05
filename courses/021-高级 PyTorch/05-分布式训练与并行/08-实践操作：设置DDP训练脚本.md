# 实践操作：设置DDP训练脚本

来源：[原文](https://apxml.com/zh/courses/advanced-pytorch/chapter-5-distributed-training-parallelism/practice-ddp-script)

[返回章节目录](README.md) · [返回课程目录](../README.md)

演示了如何将一个标准的单GPU PyTorch训练脚本转换为使用分布式数据并行（DDP）进行单机多GPU训练。这主要说明启用DDP所需的具体修改，而不是构建一个完整的先进模型训练器。我们假设您已有一个可用的单GPU脚本。

DDP 的主要思想很简单：在每个可用GPU上复制模型，为每个副本提供输入数据批次的不同切片，在每个GPU上独立计算梯度，然后在更新模型参数 (parameter)之前将这些梯度在所有GPU上平均。这保证了所有模型副本保持同步。

### 前提条件

- 一个适用于单GPU的PyTorch训练脚本。
- 一台PyTorch可检测到的、具有多个CUDA功能的GPU的机器。
- 已安装具有分布式支持的PyTorch（通常在CUDA构建中默认包含）。

### DDP转换步骤

我们将转换过程分解为易于管理的小步骤：

1. **初始化进程组：** 建立进程间的通信。
2. **配置设备放置：** 将每个进程分配给特定的GPU。
3. **数据分片：** 使用 `DistributedSampler` 为每个进程分配数据集的独特部分。
4. **包装模型：** 使用 `DistributedDataParallel` 封装模型。
5. **调整训练循环：** 处理采样器 epoch 设置和可能的指标聚合。
6. **管理检查点：** 确保只有一个进程保存模型。
7. **清理：** 妥善终止进程组。
8. **启动脚本：** 使用 `torchrun` 启动分布式进程。

让我们详细说明每个步骤。

### 1. 初始化进程组

每个DDP脚本都需要初始化分布式环境。这使得进程能够相互发现并协调。我们使用 `torch.distributed.init_process_group`。

```python
import torch
import torch.distributed as dist
import os

def setup(rank, world_size):
    """初始化分布式环境。"""
    os.environ['MASTER_ADDR'] = 'localhost' # 主节点的地址
    os.environ['MASTER_PORT'] = '12355'     # 一个可用端口

    # 初始化进程组
    # 需要rank和world_size。对于NVIDIA GPU，推荐使用后端'nccl'。
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"已为 {world_size} 个进程中的 rank {rank} 初始化进程组。")

def cleanup():
    """销毁进程组。"""
    dist.destroy_process_group()
    print("已销毁进程组。")

# --- 在您的主执行流程中 ---
# world_size = torch.cuda.device_count() # 假设使用所有可用GPU
# rank = ... # 这将由启动器（torchrun）提供
# setup(rank, world_size)
# ... 训练代码 ...
# cleanup()
```

- `rank`：当前进程的唯一标识符（从0到 `world_size - 1`）。
- `world_size`：参与分布式任务的进程总数。
- `backend`：要使用的通信库。`nccl` 为 NVIDIA GPU 进行了高度优化。`gloo` 是 CPU 或没有 `nccl` 环境的替代方案。
- `MASTER_ADDR` 和 `MASTER_PORT`：这些变量告诉进程在哪里找到主进程（rank 0）以进行初始协调。`localhost` 对于单节点训练足够。

**注意：** 使用 `torchrun` 时，`rank` 和 `world_size`（以及 `LOCAL_RANK` 等其他变量）通常会自动管理并传递给您的脚本。如果需要在其他地方使用 `rank`，您通常可以通过参数 (parameter)解析器或直接从环境变量中获取它。

### 2. 配置设备放置

每个进程都需要在其分配的GPU上运行。常见的做法是使用 `local_rank`。`local_rank` 是*当前节点内*的GPU索引。对于单节点训练，`local_rank` 通常与全局 `rank` 相同，但依靠 `local_rank` 是为了可移植性而采取的良好做法。`torchrun` 会设置 `LOCAL_RANK` 环境变量。

```python
# 在您的训练脚本或函数的开头：
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

# --- 示例用法 ---
# model = YourModel().to(device) # 将模型移动到分配的GPU
# data = data.to(device)         # 将数据移动到分配的GPU
# labels = labels.to(device)
```

通过设置 `torch.cuda.set_device(local_rank)`，该进程后续的CUDA操作和张量分配将默认使用正确的GPU。仍然需要使用 `.to(device)` 明确地移动模型和数据。

### 3. 使用 DistributedSampler 进行数据分片

为了确保每个GPU处理数据的唯一子集，请将标准的 `DataLoader` 洗牌功能替换为 `torch.utils.data.distributed.DistributedSampler`。

```python
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# 假设'train_dataset'是您的torch.utils.data.Dataset实例
# rank和world_size在init_process_group后获得

train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

# 重要：在DataLoader中设置shuffle=False，因为DistributedSampler处理洗牌。
train_loader = DataLoader(
    train_dataset,
    batch_size=per_device_batch_size, # 每个GPU的批处理大小
    sampler=train_sampler,
    num_workers=4, # 根据需要调整
    pin_memory=True # 推荐用于性能提升
)

# --- 在训练循环内部 ---
for epoch in range(num_epochs):
    # 为采样器设置 epoch，以确保在不同 epoch 之间洗牌变化
    train_sampler.set_epoch(epoch) 

    for batch_idx, (data, target) in enumerate(train_loader):
        # ... 训练步骤的其余部分...
```

- `DistributedSampler` 自动将数据集索引分配给各个进程（`num_replicas=world_size`）。
- 在采样器中设置 `shuffle=True` 可以确保数据在分区*之前*进行洗牌。
- `DataLoader` 的 `batch_size` 现在指*每个进程/GPU*的批处理大小。所有GPU上的有效总批处理大小是 `per_device_batch_size * world_size`。
- 在每个 epoch 开始时调用 `sampler.set_epoch(epoch)` 对于多个 epoch 中的正确洗牌行为很重要。

### 4. 使用 DistributedDataParallel 包装模型

在创建模型并将其移动到正确设备后，使用 `torch.nn.parallel.DistributedDataParallel` 进行包装。

```python
from torch.nn.parallel import DistributedDataParallel as DDP

# 假设'model'是您的nn.Module实例，已移动到'device'
# model = YourModel().to(device)

# 包装模型
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# 现在像往常一样使用'model'进行前向传播。
# DDP在backward()期间自动处理梯度同步。
```

- `device_ids`：指定此进程的模型副本所在的GPU。通常是 `[local_rank]`。
- `output_device`：指定前向传播的输出应该在哪里收集。通常也是 `local_rank`。DDP在内部处理此问题。

DDP 通过向模型的 `backward()` 传递添加钩子来工作。当调用 `loss.backward()` 时，梯度在每个GPU上局部计算，然后DDP触发一个all-reduce操作来汇总/平均所有进程的梯度，之后再更新模型参数 (parameter)。这确保了所有模型副本保持同步。

### 5. 调整训练循环

核心训练逻辑（前向传播、损失计算、`optimizer.step()`）基本保持不变。然而，请考虑以下几点：

- **采样器 Epoch：** 请记住调用 `train_sampler.set_epoch(epoch)`。
- **指标聚合：** 如果您按批次计算损失或准确率等指标，这些值对于每个进程是局部的。要获得全局平均值，您需要聚合它们。一种常见的方法是使用 `dist.all_reduce`。

```python
# --- 在计算损失后，训练循环内部 ---
loss = criterion(outputs, target)

# 为聚合创建一个副本，防止修改用于backward()的损失张量
loss_tensor = torch.tensor([loss.item()], device=device) 

# 汇总所有进程的损失值
dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM) 

# 平均损失（除以总数）
avg_loss = loss_tensor.item() / world_size

if rank == 0: # 仅在主进程上记录
    print(f"Epoch {epoch}, Batch {batch_idx}, Avg Loss: {avg_loss:.4f}")

# 注意：反向传播使用原始的'loss'张量
loss.backward() 
optimizer.step()
```

这个例子展示了如何规约损失。您可以对准确率或其他指标执行类似操作。`torchmetrics` 等更复杂的库通常内置对分布式环境的支持。

### 6. 管理检查点

保存检查点（模型状态、优化器状态）通常应仅由一个进程（通常是rank 0）执行，以防止多个进程同时写入同一文件。保存DDP包装的模型时，通过 `.module` 访问底层模型。

```python
# --- 在您的保存逻辑内部 ---
if rank == 0:
    checkpoint = {
        'epoch': epoch,
        # 通过.module访问原始模型的state dict
        'model_state_dict': model.module.state_dict(), 
        'optimizer_state_dict': optimizer.state_dict(),
        # 添加任何其他必要信息
    }
    torch.save(checkpoint, f"model_epoch_{epoch}.pt")
    print(f"检查点已在 epoch {epoch} 由 rank {rank} 保存。")

# --- 加载逻辑 ---
# 确保所有进程在包装模型之前加载相同的检查点
map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank} # 将保存的权重映射到当前设备
checkpoint = torch.load(checkpoint_path, map_location=map_location);

# 在用DDP包装之前加载state dict
model.load_state_dict(checkpoint['model_state_dict']) 
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# ... 加载其他信息...

# 加载后，将模型移动到设备并用DDP包装
model.to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# 确保所有进程都已加载后再继续
dist.barrier() 
```

在加载后使用 `dist.barrier()` 可确保所有进程成功加载检查点数据之前，没有进程会继续进行，从而防止潜在的竞态条件。

### 7. 清理

在脚本的最后或 `finally` 块中调用 `dist.destroy_process_group()` 以释放资源。

```python
# --- 在主执行的最后 ---
# ... 训练已完成...
cleanup() 
```

### 8. 使用 torchrun 启动脚本

启动 PyTorch DDP 脚本的标准方式是使用 `torchrun` 工具（以前称为 `torch.distributed.launch`）。它负责设置环境变量（`RANK`、`LOCAL_RANK`、`WORLD_SIZE`、`MASTER_ADDR`、`MASTER_PORT`）并生成进程。

假设您的脚本名为 `train_ddp.py`，并且您想在当前机器上使用2个GPU：

```bash
torchrun --standalone --nproc_per_node=2 train_ddp.py --arg1 value1 --arg2 value2 
```

- `--standalone`：表示单节点训练。
- `--nproc_per_node`：此节点上要使用的进程数（通常也是GPU数）。将其设置为您希望使用的GPU数量。
- `train_ddp.py`：您的脚本名称。
- `--arg1 value1 ...`：您的脚本期望的任何命令行参数 (parameter)。

`torchrun` 将生成您的脚本的 `nproc_per_node` 个副本，每个副本都设置了正确的环境变量，从而触发每个进程内的 `setup` 函数和后续的DDP逻辑。

### 示例脚本结构

以下是一个结合了这些元素的骨架：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import argparse

# --- 模拟模型和数据集 ---
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    def forward(self, x):
        return self.linear(x)

class ToyDataset(Dataset):
    def __init__(self, size=1000):
        self.size = size
        self.features = torch.randn(size, 10)
        self.labels = torch.randn(size, 1)
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
# --- 模拟结束 ---

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' # 确保此端口空闲
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank) # 为了简单起见，这里直接使用全局rank作为本地rank

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, args):
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")

    # 1. 数据集和采样器
    dataset = ToyDataset(size=args.dataset_size)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    # 有效批处理大小 = args.batch_size * world_size
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)

    # 2. 模型
    model = ToyModel().to(device)
    model = DDP(model, device_ids=[rank])

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    print(f"Rank {rank} 开始训练...")

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch) # 对于洗牌很重要
        epoch_loss = 0.0
        num_batches = 0
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward() # DDP在此处理梯度同步
            optimizer.step()

            # 聚合损失用于日志记录（可选但推荐）
            loss_tensor = torch.tensor([loss.item()], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            epoch_loss += loss_tensor.item()
            num_batches += 1

        avg_epoch_loss = epoch_loss / (num_batches * world_size) # 所有批次和进程的平均值

        if rank == 0: # 仅从 rank 0 记录
            print(f"Epoch {epoch+1}/{args.epochs}, Avg Loss: {avg_epoch_loss:.4f}")

        # --- 检查点（示例）---
        if rank == 0 and (epoch + 1) % args.save_interval == 0:
             checkpoint_path = f"model_epoch_{epoch+1}.pt"
             torch.save(model.module.state_dict(), checkpoint_path)
             print(f"Rank {rank} 已将检查点保存到 {checkpoint_path}")

        dist.barrier() # 确保所有进程完成 epoch 后再继续/保存

    cleanup()
    if rank == 0:
        print("训练完成。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5, help='训练 epoch 数量')
    parser.add_argument('--batch_size', type=int, default=64, help='每个GPU的批处理大小')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--dataset_size', type=int, default=2048, help='数据集总大小')
    parser.add_argument('--save_interval', type=int, default=2, help='每 N 个 epoch 保存一次检查点')
    # 注意：rank、world_size、local_rank 通常由启动器（torchrun）设置
    # 我们在train函数或setup内部从环境中获取它们。
    args = parser.parse_args()

    # torchrun 设置这些环境变量
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"]) # 通常用于设备分配

    # 为当前进程启动训练函数
    train(rank, world_size, args) # 显式传递 rank 和 world_size
```

使用2个GPU运行此脚本：
`torchrun --standalone --nproc_per_node=2 train_ddp.py --epochs 10 --batch_size 32`

"本实践练习演示了将单进程脚本调整为使用 `DistributedDataParallel` 进行多GPU数据并行训练所需的基本更改。尽管应用程序通常涉及更复杂的指标处理、日志记录和检查点策略，但这些核心步骤构成了 PyTorch 训练任务扩展的根基。请记住监控 GPU 利用率（`nvidia-smi`）和训练时间，以查看分布式训练的优势。"

## 参考资料

- [Getting Started with DistributedDataParallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) — Shen Li, Joe Zhu, Chirag Pandya (2024)
  PyTorch 官方指南，介绍如何将单 GPU 脚本转换为使用 DDP，涵盖环境设置、数据分片、模型封装和进程启动。
- [Distributed Communication Package - torch.distributed](https://pytorch.org/docs/stable/distributed.html) — PyTorch Developers (2024)
  Publisher: PyTorch
  PyTorch 分布式通信包的完整 API 参考，详细介绍了 init_process_group、all_reduce 和 barrier 等基本操作。
- [torch.nn.parallel.DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) — PyTorch Developers (2024)
  DistributedDataParallel 模块的详细 API 文档，解释其参数以及如何处理多 GPU 训练中的梯度同步。
- [torchrun (Elastic Launch)](https://pytorch.org/docs/stable/elastic/run.html) — PyTorch Developers (2021)
  Publisher: PyTorch Foundation
  torchrun 的官方文档，这是启动 PyTorch 分布式训练任务的推荐工具，涵盖其用法和配置。

---

[上一节](07-%E8%AE%BE%E7%BD%AE%E5%88%86%E5%B8%83%E5%BC%8F%E7%8E%AF%E5%A2%83.md) · [下一节](../06-%E8%87%AA%E5%AE%9A%E4%B9%89%E6%89%A9%E5%B1%95%E4%B8%8E%E4%BA%92%E6%93%8D%E4%BD%9C%E6%80%A7/01-%E6%9E%84%E5%BB%BA%E5%AE%9A%E5%88%B6C%2B%2B%E6%89%A9%E5%B1%95.md)
