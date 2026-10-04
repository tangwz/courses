---
course: "advanced-ai-infrastructure-design-optimization"
chapter: "engineering-distributed-model-training"
lesson: "practical-distributed-training-pytorch-fsdp"
sourceId: 6986
sourceUrl: "https://apxml.com/zh/courses/advanced-ai-infrastructure-design-optimization/chapter-2-engineering-distributed-model-training/practical-distributed-training-pytorch-fsdp"
title: "动手实践：使用 PyTorch FSDP 进行分布式训练"
description: "使用 PyTorch 的 Fully Sharded Data Parallel (FSDP) 在多 GPU 环境中运行 Transformer 模型的分布式训练任务。"
order: 6
plots: []
sourceHash: "c25bd4e72060da688864bfa13e64c29d3ea3c2c230f32bc150376ee0bc4ff9d6"
sourceCorrections: []
---

理论为我们指引方向，而实践则打造引擎。为实现先进的分布式模型训练，PyTorch 提供了 `Fully Sharded Data Parallel` (FSDP) 作为其原生解决方案。FSDP 旨在处理数据并行和模型并行带来的挑战。它提供了一个功能强大的、高度集成的 API，用于应用先进的内存节省技术，其中包括与 DeepSpeed ZeRO 相似的技术。

本次实践将引导你配置并在多 GPU 系统上运行一个 Transformer 模型的分布式训练任务。你将不仅仅是执行一个脚本；你将学习设置分布式环境的机制、正确地使用 FSDP 包装模型、管理分片检查点以及启动任务。这种动手经验对于将分布式训练原理转化为可用于生产的实现非常有益。

### 前提条件和环境配置

在编写任何代码之前，我们必须配置好环境。本实验需要至少两个 GPU 的系统，尽管代码在单个 GPU 上也能正常运行以进行语法检查。你需要 PyTorch 和 Hugging Face 的 `transformers` 库来获取预训练 (pre-training)模型和分词 (tokenization)器 (tokenizer)。

使用 pip 安装所需的库：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate
```

我们将使用 `torchrun` 启动训练脚本，它是 PyTorch 用于启动分布式任务的标准工具。`torchrun` 会自动管理每个进程所需的环境变量：

- `WORLD_SIZE`: 参与任务的进程（GPU）总数。
- `RANK`: 当前进程的全局唯一 ID，范围从 0 到 `WORLD_SIZE - 1`。
- `LOCAL_RANK`: 当前进程在给定机器上的本地唯一 ID。

理解这些变量有助于执行诸如仅从一个进程（通常是 `rank 0`）打印日志或保存检查点等任务。

### FSDP 分片机制

FSDP 通过在数据并行组中的所有 GPU 上分片模型的参数 (parameter)、对应的梯度和优化器状态来实现内存效率。在运行时，每个 GPU 只保存总模型状态的一部分。

当前向传播中需要某个层进行计算时，每个 GPU 会从所有其他 GPU 收集必要的参数分片，以重构完整的层。计算完成后，完整的层会被丢弃，释放内存。在反向传播 (backpropagation)过程中，会发生一个类似的反向过程。

> FSDP `all_gather` 操作在前向传播过程中的示意图。每个 GPU 只保留其模型状态的分片，并在需要时从其他对等 GPU 收集剩余的分片，以即时重构完整层进行计算。

FSDP 提供多种 `ShardingStrategy` 选项来控制这种行为，在内存节省和通信开销之间做出权衡。两种主要策略是：

- `FULL_SHARD`: 这种策略对模型参数、梯度和优化器状态进行分片，提供最大的内存节省。它类似于 ZeRO-3。
- `SHARD_GRAD_OP`: 这种策略只对梯度和优化器状态进行分片，在每个 GPU 上保留一份完整的模型参数。它节省的内存较少，但减少了通信。它类似于 ZeRO-2。

在本次实践中，我们将使用 `FULL_SHARD` 以实现最大的内存效率。

### 步骤 1：初始化分布式进程组

在任何 PyTorch 分布式脚本中，第一步是初始化进程组。此函数建立通信后端（例如 NVIDIA GPU 的 `nccl`），并允许进程之间相互发现。

创建一个名为 `train_fsdp.py` 的文件并添加设置函数。

```python
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import functools

def setup():
    """初始化分布式环境。"""
    dist.init_process_group("nccl")
    # 为当前进程设置设备。
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

def cleanup():
    """清理分布式环境。"""
    dist.destroy_process_group()
```

### 步骤 2：准备模型和数据

我们将使用 `transformers` 库中的 `GPT2` 模型及其分词 (tokenization)器 (tokenizer)。在真实的训练运行中，你会使用一个大型数据集；在这里，我们将创建一个简单的虚拟数据集用于演示。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_model_and_tokenizer():
    """加载预训练模型和分词器。"""
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 如果不存在，则添加填充令牌
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def get_dummy_dataloader(tokenizer, batch_size=4):
    """创建一个用于演示的虚拟数据加载器。"""
    dummy_data = ["This is a test sentence for FSDP." for _ in range(100)]
    encoded_data = tokenizer(dummy_data, return_tensors="pt", padding=True, truncation=True, max_length=128)
    dataset = torch.utils.data.TensorDataset(encoded_data.input_ids, encoded_data.attention_mask)
    # 采样器对于在 GPU 之间分发数据非常有用
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return dataloader
```

请注意 `DistributedSampler` 的使用。这是一个必不可少的组件，它确保每个 GPU 在每个 epoch 中接收到数据的一个唯一、不重叠的片段。

### 步骤 3：使用 FSDP 包装模型

这是 FSDP 的配置之处。相比于将整个模型包装成一个大的 FSDP 单元，包装单个层或块效率更高。这允许 FSDP 在层的参数 (parameter)用于前向和反向传播 (backpropagation)后立即释放内存。

`auto_wrap_policy` 使这变得简单。我们将使用 `size_based_auto_wrap_policy`，它会自动包装任何超出特定参数数量的子模块。

```python
# (这段代码放在你的主训练函数中)

# 定义自动包装策略
# 我们包装参数数量超过 100 万的 Transformer 块。
# 根据你的模型架构调整此值。
auto_wrap_policy = functools.partial(
    size_based_auto_wrap_policy,
    min_num_params=1_000_000, 
)

# 获取本地进程号
local_rank = int(os.environ["LOCAL_RANK"])
model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    sharding_strategy=torch.distributed.fsdp.ShardingStrategy.FULL_SHARD,
    device_id=torch.cuda.current_device(),
    # cpu_offload=CPUOffload(offload_params=True) # 可选：卸载到 CPU
)
```

`device_id` 参数是必不可少的；它告诉 FSDP 将模型分片移动到哪个 GPU。被注释掉的 `cpu_offload` 参数展示了如果你内存极度受限，如何将参数卸载到 CPU RAM，但这会以 PCIe 数据传输导致的性能下降为代价。

### 步骤 4：训练循环

训练循环本身与标准的非分布式 PyTorch 循环几乎相同。优化器必须在模型被 FSDP 包装 *之后* 定义，因为 FSDP 会用它自己的 `FlatParameter` 对象替换模型的参数 (parameter)。

```python
# (在你的主训练函数中，模型包装之后)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

model.train()
for epoch in range(1, 3): # 训练 2 个 epoch
    dataloader.sampler.set_epoch(epoch) # 确保每个 epoch 的洗牌不同
    for batch_idx, (input_ids, attention_mask) in enumerate(dataloader):
        input_ids = input_ids.to(local_rank)
        attention_mask = attention_mask.to(local_rank)

        optimizer.zero_grad()

        # 前向传播自动处理 all-gather
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss

        # 反向传播处理 reduce-scatter
        loss.backward()

        optimizer.step()

        if batch_idx % 10 == 0 and dist.get_rank() == 0:
            print(f"Epoch: {epoch}/{2} | Batch: {batch_idx} | Loss: {loss.item():.4f}")
```

我们使用 `dist.get_rank() == 0` 来确保打印语句只由一个进程执行，从而避免大量重复的日志消息。

### 步骤 5：处理检查点

使用 FSDP 保存和加载需要一种特定的方法，因为模型状态分布在所有 GPU 上。你必须决定是保存分片检查点（速度更快，但加载时需要相同的 `WORLD_SIZE`）还是完整的、合并的检查点（更具可移植性）。

要保存完整的检查点，我们需要在保存之前将整个模型状态收集到一个进程（通常是进程 0）上。

```python
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

# --- 保存完整状态字典 ---
if dist.get_rank() == 0:
    print("正在保存合并的模型检查点...")

# 使用上下文管理器获取完整状态字典
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
    full_state_dict = model.state_dict()

    # 只有进程 0 保存文件
    if dist.get_rank() == 0:
        torch.save(full_state_dict, "full_model_checkpoint.pt")

# --- 加载完整状态字典 ---
# 要加载，你首先使用 FSDP 包装模型，然后加载状态。
# model = FSDP(...) # 像之前一样初始化 FSDP 模型

with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
    # 首先在 CPU 上加载检查点，以避免单个 GPU 内存不足
    checkpoint = torch.load("full_model_checkpoint.pt", map_location="cpu")
    model.load_state_dict(checkpoint)
```

这种模式确保状态在保存前正确地从所有分片中收集，并在加载时正确地散布回去。

### 启动任务

现在，将所有部分组装到 `train_fsdp.py` 脚本中。主执行块应如下所示：

```python
def main():
    setup()

    rank = int(os.environ["RANK"])
    model, tokenizer = get_model_and_tokenizer()
    dataloader = get_dummy_dataloader(tokenizer)

    # ... [FSDP 包装逻辑在此] ...
    # ... [优化器定义在此] ...
    # ... [训练循环在此] ...
    # ... [检查点保存逻辑在此] ...

    cleanup()

if __name__ == '__main__':
    main()
```

从你的终端启动训练任务。此命令指示 `torchrun` 在当前机器上启动 2 个进程，每个进程运行 `train_fsdp.py` 脚本。

```bash
torchrun --nproc_per_node=2 train_fsdp.py
```

如果你有 4 个 GPU，则使用 `--nproc_per_node=4`。脚本将执行，你会看到进程 0 打印的损失值。你可以使用 `watch -n 1 nvidia-smi` 监控 GPU 内存使用情况。你会看到每个 GPU 上使用的内存明显少于容纳整个 GPT-2 模型所需的内存。

通过完成本次实践，你不仅执行了一个分布式训练任务，还接触了 FSDP 的核心机制：初始化、自动包装策略、`DistributedSampler` 和状态字典管理。这些技能可以直接应用于生产环境中大规模模型的训练。

## 参考资料

- [Fully Sharded Data Parallel (FSDP)](https://pytorch.org/docs/stable/fsdp.html) — PyTorch Team (2024)
  PyTorch FSDP的官方文档，详细说明了其API、用法以及分布式训练的配置选项。
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://doi.org/10.1145/3479069.3487532) — Samyam Rajbhandari, Cong Guo, Jeff Rasley, Shaden Smith, Yuxiong He (2021)
  Journal: SC21: International Conference for High Performance Computing, Networking, Storage and Analysis; Publisher: ACM; Pages: 70–82; DOI: [10.1145/3479069.3487532](https://doi.org/10.1145/3479069.3487532)
  这篇学术论文介绍了ZeRO优化阶段（ZeRO-2和ZeRO-3），以实现大模型训练的内存效率，FSDP的分片策略与这些优化阶段类似。
- [Distributed communication package - torch.distributed](https://pytorch.org/docs/stable/distributed.html) — PyTorch Team (2024)
  Publisher: PyTorch
  PyTorch官方分布式通信原语文档，包括`init_process_group`、`torchrun`和环境变量。
- [Distributed training with 🤗 Accelerate](https://huggingface.co/docs/accelerate/) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face关于使用Accelerate简化transformer模型分布式训练（包括FSDP集成）的实用指南。
