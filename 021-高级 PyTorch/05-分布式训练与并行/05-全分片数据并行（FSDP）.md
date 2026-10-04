# 全分片数据并行（FSDP）

来源：[原文](https://apxml.com/zh/courses/advanced-pytorch/chapter-5-distributed-training-parallelism/fully-sharded-data-parallelism)

[返回章节目录](README.md) · [返回课程目录](../README.md)

`DistributedDataParallel` (DDP) 通过复制模型和平均梯度有效地在多个GPU上扩展训练。然而，它根本上要求每个GPU都持有完整的模型、其梯度和优化器状态。这在处理包含数十亿参数 (parameter)的模型时成为一个限制因素，因为这些模型可能轻易超出甚至高端加速器的内存容量。

全分片数据并行（FSDP）通过扩展数据并行理念，同时大幅减少每个GPU的内存占用，提供了一种解决办法。FSDP不是复制整个模型，而是将模型的参数、梯度和优化器状态分片或分区到数据并行工作器（GPU）上。

### FSDP运行机制

FSDP的核心是确保数据并行组中的每个GPU在任何给定时间点只持有模型参数 (parameter)、梯度和优化器状态的一部分（一个“分片”）。完整张量仅在计算需要时才临时重建。

以下是训练过程的详细步骤：

1. **初始化：** 模型使用 `FullyShardedDataParallel` 模块进行封装。在初始化期间，参数、梯度和优化器状态被划分到参与进程组的GPU上。每个GPU负责管理其分配到的分片。
2. **前向传播：**

   - 当FSDP封装的层需要执行计算时，每个GPU使用 `all_gather` 集合通信操作从组中所有其他GPU收集该特定层所需的完整参数。
   - 该层的计算在当前本地的完整参数上进行。
   - 该层的前向计算完成后，除了GPU自己的分片外，完整参数会立即被丢弃，从而释放内存。这个过程逐层（或逐块，取决于封装策略）重复进行。
3. **反向传播 (backpropagation)：**

   - 当给定层的参数在本地计算出梯度时，它们不会像DDP那样立即在所有GPU上求平均。
   - 相反，会执行 `reduce_scatter` 操作。此操作计算所有GPU上梯度的平均值，并同时对平均结果进行分片，仅向每个GPU发送与其管理参数分片对应的梯度部分（分片）。
   - 这个分片后的梯度被存储起来，再次保持了较低的内存使用。`reduce_scatter` 后，完整梯度被丢弃。
4. **优化器步骤：**

   - 每个GPU的优化器只需要更新其负责的参数分片。
   - 由于优化器状态（如Adam中的动量缓冲区）也与参数一同分片，因此优化器步骤可以在每个GPU上仅使用其梯度和优化器状态的分片进行本地处理。

这种方法大幅减少了每个GPU所需的峰值内存，因为只有当前执行层的参数以及完整模型、梯度和优化器状态的分片才会被持久存储。

> 分布式数据并行（DDP）和全分片数据并行（FSDP）每GPU内存分配比较。DDP复制所有组件，而FSDP将其分片。

### 在PyTorch中实现FSDP

PyTorch通过 `torch.distributed.fsdp.FullyShardedDataParallel` 类提供了对FSDP的原生支持。集成它通常涉及封装您的模型定义。

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import functools

# 假设分布式环境已初始化（rank, world_size 等）
# dist.init_process_group(backend="nccl")
# torch.cuda.set_device(local_rank) # local_rank 通常获取

class LargeTransformerBlock(nn.Module):
    # 子模块定义示例
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads=8) # Simplified
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim)
        )

    def forward(self, x):
        x = self.layer_norm(x + self.attention(x, x, x)[0])
        x = x + self.ffn(x)
        return x

class BigModel(nn.Module):
    def __init__(self, num_layers, dim, ff_dim, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            [LargeTransformerBlock(dim, ff_dim) for _ in range(num_layers)]
        )
        self.output_head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output_head(x)
        return x

# --- FSDP 设置 ---
model = BigModel(num_layers=48, dim=2048, ff_dim=8192, vocab_size=50000).to(torch.cuda.current_device())

# 定义一个自动封装策略（可选但推荐用于大型模型）
# 这会根据大小封装子模块（例如 LargeTransformerBlock）
auto_wrap_policy = functools.partial(
    size_based_auto_wrap_policy, min_num_params=1_000_000 # 示例阈值
)

# 用 FSDP 封装模型
fsdp_model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    # 其他配置选项可在此处添加
    # 例如，cpu_offload=CPUOffload(offload_params=True)
    # 例如，mixed_precision=MixedPrecision(...)
    # 例如，sharding_strategy=ShardingStrategy.SHARD_GRAD_OP
)

# --- 训练循环 ---
# 优化器必须在用 FSDP 封装模型后构建
optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-4)

# 训练步骤示例（简化）
# for batch in dataloader:
#     inputs = batch['input_ids'].to(torch.cuda.current_device())
#     labels = batch['labels'].to(torch.cuda.current_device())
#
#     optimizer.zero_grad()
#     outputs = fsdp_model(inputs)
#     loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
#     loss.backward()
#     optimizer.step()
```

实现中的要点：

1. **模型定义：** 像往常一样使用 `nn.Module` 定义您的模型。
2. **封装：** 使用 `FSDP` 封装模型实例。请注意，模型应在封装 *之前* 移至目标设备。
3. **优化器：** 在使用 `FSDP` 封装模型 *之后* 构建优化器，并将 `fsdp_model.parameters()` 传递给它。这确保优化器了解分片参数 (parameter)和状态。
4. **自动封装策略：** 对于复杂模型，定义 `auto_wrap_policy` 非常重要。此策略告知FSDP如何递归地封装主模型内的子模块。封装单个块（例如Transformer层）可以实现更细粒度的分片以及更好的通信和计算重叠。`size_based_auto_wrap_policy` 是一种常见选择，用于封装参数数量超出特定阈值的模块。

### 配置选项

FSDP提供了多种配置选项来调整其行为：

- **`sharding_strategy`**：控制参数 (parameter)、梯度和优化器状态的分片程度。

  - `ShardingStrategy.FULL_SHARD`：（默认）分片参数、梯度和优化器状态。提供最大的内存节省，但通信开销可能更高。
  - `ShardingStrategy.SHARD_GRAD_OP`：仅分片梯度和优化器状态。参数被复制（类似于ZeRO 阶段2）。内存节省少于 `FULL_SHARD`，但通信开销可能更低。
  - `ShardingStrategy.NO_SHARD`：等同于DDP（复制所有内容）。有助于调试或基准比较。
  - `ShardingStrategy.HYBRID_SHARD`：在节点内结合完全分片，跨节点复制。在多节点场景中有用。
- **`cpu_offload`**：通过 `CPUOffload(offload_params=True/False)` 配置。当参数和梯度的分片未主动用于计算时，允许将其卸载到CPU内存。这以CPU和GPU之间显著的通信开销为代价，进一步增加了可行的模型大小。当GPU内存是绝对瓶颈时使用此选项。
- **`mixed_precision`**：通过 `MixedPrecision(param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float16)` 配置。将混合精度训练直接集成到FSDP封装器中，自动处理类型转换和梯度缩放。通常建议使用FSDP内置的混合精度，而不是在外部应用 `torch.cuda.amp.autocast`。
- **`auto_wrap_policy`**：如前所述，它定义了嵌套模块如何封装。`size_based_auto_wrap_policy` 的替代方案包括基于模块类型（`transformer_auto_wrap_policy`）的封装或手动封装。
- **`backward_prefetch`**：控制反向传播 (backpropagation)的参数预取，以实现通信和计算重叠。像 `BackwardPrefetch.BACKWARD_PRE` （在当前层的反向传播期间预取下一层的参数）这样的选项可以提高性能。

### 权衡与考量

尽管FSDP能够训练显著更大的模型，但它也引入了一些权衡：

- **增加的通信量**：`all_gather`（前向）和 `reduce_scatter`（反向）操作相比DDP在反向传播 (backpropagation)中的单个 `all_reduce` 引入了更大的通信量。性能影响严重依赖于GPU/节点之间的互连速度。更快的互连（例如NVLink，InfiniBand）能更有效地减轻此开销。
- **计算/通信重叠**：FSDP旨在使通信（为下一层收集参数 (parameter)）与计算（当前层的执行）重叠。使用 `auto_wrap_policy` 的有效封装策略对最大化这种重叠很重要。
- **激活检查点**：为了进一步减少正向传播期间存储的激活所带来的内存使用，FSDP通常与激活检查点（也称为梯度检查点）一起使用。PyTorch提供了 `torch.utils.checkpoint.checkpoint`，FSDP有特定的实用工具（`fsdp_checkpointing`）可以高效地将其应用于封装模块。
- **复杂性**：设置和调优FSDP可能比基本的DDP更复杂，特别是在最佳封装策略和特定硬件设置的配置方面。

总而言之，FSDP是一种强大的技术，用于训练不适合单个GPU内存的超大型模型。通过将参数、梯度和优化器状态分片到数据并行工作器上，它显著降低了每GPU的内存需求。然而，这可能会增加通信开销，使得快速互连和仔细的配置对于获得良好的训练性能非常重要。这代表了PyTorch在大规模模型训练能力方面的一个重要进展。

## 参考资料

- [FullyShardedDataParallel (FSDP) documentation](https://pytorch.org/docs/stable/fsdp.html) — PyTorch Authors (2022)
  Publisher: PyTorch Foundation
  这是 PyTorch FSDP 模块的官方 API 文档，详细说明了其类定义、参数和方法。
- [Getting Started with Fully Sharded Data Parallel (FSDP)](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) — Wei Feng, Will Constable, Yifan Mao (2024)
  Publisher: PyTorch
  这份官方 PyTorch 教程提供了实现和配置 FSDP 的分步指南和实际示例，补充了 API 文档。

---

[上一节](04-%E6%B5%81%E6%B0%B4%E7%BA%BF%E5%B9%B6%E8%A1%8C%E5%AE%9E%E7%8E%B0.md) · [下一节](06-%E4%BD%BF%E7%94%A8%20torch.distributed%20%E9%80%9A%E4%BF%A1%E5%8E%9F%E8%AF%AD.md)
