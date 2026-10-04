---
course: "distributed-training-pytorch-fsdp"
chapter: "model-wrapping-initialization-policies"
lesson: "code-practice-advanced-wrapping"
sourceId: 7892
sourceUrl: "https://apxml.com/zh/courses/distributed-training-pytorch-fsdp/chapter-2-model-wrapping-initialization-policies/code-practice-advanced-wrapping"
title: "代码实践: 高级包装配置"
description: "侧重于动手操作，编写自定义包装策略和初始化具有数十亿参数的模型。"
order: 5
plots: ["plots/7892-0.json"]
sourceHash: "0b13028f0a795c82f164a34a030def0d012bd0ffc58bf9ffa67dd490f3ad976f"
sourceCorrections: []
---

高效使用全分片数据并行 (FSDP) 与模型分区有多精细有关。尽管基于模块类的标准自动包装策略适用于统一架构，但它们在异构模型或极其深层网络中通常无法优化通信。在数十亿参数 (parameter)的规模下，分片的粒度决定了内存节省与网络开销之间的平衡。如果分片过大，all-gather 操作会导致内存使用量急剧增加；如果分片过小，内核启动和同步延迟的开销会主导训练循环。

高级配置涉及定义自定义包装策略并管理初始化生命周期，以防止主机端内存耗尽。我们将实现一种大小感知包装策略，并将其与 PyTorch 的 `meta` 设备结合，实现零内存实例化。

### 分片层级结构

FSDP 通过将包装单元内的参数 (parameter)扁平化为一个 `FlatParameter` 来运行。在前向传播期间，FSDP 收集当前单元的完整参数，执行计算，并立即释放非本地分片。这种机制依赖于递归结构，其中嵌套的 FSDP 实例管理它们各自的作用域。

次优的包装策略会创建一个扁平的层级结构，其中同时收集过多的参数。目标是创建一种平衡的树结构，使得参数的工作集（当前计算所需的峰值内存）无论模型总深度如何都能保持不变。

> 此图描绘了一个嵌套的 FSDP 结构。Transformer 块是独立包装的，这使得在任何给定时间，仅有一个块的参数在 GPU 内存中完全实体化。

### 自定义 Lambda 策略

PyTorch 提供的 `transformer_auto_wrap_policy` 是一个便捷包装器。对于专家级别的控制，您可以使用 `functools.partial` 和 lambda 函数来构建策略。这使得您可以根据参数 (parameter)数量阈值或特定模块名称进行分片，而不仅仅是类类型。

一个常见要求是，将小层（如 LayerNorm 或偏置 (bias)）排除在分片之外，以减少通信频率；或者强制对标准 Transformer 块之外的大型线性投影进行分片。

以下实现展现了一种混合策略。如果模块是特定的 Transformer 层 *或* 参数数量超过定义的阈值 $10^6$，它将包装这些模块。

```python
import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    lambda_auto_wrap_policy,
    _or_policy,
)
import functools

def custom_size_policy(module: nn.Module, recurse: bool, nonwrapped_numel: int) -> bool:
    # 如果模块本身参数超过2000万，则应用分片
    # 这会处理标准块之外的大型临时层
    return nonwrapped_numel >= 2 * 10**7

def get_hybrid_policy(transformer_layer_cls):
    """
    将主干的基于类型的包装与异构头部/嵌入的基于大小的包装相结合。
    """
    # 策略1: 包装标准Transformer块
    type_policy = functools.partial(
        transformer_auto_wrap_policy, 
        transformer_layer_cls={transformer_layer_cls}
    )
    
    # 策略2: 包装所有非Transformer块的巨型层
    size_policy = functools.partial(custom_size_policy)
    
    # 组合策略: 如果任一条件为真则包装
    return functools.partial(
        _or_policy, 
        policies=[type_policy, size_policy]
    )
```

### 在 Meta 设备上初始化

初始化一个通用 70B 参数 (parameter)模型，仅为在训练开始前保存 float16 权重 (weight)，就需要大约 140GB 的系统内存。大多数集群节点没有足够的 CPU 内存来为每个进程实例化整个模型。解决方案是使用 `meta` 设备上下文 (context)。

当模型在 `torch.device("meta")` 下初始化时，PyTorch 会记录张量形状和计算图，但不分配任何存储空间。FSDP 可以包装这些“影子”模块。实际的内存分配仅在我们明确地实体化跨 GPU 分片的参数时发生。

然而，`meta` 设备初始化带来了一个复杂之处：权重是空的。我们必须定义一个 `param_init_fn`，FSDP 会在参数在本地 GPU 上分配 *之后* 但在训练开始 *之前* 调用它来初始化参数。

### 实现: 零内存实例化

下面的代码模拟了一个高端训练环境。我们定义了一个庞大的模型结构，在 meta 设备上初始化它，用我们的混合策略包装它，然后高效地实体化参数 (parameter)。

```python
import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
)

# 1. 定义一个模拟的Transformer块 (通常从您的模型库导入)
class DecoderLayer(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
        
    def forward(self, x):
        return self.mlp(self.attn(x))

class MassiveModel(nn.Module):
    def __init__(self, num_layers=8, dim=4096):
        super().__init__()
        # 大型嵌入层
        self.embed = nn.Linear(dim, dim)
        self.layers = nn.ModuleList([
            DecoderLayer(dim, 4 * dim) for _ in range(num_layers)
        ])
        # 一个可能需要单独分片的巨大输出头
        self.head = nn.Linear(dim, 32000)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)

# 2. 定义实体化初始化逻辑
def materialization_fn(module: nn.Module):
    """
    FSDP 为每个模块调用此函数以初始化权重。
    这在存储分配后于 GPU 上运行。
    """
    # 只初始化实际已分配的参数
    for name, param in module.named_parameters(recurse=False):
        if hasattr(param, "_is_sharded"):
            # 根据版本可能需要FSDP特定的标志检查
            pass 
            
        # 标准初始化逻辑
        if "weight" in name and param.dim() > 1:
            torch.nn.init.kaiming_normal_(param)
        elif "bias" in name:
            torch.nn.init.zeros_(param)

# 3. 执行上下文
def setup_fsdp_model(rank, world_size):
    # 使用 BFloat16 以提高训练稳定性
    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # 构建针对我们特定层类的包装策略
    my_auto_wrap_policy = get_hybrid_policy(DecoderLayer)

    # A. 在 META 设备上初始化 (0内存使用)
    with torch.device("meta"):
        meta_model = MassiveModel()

    # B. 用 FSDP 包装
    # 注意: param_init_fn 处理从 meta 到真实权重的转换
    fsdp_model = FSDP(
        meta_model,
        auto_wrap_policy=my_auto_wrap_policy,
        mixed_precision=bf16_policy,
        device_id=torch.device("cuda", rank),
        param_init_fn=materialization_fn,
        sync_module_states=True # 确保所有进程初始化一致性很重要
    )
    
    return fsdp_model

# 在启动脚本中的用法:
# model = setup_fsdp_model(local_rank, world_size)
```

### 内存分析

此配置的效果在内存使用情况分析中可见。如果不进行包装（单一 FSDP），系统会尝试收集所有参数 (parameter)，导致在第一次前向传播时立即出现内存不足 (OOM) 错误。采用简单包装（例如，只包装顶层模块），内存峰值仍然危险地高。

优化后的策略创建了锯齿状的内存模式。内存使用量仅增加一个 `DecoderLayer` 的大小加上激活内存，然后随着 FSDP 释放分片而立即下降。



![GPU 内存使用: 前向传播](plots/7892-0.json)



> 优化包装策略（蓝色）动态收集参数，产生周期性峰值，这些峰值保持在硬件限制之内。简单方法（红色）不必要地保留参数，导致内存饱和。

### 验证分片结构

初始化后，验证包装是否正确应用是很重要的。简单地打印 PyTorch 中的模型会显示递归的 FSDP 包装器。

您应该会看到 `FullyShardedDataParallel` 包装了每个 `DecoderLayer`。如果您只在顶层 (`MassiveModel`) 看到 `FullyShardedDataParallel`，则表示自动包装策略失败，模型在内存方面将表现得像一个标准 DDP 模型，很可能会高效崩溃。

```python
# 诊断检查
if rank == 0:
    print(fsdp_model)

# 预期输出片段:
# MassiveModel(
#   (embed): Linear(...)
#   (layers): ModuleList(
#     (0): FullyShardedDataParallel(
#       (_fsdp_wrapped_module): DecoderLayer( ... )
#     )
#     (1): FullyShardedDataParallel( ... )
#   )
# )
```

这种结构确认 `DecoderLayer` 作为计算和通信的基本单元。`param_init_fn` 确保当这些单元首次被访问时，它们包含直接在 GPU 上初始化的有效权重 (weight)，完全绕过主机 RAM。
