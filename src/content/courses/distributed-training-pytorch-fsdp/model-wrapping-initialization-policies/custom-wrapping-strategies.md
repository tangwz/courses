---
course: "distributed-training-pytorch-fsdp"
chapter: "model-wrapping-initialization-policies"
lesson: "custom-wrapping-strategies"
sourceId: 7889
sourceUrl: "https://apxml.com/zh/courses/distributed-training-pytorch-fsdp/chapter-2-model-wrapping-initialization-policies/custom-wrapping-strategies"
title: "自定义包装策略"
description: "开发自定义 lambda 函数，以控制非标准模型架构的单元分片。"
order: 2
plots: ["plots/7889-0.json"]
sourceHash: "b6fbbf919de79119b5f70b46def470ed3d33c4c1ff67277e2cc837435c7a0272"
sourceCorrections: []
---

标准自动包装策略通常能满足 BERT 或 GPT 等同构架构的需求。然而，复杂的生产环境常常需要针对分布式训练的定制策略。混合多模态 (multimodal)模型、专家混合 (MoE) 配置以及具有非标准残差连接的架构，引入的内存模式是常见默认策略无法优化的。在这些情况下，依赖于诸如 `transformer_auto_wrap_policy` 这样的策略可能会导致次优的分片，从而引起内存峰值（当分片过大时）或通信瓶颈（当分片过小时）。

本节说明如何使用 Python lambda 函数和 `functools.partial` 工具构建自定义包装策略。通过定义明确的图分割规则，您可以控制内存粒度与网络开销之间的权衡。

### 包装策略的运作方式

在 PyTorch FSDP 中，`auto_wrap_policy` 参数 (parameter)接受一个可调用对象，用于确定特定子模块是否应包装到其自己的 FSDP 单元中。在初始化期间，FSDP 遍历模型的模块树。对于遇到的每个模块，它会调用此可调用对象，传入三个特定参数：

1. **`module`**: 当前正在评估的 `nn.Module` 实例。
2. **`recurse`**: 一个布尔值，指示遍历是否可以继续到模块的子级。
3. **`unwrapped_params`**: 当前模块中尚未分配给子 FSDP 单元的参数数量。

该函数必须返回一个 `boolean`。如果为 `True`，则模块被包装。如果为 `False`，它将保留为父 FSDP 单元（或根单元）的一部分。

该逻辑以递归方式运行。如果父模块被包装，它的参数会被分片。然而，如果该父级包含那些 *也* 被包装的子级，则子级会成为独立的 FSDP 单元。这种嵌套实现了“收集-计算-分散”的重叠。系统收集子级，计算，释放子级，然后收集父级（如果需要），计算剩余部分，依此类推。

### 实现基于逻辑的策略

自定义策略通常结合了参数 (parameter)计数和模块类型检查。设想一个情况，您正在训练一个模型，它有一个必须被分片的大型嵌入 (embedding)层，但也有许多小型投影头应该保持聚合，以避免启动数千个微小的 NCCL 内核。

以下代码演示了一种策略，它根据两个条件包装模块：它们必须是特定类型（例如，Transformer 块），或者它们必须超过特定的参数阈值。

```python
import functools
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    _or_policy,
    lambda_auto_wrap_policy,
)
import torch.nn as nn

def hybrid_shard_policy(
    module: nn.Module,
    recurse: bool,
    unwrapped_params: int,
    min_num_params: int = 1e6,
    target_module_types: tuple = ()
) -> bool:
    """
    如果满足以下条件，则进行包装的自定义策略：
    1. 模块是 `target_module_types` 的实例
    或者
    2. 模块包含的参数数量超过 `min_num_params`
    """
    # 如果模块允许递归，则始终递归以确保检查子级
    if recurse:
        return True

    # 条件 1：基于类型的包装
    if isinstance(module, target_module_types):
        return True

    # 条件 2：基于大小的包装
    # 仅当剩余参数足以构成一个新的分片时才进行包装
    if unwrapped_params >= min_num_params:
        return True

    return False

# 在 FSDP 构造函数中的应用
# 假设 MyTransformerBlock 在其他地方定义
# my_model = ...

custom_policy = functools.partial(
    hybrid_shard_policy,
    min_num_params=5 * 10**6,  # 最小 5 百万参数
    target_module_types=(nn.TransformerEncoderLayer, )
)

# fsdp_model = FSDP(my_model, auto_wrap_policy=custom_policy, ...)
```

在此实现中，该逻辑避免了“过度分片”。如果未仔细限制，纯粹的递归包装器可能会将小的 LayerNorm 或 Dropout 模块隔离。通过强制设置 `min_num_params` 下限，我们确保非常小的层仍然是其父块的一部分，减少了前向传播期间所需的独立全收集操作的数量。

### 图分区可视化

以下图表说明了不同策略如何影响包含图像编码器和文本解码器的混合架构的分片。

> 图分区结果比较。基于类型的包装（左侧）针对架构单元，而基于参数 (parameter)数量的包装（右侧）直接针对内存消耗。

### 处理异构架构

在训练诸如专家混合 (MoE) 的架构时，标准包装会失败，因为“专家”层通常是稀疏的。如果您将整个 MoE 块包装成一个单元，即使只有一个专家是活跃的，您也会强制对 *所有* 专家进行全收集，这违背了稀疏计算的目的。

对于 MoE，自定义策略必须针对单个专家。这需要检查模块名称或结构，而不是仅仅是类类型，特别是如果专家是泛型 `nn.Linear` 层的实例。

```python
def moe_sparse_policy(module, recurse, unwrapped_params):
    if recurse:
        return True

    # 识别此模块是否为特定的专家容器
    # 检查逻辑取决于具体的模型实现细节
    if hasattr(module, 'is_sparse_expert') and module.is_sparse_expert:
        return True

    return False
```

通过包装单个专家，FSDP 在前向传播期间只收集活跃专家的参数 (parameter)（假设专家在不同的执行路径或 CUDA 流上），显著减少了峰值内存使用。

### 优化分片粒度

设计自定义策略涉及平衡内存效率和通信延迟。

1. **粒度过细（过度包装）：** 包装每个小层会导致数百个 FSDP 单元。这会触发过多的 CUDA 内核启动和 NCCL 同步屏障。GPU 花在等待网络操作上的时间比计算时间更多。
2. **粒度过粗（不足包装）：** 仅包装顶层容器意味着必须一次性收集大量参数 (parameter)。这会增加峰值内存需求，可能导致内存不足 (OOM) 错误。

目标是找到“最佳点”，即分片大小既要足够大以饱和网络带宽，又要足够小以适应 GPU 内存预算。



![包装粒度对训练吞吐量的影响](plots/7889-0.json)



> 分片大小与吞吐量 (throughput)之间的关系。极小的分片会增加通信开销（红色虚线），而极大的分片由于内存碎片和计算重叠不足而降低性能。

### 排除策略

有时，需要自定义策略来 *阻止* 包装。在网络中共享的参数 (parameter)，例如与输出投影关联的嵌入 (embedding)层（在 GPT 架构中很常见），如果它们在不同的 FSDP 单元中分片，常常会导致同步问题。

lambda 策略可以明确地为已知的共享模块返回 `False`，强制它们由根 FSDP 单元或特定父级管理。这确保了共享参数在前向传播中只广播一次，而不是被不同的包装器多次收集和分散。

```python
def exclude_embeddings_policy(module, recurse, unwrapped_params):
    # 识别要排除的嵌入层
    if isinstance(module, (nn.Embedding, nn.EmbeddingBag)):
        return False  # 强制排除

    # 对其他层继续执行标准的基于大小的包装
    return unwrapped_params >= 1e7
```

这种策略非常重要，尤其是在使用 `sharding_strategy=ShardingStrategy.SHARD_GRAD_OP` 时，其中优化器状态被分片但参数可能保持不变。通过明确控制切分点，您可以确保系统行为与模型架构的特定数学要求一致。
