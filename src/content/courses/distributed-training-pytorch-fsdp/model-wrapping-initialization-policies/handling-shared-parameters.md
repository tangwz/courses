---
course: "distributed-training-pytorch-fsdp"
chapter: "model-wrapping-initialization-policies"
lesson: "handling-shared-parameters"
sourceId: 7891
sourceUrl: "https://apxml.com/zh/courses/distributed-training-pytorch-fsdp/chapter-2-model-wrapping-initialization-policies/handling-shared-parameters"
title: "处理共享参数"
description: "在分布式分片环境下管理共享权重的技术方法。"
order: 4
plots: ["plots/7891-0.json"]
sourceHash: "37a1dc44622f64360bc75abdf3f4c1155278e6c7651b7a05bd85e32dfedbd3f7"
sourceCorrections: []
---

参数 (parameter)共享，通常被称为权重 (weight)绑定，是现代Transformer模型中一种标准的架构效率措施。最常见的情况是绑定输入嵌入 (embedding)矩阵与输出语言建模头。数学上，如果输入嵌入为 $W_E \in \mathbb{R}^{V \times d}$，输出层通常会利用相同的权重，有效地将logits计算为 $y = x W_E^T$。这大幅减少了参数数量，在大型词汇模型中通常可节省数亿个参数。

然而，完全分片数据并行（FSDP）在处理这些共享引用时增加了难度。FSDP通过将参数扁平化为每个封装单元内单一的连续 `FlatParameter` 来运行。如果两个模块共享一个参数但被分配给不同的FSDP单元（分片组），系统就会面临冲突：它无法将相同的底层存储分配给由不同通信流管理的两个独立的扁平化向量 (vector)。

### 共享权重 (weight)的分片冲突

FSDP初始化时，它会遍历模块层级结构以识别参数 (parameter)。它遵循Python对象标识（通过 `id()` 检查）来检测共享权重。如果系统遇到一个已被现有FSDP单元管理的参数，它必须决定如何处理。

FSDP中的主要限制是**共享参数必须属于同一个FSDP单元**。它们不能跨越不同边界进行分片。如果模块A和模块B共享权重 $W$，你不能将模块A封装在一个FSDP实例中，而将模块B封装在另一个FSDP实例中。这样做将迫使 $W$ 存在于两个独立的 `FlatParameter` 存储中，破坏同步逻辑，并在优化过程中实际上解除了权重绑定。

下图描绘了共享参数的有效和无效封装层级结构之间的结构差异。

> 描绘所有权冲突。在无效策略中，两个FSDP实例试图分片相同的内存地址。在有效策略中，共享权重保留在单个FSDP范围内。

### 实现安全的封装策略

为了正确处理共享参数 (parameter)，你必须设计你的自动封装策略，将包含共享权重 (weight)的模块排除在单独封装之外。而是让顶层FSDP封装器来管理它们。

以标准的GPT风格架构为例，其中Transformer块计算量很大，但嵌入 (embedding)层和头共享权重。目标是单独封装Transformer层以节省内存，同时将嵌入层和头保留在根FSDP单元中。

以下是如何构建一个遵循参数共享原则的封装策略：

```python
import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    lambda_auto_wrap_policy,
)

class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, dim)
        self.mlp = nn.Linear(dim, dim)

class GPTModel(nn.Module):
    def __init__(self, vocab_size, dim, layers):
        super().__init__()
        # 共享权重逻辑在此处实现
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim) for _ in range(layers)])
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
        # 显式绑定权重
        self.lm_head.weight = self.token_emb.weight

def get_wrapping_policy():
    """
    返回一个策略，该策略封装 TransformerBlock 实例 
    但将 Embedding 和 LM Head 留给根 FSDP 单元。
    """
    return functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

# 训练循环中的设置代码
model = GPTModel(vocab_size=32000, dim=1024, layers=12)

# 该策略单独封装块。 
# token_emb 和 lm_head 在根 FSDP 调用之前保持未封装状态。
fsdp_model = FSDP(
    model,
    auto_wrap_policy=get_wrapping_policy(),
    mixed_precision=MixedPrecision(param_dtype=torch.bfloat16),
    device_id=torch.cuda.current_device()
)
```

在此配置中，`TransformerBlock`实例立即被封装和分片。`token_emb`和`lm_head`被自动封装策略忽略。因此，它们落入外部`fsdp_model`的作用域。由于它们都由同一个根FSDP单元管理，共享权重`self.token_emb.weight`被扁平化一次，并且两个模块都引用扁平化存储中正确的索引。

### 参数 (parameter)身份验证

在分布式设置中工作时，无声无息不等于一切顺利。务必检查权重 (weight)在FSDP初始化后保持绑定。如果封装逻辑意外中断了链接，模型可能会使用分离的嵌入 (embedding)和输出头进行训练，导致收敛不良。

你可以在封装后验证存储指针的身份。请注意，FSDP会修改`param.data`以指向`FlatParameter`中的一个视图。

```python
def verify_weight_tying(fsdp_model):
    # 访问底层模块（如果嵌套很深，可能需要展开）
    # 在此简单示例中，我们假设通过属性访问 
    
    # 注意：我们必须访问原始模块结构。 
    # FSDP封装了模块，因此我们可能需要查看 fsdp_model.module 或类似结构
    # 具体依据准确的层级结构。
    
    embedding_weight = fsdp_model.token_emb.weight
    head_weight = fsdp_model.lm_head.weight

    # 1. 检查Python对象身份
    # 如果 FSDP 用视图替换参数，这可能会改变， 
    # 但底层存储的 data_ptr 应该匹配。
    
    ptr_emb = embedding_weight.data_ptr()
    ptr_head = head_weight.data_ptr()
    
    if ptr_emb == ptr_head:
        print(f"成功：权重已绑定。存储指针：{ptr_emb}")
    else:
        print(f"警告：权重未绑定。{ptr_emb} != {ptr_head}")

# 在训练前在 Rank 0 上调用此函数
verify_weight_tying(fsdp_model)
```

### 共享参数 (parameter)与Meta设备上的初始化

将共享参数与`meta`设备初始化结合使用需要仔细的顺序安排。正如“延迟初始化”部分所提到的，我们通常在`meta`设备上创建模型以避免OOM错误。当你具体化这些权重 (weight)（将它们移动到CPU或CUDA）时，你必须确保在用FSDP封装*之前*重新建立绑定关系。

如果你在`meta`设备上初始化：

1. 在`meta`设备上创建模型。
2. 具体化参数（重置参数）。
3. **重新应用权重绑定。**
4. 用FSDP封装。

通常，`reset_parameters()`或具体化过程会为每个模块分配新的内存，破坏在`__init__`中创建的引用。你必须显式地重新绑定权重：

```python
# 从 meta 设备具体化权重后
model.to_empty(device="cuda")
model.apply(init_weights_fn)

# 关键：在 FSDP 封装之前显式重新绑定权重
model.lm_head.weight = model.token_emb.weight

# 现在可以安全封装了
fsdp_model = FSDP(model, ...)
```

### 权重 (weight)绑定的内存影响

权重绑定虽然节省了参数 (parameter)，但它可能在网络边界处造成通信瓶颈。在FSDP中，包含嵌入 (embedding)层和头的根单元通常相当大。对于词汇量 $V=128,000$ 且维度 $d=4096$ 的情况，仅嵌入表在BF16精度下就大约为1GB。

因为这个大型参数集在根FSDP单元中，它必须在正向传播开始时（用于嵌入层）以及结束时（用于头）进行收集（all-gathered）。如果根单元没有被有效分片的话，这会造成内存压力。

下表描绘了当嵌入层和头在根单元中与分布在各层中时的正向传播期间的内存分配时间线。



![共享嵌入层（根FSDP单元）引起的内存峰值](plots/7891-0.json)



> 正向传播期间的活跃内存使用情况。“根部峰值”的出现是因为大型共享嵌入层必须在开始和结束时被收集，超过了内部Transformer层的内存占用。

如果共享嵌入层过大，它可能迫使你对根单元使用`cpu_offload=True`，或实现**序列并行**来分片激活内存，弥补参数峰值。对于训练词汇量对总参数数量有较大影响的模型来说，妥善管理这些共享资源非常必要。
