---
course: "distributed-training-pytorch-fsdp"
chapter: "mixed-precision-memory-optimization"
lesson: "activation-checkpointing-mechanics"
sourceId: 7894
sourceUrl: "https://apxml.com/zh/courses/distributed-training-pytorch-fsdp/chapter-3-mixed-precision-memory-optimization/activation-checkpointing-mechanics"
title: "激活检查点工作原理"
description: "在FSDP封装的模块中应用激活检查点技术，以计算换取内存。"
order: 2
plots: ["plots/7894-0.json"]
sourceHash: "512566506d68231f3ff2ce4756719a752fca8e12217efd81bdff76bb8b9fe3d9"
sourceCorrections: []
---

将模型参数 (parameter)的精度降至BFloat16或Float16能显著减少模型权重 (weight)的静态内存占用。然而，训练过程中，反向传播 (backpropagation)所需的中间激活值存储产生的瞬时内存，常会超过参数本身占用的内存。对于序列长度较长的大型语言模型（LLMs），这种激活内存会随层数和批次大小线性增加，并因注意力机制 (attention mechanism)而随序列长度呈平方增长。激活检查点（又称梯度检查点）通过在前向传播时丢弃中间激活值，并在反向传播时重新计算它们来解决这一瓶颈。

### 重新计算策略

标准反向传播 (backpropagation)要求计算图中每个操作的输出都存储在GPU内存中，直到梯度计算完毕。对于一个有 $L$ 层的Transformer模型，这会带来 $O(L)$ 的内存复杂度。

激活检查点通过将特定模块指定为“检查点”来改变这种行为。只有这些被检查点标记 (token)的模块的输入会保留在内存中。模块内生成的所有中间激活值都会被丢弃。当反向传播到达一个被检查点标记的模块时，系统会使用存储的输入执行一次局部前向传播，以重新生成所需的中间状态。这实际上是用计算开销换取内存节省。

在理想检查点配置下，理论内存消耗通常与层数呈平方根关系。如果我们将一个包含 $N$ 个节点的网络分成长度为 $\sqrt{N}$ 的段，我们只存储 $\sqrt{N}$ 个段的边界。在反向传播期间，我们重新计算一个段内 $\sqrt{N}$ 个节点。

$检查点内存 \approx O(\sqrt{N}) + O(块大小)$

下面的图展示了标准训练和检查点训练在反向传播期间内存状态的差异。

> 标准训练与激活检查点方法中存储张量的比较。在检查点方法中，中间层被丢弃并按需重新计算。

### 将检查点与FSDP结合

在FSDP环境中应用激活检查点技术需要仔细规划。在子模块上直接使用PyTorch的`torch.utils.checkpoint`可能会与FSDP的分片逻辑冲突。FSDP将参数 (parameter)分片到多个GPU上，但检查点技术在重新计算阶段需要完整访问参数。

PyTorch为此专门提供了`apply_activation_checkpointing`工具函数。此函数确保了正确的封装顺序：FSDP封装器必须包含检查点封装器，反之亦然，这取决于所需的粒度。对于Transformer模型，建议的做法是封装每个Transformer块（注意力机制 (attention mechanism) + 前馈网络）。

当在FSDP中使用`apply_activation_checkpointing`时，该工具会自动处理`checkpoint_wrapper`。对于专业用户而言，一个重要配置项是选择可重入（reentrant）还是不可重入（non-reentrant）封装器。

#### 不可重入封装器

对于FSDP，强烈建议使用不可重入的检查点 (`use_reentrant=False`)。传统的（legacy）可重入版本依赖全局状态和反向钩子，这可能会干扰FSDP的状态同步，可能导致死锁或错误的梯度。不可重入的实现将检查点区域视为一个独立的自动求导函数，提供了更好的稳定性和与分布式设置的兼容性。

以下是将检查点技术应用于FSDP封装的Transformer模型时的实现模式：

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# 定义检查函数以识别Transformer块
# 假设模型结构包含一个名为'TransformerBlock'的类
check_fn = lambda submodule: isinstance(submodule, TransformerBlock)

# 应用检查点技术
apply_activation_checkpointing(
    model, 
    checkpoint_wrapper_fn=checkpoint_wrapper,
    check_fn=check_fn,
)

# 在应用检查点逻辑后用FSDP封装
# 注意：在最近的PyTorch版本中，
# 直接应用于FSDP模块也受支持，并且通常更受青睐。
fsdp_model = FSDP(model, auto_wrap_policy=...)
```

### 吞吐量 (throughput)与内存分析

激活检查点技术会引入计算成本。由于检查点段的前向传播会运行两次（一次用于损失计算，一次用于反向传播 (backpropagation)），计算开销大约在20%到30%之间。

然而，这种开销通常是合理的。内存使用的减少使得微批次大小能够大幅增长。更大的批次大小能提升GPU占用率和算术密度，可能抵消重新计算的成本。在模型可以在不使用检查点但批次大小很小的情况下勉强适应内存的场景中，GPU计算单元可能未被充分利用。通过启用检查点并增加批次大小，尽管有重新计算的代价，您通常能获得更高的总体每秒处理token量。

以下图表展示了批次大小与内存消耗（使用和不使用检查点）之间的关系。



![内存使用量与批次大小对比（7B参数模型）](plots/7894-0.json)



> 内存扩展行为。红线表示标准训练在批次大小为8时达到80GB显存 (VRAM)限制。蓝线显示在相同硬件条件下，检查点技术使得批次大小可以达到32。

### 选择性检查点与卸载

对于即便标准检查点技术也力有不逮的超大规模模型，PyTorch FSDP允许选择性检查点。您可以选择每 $n$ 层进行一次检查点，而非对每层Transformer都进行。这提供了一个精细的调节手段，以平衡内存节省和计算开销。

此外，`checkpoint_wrapper`支持CPU卸载。FSDP负责参数 (parameter)卸载，而激活值卸载则不同。通过在检查点封装器中配置`offload_to_cpu=True`，检查点模块保留的输入会被移至系统RAM，并在重新计算步骤前立即预取回GPU。当PCIe带宽不是瓶颈时，这种做法特别有效，使得能够训练远超集群GPU总内存容量的模型。

$总吞吐量 \propto \frac{批次大小}{步长耗时}$

在优化时，目标是使上述公式的值最大化。如果检查点技术使`StepTime`增加30%，但同时允许`BatchSize`增加100%而不会出现内存不足（OOM）错误，那么最终结果是训练效率的大幅提升。
