---
course: "advanced-jax"
chapter: "advanced-jax-transformations-control-flow"
lesson: "advanced-masking-techniques"
sourceId: 2805
sourceUrl: "https://apxml.com/zh/courses/advanced-jax/chapter-1-advanced-jax-transformations-control-flow/advanced-masking-techniques"
title: "高级掩码技术"
description: "实施复杂的掩码模式，以处理变长序列或选择性计算。"
order: 6
plots: []
sourceHash: "8d5fd43afd9f68f8e8af4d83a3ab4c9c73ee80d16d8eae60f348b2ecb4a6aaa3"
sourceCorrections: []
---

在处理批处理数据或序列时，经常会遇到元素不一致的情况。例如，批次中的句子长度不一，或者您可能只想根据某个条件对特定元素应用操作。虽然像 `lax.cond` 这样的控制流原语可以处理一些条件逻辑，但在大型数组上基于条件逐元素应用操作时，通常需要使用**掩码**。

掩码涉及使用辅助数组（通常是布尔型），以指定数据数组中的哪些元素应包含在计算中，哪些应被忽略或区别对待。这种方法在 `jit` 编译的函数中特别有效，因为它通常能转换为在硬件加速器上高效、无分支的逐元素操作。

### 为何使用掩码？

设想处理一批以词嵌入 (embedding)序列表示的句子。为了使用 `vmap` 或将它们输入到许多标准神经网络 (neural network)层中，这些序列通常需要填充到相同的最大长度。然而，您不希望填充标记 (token)影响序列平均值或注意力分数等操作的结果。掩码提供了一种系统地排除这些填充元素的方法。

类似地，在处理序列的 `lax.scan` 循环中，一些序列可能会比其他序列更早结束。掩码允许您仅对仍处于活动状态的序列有选择地更新状态或计算输出。

### 常见掩码技术

JAX 中最常用的掩码工具是 `jax.numpy.where`。其函数签名是：

```python
jnp.where(condition, x, y)
```

它返回一个数组，其中 `condition` 数组为 `True` 的位置从 `x` 中选择元素，为 `False` 的位置从 `y` 中选择元素。`condition`、`x` 和 `y` 会进行广播操作。

#### 填充与序列掩码

这是一个主要应用场景。我们用填充变长序列来举例说明。

1. **填充：** 将特殊填充值（通常为 0）添加到批次中较短序列的末尾，使所有序列的长度与最长序列的长度匹配。
2. **掩码创建：** 创建一个与填充数据形状相同的布尔数组，其中 `True` 表示有效数据元素，`False` 表示填充元素。
3. **应用掩码：** 使用 `jnp.where` 或算术操作来应用掩码。

```python
import jax
import jax.numpy as jnp

# 例子：批处理序列（为简化起见，用整数表示）
sequences = [
    jnp.array([1, 2, 3]),
    jnp.array([4, 5]),
    jnp.array([6, 7, 8, 9])
]

# 假设有一个填充函数（为简洁起见，省略实现）
# 用 0 填充到最大长度 (4)
padded_sequences = jnp.array([
    [1, 2, 3, 0],
    [4, 5, 0, 0],
    [6, 7, 8, 9]
])
# 填充后数据形状：(3, 4) - 批大小 3，最大长度 4

# 创建掩码：True 表示数据，False 表示填充
# 假设填充值为 0
mask = (padded_sequences != 0)
# mask:
# [[ True,  True,  True, False],
#  [ True,  True, False, False],
#  [ True,  True,  True,  True]]

# 例子 1：掩码求和（避免对填充求和）
# 在求和前使用 jnp.where 将填充替换为 0
masked_values_for_sum = jnp.where(mask, padded_sequences, 0)
sum_per_sequence = jnp.sum(masked_values_for_sum, axis=-1)
# sum_per_sequence: [ 6 9 30] (正确和：1+2+3=6, 4+5=9, 6+7+8+9=30)

# 例子 2：掩码平均（避免在分母中计算填充）
# 如上求和
sum_values = jnp.sum(jnp.where(mask, padded_sequences, 0.0), axis=-1)
# 计算每个序列的有效元素数量
num_valid_elements = jnp.sum(mask, axis=-1)
# 避免对可能完全是填充的序列进行零除（如果可能）
average_per_sequence = sum_values / jnp.maximum(num_valid_elements, 1)
# average_per_sequence: [ 2.        ,  4.5       ,  7.5       ]

# 例子 3：直接应用掩码（常用于注意力机制）
# 设想为所有位置（包括填充）计算了分数。
# 在 softmax 之前，将填充位置的分数设置为一个很大的负数。
scores = jnp.randn(3, 4) # 示例分数
masked_scores = jnp.where(mask, scores, -1e9) # Use large negative value
# 现在应用 softmax，填充分数将接近零。
attention_weights = jax.nn.softmax(masked_scores, axis=-1)

# print("Padded Sequences:\n", padded_sequences)
# print("Mask:\n", mask)
# print("Sum per sequence:", sum_per_sequence)
# print("Average per sequence:", average_per_sequence)
# print("Masked scores for softmax (example):\n", masked_scores)
# print("Attention weights (example):\n", attention_weights)
```

> 填充不同长度序列至统一长度并生成相应布尔掩码的视觉表示。`T` 代表 True（有效数据），`F` 代表 False（填充）。

#### 使用算术运算进行掩码

除了 `jnp.where`，有时也可以使用算术运算，尤其当掩码由 0 和 1 组成时。将布尔掩码转换为数据的数据类型即可实现这一点。

```python
# 将掩码转换为浮点型（True -> 1.0, False -> 0.0）
mask_float = mask.astype(padded_sequences.dtype)

# 使用乘法进行掩码求和
sum_per_sequence_alt = jnp.sum(padded_sequences * mask_float, axis=-1)
# sum_per_sequence_alt: [ 6. 9. 30.] (确保数据类型匹配，通常为浮点型)

# 使用乘法进行掩码平均
num_valid_elements_alt = jnp.sum(mask_float, axis=-1)
average_per_sequence_alt = sum_per_sequence_alt / jnp.maximum(num_valid_elements_alt, 1e-9) # 为安全起见添加 epsilon
# average_per_sequence_alt: [ 2.        ,  4.5       ,  7.5       ]

# print("Mask float:\n", mask_float)
# print("Sum via multiplication:", sum_per_sequence_alt)
# print("Average via multiplication:", average_per_sequence_alt)
```

在使用乘法进行掩码时请注意，尤其是在对数空间计算或涉及梯度时，因为与使用 `jnp.where` 明确选择值相比，乘以零可能不总会产生期望的数学或梯度行为。

### 控制流中的掩码

掩码与 JAX 转换和控制流的交互方式是可预测的：

- **`jit`：** 像 `jnp.where` 或算术运算这样的掩码操作很容易被 `jit` 编译成高效的低级代码。
- **`vmap`：** 如果您对函数在批处理维度上使用 `vmap`，并且您的输入包含填充数据和相应的掩码，`vmap` 将自动向量 (vector)化掩码操作以及主要计算。
- **`grad`：** 自动微分可以通过 `jnp.where` 正确工作。梯度将通过条件为每个元素选择的分支（`x` 或 `y`）反向传播 (backpropagation)。与未选择分支相关的梯度对于该元素实际上为零。这通常是期望的行为，可以防止填充或被掩码排除的元素对参数 (parameter)更新产生贡献。
- **`lax.scan`：** 掩码可以作为 `lax.scan` 中状态（`carry`）的一部分传递，或在 scan 主体内部计算。这使得您可以在批处理中执行序列操作时遵守每个序列的有效性边界。例如，在 RNN 中，您可以使用掩码来防止更新已结束（遇到填充）序列的隐藏状态。

```python
# lax.scan 内掩码的例子
def scan_body(carry, x):
  hidden_state, current_mask = carry
  input_element = x

  # 计算潜在的新状态（简化）
  potential_new_state = hidden_state * 0.9 + input_element * 0.1

  # 仅当此步骤的掩码为 True 时更新状态
  # 否则，保持旧状态
  new_state = jnp.where(current_mask, potential_new_state, hidden_state)

  # 基于新状态输出一些东西（可能已掩码）
  output = jnp.where(current_mask, new_state * 2.0, 0.0)

  # 注意：掩码如何演变取决于具体应用。
  # 这里我们假设掩码本身不被 scan 主体改变，
  # 但它可以是输入 'x' 的一部分，或根据状态进行更新。
  new_carry = (new_state, current_mask)
  return new_carry, output

# initial_state = ...
# sequence_inputs = ...
# sequence_masks = ... # 形状与 sequence_inputs 匹配

# 需要为 scan 正确组织掩码，可能需要堆叠它们
# 或者将它们作为 'xs' 的一部分传递给 lax.scan(scan_body, init, xs=(inputs, masks))
# final_carry, outputs = lax.scan(scan_body, (initial_state, initial_mask_state), inputs_with_masks)
```

### 性能考量

- **掩码与 `lax.cond` 的比较：** 掩码将计算应用到所有位置，然后使用 `jnp.where` 或算术运算选择结果。这在 GPU/TPU 上通常很高效，因为它避免了 SIMD/SIMT 执行单元内依赖于数据的分支，而这种分支可能导致线程/通道发散。然而，这意味着即使对于被掩码排除的元素，您也会计算结果。如果分支涉及显著不同且开销大的计算，即使存在潜在的发散成本，如果条件经常能避免大量的计算，`lax.cond` *可能*会更快。性能分析是确定特定情况下最合适的方法的最好方式。
- **梯度流：** 确保您的掩码策略正确防止不需要的梯度流。使用 `jnp.where` 通常是安全的。如果使用算术掩码，请仔细检查乘以零是否提供正确的梯度行为（对于简单的求和/平均值通常如此，但在其他情况下可能比较棘手）。如果需要，可以显式使用 `jax.lax.stop_gradient`，但首先应依赖 `jnp.where`。

"掌握掩码技术对于处理数据中常见的非规则性非常重要，尤其是在使用 JAX 的编译和向量 (vector)化能力在加速器上实现高性能时。它使您能够编写简洁、可组合的代码，正确处理不同结构的批次和序列。"

## 参考资料

- [JAX Documentation](https://jax.readthedocs.io/en/latest/) — JAX Developers (2024)
  官方文档提供了JAX核心功能的全面信息，包括`jax.numpy.where`、其他数组操作函数以及JAX转换和控制流的基础知识，这些对于高级掩码技术至关重要。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 30; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  本文介绍了Transformer架构，该架构利用掩码技术（填充掩码、前瞻掩码）高效且正确地处理序列数据，这是JAX掩码的一个常见应用领域。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  第10章“序列建模：循环和递归网络”讨论了可变长度序列、填充以及处理这些差异的需求，为深度学习中为何使用掩码提供了理论背景。
- [Composable transformations of Python+NumPy programs](http://github.com/google/jax) — James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang (2018)
  本文概述了JAX的基本设计原则，包括其自动微分、编译（`jit`）和向量化方法，为掩码策略为何能在加速器上高效执行提供了背景。
