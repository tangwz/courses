---
course: "advanced-jax"
chapter: "advanced-jax-transformations-control-flow"
lesson: "practical-complex-recurrent-logic"
sourceId: 2811
sourceUrl: "https://apxml.com/zh/courses/advanced-jax/chapter-1-advanced-jax-transformations-control-flow/practical-complex-recurrent-logic"
title: "实践：实现复杂的循环逻辑"
description: "练习结合 lax.scan 及其他基本操作来实现复杂的循环神经网络单元。"
order: 8
plots: []
sourceHash: "9aaa17d44f6bcd60ebda4cb28f3cdd6c59421587cb3a409b45331b8c5ad6d84b"
sourceCorrections: []
---

实现一个自定义的循环单元是序列建模中经常遇到的实际场景，这使得它成为像 `lax.scan` 这样的控制流原语的一个良好应用。尽管基本的 RNN 简单明了，但许多高级架构使用更复杂的门控机制。可以使用 `lax.scan` 来实现一个简化的门控循环单元（GRU）以处理序列并更新隐藏状态。这说明了如何在 `scan` 主体中构建非简单的计算。

### 理解简化的 GRU 单元

GRU 是一种循环神经网络 (neural network) (RNN)单元，它旨在通过门控机制捕获不同时间尺度上的依赖关系。这些门控制着信息的流动，决定保留过去状态的多少，以及从新输入中引入多少。

在这个例子中，我们将实现一个稍加简化的版本。设 $x_t$ 为时间步 $t$ 的输入向量 (vector)，$h_{t-1}$ 为前一时间步的隐藏状态。时间步 $t$ 的隐藏状态 $h_t$ 的计算方式如下：

1. **更新门 ($z_t$):** 决定保留多少前一个隐藏状态的信息。
   
   $$
   z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)
   $$
   
2. **重置门 ($r_t$):** 决定在计算候选状态时，忘记多少前一个隐藏状态的信息。
   
   $$
   r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)
   $$
   
3. **候选隐藏状态 ($\tilde{h}_t$):** 基于当前输入和*重置后*的前一个隐藏状态，计算一个新的隐藏状态提议。
   
   $$
   \tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h)
   $$
   
4. **最终隐藏状态 ($h_t$):** 使用更新门 $z_t$，在前一个隐藏状态 $h_{t-1}$ 和候选隐藏状态 $\tilde{h}_t$ 之间进行线性插值。
   
   $$
   h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
   $$
   

这里，$\sigma$ 表示 sigmoid 激活函数 (activation function)，$\tanh$ 是双曲正切激活函数，$\odot$ 表示按元素乘法。$W_z, U_z, b_z, W_r, U_r, b_r, W_h, U_h, b_h$ 是 GRU 单元的可学习参数 (parameter)（权重 (weight)矩阵和偏置 (bias)向量）。

### 使用 lax.scan 实现 GRU

我们可以使用 `lax.scan` 来实现 GRU 单元对整个序列的处理。其主要思想是：

- `lax.scan` 中的 `carry` 将在每个时间步保存隐藏状态 $h_t$。
- `xs` 参数 (parameter)将是输入序列 $(x_1, x_2, ..., x_T)$。
- 传递给 `lax.scan` 的函数（我们称之为 `gru_step`）将实现上述四个方程，它接受前一个隐藏状态 `h_prev`（来自 carry）和当前输入 `x_t`（来自 `xs`）来计算新的隐藏状态 `h_t`。
- `gru_step` 将返回 `(h_t, h_t)`。第一个 `h_t` 成为下一步的 `carry`，第二个 `h_t` 作为输出序列被累积。

我们来编写代码。首先是导入库和定义参数。在实际应用中，这些参数将是更大模型结构（如 Flax 或 Haiku）的一部分，但为了清晰起见，我们在这里直接定义它们。

```python
import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random

# 定义激活函数
sigmoid = jax.nn.sigmoid
tanh = jnp.tanh

def initialize_gru_params(key, input_dim, hidden_dim):
    """初始化简化 GRU 单元的参数。"""
    keys = random.split(key, 6) # 需要用于 Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh 的键（3 对 W,U + 3 个偏置）

    # 更新门参数
    Wz = random.normal(keys[0], (hidden_dim, input_dim)) * 0.01
    Uz = random.normal(keys[1], (hidden_dim, hidden_dim)) * 0.01
    bz = jnp.zeros((hidden_dim,))

    # 重置门参数
    Wr = random.normal(keys[2], (hidden_dim, input_dim)) * 0.01
    Ur = random.normal(keys[3], (hidden_dim, hidden_dim)) * 0.01
    br = jnp.zeros((hidden_dim,))

    # 候选隐藏状态参数
    Wh = random.normal(keys[4], (hidden_dim, input_dim)) * 0.01
    Uh = random.normal(keys[5], (hidden_dim, hidden_dim)) * 0.01
    bh = jnp.zeros((hidden_dim,))

    params = {
        'Wz': Wz, 'Uz': Uz, 'bz': bz,
        'Wr': Wr, 'Ur': Ur, 'br': br,
        'Wh': Wh, 'Uh': Uh, 'bh': bh
    }
    return params

def gru_step(params, h_prev, x_t):
    """执行简化 GRU 计算的一个步骤。"""
    # 更新门
    z_t = sigmoid(jnp.dot(params['Wz'], x_t) + jnp.dot(params['Uz'], h_prev) + params['bz'])

    # 重置门
    r_t = sigmoid(jnp.dot(params['Wr'], x_t) + jnp.dot(params['Ur'], h_prev) + params['br'])

    # 候选隐藏状态
    h_tilde_t = tanh(jnp.dot(params['Wh'], x_t) + jnp.dot(params['Uh'], (r_t * h_prev)) + params['bh'])

    # 最终隐藏状态
    h_t = (1.0 - z_t) * h_prev + z_t * h_tilde_t

    # 返回新的隐藏状态作为 carry 和输出
    return h_t, h_t 

def gru_sequence(params, initial_h, inputs):
    """使用 lax.scan 将 GRU 单元应用于一系列输入。"""

    # 定义扫描函数，闭包捕获参数
    scan_fn = lambda carry, x: gru_step(params, carry, x)

    # 应用 lax.scan
    final_h, outputs_h = lax.scan(scan_fn, initial_h, inputs)

    # final_h 包含最后一个隐藏状态
    # outputs_h 包含隐藏状态序列 [h_1, h_2, ..., h_T]
    return final_h, outputs_h

# 使用示例
key = random.PRNGKey(0)
seq_len = 10
input_dim = 5
hidden_dim = 8

# 初始化参数
gru_params = initialize_gru_params(key, input_dim, hidden_dim)

# 创建虚拟输入序列（序列长度，输入特征）
key, subkey = random.split(key)
input_sequence = random.normal(subkey, (seq_len, input_dim))

# 初始化隐藏状态
initial_hidden_state = jnp.zeros((hidden_dim,))

# 在序列上运行 GRU
final_state, hidden_states_sequence = gru_sequence(gru_params, initial_hidden_state, input_sequence)

print("输入序列形状:", input_sequence.shape)
print("初始隐藏状态形状:", initial_hidden_state.shape)
print("最终隐藏状态形状:", final_state.shape)
print("隐藏状态输出序列形状:", hidden_states_sequence.shape)
```

在这段代码中：

1. `initialize_gru_params` 使用小的随机值初始化，设置了必要的权重 (weight)矩阵和偏置 (bias)向量 (vector)，使其具有适当的形状。
2. `gru_step` 实现了一个时间步的核心逻辑。它接受参数、前一个隐藏状态 `h_prev` 和当前输入 `x_t`，并两次返回新的隐藏状态 `h_t`（一次作为新的 carry，一次作为该步骤的输出）。
3. `gru_sequence` 协调整个过程。它定义了 `scan_fn`，该函数实际上是 `gru_step`，其中 `params` 参数是固定的（通过闭包捕获）。然后它使用此函数、初始隐藏状态和输入序列调用 `lax.scan`。
4. 示例用法显示了如何创建参数、生成示例输入并调用 `gru_sequence` 函数。输出形状证实最终状态具有隐藏维度，输出序列具有 `(序列长度, 隐藏维度)` 的维度。

### 与 JAX 变换集成

使用 `lax.scan` 的一个好处是，由此产生的 `gru_sequence` 函数完全兼容 JAX 的其他变换，如 `jit`、`grad` 和 `vmap`。

例如，为了编译 GRU 计算以实现更快的执行，只需使用 `jax.jit` 包装调用：

```python
# 编译 GRU 函数以提高效率
jit_gru_sequence = jax.jit(gru_sequence)

# 运行编译版本（首次运行包含编译时间）
key, subkey = random.split(key)
input_sequence_2 = random.normal(subkey, (seq_len, input_dim))
final_state_jit, hidden_states_sequence_jit = jit_gru_sequence(gru_params, initial_hidden_state, input_sequence_2)

print("\n运行 JIT 编译版本:")
print("最终隐藏状态形状 (JIT):", final_state_jit.shape)
print("输出序列形状 (JIT):", hidden_states_sequence_jit.shape) 
```

如果你想同时处理一批序列，可以使用 `jax.vmap`。假设你的输入具有批次维度，例如 `(批次大小, 序列长度, 输入维度)`，你将对初始隐藏状态（`批次大小, 隐藏维度`）和输入都映射到批次维度：

```python
# VMAP 使用示例（需要批处理输入/状态）
# 假设:
# batch_size = 32
# batched_inputs = random.normal(key, (batch_size, seq_len, input_dim))
# batched_initial_h = jnp.zeros((batch_size, hidden_dim,))

# 映射到批次维度（params=None, initial_h, inputs 的轴 0）
# 注意：参数在批次中共享，因此我们在 in_axes 中使用 None
# batched_gru = jax.vmap(gru_sequence, in_axes=(None, 0, 0)) 
# final_states_batch, hidden_sequences_batch = batched_gru(gru_params, batched_initial_h, batched_inputs)

# print("批处理最终状态形状:", final_states_batch.shape) # (批次大小, 隐藏维度)
# print("批处理输出序列形状:", hidden_sequences_batch.shape) # (批次大小, 序列长度, 隐藏维度) 
```

同样，你可以使用 `jax.grad` 计算相对于参数 (parameter)（`gru_params`）或输入（`input_sequence`）的梯度，从而使得 GRU 单元可以在更大的模型中进行训练。

这个例子说明了 `lax.scan` 如何提供一个有效方法，以一种与 JAX 的编译和自动微分能力良好集成的方式，实现复杂的、有状态的序列计算。通过定义单个步骤的逻辑，并让 `lax.scan` 处理迭代，你可以高效地构建复杂的循环模型。

## 参考资料

- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio (2014)
  Journal: EMNLP 2014; DOI: [10.48550/arXiv.1406.1078](https://doi.org/10.48550/arXiv.1406.1078)
  这篇论文首次提出了门控循环单元（GRU）架构。
- [Deep Learning](http://www.deeplearningbook.org/contents/rnn.html) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 10: Sequence Modeling: Recurrent and Recursive Networks
  为循环神经网络和特定架构（如GRU）提供了全面的理论基础。
- [jax.lax.scan - JAX documentation](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html) — JAX developers (2024)
  提供了`lax.scan`的详细说明和示例，它是JAX中实现循环逻辑的重要工具。
- [JAX Quickstart - JAX documentation](https://jax.readthedocs.io/en/latest/jax-101/index.html) — JAX developers (2024)
  介绍了JAX的核心转换，例如`jit`、`grad`和`vmap`，展示了它们如何实现高效且可微分的数组编程。
