---
course: "advanced-jax"
chapter: "advanced-jax-transformations-control-flow"
lesson: "mastering-lax-scan"
sourceId: 2793
sourceUrl: "https://apxml.com/zh/courses/advanced-jax/chapter-1-advanced-jax-transformations-control-flow/mastering-lax-scan"
title: "精通 lax.scan 处理序列操作"
description: "使用 lax.scan 高效实现循环计算和序列处理，并了解它与其他转换的协同作用。"
order: 2
plots: []
sourceHash: "0bf895adb209361564fdb9f8f11e17fcdcca3ddc8f2f3bf762cf6f23b0be0b1f"
sourceCorrections: []
---

虽然 `jit`、`grad` 和 `vmap` 为加速和转换数值函数提供了有效的工具，但它们主要一次性处理整个数组或批次。许多重要算法，尤其是在诸如序列建模、信号处理和优化等方面，包含序列依赖性，一步的结果依赖于前一步的输出。在 JAX 中高效实现这些需要一个专用工具：`jax.lax.scan`。

可以想象处理时间序列、逐步模拟物理系统，或者实现循环神经网络 (neural network)（RNN）的前向传播。一个简单的 Python `for` 循环迭代这些步骤在纯 Python 中可行，但它对 JAX 的编译模型提出了挑战。当 JAX 跟踪一个包含 Python 循环且其长度取决于运行时值的函数时，它在编译时通常需要“展开”循环。这意味着在编译图中为每次迭代复制循环体的操作。对于长序列，这种展开会导致非常大的计算图，显著增加编译时间并可能超出内存限制。

`lax.scan` 提供了一个函数式替代方案，专门为这些情况设计。它允许你表达序列计算，JAX 可以将其编译成在 GPU 和 TPU 等加速器上的高效循环原语，避免了在 `jit` 编译函数中使用显式 Python 循环的弊端。

### 核心思想：传递值、输入和输出

其核心是，`lax.scan` 通过重复应用你定义的函数来工作。这个函数在每一步接收两个参数 (parameter)：

1. 当前的 `carry` 状态：这保存着从上一步传递到当前步的信息。这是在整个序列处理过程中保持状态的方式。
2. 来自输入序列 `xs` 的当前切片 `x`：这是当前步骤的特定输入（可选）。

该函数必须返回一个包含以下内容的元组：

1. 更新后的 `carry` 状态：这将传递给下一步。
2. 每一步的输出 `y`：这会在所有步骤中收集，形成最终的输出序列。

`lax.scan` 的整体签名是：

```python
jax.lax.scan(f, init, xs, length=None, reverse=False, unroll=1)
```

让我们分解一下主要参数：

- `f`：要重复应用的函数。它必须具有签名 `f(carry, x) -> (new_carry, y)`。如果 `xs` 为 `None`，则签名为 `f(carry, None) -> (new_carry, y)` 或者甚至 `f(carry) -> (new_carry, y)`（隐式接受 `None`）。
- `init`：第一步之前的 `carry` 状态的初始值。其结构必须与 `f` 输出的 `carry` 部分匹配。
- `xs`：一个可选的 PyTree（例如，数组、数组的元组/列表/字典），表示输入序列。`lax.scan` 遍历 `xs` 中数组的主轴。传递给 `f` 的每个切片 `x` 将具有与 `xs` 匹配的结构。如果 `xs` 为 `None`，`scan` 将迭代 `length` 次，不带每步输入。
- `length`：可选整数，指定步数。通常从 `xs` 的主轴维度推断。如果 `xs` 为 `None`，则必需。
- `reverse`：可选布尔值。如果为 `True`，则从 `xs` 的末尾扫描到开头。默认为 `False`。
- `unroll`：可选整数。用于性能调优，向编译器建议要展开多少次循环迭代。默认为 1。更大的值有时可以通过减少循环开销来提高某些硬件的性能，但会增加编译时间和代码大小。请谨慎使用并进行性能分析。

`lax.scan` 返回一个元组 `(final_carry, ys)`，`final_carry` 是最后一步后的 `carry` 状态，而 `ys` 是一个 PyTree，包含每一步堆叠的输出 `y`。`ys` 的结构与 `f` 输出的 `y` 部分匹配，但增加了一个与序列长度对应的主维度。

### 简单示例：累加和

让我们看看如何实现累加和，其中每个元素是输入数组中所有前面元素（包括其自身）的总和。

```python
import jax
import jax.numpy as jnp
import jax.lax as lax

# 定义扫描函数：f(carry, x) -> (new_carry, y)
# carry: 到前一个元素的总和
# x: 当前元素
# new_carry: 包含当前元素的总和 (carry + x)
# y: 此步骤的输出 (也是 carry + x)
def cumulative_sum_step(carry_sum, current_x):
  new_sum = carry_sum + current_x
  return new_sum, new_sum # 返回新的 carry 和此步骤的输出

# 输入数组
input_array = jnp.array([1, 2, 3, 4, 5])

# 初始 carry 为 0（第一个元素之前的和）
initial_carry = 0

# 应用 lax.scan
final_carry, result_sequence = lax.scan(cumulative_sum_step, initial_carry, input_array)

print("输入数组:", input_array)
print("初始 Carry:", initial_carry)
print("最终 Carry（总和）:", final_carry)
print("结果序列（累加和）:", result_sequence)
# 预期输出：
# 输入数组: [1 2 3 4 5]
# 初始 Carry: 0
# 最终 Carry（总和）: 15
# 结果序列（累加和）: [ 1  3  6 10 15]
```

在这个示例中：

- `initial_carry` 从 `0` 开始。
- **第1步：** `f(0, 1)` 返回 `(1, 1)`。`carry` 变为 `1`，`ys` 开始收集 `[1]`。
- **第2步：** `f(1, 2)` 返回 `(3, 3)`。`carry` 变为 `3`，`ys` 是 `[1, 3]`。
- **第3步：** `f(3, 3)` 返回 `(6, 6)`。`carry` 变为 `6`，`ys` 是 `[1, 3, 6]`。
- **第4步：** `f(6, 4)` 返回 `(10, 10)`。`carry` 变为 `10`，`ys` 是 `[1, 3, 6, 10]`。
- **第5步：** `f(10, 5)` 返回 `(15, 15)`。`carry` 变为 `15`，`ys` 是 `[1, 3, 6, 10, 15]`。
- `lax.scan` 返回 `final_carry` (`15`) 和收集到的 `ys` 序列 (`[1, 3, 6, 10, 15]`)。

### 应用：基本循环神经网络 (neural network)（RNN）单元

一个更实际的应用是实现一个 RNN。一个简单的 RNN 根据前一个隐藏状态 $h_{t-1}$ 和当前输入 $x_t$ 来更新其隐藏状态 $h_t$：


$$
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
$$


我们可以使用 `lax.scan` 对输入序列迭代执行此更新步骤。`carry` 将是隐藏状态 $h$，而 `xs` 将是输入序列 $x$。

```python
import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as random

key = random.PRNGKey(0)

# 定义维度
input_features = 3
hidden_features = 5
sequence_length = 4

# 初始化参数（作为 carry 的一部分或闭包）
key, w_key, b_key = random.split(key, 3)
W_hh = random.normal(w_key, (hidden_features, hidden_features)) * 0.1
W_xh = random.normal(w_key, (hidden_features, input_features)) * 0.1
b_h = random.normal(b_key, (hidden_features,)) * 0.1

# 生成一个虚拟输入序列 (sequence_length, input_features)
x_key = random.split(key)
input_sequence = random.normal(x_key, (sequence_length, input_features))

# 初始隐藏状态
h_initial = jnp.zeros((hidden_features,))

# 定义 RNN 步骤函数：f(h_prev, x_t) -> (h_new, h_new)
# 我们也将 h_new 作为每步结果 'y' 输出
def rnn_step(h_prev, x_t):
  h_new = jnp.tanh(jnp.dot(W_hh, h_prev) + jnp.dot(W_xh, x_t) + b_h)
  # 返回新状态，作为下一个 carry 和此步骤的输出
  return h_new, h_new

# 应用 lax.scan
final_hidden_state, hidden_states_sequence = lax.scan(rnn_step, h_initial, input_sequence)

print("输入序列形状:", input_sequence.shape)
print("初始隐藏状态形状:", h_initial.shape)
print("最终隐藏状态形状:", final_hidden_state.shape)
print("隐藏状态序列形状:", hidden_states_sequence.shape)

# 预期输出形状：
# 输入序列形状: (4, 3)
# 初始隐藏状态形状: (5,)
# 最终隐藏状态形状: (5,)
# 隐藏状态序列形状: (4, 5)
```

这里，`rnn_step` 封装了核心的循环关系。`lax.scan` 从 `h_initial` 开始，高效地将此函数应用于 `input_sequence`。`carry` 完美地表示了演变的隐藏状态 $h_t$，而收集到的 `ys`（这里是 `hidden_states_sequence`）则提供了每个时间步的隐藏状态。

请注意，参数 (parameter) `W_hh`、`W_xh` 和 `b_h` 是在 `rnn_step` 外部定义的。JAX 在跟踪和编译时会自动处理这些闭包。

### `lax.scan` 为何表现出色：性能与编译

`lax.scan` 相较于 `@jit` 内部的 Python `for` 循环的主要优势在于性能，尤其是在加速器上。

1. **避免展开：** JAX 将 `lax.scan` 翻译为专门的 XLA HLO（高级优化器）循环操作（例如 `While`）。XLA 随后可以非常高效地为目标硬件（GPU/TPU）编译这种循环表示，而无需为每次迭代复制计算图。这使得编译时间可控，并减小了编译后的代码大小。
2. **内存效率：** 展开一个长 Python 循环可能会消耗大量内存来存储计算图。`lax.scan` 的编译表示通常紧凑得多。此外，在执行过程中，XLA 可以优化循环内的内存使用，可能比完全展开的图更有效地重用缓冲区。
3. **硬件优化：** XLA 旨在优化并行硬件的循环。它可以对 `scan` 表示的循环体和控制流应用复杂的分析和转换，从而加快执行速度。

### `lax.scan` 与其他转换

`lax.scan` 旨在与其他 JAX 转换结合使用：

- **`jit`：** 如前所述，`lax.scan` 非常适合在 `@jit` 内部使用。它提供了 XLA 高效编译所需的结构。
- **`grad`：** JAX 可以自动对 `lax.scan` 进行求导。它计算相对于 `init` 状态、由 `f` 闭包捕获的参数 (parameter)以及输入序列 `xs` 的梯度。这对于训练 RNN 和其他序列模型非常重要。我们将在第 4 章中更详细地了解控制流的求导。
- **`vmap`：** 你可以使用 `vmap` 并行运行多个独立的扫描，例如，同时处理一批序列。如果你有一个形状为 `(batch_size, sequence_length, input_features)` 的 `batch_input_sequences`，你可以用 `vmap` 封装 `lax.scan`：`jax.vmap(lambda seq: lax.scan(rnn_step, h_initial, seq))(batch_input_sequences)`。（注意：正确处理批处理初始状态可能还需要对 `h_initial` 进行 `vmap` 操作）。

### 不带输入序列的扫描（`xs=None`）

有时，你可能希望仅基于 `carry` 状态来迭代函数，而不必在每一步消耗外部输入序列。这对于生成序列或运行固定步数的迭代过程很有用。你可以通过传递 `xs=None` 并指定 `length` 参数 (parameter)来实现这一点。

```python
import jax
import jax.numpy as jnp
import jax.lax as lax

# 生成前 10 个 2 的幂：1, 2, 4, ...
# carry: 前一个 2 的幂
# x: 为 None，被忽略
# new_carry: 下一个 2 的幂 (carry * 2)
# y: 当前 2 的幂 (carry)

def generate_powers_of_2(carry, _): # 使用 _ 表示 x 未被使用
  next_val = carry * 2
  return next_val, carry # 返回下一个 carry，输出当前值

initial_value = 1
num_steps = 10

final_val, powers_of_2 = lax.scan(generate_powers_of_2,
                                  initial_value,
                                  xs=None, # 无输入序列
                                  length=num_steps)

print("最终值 (2^10):", final_val)
print("生成的 2 的幂:", powers_of_2)
# 预期输出：
# 最终值 (2^10): 1024
# 生成的 2 的幂: [  1   2   4   8  16  32  64 128 256 512]
```

总而言之，`lax.scan` 是你高级 JAX 工具包中不可或缺的工具。它提供了表达复杂序列和循环计算的方式，以一种在用 `jit` 编译时既函数式优雅又高性能的方式，为在加速硬件上实现许多复杂模型和算法提供了方法。

## 参考资料

- [jax.lax.scan](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html) — JAX developers (2024)
  提供 jax.lax.scan 原语的官方描述和用法。
- [XLA Overview](https://www.tensorflow.org/xla) — Google Developers (2024)
  Publisher: Google
  概述了 XLA 作为线性代数领域特定编译器的作用，解释了其如何优化各种硬件的计算，这与 lax.scan 的性能优势相关。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: 365-400 (Chapter 10)
  第10章讨论了序列建模和循环神经网络，为 lax.scan 旨在解决的问题类型提供了理论背景。
