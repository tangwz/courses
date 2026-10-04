---
course: "advanced-jax"
chapter: "advanced-jax-transformations-control-flow"
lesson: "combining-control-flow-transformations"
sourceId: 2801
sourceUrl: "https://apxml.com/zh/courses/advanced-jax/chapter-1-advanced-jax-transformations-control-flow/combining-control-flow-transformations"
title: "结合控制流与变换"
description: "了解像 grad 和 vmap 这样的变换如何与像 scan、cond 和 while_loop 这样的控制流原语配合使用。"
order: 5
plots: []
sourceHash: "bd807b9484c0d9cff903575480ac8d55f2095de4a6a5098f9a9aefa1b1b0935c"
sourceCorrections: []
---

JAX 的优势不仅在于其独立的变换，例如 `jit`、`grad` 和 `vmap`，更在于它们能够组合使用。为了有效地构建复杂模型，理解函数式控制流原语，例如 `lax.scan`、`lax.cond` 和 `lax.while_loop` 如何与这些核心变换配合使用，具有实际意义。这种可组合性使你能够在统一、可微分和可矢量化 (quantization)的框架内，构建具有循环和条件逻辑等功能的复杂高性能模型。

### 与 `jit` 的配合

使用 `lax.scan`、`lax.cond` 和 `lax.while_loop` 的主要目的在于它们是*专门*设计用于与 `jit` 兼容的。与标准的 Python `for` 循环或 `if` 语句不同，后者如果行为依赖于输入值，可能导致追踪问题或重复编译，而这些 `lax` 原语向 JAX 追踪器提供一个静态结构。

当你对包含这些原语的函数应用 `jit` 时：

1. **追踪：** JAX 追踪函数，包括扫描、循环的循环体以及条件语句的两个分支。它将操作转换为 `jaxpr` 中间表示。
2. **编译：** XLA 获取 `jaxpr` 并将整个计算图（包括控制流逻辑）编译成目标加速器（GPU/TPU/CPU）的优化低级代码（如 HLO）。

这意味着控制流本身会被编译，从而避免了执行期间的 Python 解释器开销。对于 `lax.cond`，XLA 通常会编译*两个*分支。虽然在执行期间只会根据谓词选择一个分支的结果，但编译保证了代码已为两种可能的结果做好准备。类似地，`lax.while_loop` 和 `lax.scan` 的循环体只编译一次，然后编译好的代码会迭代执行。

看一个使用 `lax.cond` 的简单例子：

```python
import jax
import jax.numpy as jnp
import jax.lax as lax

def conditional_computation(x, threshold=5.0):
  """根据 x 的总和应用不同的函数。"""
  total = jnp.sum(x)

  def true_fun(operand):
    # 如果 total >= threshold，则执行此分支
    return operand * 2.0

  def false_fun(operand):
    # 如果 total < threshold，则执行此分支
    return operand / 2.0

  # lax.cond 选择应用于 x 的函数
  return lax.cond(total >= threshold, true_fun, false_fun, x)

# 对函数进行 JIT 编译
jitted_conditional_computation = jax.jit(conditional_computation)

# 示例用法
data1 = jnp.array([1.0, 2.0, 3.0]) # 总和 = 6.0 >= 5.0 -> true_fun
data2 = jnp.array([1.0, 1.0, 1.0]) # 总和 = 3.0 < 5.0 -> false_fun

print("Result 1:", jitted_conditional_computation(data1))
print("Result 2:", jitted_conditional_computation(data2))
# 两个分支都已编译，执行时选择合适的那个。
```

主要的一点是，`lax` 控制流原语允许基于值的动态*运行时*行为，同时为 `jit` 和 XLA 的*编译时*分析提供静态结构。

### 与 `grad` 的配合

自动微分 (`grad`、`vjp`、`jvp`) 自然地与 `lax` 控制流组合。JAX 通过执行的路径追踪操作，以正确计算梯度。

- **`grad` 与 `lax.scan`**：这种组合对于训练循环模型很重要。当你对包含 `lax.scan` 的函数求导时，JAX 会有效地通过扫描执行的序列步骤反向应用链式法则。这类似于传统 RNN 框架中的时间反向传播 (backpropagation) (BPTT)。请记住，为反向传播存储中间激活会占用大量内存，特别是对于长序列。像梯度检查点（稍后介绍）这样的技术可以减轻这个问题。

  ```python
  import jax
  import jax.numpy as jnp
  import jax.lax as lax

  def simple_rnn_step(carry, x_t):
    """一个非常基础的 RNN 步骤。"""
    prev_hidden = carry
    # 简单更新：new_hidden = tanh(W*x_t + U*prev_hidden + b)
    # 为简化起见，这里假设 W、U、b 是固定标量
    W, U, b = 0.5, 0.8, 0.1
    new_hidden = jnp.tanh(W * x_t + U * prev_hidden + b)
    return new_hidden, new_hidden # carry = new_hidden, output = new_hidden

  def run_rnn(initial_hidden, inputs):
    """在输入序列上运行 RNN。"""
    final_hidden, outputs = lax.scan(simple_rnn_step, initial_hidden, inputs)
    return jnp.sum(outputs) # 目标示例：输出之和

  # 计算目标函数对初始隐藏状态的梯度
  grad_run_rnn = jax.grad(run_rnn, argnums=0)

  # 示例数据
  hidden_init = 0.0
  input_sequence = jnp.array([1.0, -0.5, 2.0])

  gradient_wrt_h0 = grad_run_rnn(hidden_init, input_sequence)
  print(f"Gradient w.r.t. initial hidden state: {gradient_wrt_h0}")
  # 输出：对初始隐藏状态的梯度：0.313...
  ```

  这里，`jax.grad` 通过将梯度反向传播通过 `lax.scan` 的步骤，计算最终输出和相对于 `initial_hidden` 状态的变化。
- **`grad` 与 `lax.cond`**：微分通过前向传播期间执行的分支进行。未执行分支中的计算不影响该特定执行中*输出*相对于*输入*的梯度。JAX 正确处理选择过程。如果条件本身可微分地依赖于函数输入，其导数也会被纳入计算。

  ```python
  def conditional_loss(params, x):
    # 条件依赖于输入 x
    pred = jnp.sum(x) > 0

    def loss1(p): # 如果 pred 为 True
        return jnp.sum(p * x)

    def loss2(p): # 如果 pred 为 False
        return jnp.sum(p / (x + 1e-5)) # 避免除以零

    return lax.cond(pred, loss1, loss2, params)

  grad_conditional_loss = jax.grad(conditional_loss)

  params = jnp.array([0.5, -0.5])
  data_pos = jnp.array([1.0, 1.0]) # 总和 > 0 -> loss1
  data_neg = jnp.array([-1.0, -1.0]) # 总和 <= 0 -> loss2

  print("Grad (pos):", grad_conditional_loss(params, data_pos))
  # 输出：梯度（正）：[1. 1.] (梯度来自 loss1: d(p*x)/dp = x)
  print("Grad (neg):", grad_conditional_loss(params, data_neg))
  # 输出：梯度（负）：[-0.99999 -0.99999] (梯度来自 loss2: d(p/x)/dp = 1/x)
  ```
- **`grad` 与 `lax.while_loop`**：与 `lax.scan` 相似，微分会展开前向传播期间执行的循环迭代并应用链式法则。迭代次数可以依赖于输入值。同样，如果循环执行多次迭代，请注意潜在的内存使用。梯度计算会正确考虑循环体内的计算，如果条件函数依赖于被微分的变量，也会考虑其影响。

### 与 `vmap` 的配合

使用 `vmap` 矢量化 (quantization)包含控制流的函数功能强大，但需要仔细考虑。`vmap` 将映射轴*推入*计算内部。

- **`vmap` 与 `lax.scan`**：这是一个常见模式，例如在使用 RNN 同时处理一批序列时。`vmap` 通常会将 `lax.scan` 转换为在 `in_axes` 指定的批次维度上操作。批次中的每个元素都会经历自己独立的扫描。

  ```python
  # 使用来自 grad 示例的 simple_rnn_step 和 run_rnn

  # 在一批初始隐藏状态和输入序列上对 run_rnn 进行矢量化
  # 假设 hidden_init 的形状为 (batch,)，inputs 的形状为 (batch, seq_len)
  # 我们对两个参数都映射轴 0
  batched_run_rnn = jax.vmap(run_rnn, in_axes=(0, 0))

  # 示例批次数据
  batch_size = 4
  seq_len = 3
  batch_hidden_init = jnp.zeros(batch_size)
  batch_input_sequence = jnp.arange(batch_size * seq_len, dtype=jnp.float32).reshape((batch_size, seq_len))

  # 运行批处理 RNN
  batched_output_sum = batched_run_rnn(batch_hidden_init, batch_input_sequence)
  print("Batched RNN output sum shape:", batched_output_sum.shape)
  # 输出：批处理 RNN 输出和的形状：(4,)
  # 每个元素对应于批次中一个序列的输出总和
  ```
- **`vmap` 与 `lax.cond`**：如果条件谓词依赖于映射轴，这种配合可能会更复杂。如果条件对于映射维度上的不同元素求值结果不同，`vmap` 需要处理矢量化计算的不同“通道”执行可能不同的分支。JAX 通过有效地在映射轴上评估*两个*分支，然后根据每个通道对应的谓词值选择适当的结果来实现这一点。这意味着计算成本可能高于所有元素都走相同分支的情况，因为两个路径都会被处理。

  ```python
  # 使用来自 jit 示例的 conditional_computation

  # 对 x 进行矢量化，阈值保持标量
  # 映射 x 的轴 0
  vmapped_conditional = jax.vmap(conditional_computation, in_axes=(0, None))

  # 条件不同的批次数据
  batch_data = jnp.array([
      [1.0, 2.0, 3.0],  # 总和 = 6.0 >= 5.0 -> true_fun
      [1.0, 1.0, 1.0],  # 总和 = 3.0 < 5.0 -> false_fun
      [10.0, 1.0, 1.0], # 总和 = 12.0 >= 5.0 -> true_fun
  ])
  threshold = 5.0

  results = vmapped_conditional(batch_data, threshold)
  print("Vmapped conditional results:\n", results)
  # 输出：
  # 矢量化条件结果：
  # [[ 2.  4.  6.]  <- 应用 true_fun
  #  [ 0.5 0.5 0.5]  <- 应用 false_fun
  #  [20.  2.  2.]] <- 应用 true_fun
  ```

  即使不同行走了不同的路径，`vmap` + `lax.cond` 也处理了这种情况。
- **`vmap` 与 `lax.while_loop`**：矢量化 `while_loop` 的行为类似于 `lax.cond`。如果循环条件或循环体对状态的影响依赖于映射轴，不同的通道可能执行不同次数的迭代。JAX/XLA 会处理这种情况，通常涉及诸如掩蔽非活动通道或让所有通道运行批次中观察到的最大迭代次数的机制。这可能导致在已满足退出条件的通道上进行计算工作，从而影响性能，与所有通道迭代相同次数的情况相比。

### 组合多种变换

你可以自由地组合这些变换。例如，你可以计算使用扫描的矢量化 (quantization)函数的梯度：`jax.grad(jax.vmap(run_rnn))`。或者你可以 JIT 编译带有条件语句的函数的梯度：`jax.jit(jax.grad(conditional_loss))`。JAX 的函数式特性及其原语的设计保证了这些组合是明确的。

```python
# 示例：矢量化条件计算的 JIT 编译梯度
# 使用上面定义的 conditional_computation

# 参数影响阈值的函数
def threshold_from_params(params, x):
    # 阈值 = 参数的平均值
    threshold = jnp.mean(params)
    return conditional_computation(x, threshold)

# 目标：计算批次 x 中对参数的梯度
# 1. 对 x 进行矢量化（轴 0）
# 2. 计算对参数的梯度（参数 0）
# 3. JIT 编译结果
grad_vmap_fn = jax.grad(jax.vmap(threshold_from_params, in_axes=(None, 0)), argnums=0)
jitted_grad_vmap_fn = jax.jit(grad_vmap_fn)

# 示例数据
batch_data = jnp.array([
    [1.0, 2.0, 3.0],  # 总和 = 6.0
    [1.0, 1.0, 1.0],  # 总和 = 3.0
])
params = jnp.array([4.0, 6.0]) # 平均值 = 5.0

gradient = jitted_grad_vmap_fn(params, batch_data)
print("Jitted gradient of vmapped conditional function w.r.t params:\n", gradient)
# 梯度计算依赖于批次中每个项所采取的分支，
# 受参数派生阈值的影响。
```

了解 `jit`、`grad` 和 `vmap` 如何与 `lax` 控制流原语配合使用，对于编写高效且正确的进阶 JAX 代码来说很重要。它使你能够构建需要序列处理、条件逻辑或动态迭代次数的复杂模型，同时仍能从编译、自动微分和矢量化中获益。请注意潜在的性能或内存影响，尤其是在矢量化发散的控制流或对长扫描或循环进行微分时。

## 参考资料

- [JAX Documentation](https://jax.readthedocs.io/en/latest/) — JAX core developers (2023)
  JAX官方文档，涵盖了JAX的核心转换（jit、grad、vmap）和函数式控制流原语（lax.scan、lax.cond、lax.while_loop）。对于理解它们的用法和相互作用至关重要。
- [The JAX Autodiff Cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html) — Google Developers (2023)
  JAX官方文档中的实用指南，通过详细示例说明自动微分概念，包括与函数式控制流的交互。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本关于深度学习的综合教科书，提供了递归神经网络和通过时间反向传播相关的理论基础。
