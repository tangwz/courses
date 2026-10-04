---
course: "getting-started-with-jax"
chapter: "managing-state-in-jax"
lesson: "challenge-of-state-functional"
sourceId: 1801
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-jax/chapter-6-managing-state-in-jax/challenge-of-state-functional"
title: "函数式代码中的状态挑战"
description: "为什么在函数式背景下管理变化的状态需要特定的模式。"
order: 2
plots: []
sourceHash: "ac182da0c6a1260ee03c4685f64b5100184a51cc0312033387a3214ff6068482"
sourceCorrections: []
---

JAX 基于纯函数的理念构建，纯函数的输出*仅*取决于其明确的输入，不会引起外部变化或副作用，例如修改全局变量或改变输入对象的状态。这种纯粹性是 `jax.jit` 和 `jax.grad` 等变换运作的根本。它们使用符号输入对函数的执行进行一次追踪，以便了解操作序列，然后可以高效地编译、求导或并行化这些操作。

现在，考虑标准 Python 中常见的编程模式。我们通常通过直接修改对象来管理变化的信息（即状态）。比如使用列表的 `.append()` 方法原地更新列表，修改字典中的值，或修改类实例的属性。

```python
# 标准 Python 状态修改（不适合 JAX）
my_data = {'values': [10, 20]}

def update_data(data_dict, new_value):
  # 修改传入的字典——这是一个副作用
  data_dict['values'].append(new_value)
  data_dict['last_added'] = new_value

update_data(my_data, 30)
print(my_data)
# 输出: {'values': [10, 20, 30], 'last_added': 30}
```

这种命令式风格，以原地修改和隐藏状态变化为特点，虽然在许多场合很自然，但在 JAX 的函数式框架内带来了明显的挑战。为什么会这样？

1. **破坏 `jax.jit`**: 当 `jax.jit` 追踪函数以进行编译时，它基本假定函数行为是纯粹的。如果函数修改了某个外部状态（例如全局变量，或未明确传入和返回的对象属性），追踪将捕获*追踪时的*状态。编译后的版本旨在提高速度，很可能会重用这个初始追踪。它不会自动追踪在函数外部或函数调用之间对该外部状态进行的后续更改。这可能导致结果轻微错误或完全出乎意料，因为编译代码的行为取决于陈旧、隐藏的历史，而非仅仅当前的输入。
2. **使 `jax.grad` 困惑**: 自动微分是 `jax.grad` 背后的引擎，它需要追踪数据在计算中的精确流动以正确计算梯度。原地修改会模糊这种数据流动。如果数组或对象中的值在函数执行中途被改变，`grad` 如何能准确确定最终输出与原始输入之间的数学关系？这实际上破坏了反向模式微分所需的运算链。尝试对具有此类副作用的函数进行微分通常会导致错误，或者更糟的是，导致数学上不正确的梯度，从而悄悄地破坏模型训练等优化过程。
3. **阻碍 `jax.vmap` 和 `jax.pmap`**: 向量 (vector)化 (`jax.vmap`) 和多设备并行化 (`jax.pmap`) 依赖于一个假设，即函数可以有多个实例独立运行，可能对不同数据切片进行操作，或在不同硬件加速器上同时执行。如果这些操作都试图读写*同一个*可变状态对象，就会引入冲突和潜在的竞争条件。这从根本上破坏了安全和正确并行执行所需的独立性假设。每个向量化 (quantization)或并行实例实际上需要自己的隔离状态，而直接修改使其难以可靠地管理。

考虑一个简化场景，尝试 `jit` 一个修改传入对象属性的函数：

```python
import jax
import jax.numpy as jnp

class StateHolder:
    def __init__(self, value):
        self.value = jnp.array(value)

def update_state_impurely(state_obj, increment):
  # 不纯：原地修改对象的属性
  state_obj.value = state_obj.value + increment
  return state_obj.value # 返回新值，但修改已经发生

my_state = StateHolder(10.0)

# 不使用 JIT 时，这通过修改按预期工作
print("Without JIT:")
print(update_state_impurely(my_state, 5.0)) # 输出: 15.0
print(my_state.value)                     # 输出: 15.0
print(update_state_impurely(my_state, 3.0)) # 输出: 18.0
print(my_state.value)                     # 输出: 18.0

# 重置状态并尝试使用 JIT
my_state_jit = StateHolder(10.0)
jitted_update = jax.jit(update_state_impurely)

print("\nWith JIT (potential issues):")
try:
    # 第一次调用触发追踪和编译
    print(jitted_update(my_state_jit, 5.0))
    # 编译后的函数可能作用于*追踪到的*值 (10.0)
    # 修改可能发生在编译函数的作用范围之外，
    # 或者 JAX 可能会就副作用引发错误/警告。
    print(my_state_jit.value) # 状态可能未按预期更新
    print(jitted_update(my_state_jit, 3.0)) # 可能重用陈旧的追踪
    print(my_state_jit.value)
except Exception as e:
    print(f"JIT 尝试遇到问题: {e}")
```

JIT 编译过程可能会使用初始值 (`10.0`) 来追踪函数。编译后的函数可能会根据该追踪固定下来。原地更新 `state_obj.value = ...` 代表了一种 JAX 变换难以处理的副作用。根据具体情况和 JAX 版本，这可能导致编译时出错、关于不纯回调的警告，或在后续调用中编译函数无法反映预期状态更新的静默不正确行为。

因此，JAX 中任何涉及信息随时间或跨计算步骤变化的任务，例如训练期间更新模型参数 (parameter)、管理优化器中的动量值或在循环神经网络 (neural network) (RNN)中传递隐藏状态，都需要不同的编程模式。如果我们希望代码与 JAX 强大的变换 (`jit`, `grad`, `vmap`, `pmap`) 可靠兼容，就不能依赖于原地修改对象或变量。

我们需要能清晰处理状态的模式，使其流转在函数式方法中透明且可控。这意味着需要更新状态的函数通常应将当前状态作为输入，并返回新的、已更新的状态作为输出，而保持原始状态不变。后续章节将介绍 JAX 中状态管理的这些核心函数式模式。

## 参考资料

- [JAX - The JAX core library documentation](https://jax.readthedocs.io/en/latest/) — The JAX Developers (2024)
  Publisher: Google
  JAX官方文档，提供JAX函数式编程模型、纯函数以及副作用对其转换影响的基础信息。
- [Structure and Interpretation of Computer Programs](https://mitp.mit.edu/sites/default/files/sicp/full-text/book/book.html) — Harold Abelson, Gerald Jay Sussman, Julie Sussman (1996)
  Publisher: MIT Press; Pages: 688
  计算机科学的经典著作，详细阐述了函数式编程原则，包括纯函数和状态管理的挑战。
- [Automatic Differentiation in Machine Learning: a Survey](http://jmlr.org/papers/v18/17-468.html) — Atilim Gunes Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, Jeffrey Mark Siskind (2018)
  Journal: Journal of Machine Learning Research; Publisher: Journal of Machine Learning Research; Volume: 18; Pages: 1-43
  自动微分的全面调查，解释了其基本原理以及定义良好的计算图对于正确梯度计算的重要性。
