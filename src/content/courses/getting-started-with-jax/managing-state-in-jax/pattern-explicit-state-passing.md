---
course: "getting-started-with-jax"
chapter: "managing-state-in-jax"
lesson: "pattern-explicit-state-passing"
sourceId: 1805
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-jax/chapter-6-managing-state-in-jax/pattern-explicit-state-passing"
title: "模式：显式状态传递"
description: "通过显式地将状态作为输入传递并接收更新后的状态作为输出来处理状态。"
order: 3
plots: []
sourceHash: "0ac6e35b83c5b60f3e2c248bbdd62a093640fa95d6a59c0e7f067795fdaf738f"
sourceCorrections: []
---

JAX 依赖纯函数运行良好。纯函数在给定相同输入时，总是产生相同的输出，并且没有副作用。副作用包括修改全局变量、向控制台打印信息，或，最相关的是，改变函数作用域外对象的内部状态（例如更新类实例的属性）。

然而，许多有用的计算，特别是在机器学习 (machine learning)中，涉及随时间变化的状态。例如，训练期间更新的模型参数 (parameter)、优化器中的动量值，或循环神经网络 (neural network) (RNN)中的隐藏状态。我们如何调和对状态变化的需求与 JAX 对纯函数的要求呢？

JAX 中处理状态最主要的方式是**显式状态传递**。函数不是在原地修改状态，而是这样编写的：

1. 接受当前状态作为输入参数。
2. 使用此状态和可能的其他输入执行计算。
3. 返回*新的*、已更新的状态作为输出值，通常与任何其他结果一起返回。

这种模式将状态视为通过函数传递的任何其他数据。它避免了副作用，因为函数不修改*原始*状态对象；它产生一个*新*的状态对象。

### 工作原理：一个例子

我们来看一个非常简单的例子：一个计数器。在典型的命令式 Python 中，你可能会使用类来实现它：

```python
# 命令式（有状态对象）方法 - 不适合JAX
class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1
        return self.count

# 用法
counter_obj = Counter()
print(counter_obj.increment()) # 输出: 1
print(counter_obj.increment()) # 输出: 2
print(counter_obj.count)       # 输出: 2
```

这个 `increment` 方法修改了对象的内部 `self.count`。这是一个副作用。如果你尝试直接对这样的方法使用 `jax.jit` 或 `jax.grad`，它不会按预期工作，因为 JAX 转换需要根据函数的输入来追踪其操作，而对外部或内部对象状态的修改对于这个追踪过程是不可见的。

现在，我们使用适合 JAX 的显式状态传递模式来实现计数器：

```python
import jax

# 函数式（显式状态传递）方法 - 适合JAX
def init_counter_state():
  """初始化状态。"""
  return 0 # 状态只是一个整数

def increment(current_state):
  """接受当前状态，返回新状态。"""
  print(f"Running increment function with state: {current_state}") # 用于演示
  new_state = current_state + 1
  # 在更复杂的情况下，我们可能返回 (new_state, result)
  return new_state

# 用法
state = init_counter_state()
print(f"Initial state: {state}")

# 调用 1
state = increment(state)
print(f"State after call 1: {state}")

# 调用 2
state = increment(state)
print(f"State after call 2: {state}")
```

注意主要区别：

- 状态（整数 `count`）被显式地传递*给* `increment` 函数。
- `increment` 函数执行计算 (`current_state + 1`)。
- 它返回状态的*新值*。
- 调用代码负责捕获这个返回的新状态并将其用于下一次调用。

`increment` 函数本身是纯函数。给定输入 `0`，它总是返回 `1`。给定 `1`，它总是返回 `2`。它不修改其局部作用域之外的任何东西。

### JAX 转换的优势

这种显式状态传递模式与 JAX 的函数式特性完美契合，使得有状态的计算能够兼容 `jax.jit` 等转换：

```python
# 将 JIT 应用于函数式计数器
jitted_increment = jax.jit(increment)

state = init_counter_state()
print(f"\nJIT 编译并运行:")

# 第一次调用: 触发 JIT 编译 (并运行 Python 代码)
state = jitted_increment(state)
print(f"State after JIT call 1: {state}")

# 第二次调用: 使用已编译版本 (内部的 Python print 不会执行)
state = jitted_increment(state)
print(f"State after JIT call 2: {state}")

# 第三次调用: 仍使用已编译版本
state = jitted_increment(state)
print(f"State after JIT call 3: {state}")
```

当 `jitted_increment` 首次以特定类型和形状的状态（这里是一个标量整数）被调用时，JAX 会追踪 `increment` 函数，使用 XLA 编译它，并执行。`increment` 内部的 `print` 语句在此初始追踪期间运行。在后续具有兼容输入（相同类型/形状）的调用中，JAX 直接使用缓存的、高度优化的编译代码，跳过 Python 执行（包括 `print`）。状态更新在编译后的计算中正确发生，因为状态的流向（输入 -> 输出）是追踪逻辑的一部分。

下图说明了这种流向：

> 纯函数接受当前状态和任何其他输入，执行计算，并返回更新后的状态以及任何其他结果。

此模式也适用于 `jax.grad`。如果状态更新涉及可微分操作，JAX 可以针对其输入（如果需要，包括状态）对函数进行微分，因为整个数据流是显式的。

"虽然我们的计数器示例使用了一个简单的整数状态，但在许多实际使用中，状态结构通常更复杂，例如包含模型参数 (parameter)、优化器统计数据（均值、方差）等的嵌套字典。JAX 提供了 PyTrees 等工具（将在下一节讨论），以便使用显式传递模式方便地管理这些结构化状态。"

显式状态传递是在 JAX 函数式框架中管理可变过程的根本。通过使状态流显式化，我们保留了使用 JAX 强大函数转换进行加速和微分的能力。

## 参考资料

- [Side effects and mutation](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#side-effects-and-mutation) — JAX core contributors (2024)
  JAX 官方文档的此部分阐明了为何 JAX 函数必须是纯函数，并解释了状态修改等副作用如何与 JAX 转换不兼容。
- [Structure and Interpretation of Computer Programs](https://mitp-content-server.mit.edu/books/content/sicp/full-text/book/book.html) — Harold Abelson, Gerald Jay Sussman, and Julie Sussman (1996)
  Publisher: MIT Press
  这是一本经典的计算机科学教材，介绍了函数式编程的基本概念，包括通过显式传递处理状态，这对理解 JAX 范式至关重要。
