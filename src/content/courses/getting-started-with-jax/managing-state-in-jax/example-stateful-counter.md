---
course: "getting-started-with-jax"
chapter: "managing-state-in-jax"
lesson: "example-stateful-counter"
sourceId: 1809
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-jax/chapter-6-managing-state-in-jax/example-stateful-counter"
title: "示例：有状态计数器"
description: "一个简单示例，展示计数器函数的状态管理。"
order: 5
plots: []
sourceHash: "202cb17e0c930a235bc47210981e8ff8f9e299e15a0850f4e5c7c43371ca89d7"
sourceCorrections: []
---

显式状态传递模式可以用一个简单的有状态操作来演示：递增计数器。在标准 Python 中，可能会使用一个带有方法来修改内部属性的类。

```python
# 标准 Python（可变状态）- 不兼容 JAX
class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1
        return self.count

counter = Counter()
print(counter.increment()) # 输出: 1
print(counter.increment()) # 输出: 2
```

这种方法依赖于*副作用*（修改 `self.count`）。如前所述，JAX 变换（如 `jit`）与纯函数配合最佳。如果我们尝试直接对 `increment` 方法应用 `jit`，JAX 将无法在编译环境中追踪 `self.count` 在多次调用中的变化，因为这种变化是*原地发生*的。

### 函数式实现

为了使其与 JAX 的函数式方法兼容，我们将该操作重新定义为纯函数。这个函数将*当前状态*（计数）作为参数 (parameter)，并返回*新状态*（递增后的计数）。

```python
import jax
import jax.numpy as jnp

# 状态只是一个整数（或 JAX 标量）
initial_state = 0

# 纯函数：接收状态，返回新状态
def update_counter(current_state):
  """将输入状态递增 1。"""
  print(f"Tracing update_counter with state: {current_state}") # 为追踪演示添加
  new_state = current_state + 1
  return new_state

# 通过函数显式传递状态
state_after_1_update = update_counter(initial_state)
state_after_2_updates = update_counter(state_after_1_update)
state_after_3_updates = update_counter(state_after_2_updates)

print(f"\n初始状态: {initial_state}")
print(f"1 次更新后的状态: {state_after_1_update}")
print(f"2 次更新后的状态: {state_after_2_updates}")
print(f"3 次更新后的状态: {state_after_3_updates}")
```

注意状态是如何显式地通过函数调用传递的。`update_counter` 函数本身不*改变*任何东西；它只是根据其输入计算一个新值。

> 状态通过纯函数 `update_counter` 的流动。一个调用的输出状态成为下一个调用的输入。

### 与 JIT 的兼容性

因为 `update_counter` 是一个纯函数，我们可以安全地对其应用 `jax.jit` 等 JAX 变换。

```python
# JIT 编译纯函数
jitted_update_counter = jax.jit(update_counter)

# 首次调用：JAX 追踪并编译函数
print("\n应用 JIT：")
jitted_state_1 = jitted_update_counter(initial_state)
print(f"1 次 JIT 优化更新后的状态: {jitted_state_1}")

# 第二次调用：使用缓存的编译版本（内部无打印输出）
jitted_state_2 = jitted_update_counter(jitted_state_1)
print(f"2 次 JIT 优化更新后的状态: {jitted_state_2}")

# 第三次调用：也使用编译版本
jitted_state_3 = jitted_update_counter(jitted_state_2)
print(f"3 次 JIT 优化更新后的状态: {jitted_state_3}")
```

你会注意到“Tracing update\_counter...”消息只出现*一次*（或者如果输入类型略有变化，例如从 Python int 变为 JAX 追踪器，可能会出现几次）。后续调用使用 XLA 生成的优化编译代码，这表明 `jit` 成功处理了我们的有状态计算，因为状态是以函数方式管理的。

### 使用 `lax.scan` 进行高效迭代

手动链式调用函数是可行的，但对于多步操作来说，它冗长且效率不高。在 JAX 中处理有状态更新序列，更符合 JAX 习惯且性能更好的方法是使用 `jax.lax.scan`。这个函数本质上是一个为编译而优化的函数式循环结构。

`jax.lax.scan` 反复将一个函数（“主体函数”）应用于一个累积状态。主体函数接收当前的 `carry` 状态和一个可选的输入切片 `x`（来自输入序列 `xs`），并返回 `new_carry` 状态和该步骤的可选输出 `y`。

```python
# 为 scan 定义主体函数
# 接收 (累积状态, 可选输入切片)
# 返回 (新累积状态, 可选的每步输出)
def scan_body(carry_state, _): # 这里没有每步输入，所以使用 _
  """在扫描的每一步应用的函数。"""
  next_state = update_counter(carry_state)
  # 我们不需要每步输出，只需要最终状态
  per_step_output = None
  return next_state, per_step_output

num_steps = 10
initial_scan_state = 0

print(f"\n使用 lax.scan 进行 {num_steps} 步操作：")
# 运行扫描
# scan(f, 初始累积, xs, 长度)
# 这里 xs 为 None，所以我们通过 length 指定步数
final_state, accumulated_outputs = jax.lax.scan(
    scan_body,
    initial_scan_state,
    xs=None,        # 简单计数器不需要输入序列
    length=num_steps
)

print(f"扫描的初始状态: {initial_scan_state}")
print(f"扫描后的最终状态: {final_state}")
# accumulated_outputs 将为 None，因为我们每步返回了 None

# 我们也可以对整个扫描操作进行 JIT 优化以获得最高效率
@jax.jit
def run_scan_jitted(init_state, steps):
  final_st, _ = jax.lax.scan(scan_body, init_state, xs=None, length=steps)
  return final_st

print("\n使用 JIT 优化后的 lax.scan：")
final_state_jitted = run_scan_jitted(initial_scan_state, num_steps)
# 注意：来自 update_counter 的“追踪...”消息可能会在
# 对 run_scan_jitted 进行 jax.jit 追踪时出现，但在执行时不会出现。
print(f"JIT 优化扫描后的最终状态: {final_state_jitted}")
```

结合 `jit` 使用 `lax.scan`，JAX 能够将整个循环编译成一个单一的优化内核，这比执行一个由 JIT 优化函数组成的 Python 循环快得多。

### 扩展到复杂状态

尽管这个计数器使用了简单的整数状态，但完全相同的模式适用于更复杂的状态，例如用于模型参数 (parameter)或优化器统计数据的嵌套字典或 JAX 数组列表。只要你的更新函数将整个状态结构（作为 PyTree）作为输入并返回整个更新后的状态结构，JAX 变换就会正确处理它。接下来当我们查看优化器状态管理时，我们将看到它如何发挥作用。

这个有状态计数器示例展示了 JAX 中状态管理的核心函数模式：将状态视为一个不可变的值，显式地传入和传出纯函数。这种方法确保了与 `jit` 等 JAX 变换的兼容性，并使得 `lax.scan` 等高效执行模式成为可能。

## 参考资料

- [JAX Documentation: The JAX Way - Pure Functions](https://jax.readthedocs.io/en/latest/the_sharp_bits.html#pure-functions) — JAX Contributors (2024)
  解释了JAX对纯函数和不可变数据结构的依赖，这是显式状态管理和启用JAX转换的基础。
- [JAX Documentation: jax.lax.scan](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html) — JAX Contributors (2024)
  提供了`jax.lax.scan`的详细文档和示例，这是编写高效、JAX兼容的迭代和状态操作循环的重要函数。
- [Functional Python Programming](https://www.packtpub.com/product/functional-python-programming-third-edition/9781803232577) — Steven F. Lott (2022)
  Publisher: Packt Publishing; Pages: Chapter 1: Understanding Functional Programming
  为Python中的函数式编程概念提供了实用介绍，强调了纯函数和不可变状态对于可维护和可测试代码的益处。
