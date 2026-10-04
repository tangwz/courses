---
course: "advanced-jax"
chapter: "advanced-jax-transformations-control-flow"
lesson: "looping-lax-while-loop"
sourceId: 2798
sourceUrl: "https://apxml.com/zh/courses/advanced-jax/chapter-1-advanced-jax-transformations-control-flow/looping-lax-while-loop"
title: "使用 lax.while_loop 进行循环"
description: "使用 lax.while_loop 及其相关限制在编译的 JAX 代码中处理动态循环。"
order: 4
plots: []
sourceHash: "c1b6bcd2f074cfad3b1694acd09a5ef4235df9c7f7b1591700bc1d810a5146ec"
sourceCorrections: []
---

尽管 `lax.scan` 擅长执行固定数量的顺序步骤，但许多算法需要循环，直到满足特定条件才停止。标准Python `while` 循环给JAX的编译过程带来了难题。因为JAX在执行*之前*会跟踪函数以生成计算图，所以它需要事先了解计算的结构。如果Python `while` 循环的终止条件依赖于循环内部计算的中间值，则无法直接有效地进行跟踪和JIT编译。

为了在编译代码中处理此类动态循环结构，JAX提供了 `jax.lax.while_loop`。此函数允许你表达迭代次数根据运行时值动态确定的循环，同时仍能进行JIT编译并在加速器上执行。

### `lax.while_loop` 的结构

`lax.while_loop` 函数接受三个主要参数 (parameter)：

1. `cond_fun`: 一个Python可调用对象，接收当前循环状态（“carry”），并返回一个布尔型标量JAX值。只要 `cond_fun` 返回 `True`，循环就继续。
2. `body_fun`: 一个Python可调用对象，定义单次迭代中执行的操作。它接收当前循环状态（carry）作为输入，并且必须返回一个具有相同结构（形状和数据类型）的更新循环状态。
3. `init_val`: 提供给循环第一次迭代的初始状态或“carry”值。

基本形式如下所示：

```python
final_val = jax.lax.while_loop(cond_fun, body_fun, init_val)
```

循环按以下方式进行：

- `cond_fun` 传入 `init_val` 调用。
- 如果为 `True`，则 `body_fun` 传入 `init_val` 调用。它返回 `next_val`。
- `cond_fun` 传入 `next_val` 调用。
- 如果为 `True`，则 `body_fun` 传入 `next_val` 调用。它返回 `another_val`。
- 这会持续直到 `cond_fun` 返回 `False`。
- `body_fun` 返回的最后一个值（即传入 `cond_fun` 并使其返回 `False` 的那个值）是 `lax.while_loop` 返回的最终结果。

### 函数式状态管理

重要的是，像其他JAX变换和控制流原语一样，`lax.while_loop` 以函数式方式运行。循环状态不会原地修改；相反，`body_fun` 必须显式返回下一个迭代的*新*状态。此状态（`init_val` 和 `body_fun` 的后续输出）可以是任何JAX兼容类型，包括标量、数组或数组的pytree（嵌套元组、列表、字典）。状态的结构和类型/形状必须在迭代之间保持一致。

### 示例：寻找最小的2的幂

让我们用一个简单的例子来说明：寻找大于或等于给定数字 `n` 的最小2的幂。

```python
import jax
import jax.numpy as jnp

def find_power_of_two(n):
  # 条件函数：当 current_power < n 时继续
  def cond_fun(loop_state):
    current_power = loop_state
    return current_power < n

  # 循环体函数：将当前幂翻倍
  def body_fun(loop_state):
    current_power = loop_state
    return current_power * 2

  # 初始状态：从幂为1开始
  init_val = 1
  
  # 运行 while 循环
  final_power = jax.lax.while_loop(cond_fun, body_fun, init_val)
  return final_power

# 示例用法
target_number = 100
result = find_power_of_two(target_number)
print(f"最小的2的幂 >= {target_number}: {result}") 
# 输出: 最小的2的幂 >= 100: 128

# 我们可以JIT编译此函数
jit_find_power_of_two = jax.jit(find_power_of_two)
result_jit = jit_find_power_of_two(target_number)
print(f"JIT 结果: {result_jit}")
# 输出: JIT 结果: 128 
```

在此示例中：

- `init_val` 是 `1`。
- `cond_fun` 检查当前幂（`loop_state`）是否小于 `n` (100)。
- `body_fun` 接收当前幂并返回翻倍的值作为新状态。
- 循环运行如下：
  - `cond_fun(1)` -> True。`body_fun(1)` -> 2。
  - `cond_fun(2)` -> True。`body_fun(2)` -> 4。
  - ...
  - `cond_fun(64)` -> True。`body_fun(64)` -> 128。
  - `cond_fun(128)` -> False。循环终止。
- `lax.while_loop` 返回导致 `cond_fun` 返回 `False` 的最后一个传入状态，即 `128`。

### 跟踪和编译

当JAX在一个正在JIT编译的函数内部遇到 `lax.while_loop` 时，它会一次性跟踪 `cond_fun` 和 `body_fun`，以了解它们执行的操作。循环状态（`init_val` 和 `body_fun` 的返回值）的形状和数据类型必须是静态的，并且在跟踪期间可确定。*值*可以动态变化，但*结构*不能。

整个循环随后被编译成一个单一的优化操作（通常是底层XLA表示中的 `while` 操作）。这与Python `while` 循环有本质区别，Python `while` 循环通常会导致JAX在迭代次数固定且在跟踪期间已知时展开循环，或者在条件依赖于跟踪值时编译失败。`lax.while_loop` 明确告诉JAX，这是一个旨在编译的动态循环结构。

### 与 `lax.scan` 的比较

对比 `lax.while_loop` 和 `lax.scan` 会有所帮助：

- **迭代次数：** `lax.scan` 执行*固定*数量的迭代，由输入序列的长度决定。`lax.while_loop` 执行*可变*数量的迭代，直到 `cond_fun` 返回 `False`。
- **输出：** `lax.scan` 通常累积每一步的输出。`lax.while_loop` 只返回循环carry的*最终*状态。
- **用例：** `lax.scan` 非常适合处理序列（如RNN中）或为已知次数重复应用操作。`lax.while_loop` 适合带有收敛准则的迭代算法或运行直到满足条件的模拟。

### 注意事项

- **终止：** 确保你的 `cond_fun` 最终会变为 `False`。`lax.while_loop` 中的无限循环将导致编译后的程序挂起。JAX无法静态验证所有可能循环的终止。
- **性能：** 尽管 `lax.while_loop` 实现了动态迭代计数，但与静态大小的操作或在编译时已知总工作量的 `lax.scan` 相比，迭代次数非常大或高度可变的循环可能会影响性能。循环开销确实存在，但XLA对其优化得很好。
- **求导：** `lax.while_loop` 支持自动求导（`jax.grad`）。对 `while_loop` 进行求导通常涉及在反向传播 (backpropagation)过程中展开循环迭代，这可能会根据执行的迭代次数产生内存影响。这种关联将在第4章中进一步讨论。

`lax.while_loop` 为在JAX中表达复杂的计算增加了一个重要的工具，让你能够处理其流程动态依赖于计算值的算法，同时保持JAX编译和硬件加速的优势。

## 参考资料

- [jax.lax.while_loop](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.while_loop.html) — JAX developers (2024)
  Publisher: Google
  JAX官方文档，提供`jax.lax.while_loop`的API详情和使用示例。
- [Control Flow in JAX](https://docs.jax.dev/en/latest/control-flow.html) — JAX developers (2024)
  Publisher: Google
  JAX控制流指南，解释了`while_loop`等`lax`原语在JIT编译代码中的动机和用法。
- [XLA Overview](https://openxla.org/xla/architecture) — Google Brain Team (2024)
  Publisher: OpenXLA Project
  XLA的文档，JAX使用的线性代数编译器，为JAX高性能执行控制流原语提供了背景信息。
