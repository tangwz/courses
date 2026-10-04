---
course: "getting-started-with-jax"
chapter: "introduction-to-jax"
lesson: "jax-vs-numpy"
sourceId: 1677
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-jax/chapter-1-introduction-to-jax/jax-vs-numpy"
title: "JAX 对比 NumPy"
description: "了解 JAX 的 NumPy API 与标准 NumPy 之间的异同。"
order: 2
plots: []
sourceHash: "0f620af0c2e698599d9db506ac21dbf0727dde83b3c1cd245e0656543b49c0bf"
sourceCorrections: []
---

如果你用过 NumPy，你会发现 JAX 的数组库 `jax.numpy` 用起来非常熟悉。这是特意为之。JAX 旨在提供一个高性能的数值计算环境，让已经熟悉 Python 科学计算核心库 NumPy 的 Python 用户感到得心应手。

通常，`jax.numpy` 会被导入为 `jnp`：

```python
import numpy as np
import jax.numpy as jnp
import jax

# 检查可用设备（CPU 始终存在）
print(f"JAX Devices: {jax.devices()}")

# NumPy 数组创建
np_array = np.array([1.0, 2.0, 3.0])
print(f"NumPy Array: {np_array}, Type: {type(np_array)}")

# JAX 数组创建
jnp_array = jnp.array([1.0, 2.0, 3.0])
print(f"JAX Array: {jnp_array}, Type: {type(jnp_array)}")
```

你会发现许多函数名称和行为都一样。创建数组、进行逐元素操作、计算求和、平均值或标准差，通常都涉及完全相同的函数调用，只是将 `np` 换成了 `jnp`。

```python
# 基本操作看起来相似
a_np = np.arange(6).reshape((2, 3))
b_np = np.array([[1, 1, 1], [2, 2, 2]])
c_np = a_np + b_np * 2

a_jnp = jnp.arange(6).reshape((2, 3))
b_jnp = jnp.array([[1, 1, 1], [2, 2, 2]])
c_jnp = a_jnp + b_jnp * 2 # 与 NumPy 语法相同

print(f"NumPy 结果:\n{c_np}")
print(f"JAX 结果:\n{c_jnp}")

# 检查结果是否接近（比较浮点数结果时有用）
print(f"结果接近: {np.allclose(c_np, c_jnp)}")
```

这种相似性大大降低了使用门槛。你现有的 NumPy 知识大部分可以直接沿用。然而，在这熟悉的外表之下，存在着一些根本区别，它们是理解 JAX 用途和功能的重要部分。

### 区别

尽管 API 力求兼容，但 JAX 在几个重要方面与 NumPy 的运行方式不同：

1. **不可变性：** 这或许是日常编程中最重要的区别。NumPy 数组是*可变*的，这意味着你可以原地修改它们的值。而 JAX 数组是*不可变*的。创建 JAX 数组后，你无法修改它；看似修改数组的操作实际上会返回一个带有更新值的新数组。

   ```python
   np_array = np.array([1, 2, 3])
   np_array[0] = 100 # 这在 NumPy 中运行良好
   print(f"修改后的 NumPy 数组: {np_array}")

   jnp_array = jnp.array([1, 2, 3])
   try:
       # 这将在 JAX 中引发 TypeError
       jnp_array[0] = 100
   except TypeError as e:
       print(f"\n原地修改 JAX 数组出错: {e}")

   # JAX 的方法：创建一个带有更新值的新数组
   # 使用索引更新语法：.at[index].set(value)
   updated_jnp_array = jnp_array.at[0].set(100)
   print(f"原始 JAX 数组（未改变）: {jnp_array}")
   print(f"更新后的 JAX 数组（新对象）: {updated_jnp_array}")
   ```

   不可变性是函数式编程的核心原则，对于 JAX 的函数转换（如 `jit` 和 `grad`）来说，它在不同硬件加速器上正确可靠地运行非常重要。它避免了副作用，使代码更容易理解、并行化和微分。
2. **硬件加速：** 标准 NumPy 操作只在 CPU 上运行。JAX 从设计之初就考虑了在不同类型的硬件加速器上运行，例如图形处理器（GPU）和张量处理器（TPU），以及 CPU。JAX 通常会自动处理设备放置，将计算发送到最快的可用加速器。你通常不需要大幅度修改代码就能获得 GPU/TPU 的加速好处。我们将在本章后面讨论设备管理。
3. **执行模型（惰性求值与编译）：** NumPy 操作是*即时执行*的。当你输入 `c = a + b` 时，加法会立即发生。JAX 操作，尤其是在与 `jax.jit`（即时编译）等转换结合使用时，通常会使用*惰性求值*。JAX 可能会构建一个内部计算图，并只在实际需要结果时（例如，打印或保存时）才执行它。`jax.jit` 会将你的 Python 函数编译成优化过的 XLA（加速线性代数）代码，专门针对可用的硬件（CPU/GPU/TPU）。这个编译步骤在幕后自动发生，是 JAX 在计算密集型任务上比标准 NumPy 具有性能优势的主要原因。我们将在第 2 章详细研究 `jax.jit`。
4. **函数转换：** 最重要的区别不在于 `jax.numpy` API 本身，而在于 JAX 在其周围提供的能力。JAX 提供了可组合的函数转换：

   - `jax.jit`：用于即时（JIT）编译以加速代码。
   - `jax.grad`：用于自动微分（计算梯度）。
   - `jax.vmap`：用于自动向量 (vector)化（将函数映射到数组轴上）。
   - `jax.pmap`：用于跨多个设备的并行化（SPMD 编程）。
     标准 NumPy 没有这些转换的对应功能。它们是 JAX 在机器学习 (machine learning)研究和高性能计算方面如此高效的核心原因。我们将在后续章节中介绍这些转换。
5. **类型提升和精度：** 尽管通常兼容，但在默认数据类型（例如，JAX 启动时默认使用 32 位浮点数，除非另行配置，这可能与 NumPy 的默认 64 位浮点数不同）以及涉及混合类型操作时类型提升的方式上，可能存在细微的差别。当精度很重要时，使用 `dtype` 参数 (parameter)明确指定数据类型是一个好的做法。

   ```python
   # 检查默认浮点类型（可能因 JAX 配置而异）
   np_float = np.array([1.0]).dtype
   jnp_float = jnp.array([1.0]).dtype
   print(f"\n默认 NumPy 浮点类型: {np_float}")
   print(f"默认 JAX 浮点类型: {jnp_float}")

   # 明确设置 dtype
   jnp_float64 = jnp.array([1.0, 2.0], dtype=jnp.float64)
   print(f"具有明确 float64 的 JAX 数组: {jnp_float64.dtype}")
   ```

### 总结表格

这里是一个快速比较：

| 特性 | NumPy (`numpy`) | JAX (`jax.numpy`) |
| --- | --- | --- |
| **API 相似度** | - | 高，模仿 NumPy API |
| **可变性** | 可变（原地修改） | 不可变 |
| **硬件** | CPU | CPU, GPU, TPU |
| **执行模型** | 即时执行 | 惰性（常通过 JIT），通过 XLA 编译 |
| **转换** | 无 | `jit`, `grad`, `vmap`, `pmap` |
| **主要目标** | 通用数值计算 | 高性能，可微分计算 |
| **状态处理** | 允许副作用 | 倾向纯函数（明确传递状态） |

### 何时使用

你不一定需要用 `jax.numpy` 替换所有 NumPy 代码。它们可以共存。

- 对于数据加载、预处理或后处理流程中不需要加速或微分的部分，请使用标准 **NumPy**，特别是如果它们涉及不易用函数式表达的操作或大量依赖原地修改。
- 对于算法中的核心数值计算，请使用 **`jax.numpy`**，特别是那些你打算用 `jit` 加速、用 `grad` 微分、用 `vmap` 向量 (vector)化或用 `pmap` 并行化的部分。

可以将 `jax.numpy` 看作是 JAX 强大编译和转换引擎的类 NumPy 接口。通过理解相同点以及不可变性、执行方式和硬件能力方面的根本区别，你可以有效使用 JAX 来处理要求高的计算任务。

## 参考资料

- [JAX Basics](https://jax.readthedocs.io/en/latest/jax-101/01-jax-basics.html) — JAX core developers (2024)
  Publisher: Google
  权威性地介绍了 JAX，包括其设计理念、`jax.numpy` 数组库、硬件加速能力以及与 NumPy 的基本区别概述。
- [NumPy Documentation](https://numpy.org/doc/stable/) — NumPy community (2024)
  理解标准 NumPy 数组对象（`ndarray`）、其可变行为和受限于 CPU 的执行模型的必备参考资料，为与 JAX 进行比较提供了基础。
- [JAX Array Semantics](https://jax.readthedocs.io/en/latest/jax.html#jax.Array) — JAX core developers (2024)
  详细说明了 JAX 数组和 NumPy 数组之间的特定语义差异，重点在于不变性、使用 `.at` 进行的索引更新以及对有效 JAX 编程至关重要的其他区别。
