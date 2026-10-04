---
course: "getting-started-with-jax"
chapter: "parallelization-across-devices-pmap"
lesson: "pmap-axes-devices"
sourceId: 1777
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-jax/chapter-5-parallelization-across-devices-pmap/pmap-axes-devices"
title: "将数据映射到设备 (`in_axes`, `out_axes`)"
description: "指定输入数据如何分发以及输出数据如何在设备间收集。"
order: 3
plots: []
sourceHash: "0b38a345981c802d2b76118952faff9ba8c8d2588d919f8a7782288451083916"
sourceCorrections: []
---

`jax.pmap` 函数使用单一程序多数据（SPMD）模型运行。这意味着*相同*的 Python 函数代码在多个设备（如 GPU 或 TPU 核心）上执行，但每个设备通常处理输入数据的不同部分。在这个模型中，一个主要挑战是如何指定*哪些*数据去到*哪个*设备，以及如何将每个设备的结果组合起来。`pmap` 的 `in_axes` 和 `out_axes` 参数 (parameter)正是为了解决这些问题而设计的。

可以将 `in_axes` 和 `out_axes` 看作是 JAX 处理 SPMD 模型中“多数据”部分的指令。它们指定了输入数组应沿哪个轴进行拆分，以及输出数组应沿哪个轴进行堆叠。

### 使用 `in_axes` 分发输入

`in_axes` 参数 (parameter)告诉 `pmap` 如何在可用设备上分发函数的输入参数。它为每个参数指定了应拆分（或映射）哪个轴。

- **规定：** `in_axes` 通常是一个整数、`None`，或是一个（可能嵌套的）结构（如元组、列表或字典），其结构与函数参数的结构（一个 PyTree）相匹配。
- **整数值（例如 `0`）：** 如果特定参数的 `in_axes` 是 `0`，这意味着该 NumPy 或 JAX 数组的第一个轴（轴 0）将在设备间进行拆分。如果您有 $N$ 个设备，形状为 `(B, ...)` 的数组将被切分为 $N$ 个块，每个块的形状为 `(B/N, ...)`，每个块发送到一个不同的设备。这是分发批量数据最常见的情形。如果您指定 `1`，则第二个轴将被拆分，以此类推。
- **`None` 值：** 如果参数的 `in_axes` 是 `None`，则整个参数会被复制并提供给 *每个* 设备上的函数。这通常用于需要在所有并行计算中保持相同的数据，例如模型参数或共享配置值。
- **PyTree 结构：** 如果您的函数接受多个参数，`in_axes` 应该是一个元组（或列表、字典），其结构与参数相对应。例如，如果您的函数是 `def my_func(x, y): ...`，您可以使用 `in_axes=(0, None)`。这将沿着第一个参数 `x` 的第一个轴进行拆分，但将第二个参数 `y` 复制到所有设备上。

让我们通过 `in_axes=0` 可视化将形状为 `(8, 100)` 的数组 `data` 分割到 4 个设备上的情形：

> 使用 `in_axes=0` 在 4 个设备间分发数据。输入数组的第一个轴被平均拆分。

如果一个参数没有指定的映射轴（例如，对于 0 维标量，`in_axes=0`），或者映射轴的大小不能被设备数量整除，JAX 将会引发错误。

### 使用 `out_axes` 收集输出

正如 `in_axes` 控制输入如何分发一样，`out_axes` 控制函数从每个设备返回的结果如何组合回主机上的单个输出值。

- **规定：** 与 `in_axes` 类似，`out_axes` 通常是一个整数、`None`，或是一个与函数返回值结构相匹配的 PyTree 结构。
- **整数值（例如 `0`）：** 如果 `out_axes` 是 `0`，则来自每个设备的输出将沿着一个新的轴 0 堆叠在一起。如果每个设备生成了一个形状为 `(S, ...)` 的数组，最终组合结果的形状将是 `(N, S, ...)`，其中 $N$ 是设备的数量。通常，如果输入是沿着轴 0 拆分的 (`in_axes=0`)，您会希望沿着轴 0 堆叠输出 (`out_axes=0`) 以重建完整的批量维度。
- **`None` 值：** 如果 `out_axes` 是 `None`，JAX 假定输出值在所有设备上是相同的。当函数计算的值已经通过设备间聚合（例如，使用像 `lax.psum` 或 `lax.pmean` 这样的集体操作，我们将在接下来介绍）时，这种情况很常见。在这种情况下，JAX 只返回第一个设备的输出。
- **PyTree 结构：** 如果函数返回多个值（例如 `return loss, accuracy`），`out_axes` 应该是一个元组（或列表、字典），指定如何处理每个返回的元素，例如 `out_axes=(0, None)`。

继续前面的例子，假设每个设备上的函数处理其 `(2, 100)` 切片并返回形状为 `(2, 50)` 的结果。使用 `out_axes=0` 时：

> 使用 `out_axes=0` 收集输出。来自每个设备的结果沿第一个轴堆叠。

### 用法示例

让我们考虑一个简单的函数，以及 `in_axes` 和 `out_axes` 如何控制 `pmap`。假设我们有 2 个可用设备。

```python
import jax
import jax.numpy as jnp
import numpy as np

# 假设本例中有 2 个设备可用
# 您可以使用 jax.local_device_count() 查看可用设备

# 示例函数：将输入 x 按标量因子 'k' 进行缩放
def scale(x, k):
  return x * k

# 输入数据：4 个条目，特征大小为 3
data = jnp.arange(12, dtype=jnp.float32).reshape((4, 3))
# 标量因子
scalar = jnp.float32(10.0)

# 对函数进行 pmap
# 沿轴 0 拆分 'data' (x)
# 在两个设备上复制 'scalar' (k)
# 沿轴 0 堆叠结果
pmapped_scale = jax.pmap(scale, in_axes=(0, None), out_axes=0)

# 执行 pmap 函数
result = pmapped_scale(data, scalar)

print("设备数量:", jax.local_device_count()) # 示例输出: 2
print("原始数据形状:", data.shape)
print("标量值:", scalar)
print("pmap 结果形状:", result.shape)
print("pmap 结果:\n", result)

# 预期输出（如果在 2 个设备上运行）：
# Number of devices: 2
# Original data shape: (4, 3)
# Scalar value: 10.0
# Pmapped result shape: (4, 3)
# Pmapped result:
#  [[  0.  10.  20.]
#   [ 30.  40.  50.]
#   [ 60.  70.  80.]
#   [ 90. 100. 110.]]
```

在此示例中：

1. `data`（形状 `(4, 3)`）的 `in_axes=0`。在 2 个设备上，设备 0 获得 `data[0:2, :]`（形状 `(2, 3)`），设备 1 获得 `data[2:4, :]`（形状 `(2, 3)`）。
2. `scalar`（形状 `()`）的 `in_axes=None`。两个设备都接收值 `10.0`。
3. `scale` 函数在每个设备上运行，使用其 `x` 的切片和复制的 `k`。设备 0 计算 `data[0:2, :] * 10.0`，设备 1 计算 `data[2:4, :] * 10.0`。每个都产生形状为 `(2, 3)` 的结果。
4. `out_axes=0` 指示 JAX 沿第一个轴堆叠这些 `(2, 3)` 结果，从而得到最终的 `(4, 3)` 结果。

掌握 `in_axes` 和 `out_axes` 对于控制 `pmap` 中的数据流非常重要。通过正确指定数据应如何分发以及结果如何收集，您可以有效使用多个设备进行并行计算，特别是对于机器学习 (machine learning)中常见的数据并行模式。在下一节中，我们将介绍集体操作，这些操作使得在 `pmap` 计算中设备之间能够进行通信。

## 参考资料

- [\`jax.pmap\` documentation](https://jax.readthedocs.io/en/latest/jax.html#jax.pmap) — The JAX Authors (2024)
  `jax.pmap` 的官方 API 参考文档，详细说明了其参数，如 `in_axes` 和 `out_axes`。
- [Accelerated ML with JAX](https://www.packtpub.com/product/accelerated-ml-with-jax/9781803233869) — Yury Vasilev, Rohit Singh (2023)
  Publisher: Packt Publishing
  一本关于 JAX 的实用指南，涵盖其核心功能，包括使用 `pmap` 的并行编程概念。
