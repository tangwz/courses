# `vmap`的性能考量

来源：[原文](https://apxml.com/zh/courses/getting-started-with-jax/chapter-4-automatic-vectorization-vmap/vmap-performance)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管 `jax.vmap` 提供了一种方便的函数向量 (vector)化方法，但了解其性能特点有助于您有效使用它。它并非万能，并非总能比其他所有方法都更快，但在某些特定且常见的情况下，它表现出色。`vmap` 对执行速度和内存使用的影响是重要的性能考量。

### `vmap` 的内部工作原理（简述）

与标准 Python `for` 循环每次迭代执行 Python 字节码不同，`vmap` 在执行 *之前* 转换您的函数代码。它本质上将循环逻辑下沉到 JAX 的内部机制中，通常允许 JAX 的底层编译器 XLA（加速线性代数）为整个批处理操作生成高度优化、并行化的代码，尤其是在面向 GPU 或 TPU 时。这避免了重复的 Python 函数调用和解释器交互带来的开销。

### `vmap` 与手动向量 (vector)化

考虑一个您想对数据批次进行逐元素应用的功能简单的函数。您通常有两种选择：

1. **手动向量化 (quantization)：** 使用 `jax.numpy` 操作重新编写您的函数，这些操作能够自然地处理数组（例如，使用 `jnp.add(batch_a, batch_b)` 而不是对标量加法进行循环）。
2. **`vmap`：** 为单个数据点编写函数，并使用 `vmap` 来处理批次维度。

```python
import jax
import jax.numpy as jnp
import timeit

# 示例数据
batch_size = 1000
a = jnp.ones(batch_size)
b = jnp.arange(batch_size, dtype=jnp.float32)

# 1. 手动向量化
def manual_vectorized_add(x_batch, y_batch):
  return jnp.add(x_batch, y_batch) # jnp.add 直接处理数组

# 2. 使用 vmap
def scalar_add(x, y):
  return x + y

vmapped_add = jax.vmap(scalar_add)

# --- 性能比较 ---
# 注意：实际计时结果很大程度上取决于硬件以及 JAX/XLA 版本。
# JIT 编译通常会使这些差异更明显。

# timeit.timeit(lambda: manual_vectorized_add(a, b).block_until_ready(), number=1000)
# timeit.timeit(lambda: vmapped_add(a, b).block_until_ready(), number=1000)
```

对于像加法这样 `jax.numpy` 已经为数组高效实现的简单操作，手动向量化通常与使用 `vmap` 一样快，甚至略快。`vmap` 本身在转换过程中会增加少量开销。

然而，`vmap` 真正的优势在于向量化那些 *不容易* 手动向量化的函数。想象一个包含 Python 控制流（`if`/`else`、对单个元素操作的 `for` 循环）或复杂逻辑的函数，这些逻辑不能直接映射到单个 `jax.numpy` 操作。手动重写此类函数以处理批次可能会很困难且容易出错。`vmap` 自动化了这一过程，通过一次追踪然后生成处理批次维度的代码，来向量化 *整个函数的逻辑*，包括控制流。

### `vmap` 与 Python 循环

原生 Python `for` 循环遍历数据并重复调用 JAX 函数会产生显著开销。每次调用都涉及 Python 解释器开销，并且 JAX 可能无法有效优化跨迭代的计算。

```python
# 3. Python 循环（通常对数值任务效率低下）
def loop_add(x_batch, y_batch):
  results = []
  for i in range(len(x_batch)):
    results.append(scalar_add(x_batch[i], y_batch[i]))
  return jnp.stack(results)

# --- 性能比较 ---
# timeit.timeit(lambda: loop_add(a, b).block_until_ready(), number=100) # 通常慢得多
```

`vmap` 在 JAX 中几乎总能以显著优势胜过显式 Python 循环进行批处理计算。这是因为 `vmap` 允许将计算表示为更大型的、融合的操作，这些操作执行时更接近硬件。

### 与 `jit` 的配合

`vmap` 的性能优势在与 `jax.jit` 结合使用时最明显。当您将 `jit` 应用于 `vmap` 过的函数时，例如 `jax.jit(jax.vmap(my_func))`，JAX 会首先执行 `vmap` 转换，然后 JIT 编译生成的向量 (vector)化函数。

XLA 随后可以对这个更大的批处理计算图进行积极优化。它可以融合操作，针对特定硬件（CPU、GPU、TPU）优化内存访问模式，并高效地在批次维度上并行执行。编译 `vmap` 过的函数可以避免从 Python 启动许多小型、独立计算的开销。

```python
# 结合 vmap 和 jit
jit_vmapped_add = jax.jit(vmapped_add)

# --- 性能比较 ---
# %timeit jit_vmapped_add(a, b).block_until_ready() # 通常比单独使用 vmap 快得多
```

我们来可视化典型的性能关系：



[交互图表：性能：vmap 对比 循环 对比 jit(vmap)](https://apxml.com/zh/courses/getting-started-with-jax/chapter-4-automatic-vectorization-vmap/vmap-performance#plot-fhyh28)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "性能：vmap 对比 循环 对比 jit(vmap)",
    "xaxis": {
      "title": "方法"
    },
    "yaxis": {
      "title": "相对执行时间（越低越好）",
      "type": "log"
    },
    "showlegend": false,
    "margin": {
      "l": 60,
      "r": 20,
      "t": 40,
      "b": 40
    }
  },
  "data": [
    {
      "x": [
        "Python 循环",
        "手动向量化",
        "vmap",
        "jit(vmap)"
      ],
      "y": [
        1000,
        1.1,
        1.5,
        1
      ],
      "type": "bar",
      "marker": {
        "color": [
          "#ff6b6b",
          "#4dabf7",
          "#74c0fc",
          "#1c7ed6"
        ]
      }
    }
  ]
}
```

</details>



> 使用不同方法添加两个向量的相对执行时间。条形越低表示执行越快。请注意对数刻度。实际结果会根据操作复杂性、批次大小和硬件而有所不同。将 `jit` 与 `vmap` 结合通常能获得最佳性能。

### 内存使用考量

尽管 `vmap` 可以显著加快计算速度，但与顺序循环相比，它也可能增加峰值内存使用量。当您应用 `vmap` 过的函数时，JAX 通常需要创建中间数组，这些数组同时容纳 *整个批次* 的结果。

考虑一个函数 `f`，它接受一个向量 (vector)并在最终结果之前生成一个大型中间矩阵。

- **`vmap(f)`：** 如果您将 `vmap(f)` 应用于一批输入向量，JAX 可能会一次性实例化 *所有* 批次元素的大型中间矩阵，需要 `batch_size * memory_per_intermediate_matrix` 的内存。
- **Python 循环：** 顺序调用 `f` 的循环一次只需要一个中间矩阵的内存。

这在 GPU 等内存受限设备上尤其相关。如果您的 `vmap` 过的函数内存不足，应对方法包括：

1. **减小批次大小：** 以更小的块处理数据。
2. **优化函数：** 查看函数本身的中间内存使用量是否可以减少。
3. **手动循环（如有必要）：** 作为最后的手段，如果内存限制非常严格且性能可接受，可以恢复使用 Python 循环（可能对内部函数调用进行 JIT 编译：`jax.jit(f)`）。

### 何时不使用 `vmap`

- **简单、已向量 (vector)化的操作：** 如果您的计算可以使用操作整个数组的标准 `jax.numpy` 函数轻松表达，手动向量化 (quantization)可能稍微简单且性能相当。
- **严苛的内存限制：** 如果整个批次的中间结果超出可用内存，`vmap` 在不调整的情况下可能不可行。
- **固有的顺序依赖：** 如果一个批次项的计算 *依赖于* 前一个项的结果，`vmap` 不适用。对于此类情况，您通常会使用 `jax.lax.scan`。

总而言之，`vmap` 是一个功能强大的工具，用于自动向量化可能复杂的 Python 函数，尤其是在与 `jit` 结合使用时。通过启用底层优化，它通常比 Python 循环提供显著的速度提升。然而，始终要考虑它对内存使用的潜在影响，并使用性能分析工具将其性能与简单情况下的手动向量化进行比较。

## 参考资料

- [jax.vmap - JAX documentation](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html) — JAX core team (2024)
  提供 vmap 自动批处理的官方 API 参考和使用示例，包括其输入/输出映射。
- [jax.jit - JAX documentation](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html) — JAX core team (2024)
  解释了 JAX 中的即时编译机制及其性能优势，尤其是在与 vmap 结合进行整体函数优化时。
- [JAX: Autograd and XLA, Accelerated](https://arxiv.org/abs/1810.08298) — James Bradbury, Roy Frostig, Peter Hawkins, Matthew J. Johnson, Chris Leary, Skye Wanderman-Milne, and Kevin Brooks (2018)
  Journal: arXiv; Publisher: Google Research; DOI: [10.48550/arXiv.1810.08298](https://doi.org/10.48550/arXiv.1810.08298)
  介绍了 JAX 的设计原则，包括自动微分、XLA 即时编译和向量化，它们构成了 vmap 性能特点的基础。
- [XLA Overview](https://openxla.org/) — Google Developers (2024)
  Publisher: OpenXLA Project
  描述了 XLA 作为线性代数领域专用编译器的作用，阐明了它如何优化模型以适应各种硬件加速器，这是 vmap 效率的关键。

---

[上一节](06-%E7%BB%93%E5%90%88%20%60vmap%60%20%E4%B8%8E%20%60jit%60%20%E5%92%8C%20%60grad%60.md) · [下一节](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%87%BD%E6%95%B0%E5%90%91%E9%87%8F%E5%8C%96.md)
