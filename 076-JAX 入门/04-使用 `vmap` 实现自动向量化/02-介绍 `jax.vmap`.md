# 介绍 `jax.vmap`

来源：[原文](https://apxml.com/zh/courses/getting-started-with-jax/chapter-4-automatic-vectorization-vmap/introducing-jax-vmap)

[返回章节目录](README.md) · [返回课程目录](../README.md)

将为单个数据点设计的函数应用于整个批次是一个常见任务。尽管Python循环或NumPy自带的向量 (vector)化可以处理某些情况，但显式管理维度和循环可能会使核心逻辑不清晰，特别是对于更复杂的函数或处理多个批次输入时。

JAX 提供了 `jax.vmap`（“向量化 (quantization)映射”）作为一种专门为此目的构建的函数变换。可以将 `vmap` 视为一种无需重写函数内部逻辑即可自动为其添加批次维度的方式。您编写函数时就像它作用于单个例子一样，而 `vmap` 会将其转换为一个能高效作用于整个批次例子的函数。

让我们看看它的实际应用。假设我们有一个函数，用于计算单个向量中元素的平方和：

```python
import jax
import jax.numpy as jnp

# 专为单个向量输入设计的函数
def sum_of_squares(vector):
  # 此函数期望一个一维数组（向量）
  print(f"Running sum_of_squares for a vector of shape: {vector.shape}")
  return jnp.sum(vector**2)

# 单个向量示例
single_vector = jnp.array([1., 2., 3.])
result_single = sum_of_squares(single_vector)
print(f"Result for single vector: {result_single}")
# 预期输出: Running sum_of_squares for a vector of shape: (3,)
# 预期输出: Result for single vector: 14.0 (1^2 + 2^2 + 3^2)
```

现在，设想我们有一 *批* 这样的向量，可能表示为一个矩阵，其中每一行都是我们希望独立处理的向量：

```python
# 包含 4 个向量的批次，每个向量大小为 3
batch_of_vectors = jnp.array([
    [1., 2., 3.],
    [4., 5., 6.],
    [7., 8., 9.],
    [0., 1., 0.]
])
print(f"Batch shape: {batch_of_vectors.shape}")
# 预期输出: Batch shape: (4, 3)
```

我们如何将 `sum_of_squares` 应用到 `batch_of_vectors` 中的每一行（向量）？如果没有 `vmap`，我们可能会编写一个循环：

```python
# 手动循环方法（演示，不推荐在 JAX 中使用）
manual_results = []
for i in range(batch_of_vectors.shape[0]):
  vector = batch_of_vectors[i]
  result = sum_of_squares(vector) # 函数为每一行运行（并可能进行追踪）
  manual_results.append(result)
manual_output = jnp.stack(manual_results)

print(f"Manual loop output: {manual_output}")
# 预期输出:
# Running sum_of_squares for a vector of shape: (3,)
# Running sum_of_squares for a vector of shape: (3,)
# Running sum_of_squares for a vector of shape: (3,)
# Running sum_of_squares for a vector of shape: (3,)
# Manual loop output: [ 14.  77. 194.   1.]
```

这种方法可行，但冗长，且使用了通常较慢的Python级别迭代，在JAX中，如果在循环内这样重复调用函数，与 `jit` 结合时可能导致低效的追踪（如第2章所述）。

`jax.vmap` 的出场。我们可以通过简单地包装我们的函数来创建其向量化版本：

```python
# 使用 vmap 创建向量化版本
vectorized_sum_of_squares = jax.vmap(sum_of_squares)

# 直接应用于批次
vmap_output = vectorized_sum_of_squares(batch_of_vectors)

print(f"vmap output: {vmap_output}")
# 预期输出:
# Running sum_of_squares for a vector of shape: (3,)  <-- 注意：这只打印了一次！
# vmap output: [ 14.  77. 194.   1.]
```

请注意以下几点：

1. **简洁性：** 我们没有重写 `sum_of_squares`。我们只是简单地将 `jax.vmap` 应用于它。
2. **直接应用：** 我们将整个 `batch_of_vectors`（形状 (4, 3)）传递给了 `vectorized_sum_of_squares` 函数。
3. **正确输出：** 结果 `[ 14. 77. 194. 1.]` 包含了每一行的平方和，与手动循环的结果一致。
4. **效率提示：** `sum_of_squares` 内部的打印语句在 JAX 对 `vmap` 的追踪过程中可能只执行了 *一次*，而不是像 Python 循环那样每行一次。`vmap` 理解批次处理模式并优化执行。

简而言之，`jax.vmap` 接收函数 `sum_of_squares`（它期望输入形状为 `(3,)` 并产生输出形状为 `()`），并将其转换为 `vectorized_sum_of_squares`。这个新函数知道，如果您给它一个形状为 `(4, 3)` 的输入，它应该沿着最前面的轴（轴 0）映射原始函数，从而有效地将其应用于 4 个大小为 3 的向量中的每一个。然后，它将标量结果 `()` 沿着一个新的最前面的轴重新堆叠起来，产生形状为 `(4,)` 的输出。

默认情况下，`vmap` 会对输入参数 (parameter)的第一个轴（轴 0）进行映射，并将结果沿输出的新的第一个轴堆叠。这种默认行为通常正是机器学习 (machine learning)和数值计算中常见批处理场景所需的。

在接下来的部分中，我们将考察如何使用 `in_axes` 和 `out_axes` 定制这种映射行为，以应对涉及多个参数和不同批次维度的更复杂情况。

## 参考资料

- [jax.vmap documentation](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html) — JAX core developers (2024)
  JAX `vmap` 转换的官方文档，详细介绍了其机制和用法。
- [JAX: Composable transformations of Python & NumPy programs](https://github.com/google/jax) — JAX core developers (2024)
  Publisher: JAX-ML Team
  JAX 的官方 GitHub 仓库，概述了其设计原则，包括可组合的函数变换。

---

[上一节](01-%E5%90%91%E9%87%8F%E5%8C%96%E7%9A%84%E5%8E%9F%E7%90%86.md) · [下一节](03-%E5%AF%B9%E7%89%B9%E5%AE%9A%E5%8F%82%E6%95%B0%E8%BF%9B%E8%A1%8C%E6%98%A0%E5%B0%84%EF%BC%88%60in_axes%60%EF%BC%8C%60out_axes%60%EF%BC%89.md)
