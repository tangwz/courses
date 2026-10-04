# 对特定参数进行映射（`in_axes`，`out_axes`）

来源：[原文](https://apxml.com/zh/courses/getting-started-with-jax/chapter-4-automatic-vectorization-vmap/vmap-axes-arguments)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管 `jax.vmap` 自动处理批次维度的添加，但你通常需要更多掌控。如果只有部分参数 (parameter)表示数据批次怎么办？如果你的数据不是方便地沿第一个轴（轴 0）进行批处理怎么办？为了处理这些情况，可以使用 `in_axes` 和 `out_axes` 参数。它们提供了对 `vmap` 如何转换函数输入和输出的精细控制。

### 使用 `in_axes` 指定输入映射

`in_axes` 参数 (parameter)告诉 `vmap` 每个输入参数的 *哪个轴* 应该被映射（向量 (vector)化）。它通常作为元组或列表提供，其长度与被向量化 (quantization)函数的参数数量匹配。

`in_axes` 元组中的每个元素对应一个参数：

- 一个整数 `i` 表示相应参数的第 `i` 个轴是批次维度。`vmap` 将有效地沿此轴对切片进行迭代。
- `None` 表示相应参数*不*应该被映射。相反，整个参数将被广播并在所有向量化调用中重复使用。这对于批次中共享的参数或常量很有用。

让我们看一个例子。假设我们有一个函数，它将一个标量值加到向量的每个元素上：

```python
import jax
import jax.numpy as jnp

def add_scalar(vector, scalar):
  # 将一个标量加到向量的每个元素上
  return vector + scalar

# 示例数据
vectors = jnp.arange(12).reshape(4, 3) # 一个包含4个向量的批次，每个向量大小为3
scalar_val = 100.0                    # 一个单独的标量值
```

如果我们想将 `add_scalar` 应用于 `vectors` 批次中的每个向量，并对每个向量使用*相同*的 `scalar_val`，我们告诉 `vmap` 映射 `vectors` 的轴 0，但不映射 `scalar_val`：

```python
# 映射第一个参数（vectors）的轴 0
# 广播第二个参数（scalar_val）
vectorized_add_scalar = jax.vmap(add_scalar, in_axes=(0, None))

result = vectorized_add_scalar(vectors, scalar_val)

print("输入向量（形状 {}）：\n{}".format(vectors.shape, vectors))
print("输入标量：", scalar_val)
print("结果（形状 {}）：\n{}".format(result.shape, result))
```

```text
输入向量（形状 (4, 3)）：
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]
输入标量： 100.0
结果（形状 (4, 3)）：
[[100. 101. 102.]
 [103. 104. 105.]
 [106. 107. 108.]
 [109. 110. 111.]]
```

正如你所见，`vmap` 应用了 `add_scalar` 四次。在每次应用中，它从 `vectors` 取出一行（轴 0）以及整个 `scalar_val`。输出 `result` 收集这些单独结果，沿轴 0 堆叠，与输入批次维度匹配。

如果我们也有一个标量批次，并想将第 i 个标量加到第 i 个向量怎么办？我们会指定 `in_axes=(0, 0)`：

```python
scalars = jnp.array([100., 200., 300., 400.]) # 一个包含4个标量的批次

# 映射第一个参数（vectors）的轴 0
# 映射第二个参数（scalars）的轴 0
vectorized_add_scalar_batch = jax.vmap(add_scalar, in_axes=(0, 0))

result_batch = vectorized_add_scalar_batch(vectors, scalars)

print("输入向量（形状 {}）：\n{}".format(vectors.shape, vectors))
print("输入标量（形状 {}）：\n{}".format(scalars.shape, scalars))
print("结果（形状 {}）：\n{}".format(result_batch.shape, result_batch))
```

```text
输入向量（形状 (4, 3)）：
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]
输入标量（形状 (4,)）：
[100. 200. 300. 400.]
结果（形状 (4, 3)）：
[[100. 101. 102.]
 [203. 204. 205.]
 [306. 307. 308.]
 [409. 410. 411.]]
```

注意 JAX 自动处理了标量 `scalars[i]` 在每个映射函数调用中对 `vectors[i]` 元素进行广播的情况。

你也可以映射非 0 轴。例如，`in_axes=(1, None)` 会映射第一个参数的轴 1。这要求形状正确对齐 (alignment)。所有映射输入参数的映射轴大小必须相同。如果它们不匹配，JAX 将引发错误。

```python
# 示例：映射向量的轴 1
vectors_T = vectors.T # 形状 (3, 4)
# 映射 vectors_T 的轴 1（列），广播 scalar_val
vectorized_add_scalar_axis1 = jax.vmap(add_scalar, in_axes=(1, None))
result_axis1 = vectorized_add_scalar_axis1(vectors_T, scalar_val)

print("输入 vectors_T（形状 {}）：\n{}".format(vectors_T.shape, vectors_T))
print("输入标量：", scalar_val)
# 默认情况下，输出批次维度（大小为 4）将是轴 0
print("结果（形状 {}）：\n{}".format(result_axis1.shape, result_axis1))
```

```text
输入 vectors_T（形状 (3, 4)）：
[[ 0  3  6  9]
 [ 1  4  7 10]
 [ 2  5  8 11]]
输入标量： 100.0
结果（形状 (4, 3)）：
[[100. 101. 102.]
 [103. 104. 105.]
 [106. 107. 108.]
 [109. 110. 111.]]
```

即使我们映射了输入 `vectors_T` 的轴 1，默认情况下，输出中的结果批次维度仍是轴 0。我们可以使用 `out_axes` 进行控制。

### 使用 `out_axes` 控制输出轴

默认情况下，`vmap` 沿轴 0 堆叠结果。`out_axes` 参数 (parameter)允许你指定输出中哪个轴应与映射维度对应。

让我们考虑一个处理向量 (vector)并返回转换后向量的函数：

```python
def process_vector(v):
  # 示例：将向量加倍
  return v * 2

input_vectors = jnp.arange(12).reshape(4, 3) # 包含 4 个向量的批次
```

使用默认的 `out_axes=0`：

```python
# 默认：将输入轴 0 映射到输出轴 0
vectorized_process_default = jax.vmap(process_vector, in_axes=0, out_axes=0)
result_default = vectorized_process_default(input_vectors)
print("输入向量（形状 {}）：\n{}".format(input_vectors.shape, input_vectors))
print("结果（out_axes=0，形状 {}）：\n{}".format(result_default.shape, result_default))
```

```text
输入向量（形状 (4, 3)）：
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]
结果（out_axes=0，形状 (4, 3)）：
[[ 0  2  4]
 [ 6  8 10]
 [12 14 16]
 [18 20 22]]
```

输出形状是 `(4, 3)`，4 是位于轴 0 的批次维度。

现在，让我们指定 `out_axes=1`：

```python
# 将输入轴 0 映射到输出轴 1
vectorized_process_out1 = jax.vmap(process_vector, in_axes=0, out_axes=1)
result_out1 = vectorized_process_out1(input_vectors)
print("输入向量（形状 {}）：\n{}".format(input_vectors.shape, input_vectors))
print("结果（out_axes=1，形状 {}）：\n{}".format(result_out1.shape, result_out1))
```

```text
输入向量（形状 (4, 3)）：
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]
结果（out_axes=1，形状 (3, 4)）：
[[ 0  6 12 18]
 [ 2  8 14 20]
 [ 4 10 16 22]]
```

输出形状现在是 `(3, 4)`。原始向量维度（大小为 3）现在是轴 0，映射的批次维度（大小为 4）已放置在轴 1。

#### 带多个返回值的 `out_axes`（PyTree）

如果你的函数返回多个值（例如，在元组或字典中，JAX 将其称为 PyTree），`out_axes` 也可以是与输出匹配的 PyTree 结构。这允许你为不同的返回值指定不同的输出轴。

```python
def process_vector_pytree(v):
  # 返回一个包含和与加倍向量的字典
  return {'sum': v.sum(), 'doubled': v * 2}

# 映射输入轴 0。将“和”的批次轴放在 0，将“加倍”的批次轴放在 1。
vectorized_pytree = jax.vmap(
    process_vector_pytree,
    in_axes=0,
    out_axes={'sum': 0, 'doubled': 1}
)

result_pytree = vectorized_pytree(input_vectors)

print("输入向量（形状 {}）：\n{}".format(input_vectors.shape, input_vectors))
print("结果 PyTree：")
print("  和（形状 {}）：\n{}".format(result_pytree['sum'].shape, result_pytree['sum']))
print("  加倍（形状 {}）：\n{}".format(result_pytree['doubled'].shape, result_pytree['doubled']))
```

```text
输入向量（形状 (4, 3)）：
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]
结果 PyTree：
  和（形状 (4,)）：
[ 3. 12. 21. 30.]
  加倍（形状 (3, 4)）：
[[ 0  6 12 18]
 [ 2  8 14 20]
 [ 4 10 16 22]]
```

在这里，和的批次形状为 `(4,)`（批次轴 0），而加倍向量的批次形状为 `(3, 4)`（批次轴 1），完全符合 `out_axes` 中的指定。

### 结合使用 `in_axes` 和 `out_axes`

你经常结合使用 `in_axes` 和 `out_axes` 来精准控制向量 (vector)化过程。这种组合提供了必要的灵活性，可以将期望单个输入的函数适应复杂的批处理场景，而无需重写核心逻辑或诉诸手动维度重排。通过理解如何指定要映射的输入轴以及结果批次维度应该出现在输出的何处，你可以为批处理计算编写更简洁且通常更高效的 JAX 代码。

## 参考资料

- [jax.vmap](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html) — JAX core contributors (2024)
  `jax.vmap`的官方API参考，详细介绍了其参数和行为，包括用于精确控制向量化的`in_axes`和`out_axes`参数。
- [JAX 101: Automatic Vectorization with \`vmap\`](https://jax.readthedocs.io/en/latest/jax-101/03-vectorization.html) — JAX core contributors (2024)
  一份易于理解的官方教程，解释了JAX中自动向量化的原理，并包含实用示例，展示了`in_axes`和`out_axes`在不同批处理场景中的有效用法。
- [Machine Learning with JAX: Learn, Implement, and Deploy Advanced ML Models with JAX](https://learning.oreilly.com/library/view/machine-learning-with/9781098131376/) — Michael H. Lim, Sam Stites, and Ketan Singh (2023)
  Publisher: O'Reilly Media
  本书系统介绍了JAX用于机器学习的核心功能，包括关于`vmap`自动向量化以及`in_axes`和`out_axes`在管理数据流中确切作用的详细章节。

---

[上一节](02-%E4%BB%8B%E7%BB%8D%20%60jax.vmap%60.md) · [下一节](04-%E5%A4%84%E7%90%86%E5%A4%9A%E4%B8%AA%E6%89%B9%E5%A4%84%E7%90%86%E5%8F%82%E6%95%B0.md)
