---
course: "getting-started-with-jax"
chapter: "automatic-vectorization-vmap"
lesson: "nesting-vmap"
sourceId: 1756
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-jax/chapter-4-automatic-vectorization-vmap/nesting-vmap"
title: "嵌套 `vmap`"
description: "将 `vmap` 自身进行组合以实现多维度批处理。"
order: 5
plots: []
sourceHash: "ce81510eda62c0bd6e29658ab00290415e6f5242ff4d70cd607028b4b3f40826"
sourceCorrections: []
---

尽管 `vmap` 在添加单个批次维度方面表现出色，但有时您的数据或计算涉及多层批处理或映射。例如，您可能有一批句子，其中每个句子都是一个单词序列（批次），并且您希望对每个词嵌入 (embedding)应用一个函数。又或者您可能需要计算一个批次中每个元素与另一个批次中每个元素之间的对应关系。

JAX 通过*嵌套* `vmap` 优雅地处理这些情况。由于 `vmap` 本身是一个函数转换，您可以多次应用它。将 `vmap` 应用于一个已经经过 `vmap` 转换的函数，可以同时映射多个维度。

### 映射多个批处理输入

让我们重新考虑一个简单函数，例如矩阵-向量 (vector)乘法。假设我们有一个设计用于单个矩阵和单个向量的函数：

```python
import jax
import jax.numpy as jnp

def matvec_multiply(matrix, vector):
  """计算矩阵和向量的乘积。"""
  # 矩阵: [M, N], 向量: [N] -> 结果: [M]
  return jnp.dot(matrix, vector)

# 单个输入示例
matrix = jnp.arange(6.).reshape(2, 3) # 形状 (2, 3)
vector = jnp.arange(3.)             # 形状 (3,)

print("单个矩阵向量乘法:", matvec_multiply(matrix, vector))
# 预期输出: [ 5. 14.] (形状 (2,))
```

现在，设想您有一批矩阵和一个*对应*的向量批次，并且您希望计算每对的乘积。

```python
# 矩阵批次 (3 个矩阵, 每个 2x3)
batch_of_matrices = jnp.arange(18.).reshape(3, 2, 3)
# 向量批次 (3 个向量, 每个 3)
batch_of_vectors = jnp.arange(9.).reshape(3, 3)
```

我们可以通过应用一次 `vmap` 来实现这一点，映射两个输入的第一维（轴 0）：

```python
# 对两个参数都映射轴 0
batched_matvec = jax.vmap(matvec_multiply, in_axes=(0, 0))

print("批处理矩阵向量乘法（成对）:", batched_matvec(batch_of_matrices, batch_of_vectors))
# 预期输出形状: (3, 2) -> 3 个结果, 每个形状 (2,)
```

这之所以可行，是因为 `vmap` 添加了一个批次维度。但是，如果您想计算批次中*每个*矩阵与可能不同批次中*每个*向量的乘积呢？这就像在批次级别计算一个“外积”。

这需要映射矩阵批次，并独立映射向量批次。这就是 `vmap` 嵌套的作用。

```python
# 矩阵批次 (3 个矩阵, 每个 2x3)
batch_of_matrices = jnp.arange(18.).reshape(3, 2, 3)
# 另一个向量批次 (例如, 4 个向量, 每个 3)
batch_of_vectors_outer = jnp.arange(12.).reshape(4, 3)

# 目标: 计算所有 3x4 组合的矩阵向量乘积。结果形状应为 (3, 4, 2)

# 内部 vmap: 映射向量 (轴 0), 保持矩阵固定 (None)
inner_mapped_matvec = jax.vmap(matvec_multiply, in_axes=(None, 0))
# inner_mapped_matvec 接受一个矩阵和一批向量

# 外部 vmap: 将 inner_mapped_matvec 映射到矩阵上 (轴 0)
# inner_mapped_matvec 的第二个参数 (batch_of_vectors_outer) 对于这个外部映射是常量。
# 然而, JAX 通常需要显式处理, 因此我们通过 in_axes 来指定它。
# 由于 batch_of_vectors_outer 已经由内部映射处理,
# 它不会随外部映射的迭代而变化。
# 但最常见和更清晰的方式是直接定义它:

# 映射矩阵 (轴 0), 然后映射向量 (轴 0)
# 从内到外阅读:
# 1. vmap(matvec_multiply, in_axes=(None, 0)):
#    创建一个函数, 它接受一个矩阵和一批向量,
#    将 matvec_multiply 应用于该矩阵和每个向量。
#    一个矩阵和批次中 4 个向量的输出形状: (4, 2)
# 2. vmap(..., in_axes=(0, None)):
#    接受步骤 1 中的函数。将此函数映射到一批矩阵上 (轴 0)。
#    第二个参数 (向量批次) 完整地传递给内部映射函数的每次调用
#    (因此是 None 轴)。
#    批次中 3 个矩阵和批次中 4 个向量的输出形状: (3, 4, 2)
pairwise_matvec = jax.vmap(jax.vmap(matvec_multiply, in_axes=(None, 0)), in_axes=(0, None))

result = pairwise_matvec(batch_of_matrices, batch_of_vectors_outer)

print("成对矩阵向量乘法结果的形状:", result.shape)
# 预期输出: 成对矩阵向量乘法结果的形状: (3, 4, 2)

# 让我们验证一个元素: result[i, j] 应该等于 matvec_multiply(batch_of_matrices[i], batch_of_vectors_outer[j])
manual_calc = matvec_multiply(batch_of_matrices[1], batch_of_vectors_outer[2])
print("针对 (1, 2) 的手动计算:", manual_calc)
print("来自嵌套 vmap 的 (1, 2) 结果:", result[1, 2])
```

在嵌套的 `vmap` 调用 `jax.vmap(jax.vmap(f, in_axes=(ax1, ax2)), in_axes=(ax3, ax4))` 中：

- 内部的 `vmap(f, in_axes=(ax1, ax2))` 创建一个函数，它将 `f` 映射到其输入的 `ax1` 和 `ax2` 轴上。
- 外部的 `vmap(..., in_axes=(ax3, ax4))` 接受这个*新创建的函数*，并将其映射到*其*输入的 `ax3` 和 `ax4` 轴上。
- `in_axes` 中的 `None` 表示相应的参数 (parameter)在该层级上*不*被映射；它在该映射维度上被广播或保持不变。

`vmap(vmap(f, in_axes=(None, 0)), in_axes=(0, None))`（或反之）这种模式在计算两个不同批次元素之间的成对关联时非常常见。

### 示例：成对距离计算

让我们计算集合 `A` 中的每个点与集合 `B` 中的每个点之间的欧几里得距离。

```python
def euclidean_distance_sq(point_a, point_b):
  """计算两点之间的欧几里得距离平方。"""
  # 假设点是 1D 向量
  return jnp.sum((point_a - point_b)**2)

# 批次 A: 3 个 2D 空间中的点
points_a = jnp.array([[1., 0.],
                      [0., 1.],
                      [-1., 0.]]) # 形状 (3, 2)

# 批次 B: 4 个 2D 空间中的点
points_b = jnp.array([[2., 2.],
                      [-2., 2.],
                      [2., -2.],
                      [-2., -2.]]) # 形状 (4, 2)

# 目标: 计算一个 3x4 矩阵, 其中 (i, j) 元素是 distance(points_a[i], points_b[j])

# 内部映射: 计算来自 A 的一个点与 B 中所有点之间的距离
# 映射 points_b (轴 0), 保持 point_a 固定 (None)
inner_map = jax.vmap(euclidean_distance_sq, in_axes=(None, 0))
# inner_map(points_a[0], points_b) 将计算 points_a[0] 到所有 points_b 的距离

# 外部映射: 将 inner_map 应用于 A 中的每个点
# 映射 points_a (轴 0), 将整个 points_b 批次 (None 轴) 传递给内部映射
pairwise_distance_sq = jax.vmap(inner_map, in_axes=(0, None))

# 等效的直接定义:
pairwise_distance_sq_direct = jax.vmap(
    jax.vmap(euclidean_distance_sq, in_axes=(None, 0)), # 对固定的 A 映射 B
    in_axes=(0, None)                                  # 映射 A, 传递完整的 B 批次
)

distance_matrix = pairwise_distance_sq(points_a, points_b)
distance_matrix_direct = pairwise_distance_sq_direct(points_a, points_b)

print("距离矩阵的形状:", distance_matrix.shape)
# 预期输出: 距离矩阵的形状: (3, 4)

print("距离矩阵（平方）:\n", distance_matrix)

# 手动验证一个元素: distance_sq(points_a[0], points_b[0])
manual_dist_sq_00 = euclidean_distance_sq(points_a[0], points_b[0]) # (1-2)^2 + (0-2)^2 = 1 + 4 = 5
print("手动计算的 dist_sq[0, 0]:", manual_dist_sq_00)
print("矩阵元素 [0, 0]:", distance_matrix[0, 0])
# 预期输出: 手动计算的 dist_sq[0, 0]: 5.0, 矩阵元素 [0, 0]: 5.0
```

这种嵌套的 `vmap` 清晰地表达了所需的成对计算，无需手动广播或循环。JAX 高效地编译这种嵌套结构。

### 嵌套与其他转换

嵌套的 `vmap` 可以与其他转换（如 `jit` 和 `grad`）结合使用。您可以对使用嵌套 `vmap` 的函数进行 JIT 编译以提升性能，或通过它们计算梯度。

```python
# JIT 编译成对距离函数
jitted_pairwise_distance_sq = jax.jit(pairwise_distance_sq_direct)

distance_matrix_jitted = jitted_pairwise_distance_sq(points_a, points_b)
print("JIT 编译后的距离矩阵（平方）:\n", distance_matrix_jitted)

# 示例: 成对距离总和相对于 points_a 的梯度
def sum_of_distances(p_a, p_b):
    # 计算所有成对距离
    dist_matrix = pairwise_distance_sq_direct(p_a, p_b)
    # 返回总和
    return jnp.sum(dist_matrix)

# 计算相对于第一个参数 (points_a) 的梯度
grad_sum_dist_wrt_a = jax.grad(sum_of_distances, argnums=0)

gradients_a = grad_sum_dist_wrt_a(points_a, points_b)
print("相对于 points_a 的梯度形状:", gradients_a.shape)
# 预期输出: 相对于 points_a 的梯度形状: (3, 2) (与 points_a 形状相同)
print("相对于 points_a 的梯度:\n", gradients_a)
```

嵌套 `vmap` 提供了一种强大的方法，以函数式和可组合的方式处理复杂的批处理结构。虽然 `in_axes` 的指定起初可能看起来复杂，但理解每个 `vmap` 都引入一个映射层级有助于构建所需的嵌套转换。特别是成对计算的模式，是许多科学计算和机器学习 (machine learning)任务中一个有价值的工具。

## 参考资料

- [Automatic Vectorization with \`vmap\`](https://jax.readthedocs.io/en/latest/notebooks/quick_start.html#automatic-vectorization-with-vmap) — JAX core contributors (2024)
  Publisher: JAX Project
  介绍了 `vmap` 的基本用法，解释了其在批量计算中的应用，并说明了 `in_axes` 参数在处理多维输入（包括嵌套）时的作用。
- [JAX: Composable transformations of Python+NumPy programs](https://dl.acm.org/doi/10.1145/3278972.3278996) — James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Skye Wanderman-Milne, and Adam Paszke (2018)
  Journal: SysML '18: Proceedings of the 1st Systems for Machine Learning Workshop; Publisher: ACM; Pages: 1-16; DOI: [10.1145/3278972.3278996](https://doi.org/10.1145/3278972.3278996)
  阐述了 JAX 的核心设计原则和架构，详细说明了其函数式变换（包括 `vmap`）如何实现高效且可组合的数值计算。
