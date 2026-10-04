# 集合通信原语 (psum, pmean 等)

来源：[原文](https://apxml.com/zh/courses/advanced-jax/chapter-3-distributed-computing-jax/collective-communication-primitives)

[返回章节目录](README.md) · [返回课程目录](../README.md)

`jax.pmap` 用于 JAX 中的分布式计算，它允许每个设备在其本地数据切片上执行相同的程序，遵循单程序多数据 (SPMD) 模型。分布式算法通常需要设备间进行通信和同步。比如，在数据并行训练中，每个设备会根据其本地数据批次计算梯度，但这些梯度需要在更新模型参数 (parameter)之前，在所有设备上进行汇总（通常是求平均）。此时，集合通信操作就变得非常必要了。

JAX 中的集合操作 (`jax.lax` 集合) 使得分散在多个设备上的数组（沿 `pmap` 定义的映射轴）能够参与到联合计算中。集合操作的结果通常会复制回所有参与的设备。这些操作必须在被 `pmap` 转换的函数 *内部* 调用。

### 使用 `axis_name` 指定通信轴

在 `pmap` 中使用集合操作的一个基本点是，要指定通信应沿着 *哪个* 轴进行。由于 `pmap` 本身会创建一个表示设备的新映射轴，你需要告诉集合操作使用这个特定的轴。这通过 `axis_name` 参数 (parameter)完成，该参数必须与外部 `pmap` 函数提供的 `axis_name` 匹配。这种明确的命名方式避免了歧义，尤其是在处理嵌套的 `pmap` 调用或其他复杂的转换时。

### 核心归约集合操作：`psum`、`pmean`、`pmax`、`pmin`

最常见的集合操作执行跨设备的归约运算。

#### `jax.lax.psum` (并行求和)

`jax.lax.psum` 计算参与映射轴的所有设备上数组的元素级总和。每个设备贡献其本地版本的数组，并且每个设备都接收到包含总和的相同结果数组。

假设有四个设备，每个设备持有一个标量值：

> 在命名轴 'devices' 上，使用 `psum` 对四个设备进行汇总。每个设备都从一个本地值开始，在执行 `psum` 后，每个设备都持有总和 (2 + 3 + 1 + 4 = 10)。

以下是使用 `psum` 定义一个与 `pmap` 配合使用的函数的例子：

```python
import jax
import jax.numpy as jnp

# 旨在通过 pmap 运行的函数
def sum_across_devices(local_value):
  # 'batch_axis' 必须与 pmap 调用中的 axis_name 匹配
  total_sum = jax.lax.psum(local_value, axis_name='batch_axis')
  # 现在每个设备都拥有总和。
  # 比如，我们可以用它来缩放本地计算
  return total_sum

# 用法 (假设有 4 个设备):
# values = jnp.arange(4.) # [0., 1., 2., 3.] -> 每个设备一个值
# pmapped_sum = jax.pmap(sum_across_devices, axis_name='batch_axis')
# result = pmapped_sum(values)
# # 结果将是 DeviceArray([6., 6., 6., 6.], dtype=float32)
# # 每个设备都得到 0+1+2+3 = 6
```

一个常见的应用是在优化器步骤之前，在数据并行训练中对梯度求和。

#### `jax.lax.pmean` (并行平均)

`jax.lax.pmean` 的工作方式与 `psum` 类似，但它不是返回总和，而是返回沿着命名轴在所有设备上数组的元素级 *平均值*。它等同于执行 `psum`，然后除以参与该轴的设备数量（JAX 会自动跟踪）。

```python
import jax
import jax.numpy as jnp

# 旨在通过 pmap 运行的函数
def average_across_devices(local_value):
  # 'batch_axis' 必须与 pmap 调用中的 axis_name 匹配
  average_value = jax.lax.pmean(local_value, axis_name='batch_axis')
  return average_value

# 用法 (假设有 4 个设备):
# values = jnp.arange(4.) # [0., 1., 2., 3.]
# pmapped_mean = jax.pmap(average_across_devices, axis_name='batch_axis')
# result = pmapped_mean(values)
# # 结果将是 DeviceArray([1.5, 1.5, 1.5, 1.5], dtype=float32)
# # 每个设备都得到 (0+1+2+3)/4 = 1.5
```

`pmean` 是数据并行中平均梯度的标准操作。直接使用 `pmean` 通常比先 `psum` 再手动除法更受推荐，因为后端有时可以更高效地实现它。

#### `jax.lax.pmax` 和 `jax.lax.pmin` (并行最大/最小)

这些集合操作计算沿着命名轴跨设备数组的元素级最大值 (`pmax`) 或最小值 (`pmin`)。每个设备都接收到包含所有参与者中找到的最大或最小值的相同结果数组。

用途包括同步标志（例如，*是否有* 设备遇到错误？）、寻找批次中观察到的最大损失，或进行其他形式的分布式协作。

```python
import jax
import jax.numpy as jnp

# 旨在通过 pmap 运行的函数
def get_max_value(local_value):
  max_val = jax.lax.pmax(local_value, axis_name='data_split')
  return max_val

# 用法 (假设有 4 个设备):
# values = jnp.array([2., 5., 1., 4.]) # 每个设备一个值
# pmapped_max = jax.pmap(get_max_value, axis_name='data_split')
# result = pmapped_max(values)
# # 结果将是 DeviceArray([5., 5., 5., 5.], dtype=float32)
```

### 其他集合操作

归约集合操作非常常见，JAX 也提供了其他用于不同通信模式的操作。

- **`jax.lax.all_gather`**: 此操作从所有设备收集输入数组，并沿映射轴将它们连接起来。每个设备都接收到 *完整* 的、连接后的数组。如果每个设备都需要访问所有其他设备的数据，此操作很有用，但请注意，输出数组的大小会随设备数量线性增长，可能占用大量内存。
- **`jax.lax.ppermute`**: 这是一种更通用的集合操作，允许设备根据 `permute` 参数 (parameter)指定的置换规则交换数据。它描述了数据交换的源/目标设备索引对。`ppermute` 对于实现更复杂的并行方案（如环形 all-reduce 或模型并行通信模式）是基本操作，尽管直接使用它需要仔细的索引管理。
- **`jax.lax.axis_index`**: 尽管不严格来说是通信原语，`jax.lax.axis_index(axis_name)` 常常与集合操作结合使用。它返回当前设备在指定映射轴内的整数索引（ID）。这使得设备可以根据其在组内的位置表现出不同的行为，这对于特定的通信模式或工作负载均衡很有帮助。

### 集合操作在分布式工作流中的意义

集合通信原语是 `pmap` 内部协调工作的构成要素。它们支持数据并行中的梯度聚合、同步点以及用于更复杂分布式算法的数据交换等基本模式。理解 `psum`、`pmean` 以及 `axis_name` 的作用，对于在多个加速器上高效扩展 JAX 计算来说是基础。尽管这些操作会引入通信开销，但 JAX 和 XLA 会努力优化它们在 GPU 和 TPU 等底层硬件上的执行，通常会使用 NVLink 或芯片间互联 (ICI) 等高速互联。

## 参考资料

- [JAX Documentation: Parallelization (pmap) and Collective Operations (\`jax.lax\` collectives)](https://jax.readthedocs.io/en/latest/jax.html#jax.pmap) — JAX core contributors (2024)
  官方指南，解释JAX的SPMD编程模型、`pmap`、`axis_name`以及集体通信原语的API。
- [XLA: Accelerated Linear Algebra - Operation Semantics (Collectives)](https://www.tensorflow.org/xla/operation_semantics#collectives) — Google Brain Team (2024)
  提供对JAX使用的底层XLA编译器如何实现和优化集体通信操作的见解。
- [Distributed Deep Learning: A Brief Taxonomy](https://arxiv.org/abs/1903.04233) — Anees Kazi, Shayan shekarforoush, S.Arvind krishna, Hendrik Burwinkel, Gerome Vivar, Karsten Kortuem, Seyed-Ahmad Ahmadi, Shadi Albarqouni, Nassir Navab (2019)
  Journal: arXiv; Volume: 1; DOI: [10.48550/arXiv.1903.04233](https://doi.org/10.48550/arXiv.1903.04233)
  综述论文，回顾不同的分布式深度学习策略，包括数据并行以及集体操作在梯度同步中的作用。

---

[上一节](04-%E4%BD%BF%E7%94%A8%20pmap%20%E5%AE%9E%E7%8E%B0%E6%95%B0%E6%8D%AE%E5%B9%B6%E8%A1%8C.md) · [下一节](06-pmap%20%E4%B8%AD%E8%BD%B4%E5%90%8D%E7%A7%B0%E7%9A%84%E5%A4%84%E7%90%86.md)
