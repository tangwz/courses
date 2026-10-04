---
course: "getting-started-with-jax"
chapter: "introduction-to-jax"
lesson: "working-with-jax-arrays"
sourceId: 1685
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-jax/chapter-1-introduction-to-jax/working-with-jax-arrays"
title: "使用 JAX 数组"
description: "使用 `jax.numpy` 创建和操作数组。"
order: 5
plots: []
sourceHash: "ffbdb8f211ea37429fa34395be1468a399adcf37bf39743f093d8273542a1bbf"
sourceCorrections: []
---

JAX 数组是你使用 JAX 时的基本组成部分。JAX 是一种基于转换的函数式编程方法。如果你熟悉 NumPy，你会发现 `jax.numpy` API（通常导入为 `jnp`）非常熟悉。它与标准 NumPy 接口非常相似，提供平稳过渡。

```python
import jax
import jax.numpy as jnp

# 标准约定
print(f"JAX 版本: {jax.__version__}")
```

### 创建 JAX 数组

与 NumPy 一样，你可以从现有 Python 列表或元组创建 JAX 数组，或使用专门的函数。

**从 Python 列表/元组：**

```python
# 从列表创建数组
python_list = [1.0, 2.0, 3.0]
jax_array_from_list = jnp.array(python_list)
print(f"从列表创建的数组: {jax_array_from_list}")
print(f"类型: {type(jax_array_from_list)}")

# 从嵌套列表创建二维数组
python_nested_list = [[1, 2], [3, 4]]
jax_2d_array = jnp.array(python_nested_list)
print(f"二维数组:\n{jax_2d_array}")
```

**使用 `jax.numpy` 函数：**

`jax.numpy` 提供与常见 NumPy 数组创建例程等效的功能：

```python
# 全零数组
zeros_array = jnp.zeros((2, 3)) # 形状 (2 行, 3 列)
print(f"全零数组:\n{zeros_array}")

# 全一数组
ones_array = jnp.ones((3,), dtype=jnp.int32) # 形状 (3,)，整数类型
print(f"全一数组: {ones_array}")
print(f"全一数组数据类型: {ones_array.dtype}")

# 具有一系列值的数组
range_array = jnp.arange(5) # 类似于 Python 的 range(5) -> [0, 1, 2, 3, 4]
print(f"Arange 数组: {range_array}")

# 线性间隔数组
linspace_array = jnp.linspace(0, 1, 5) # 从 0 到 1（包含）的 5 个点
print(f"Linspace 数组: {linspace_array}")
```

### JAX 中的随机数

在 JAX 中生成随机数与 NumPy 有显著不同，因为 JAX 函数必须是*纯粹的*。纯函数对于相同的输入总是返回相同的输出，并且没有副作用。NumPy 的随机函数维护一个全局状态，这违反了纯度。

JAX 使用显式的伪随机数生成器（PRNG）键来处理随机性。你创建一个初始键，然后当你需要更多随机数时，从现有键生成新键。这使得随机数生成可重现并与 JAX 转换兼容。

```python
from jax import random

# 创建初始 PRNG 种子。种子通常是整数。
key = random.PRNGKey(0)
print(f"初始: {key}")

# 生成随机数（例如，来自正态分布）
# 此操作会消耗键，但不会直接修改它。
normal_random_numbers = random.normal(key, shape=(2, 2))
print(f"正态随机数:\n{normal_random_numbers}")

# 要生成更多随机数，请“拆分”键
key, subkey = random.split(key) # 创建一个新键和一个子键
uniform_random_numbers = random.uniform(subkey, shape=(3,))
print(f"\n拆分后: {key}")
print(f"子键: {subkey}")
print(f"均匀随机数: {uniform_random_numbers}")

# 再次使用*相同*的原始键调用 random.normal 会产生*相同*的结果
# 这展示了纯度和可重现性
same_normal_numbers = random.normal(random.PRNGKey(0), shape=(2, 2))
print(f"\n相同的正态随机数:\n{same_normal_numbers}")
assert jnp.allclose(normal_random_numbers, same_normal_numbers)
```

显式管理这些键对于编写正确且可重现的 JAX 代码非常重要，尤其是在使用 `jit` 或 `vmap` 等转换时。

### 数组属性

JAX 数组与 NumPy 数组共享类似的属性：

```python
x = jnp.arange(12).reshape((3, 4))
print(f"数组 x:\n{x}")

print(f"形状: {x.shape}")      # 指示维度的元组（行数，列数）
print(f"数据类型: {x.dtype}")  # 元素的类型（例如，float32, int32）
print(f"维度数量: {x.ndim}")   # 轴的数量（矩阵为 2）
print(f"元素总数: {x.size}")  # 元素总数（3 * 4 = 12）
```

JAX 通常默认使用 32 位精度（`float32`、`int32`）以提高性能，尤其是在加速器上。如果需要，你可以启用 64 位精度，但这可能会影响速度。

### 差异：不可变性

这可能是与标准 NumPy 最重要的区别。**JAX 数组是不可变的。** 一旦创建，它们的值就不能原地修改。

```python
numpy_array = np.array([1, 2, 3])
numpy_array[0] = 100 # 在 NumPy 中运行正常
print(f"修改后的 NumPy 数组: {numpy_array}")

jax_array = jnp.array([1, 2, 3])
try:
    # 这将引发错误，因为 JAX 数组是不可变的
    jax_array[0] = 100
except TypeError as e:
    print(f"\n尝试原地修改 JAX 数组时出错: {e}")

# 相反，使用函数式“索引更新”语法
# 这会创建一个带有更新值的新数组。
updated_jax_array = jax_array.at[0].set(100)

print(f"原始 JAX 数组（未改变）: {jax_array}")
print(f"更新后的 JAX 数组（新对象）: {updated_jax_array}")

# 你也可以执行更复杂的更新：
incremented_array = jax_array.at[1].add(10) # 将索引 1 处的元素增加 10
print(f"递增后的数组: {incremented_array}")
```

这种不可变性对于 JAX 的函数转换（`jit`、`grad`、`vmap`、`pmap`）非常重要。它确保函数保持纯粹，不会因直接修改输入而产生隐藏的副作用。虽然这需要以稍微不同的方式思考更新，但 `array.at[index].set(value)` 模式通过练习会变得自然，并且与函数式方法非常契合。

### 基本操作和索引

算术运算和通用函数（ufuncs）的工作方式与 NumPy 非常相似：

```python
a = jnp.array([[1, 2], [3, 4]])
b = jnp.array([[5, 6], [7, 8]])

# 元素级操作
print(f"a + b:\n{a + b}")
print(f"a * b:\n{a * b}") # 元素级乘法

# 矩阵乘法
print(f"矩阵积 (jnp.dot):\n{jnp.dot(a, b)}")
print(f"矩阵积 (@):\n{a @ b}")

# 通用函数
print(f"a 的正弦:\n{jnp.sin(a)}")
print(f"a 的指数:\n{jnp.exp(a)}")
```

标准的 NumPy 风格的索引和切片用于*读取*数据：

```python
data = jnp.arange(10)
print(f"\n原始数据: {data}")

# 获取单个元素
print(f"索引 3 处的元素: {data[3]}")

# 获取切片
print(f"从索引 2 到 5 的切片: {data[2:5]}")

# 多维索引
matrix = jnp.arange(9).reshape((3, 3))
print(f"矩阵:\n{matrix}")
print(f"第 1 行第 2 列的元素: {matrix[1, 2]}")
print(f"前两行:\n{matrix[:2, :]}")
print(f"第一列:\n{matrix[:, 0]}")
```

请记住，由于不可变性，如果你需要修改数组的部分，必须使用 `.at[...].set(...)`（或 `.add`、`.multiply` 等）语法来创建新的、已更新的数组。

### 设备放置

JAX 自动处理数组的放置以及在可用硬件（CPU、GPU 或 TPU）上执行计算。对于基本操作，你通常不需要手动管理此项。你可以查看数组位于何处：

```python
x = jnp.ones(3)
try:
    # device() 方法给出缓冲区所在的设备
    print(f"\n数组 x 位于设备: {x.device()}")
except AttributeError:
    # 较旧的 JAX 版本可能无法直接使用 .device()
    # 或者对象在执行前可能是抽象值跟踪器。
    # 更多检查通常涉及检查 jax.devices()
    print("\n设备信息可能需要特定上下文或检查 jax.devices()。")

print(f"可用设备: {jax.devices()}")
```

设备放置的理解在使用 `pmap` 进行多设备并行时变得更加重要，这将在后面的章节中介绍。

有了对 JAX 数组、它们与 NumPy 数组的相似之处以及不可变性这一重要理解，你就可以开始学习 JAX 强大的函数转换了，从使用 `jax.jit` 加速代码开始。

## 参考资料

- [JAX Documentation: JAX NumPy Quickstart](https://jax.readthedocs.io/en/latest/jax.numpy.html) — The JAX Team (2024)
  概述了jax.numpy API、数组创建、操作及其与NumPy的关系。
- [JAX Documentation: Common Gotchas and FAQs - Random numbers are different in JAX](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#random-numbers) — Matteo Hessel, Rosalia Schneider (2024)
  Publisher: The JAX Team
  解释了JAX使用PRNG密钥生成伪随机数的方法，并与NumPy的有状态方法进行了对比。
- [JAX Documentation: Common Gotchas and FAQs - In-place array updates not supported](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_and_FAQs.html#in-place-updates) — The JAX Team (2024)
  讨论了JAX数组的不可变性概念以及更新数组元素的函数式模式。
- [JAX: Autograd and XLA, Together](https://arxiv.org/abs/1810.04838) — James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Skye Li, Gregory S. Wright, Adam Paszke, Sarah J. Adams, Sam Bourne, Sander Dieleman, Frederic Eger, Shixiang Gu, Simon Kornblith, Roman Ring, Kevin Smith, Marc van der Maaten, Balaji V. V. N., and Parker Evans (2018)
  Journal: arXiv preprint arXiv:1810.04838; DOI: [10.48550/arXiv.1810.04838](https://doi.org/10.48550/arXiv.1810.04838)
  介绍JAX的基础论文，详细阐述了其设计原则，包括函数式纯粹性和变换。
