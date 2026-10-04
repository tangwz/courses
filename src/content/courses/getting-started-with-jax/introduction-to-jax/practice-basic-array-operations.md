---
course: "getting-started-with-jax"
chapter: "introduction-to-jax"
lesson: "practice-basic-array-operations"
sourceId: 1690
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-jax/chapter-1-introduction-to-jax/practice-basic-array-operations"
title: "动手练习：基本数组操作"
description: "动手练习对 JAX 数组执行基本操作。"
order: 7
plots: ["plots/1690-0.json"]
sourceHash: "a789ce7eb450ca60793602c1a25ccb14c69fba1f1145ab2e52d710edd1464a58"
sourceCorrections: []
---

JAX 通过 `jax.numpy` 提供类似 NumPy 的 API。JAX 数组是不可变的，它们驻留在特定设备上（CPU、GPU 或 TPU）。我们将逐步进行一些基本的数组操作。请确保你已安装 JAX。

首先，让我们导入 `jax.numpy`：

```python
import jax.numpy as jnp
import jax
import numpy as np # 通常用于比较，或用于 JAX 不支持的操作

print(f"JAX 版本: {jax.__version__}")
print(f"JAX 默认后端: {jax.default_backend()}")
# 检查可用设备（可能因你的设置而异）
print(f"可用 JAX 设备: {jax.devices()}")
```

### 创建 JAX 数组

与 NumPy 类似，你可以从 Python 列表或元组创建 JAX 数组，或使用专门的创建函数。

```python
# 从 Python 列表
py_list = [1.0, 2.5, 3.0, 4.2]
jax_array_from_list = jnp.array(py_list)
print("从列表创建的 JAX 数组:", jax_array_from_list)
print("类型:", type(jax_array_from_list))
print("数据类型:", jax_array_from_list.dtype)

# 创建特定数组
zeros_array = jnp.zeros((2, 3)) # 形状 (2 行, 3 列)
print("\n零数组:\n", zeros_array)

ones_array = jnp.ones((3, 2), dtype=jnp.int32) # 指定数据类型
print("\n全一数组 (int32):\n", ones_array)
print("数据类型:", ones_array.dtype)

range_array = jnp.arange(0, 10, 2) # 起始值, 停止值, 步长
print("\n范围数组:", range_array)

linspace_array = jnp.linspace(0, 1, 5) # 起始值, 停止值, 点数
print("\n等差数组:", linspace_array)
```

请注意，输出类型是 JAX 特有的（通常是 `jaxlib.xla_extension.DeviceArray` 或类似类型）。出于性能考虑，JAX 通常在 CPU 上默认为 64 位浮点数 (`float64`)，在 GPU/TPU 上默认为 32 位浮点数 (`float32`)，但你可以像 `jnp.ones` 所示那样显式设置 `dtype`。

对于随机数，JAX 使用显式的有状态伪随机数生成器 (PRNG) 方法，这与 NumPy 的全局状态不同。你需要创建一个 `PRNGKey`。

```python
# 创建伪随机数生成器
key = jax.random.PRNGKey(42) # 种子为 42

# 生成随机数（例如，0 到 1 之间的均匀分布）
random_array = jax.random.uniform(key, shape=(2, 2))
print("\n随机数组 (均匀分布):\n", random_array)

# 重要提示：要获得新的随机数，你必须“分割”密钥
key, subkey = jax.random.split(key)
random_normal_array = jax.random.normal(subkey, shape=(3,)) # 标准正态分布
print("\n随机数组 (正态分布):\n", random_normal_array)
```

使用和分割密钥可确保结果的可复现性，这对于调试和获取一致结果非常重要，尤其是在后续涉及函数转换时。

### 检查数组

你可以像在 NumPy 中一样检查数组属性：

```python
print("\n检查 random_array:")
print("形状:", random_array.shape)
print("大小:", random_array.size) # 元素总数
print("维度数:", random_array.ndim)
print("数据类型:", random_array.dtype)

# 检查数组所在的设备
# 这将根据你的设置和 JAX 配置显示 CPU、GPU 或 TPU
print("设备:", random_array.device())
```

### 基本算术操作

算术运算符按元素操作，创建 *新* 数组。请记住，JAX 数组是不可变的。

```python
a = jnp.array([1, 2, 3])
b = jnp.array([4, 5, 6])

# 按元素操作
c_add = a + b
print("\na + b =", c_add)

c_mul = a * b
print("a * b =", c_mul)

# 标量操作
c_scalar_add = a + 10
print("a + 10 =", c_scalar_add)

c_scalar_mul = a * 2
print("a * 2 =", c_scalar_mul)

# 检查 'a' 是否保持不变（不可变性）
print("原始 'a':", a)
```

矩阵乘法使用 `@` 运算符或 `jnp.matmul`：

```python
mat_a = jnp.array([[1, 2], [3, 4]])
mat_b = jnp.array([[5, 6], [7, 8]])

mat_product = mat_a @ mat_b
# 等价于: mat_product = jnp.matmul(mat_a, mat_b)
print("\n矩阵乘积 (mat_a @ mat_b):\n", mat_product)
```

### 索引和切片

索引和切片的工作方式与 NumPy 类似，但它们返回 *新* 数组（或者更确切地说，是 DeviceArray，它们在可变性方面表现得像副本），而不是原始数据的视图，这是因为不可变性。直接修改 JAX 数组的切片是不可能的。

```python
data = jnp.arange(12).reshape((3, 4))
print("\n原始数据:\n", data)

# 获取单个元素
element = data[1, 2] # 第 1 行, 第 2 列
print("位于 [1, 2] 的元素:", element)
# 注意: 访问单个元素会返回一个 0 维数组

# 获取一行
row_1 = data[1, :] # 第 1 行, 所有列
print("第 1 行:", row_1)

# 获取一列
col_2 = data[:, 2] # 所有行, 第 2 列
print("第 2 列:", col_2)

# 获取子数组（切片）
sub_array = data[0:2, 1:3] # 第 0-1 行, 第 1-2 列
print("子数组 (0:2, 1:3):\n", sub_array)

# 尝试修改切片会失败或行为异常
# 这与 NumPy 不同，在 NumPy 中切片通常是视图
try:
    # 此操作通常不受支持或不修改 'data'
    # 在标准 JAX 中，这会引发错误，因为 DeviceArray 不支持项赋值
    # data[0, 0] = 99
    # 更符合 JAX 习惯的更新方式需要索引更新函数：
    updated_data = data.at[0, 0].set(99)
    print("\n原始数据（仍未更改）:\n", data)
    print("更新后的数据（新数组）:\n", updated_data)
except TypeError as e:
    print(f"\n正如预期，直接项赋值失败: {e}")
    print("使用 `.at[index].set(value)` 进行函数式更新。")
```

`.at[...].set(...)` 语法是 JAX 执行“就地外”更新的方式，它返回一个修改后的副本，同时保持原始数组不变。这种函数式方法对于与 `jit` 和 `grad` 等 JAX 转换的兼容性是必要的。

### 通用函数 (ufuncs)

JAX 提供了许多 NumPy 中按元素的通用函数：

```python
x = jnp.linspace(0, jnp.pi * 2, 5)
print("\nx:", x)

y_sin = jnp.sin(x)
print("sin(x):", y_sin)

y_exp = jnp.exp(x / (jnp.pi * 2)) # 在进行 exp 运算前将 x 缩放到 0 到 1
print("exp(缩放后的 x):", y_exp)
```

让我们可视化 `sin(x)`：



![使用 JAX 数组的正弦函数](plots/1690-0.json)



> 值由 `jnp.linspace` 生成并由 `jnp.sin` 转换。

### 规约操作

提供了聚合数组值的函数，如 `sum`、`mean`、`max`、`min`。你可以对整个数组或沿着特定轴执行规约。

```python
matrix = jnp.arange(12).reshape((3, 4))
print("\n矩阵:\n", matrix)

total_sum = jnp.sum(matrix)
print("总和:", total_sum)

sum_along_rows = jnp.sum(matrix, axis=0) # 对每列的元素求和
print("按列求和 (axis=0):", sum_along_rows)

mean_along_cols = jnp.mean(matrix, axis=1) # 对每行的元素求平均
print("按行求平均 (axis=1):", mean_along_cols)

max_val = jnp.max(matrix)
print("最大值:", max_val)
```

### 重塑数组

你可以改变数组的形状而不改变其数据，这同样会生成一个新数组。

```python
original = jnp.arange(6)
print("\n原始一维数组:", original)

reshaped_2x3 = original.reshape((2, 3))
# 等价于: reshaped_2x3 = jnp.reshape(original, (2, 3))
print("重塑为 (2, 3):\n", reshaped_2x3)

reshaped_3x2 = jnp.reshape(original, (3, 2))
print("重塑为 (3, 2):\n", reshaped_3x2)
```

这次动手练习涵盖了最常见的数组操作。你应该能熟练地使用 `jax.numpy` 接口创建、操作和检查 JAX 数组。与 NumPy 的相似性使得这种转变相对平稳，但要记住其核心区别，特别是不可变性和显式 PRNG 处理。这些特性是 JAX 达成高性能并支持强大函数转换的根本，我们将在后续章节中进一步了解。亲自试验这些操作，以巩固你的理解。

## 参考资料

- [JAX documentation](https://jax.readthedocs.io/en/latest/) — The JAX Team (2024)
  理解 JAX API 的重要资源，特别是 `jax.numpy`、数组不可变性、设备放置和伪随机数生成。
- [JAX for NumPy Users](https://docs.jax.dev/en/latest/jax-for-numpy-users/index.html) — The JAX Team (2025)
  阐明 JAX 的 `jax.numpy` 与标准 NumPy 之间的异同点，包括数组不可变性和伪随机数生成。
- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html) — The NumPy Developers (2024)
  NumPy 数组和操作的综合指南，为 JAX 的 NumPy 风格接口提供概念基础。
