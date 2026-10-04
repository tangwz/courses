---
course: "getting-started-with-tensorflow"
chapter: "core-tensorflow-concepts"
lesson: "understanding-tensors"
sourceId: 1688
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-2-core-tensorflow-concepts/understanding-tensors"
title: "理解张量"
description: "了解 TensorFlow 张量、它们的特性（形状、数据类型）以及与 NumPy 数组的区别。"
order: 1
plots: []
sourceHash: "e1fa46ccbf1db3dafc9c3e525a706e789e131eb14e3611e5414f347567606fa0"
sourceCorrections: []
---

TensorFlow 的核心是**张量**，一种多维数组，作为主要的数据结构。如果你熟悉 NumPy 数组，你会觉得张量相当好理解。然而，TensorFlow 张量提供额外的功能，专为大规模机器学习 (machine learning)和深度学习 (deep learning)设计，例如高效的 GPU 加速并与自动微分功能结合。

可以把张量看作是向量 (vector)和矩阵到可能更高维度情况的推广。

- **标量**（一个单个数字，如 5 或 3.14）是 0 阶张量。
- **向量**（一个一维数组，如 `[1, 2, 3]`）是 1 阶张量。
- **矩阵**（一个二维数组，如 `[[1, 2], [3, 4]]`）是 2 阶张量。
- 你可以拥有三维或更高维度的张量（3 阶、4 阶等）。例如，一批彩色图像可以表示为 4 阶张量，其维度分别对应（批次大小、高度、宽度、颜色通道）。

### 张量特性：形状和数据类型

每个张量都有两个基本属性：

1. **形状**：描述了张量的维度以及每个维度的大小。通常表示为整数元组或列表。例如：

   - 标量有一个空形状：`()`
   - 长度为 5 的向量 (vector)形状为：`(5,)`
   - 一个 3 行 4 列的矩阵形状为：`(3, 4)`
   - 一批 32 张大小为 28x28 的灰度图像的形状将是：`(32, 28, 28)`，如果明确包含通道维度，则可能是 `(32, 28, 28, 1)`。
     形状对于确保模型中操作的兼容性很重要。你可以使用张量的 `.shape` 属性来获取其形状。
2. **数据类型（`dtype`）**：指定张量中包含的数据类型，例如浮点数、整数、布尔值或字符串。常见的数据类型包括：

   - `tf.float32`：标准 32 位浮点数。大多数操作的默认值，广泛用于神经网络 (neural network)以平衡精度和计算成本。
   - `tf.int32`：标准 32 位整数。
   - `tf.bool`：布尔值（`True` 或 `False`）。
   - `tf.string`：可变长度字节字符串。
     TensorFlow 操作通常要求张量具有兼容的数据类型。你可以使用张量的 `.dtype` 属性来获取其数据类型。

### 创建张量

你可以通过几种方式创建张量。最直接的方法是使用 `tf.constant()`，它从 Python 对象（如列表或元组）或 NumPy 数组创建不可变张量。

```python
import tensorflow as tf
import numpy as np

# 创建一个标量（0 阶张量）
scalar = tf.constant(10)
print("标量：", scalar)
print("形状：", scalar.shape)
print("数据类型：", scalar.dtype)

# 创建一个向量（1 阶张量）
vector = tf.constant([1.0, 2.0, 3.0])
print("\n向量：", vector)
print("形状：", vector.shape)
print("数据类型：", vector.dtype) # 从输入推断为 float32

# 创建一个指定 dtype 的矩阵（2 阶张量）
matrix = tf.constant([[1, 2], [3, 4]], dtype=tf.int16)
print("\n矩阵：", matrix)
print("形状：", matrix.shape)
print("数据类型：", matrix.dtype)

# 创建一个 3 阶张量
tensor3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("\n3 阶张量：", tensor3d)
print("形状：", tensor3d.shape)
print("数据类型：", tensor3d.dtype) # 推断为 int32
```

TensorFlow 还提供函数来创建具有特定值的张量，类似于 NumPy：

```python
# 零张量
zeros_tensor = tf.zeros(shape=(2, 3))
print("\n零张量：\n", zeros_tensor)

# 一张量
ones_tensor = tf.ones(shape=(3, 2), dtype=tf.float32)
print("\n一张量：\n", ones_tensor)

# 填充特定值的张量
fill_tensor = tf.fill(dims=(2, 2), value=99)
print("\n填充张量：\n", fill_tensor)

# 从正态分布中取值的张量
random_normal = tf.random.normal(shape=(2, 2), mean=0.0, stddev=1.0)
print("\n随机正态分布张量：\n", random_normal)

# 从均匀分布中取值的张量
random_uniform = tf.random.uniform(shape=(2, 2), minval=0, maxval=10, dtype=tf.int32)
print("\n随机均匀分布张量：\n", random_uniform)
```

你也可以轻松地将 NumPy 数组转换为 TensorFlow 张量，反之亦然。TensorFlow 通常直接处理传递给其操作的 NumPy 数组。

```python
# 将 NumPy 数组转换为张量
numpy_array = np.array([[1.0, 2.0], [3.0, 4.0]])
tensor_from_numpy = tf.convert_to_tensor(numpy_array)
print("\n来自 NumPy 的张量：\n", tensor_from_numpy)
print("数据类型：", tensor_from_numpy.dtype) # 保留 NumPy 的 dtype（此处为 float64）

# 将张量转换回 NumPy 数组
numpy_from_tensor = tensor_from_numpy.numpy()
print("\n来自张量的 NumPy：\n", numpy_from_tensor)
print("类型：", type(numpy_from_tensor))
```

### 与 NumPy 数组的区别

虽然张量与 NumPy 数组有相似之处，但有些区别很重要：

1. **硬件加速**：张量可以明确地放置在硬件加速器（如 GPU (图形处理单元) 和 TPU (张量处理单元)）上进行计算，显著加快深度学习 (deep learning)中常见的计算。NumPy 数组主要在 CPU 上运行。
2. **不可变性**：标准 `tf.Tensor` 对象是不可变的。张量一旦创建，你无法更新其内容；操作通常会创建新的张量。这与可变的 NumPy 数组不同。对于 TensorFlow 中的可变状态，例如在训练期间需要更新的模型权重 (weight)，你会使用 `tf.Variable`（下一节会讲到）。
3. **自动微分**：TensorFlow 可以自动计算涉及张量操作的梯度。这项能力由 `tf.GradientTape` 协助（本章稍后讨论），是训练机器学习 (machine learning)模型通过梯度下降 (gradient descent)的重要组成部分。

### 阶、形状和大小再看

我们来澄清一下这些相关术语：

- **阶**：张量的维度或轴的数量。你可以使用 `tf.rank(tensor)` 来获取阶。
  - 标量：0 阶
  - 向量 (vector)：1 阶
  - 矩阵：2 阶
- **形状**：一个元组，表示每个维度（轴）上的元素数量。通过 `tensor.shape` 获取。
- **大小**：张量中元素的总数。你可以使用 `tf.size(tensor)` 来获取大小。这是形状元组中元素的乘积。

```python
rank_2_tensor = tf.constant([[1, 2, 3], [4, 5, 6]])

print("\n张量：\n", rank_2_tensor)
print("形状：", rank_2_tensor.shape)         # Output: (2, 3)
print("阶：", tf.rank(rank_2_tensor))      # Output: tf.Tensor(2, shape=(), dtype=int32)
print("大小：", tf.size(rank_2_tensor))        # Output: tf.Tensor(6, shape=(), dtype=int32)
print("元素数量：", tf.size(rank_2_tensor).numpy()) # 获取 Python 值
```

### 索引和切片

访问张量中的元素与索引 NumPy 数组或 Python 列表非常相似，都使用零基索引和切片。

```python
# 使用上面定义的 2 阶张量：形状=(2, 3)
print("\n张量：\n", rank_2_tensor)

# 获取第 0 行第 1 列的元素
print("[0, 1] 处的元素：", rank_2_tensor[0, 1]) # Output: tf.Tensor(2, ...)

# 获取第一行
print("第一行：", rank_2_tensor[0, :]) # Output: tf.Tensor([1 2 3], ...)

# 获取第二列
print("第二列：", rank_2_tensor[:, 1]) # Output: tf.Tensor([2 5], ...)

# 获取子矩阵（第一行，第 1 和第 2 列）
print("子矩阵：", rank_2_tensor[0, 1:3]) # Output: tf.Tensor([2 3], ...)
```

理解张量、它们的形状、数据类型以及如何创建和操作它们，对于使用 TensorFlow 很重要。它们是你将构建和执行的计算图中流动的数据容器。在接下来的部分，我们将介绍你可以对这些张量执行的操作，并引入 `tf.Variable` 来处理可变模型参数 (parameter)。

## 参考资料

- [Tensors](https://www.tensorflow.org/guide/tensor) — TensorFlow Authors (2024)
  官方指南，介绍 TensorFlow 的张量对象、属性、创建和操作。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  基础教科书，阐释深度学习的数学和概念基础，包含用于数据和模型参数的多维数组（张量）。
- [Arrays](https://numpy.org/doc/stable/user/basics.html) — NumPy Developers (2024)
  详细介绍了 NumPy 中的 N 维数组对象，它在概念上与 TensorFlow 张量相似，且是其常见对应物。
