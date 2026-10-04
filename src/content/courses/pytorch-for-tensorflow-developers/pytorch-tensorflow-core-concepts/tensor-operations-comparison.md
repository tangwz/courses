---
course: "pytorch-for-tensorflow-developers"
chapter: "pytorch-tensorflow-core-concepts"
lesson: "tensor-operations-comparison"
sourceId: 5739
sourceUrl: "https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-1-pytorch-tensorflow-core-concepts/tensor-operations-comparison"
title: "基础张量运算：对比视角"
description: "了解 PyTorch 与 TensorFlow 中如何执行常见的张量运算（数学、索引、形状变换）。"
order: 4
plots: []
sourceHash: "a9cebf83cc91a3d1a33236f2293e0a8750029ca8a4cf7bdb1632876158c2a455"
sourceCorrections: []
---

操作 `torch.Tensor` 对象（与 TensorFlow 中的 `tf.Tensor` 类似）涉及执行各种运算。您在 TensorFlow 和 NumPy 中熟悉的许多数学运算在 PyTorch 中有直接的对应物。通过比较这些基础张量运算的执行方式，可以帮助您将 TensorFlow 知识映射到 PyTorch 的语法。

### 算术运算

基本的算术运算，如加法、减法、乘法和除法，都按元素执行，正如在 TensorFlow 中一样。PyTorch 支持运算符重载和显式函数。

```python
import torch
import tensorflow as tf # 用于说明性比较

# PyTorch
a_pt = torch.tensor([[1., 2.], [3., 4.]])
b_pt = torch.tensor([[5., 6.], [7., 8.]])

# 按元素相加
sum_pt = a_pt + b_pt
# sum_pt = torch.add(a_pt, b_pt)
print("PyTorch Sum:\n", sum_pt)

# 按元素相乘
prod_pt = a_pt * b_pt
# prod_pt = torch.mul(a_pt, b_pt)
print("PyTorch Product:\n", prod_pt)

# TensorFlow 等效代码（供参考）
# a_tf = tf.constant([[1., 2.], [3., 4.]])
# b_tf = tf.constant([[5., 6.], [7., 8.]])
# sum_tf = tf.add(a_tf, b_tf) # or a_tf + b_tf
# prod_tf = tf.multiply(a_tf, b_tf) # or a_tf * b_tf
```

PyTorch 在 `torch` 模块中提供了一套全面的数学函数，例如 `torch.sin()`、`torch.cos()`、`torch.exp()`、`torch.log()`，它们对张量按元素进行运算。

### 矩阵乘法

矩阵乘法是神经网络 (neural network)的基础。在 PyTorch 中，您可以使用 `torch.matmul()` 或 `@` 运算符。

```python
# PyTorch
mat1_pt = torch.randn(2, 3)
mat2_pt = torch.randn(3, 4)

# 矩阵乘法
result_pt = torch.matmul(mat1_pt, mat2_pt)
# result_pt = mat1_pt @ mat2_pt
print("PyTorch Matrix Multiplication (2x3 @ 3x4):\n", result_pt)
print("Result shape:", result_pt.shape) # torch.Size([2, 4])

# TensorFlow 等效代码
# mat1_tf = tf.random.normal((2, 3))
# mat2_tf = tf.random.normal((3, 4))
# result_tf = tf.matmul(mat1_tf, mat2_tf) # or mat1_tf @ mat2_tf
```

### 索引、切片和连接

PyTorch 张量支持 NumPy 风格的索引和切片，如果您使用过 NumPy 或 TensorFlow 张量，会感到非常熟悉。

```python
# PyTorch
data_pt = torch.arange(0, 10) # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

print("索引 3 处的元素:", data_pt[3])
print("从索引 2 到 5（不含）的切片:", data_pt[2:5])
print("从索引 5 开始的所有元素:", data_pt[5:])
print("最后一个元素:", data_pt[-1])

# 多维索引
matrix_pt = torch.randn(3, 4)
print("第一行:", matrix_pt[0])        # 获取第一行
print("第一列:", matrix_pt[:, 0])  # 获取第一列
print("位于 (1,1) 的元素:", matrix_pt[1, 1])

# 修改元素
data_pt[0] = 100
print("修改后的 data_pt:", data_pt)
```

对于连接张量，PyTorch 提供 `torch.cat()` 沿现有维度进行拼接，以及 `torch.stack()` 沿新维度进行堆叠。这与 TensorFlow 中的 `tf.concat()` 和 `tf.stack()` 类似。

```python
# PyTorch
t1_pt = torch.randn(2, 3)
t2_pt = torch.randn(2, 3)

# 沿维度 0（行）拼接
cat_dim0_pt = torch.cat((t1_pt, t2_pt), dim=0) # Shape: [4, 3]
print("沿维度 0 拼接后的形状:", cat_dim0_pt.shape)

# 沿维度 1（列）拼接
cat_dim1_pt = torch.cat((t1_pt, t2_pt), dim=1) # Shape: [2, 6]
print("沿维度 1 拼接后的形状:", cat_dim1_pt.shape)

# 沿新维度堆叠（默认为维度 0）
stacked_pt = torch.stack((t1_pt, t2_pt), dim=0) # Shape: [2, 2, 3]
print("沿新维度 0 堆叠后的形状:", stacked_pt.shape)

# TensorFlow 等效代码
# t1_tf = tf.random.normal((2,3))
# t2_tf = tf.random.normal((2,3))
# tf.concat([t1_tf, t2_tf], axis=0)
# tf.stack([t1_tf, t2_tf], axis=0)
```

### 形状变换运算

在不改变张量数据的情况下修改其形状是一个常见需求。PyTorch 为此提供了几个函数：

- `reshape()`: 返回具有指定形状的张量。如果新形状与原始元素数量兼容且张量在内存中是连续的，它通常返回一个视图（共享底层数据）。否则，它可能返回一个副本。
- `view()`: 类似于 `reshape()`，但严格返回一个视图。张量必须是连续的，并且新形状必须兼容。这更节省内存，因为它避免了数据复制，但对视图的更改会影响原始张量。
- `squeeze()`: 移除大小为 1 的维度。
- `unsqueeze()`: 添加大小为 1 的维度。

```python
# PyTorch
original_pt = torch.arange(12) # tensor([ 0,  1, ..., 11])

# 重塑为 3x4
reshaped_pt = original_pt.reshape(3, 4)
print("重塑后 (3x4):\n", reshaped_pt)

# 使用 view
view_pt = original_pt.view(3, 4)
# view_pt[0,0] = 99 # 这也会改变 original_pt[0]

# Squeeze 和 unsqueeze
x_pt = torch.randn(1, 3, 1, 4) # Shape: [1, 3, 1, 4]
squeezed_pt = x_pt.squeeze()     # Shape: [3, 4] (移除大小为 1 的维度)
print("Squeeze 后的形状:", squeezed_pt.shape)

unsqueezed_pt = squeezed_pt.unsqueeze(dim=0) # Shape: [1, 3, 4] (在位置 0 添加维度)
print("Unsqueezed 后的形状:", unsqueezed_pt.shape)

# TensorFlow 等效代码
# original_tf = tf.range(12)
# reshaped_tf = tf.reshape(original_tf, (3, 4))
# x_tf = tf.random.normal((1, 3, 1, 4))
# squeezed_tf = tf.squeeze(x_tf)
# unsqueezed_tf = tf.expand_dims(squeezed_tf, axis=0)
```

在 TensorFlow 中，`tf.reshape` 是修改形状的主要方式，而 `tf.squeeze` 和 `tf.expand_dims` 对应于 PyTorch 的 `squeeze` 和 `unsqueeze`。

### 归约运算

归约运算聚合张量值，例如 `sum()`、`mean()`、`max()`、`min()` 和 `std()`。您可以在整个张量上或沿特定维度执行这些运算。

```python
# PyTorch
tensor_pt = torch.tensor([[1., 2., 3.], [4., 5., 6.]])

# 所有元素的和
sum_all_pt = tensor_pt.sum()
print("所有元素的和:", sum_all_pt) # tensor(21.)

# 沿维度 0 求和（折叠行，每列的和）
sum_cols_pt = tensor_pt.sum(dim=0)
print("沿维度 0 求和（列）:", sum_cols_pt) # tensor([5., 7., 9.])

# 沿维度 1 求平均（折叠列，每行的平均值）
mean_rows_pt = tensor_pt.mean(dim=1)
print("沿维度 1 求平均（行）:", mean_rows_pt) # tensor([2., 5.])

# 最大元素及其索引
max_val_pt, max_idx_pt = torch.max(tensor_pt, dim=1)
print("每行的最大值:", max_val_pt)
print("每行的最大值索引:", max_idx_pt)

# TensorFlow 等效代码
# tensor_tf = tf.constant([[1., 2., 3.], [4., 5., 6.]])
# tf.reduce_sum(tensor_tf)
# tf.reduce_sum(tensor_tf, axis=0)
# tf.reduce_mean(tensor_tf, axis=1)
# tf.argmax(tensor_tf, axis=1) for indices, tf.reduce_max(tensor_tf, axis=1) for values
```

TensorFlow 对于这些运算使用 `tf.reduce_sum`、`tf.reduce_mean` 等。`tf.argmax` 和 `tf.argmin` 查找最大/最小值索引。

### 比较运算

按元素的比较（`>`、`<`、`==`、`!=` 等）会生成布尔张量。PyTorch 提供诸如 `torch.eq()`、`torch.gt()` 等函数。

```python
# PyTorch
a_pt = torch.tensor([1, 2, 3, 4])
b_pt = torch.tensor([4, 3, 2, 1])

# 按元素大于
gt_pt = a_pt > b_pt
print("a_pt > b_pt:", gt_pt) # tensor([False, False,  True,  True])

# 按元素相等
eq_pt = torch.eq(a_pt, torch.tensor([1, 3, 3, 5]))
print("torch.eq(a_pt, [1,3,3,5]):", eq_pt) # tensor([ True, False,  True, False])

# TensorFlow 等效代码
# a_tf = tf.constant([1, 2, 3, 4])
# b_tf = tf.constant([4, 3, 2, 1])
# tf.greater(a_tf, b_tf)
# tf.equal(a_tf, tf.constant([1, 3, 3, 5]))
```

### 就地运算

PyTorch 支持就地运算，即直接修改张量而不会创建新张量。这些运算通常由一个尾随下划线表示（例如 `add_()`、`mul_()`）。虽然它们可以节省内存，但请谨慎使用，尤其是在需要梯度的运算中，因为修改反向传播 (backpropagation)所需的张量可能会导致错误。

```python
# PyTorch
x_pt = torch.ones(3)
y_pt = torch.tensor([1., 2., 3.])

print("原始 y_pt:", y_pt)
y_pt.add_(x_pt) # 就地加法: y_pt = y_pt + x_pt
print("add_() 后的 y_pt:", y_pt) # y_pt 已被修改

# 这与以下情况不同：
# z_pt = y_pt.add(x_pt) # 非就地操作: z_pt 是一个新张量，y_pt 保持不变
```

TensorFlow 的 `tf.Tensor` 对象是不可变的。运算通常会创建新张量。TensorFlow 中的可变性主要通过 `tf.Variable` 对象处理，它们具有 `assign()`、`assign_add()` 等方法。

### 广播

PyTorch 支持广播，类似于 NumPy 和 TensorFlow。如果两个张量形状不同但根据广播规则兼容（维度相等，或其中一个为 1，或其中一个缺失），运算仍然可以按元素执行。

```python
# PyTorch
# 张量 m_pt: 形状 (3, 1)
# [[1],
#  [2],
#  [3]]
m_pt = torch.arange(1, 4).reshape(3, 1).float()

# 张量 n_pt: 形状 (1, 2)
# [[10, 20]]
n_pt = torch.tensor([[10., 20.]])

# m_pt 广播到 (3,2)，n_pt 广播到 (3,2)
# m_pt + n_pt:
# [[1+10, 1+20],   [[11, 21],
#  [2+10, 2+20], =  [12, 22],
#  [3+10, 3+20]]    [13, 23]]
result_pt = m_pt + n_pt
print("广播后的和 (3,1) + (1,2):\n", result_pt)
print("结果形状:", result_pt.shape) # torch.Size([3, 2])

# TensorFlow 等效代码
# m_tf = tf.constant([[1.],[2.],[3.]]) # Shape (3,1)
# n_tf = tf.constant([[10., 20.]])     # Shape (1,2)
# result_tf = m_tf + n_tf              # Shape (3,2) via broadcasting
```

### 关于数据类型的说明

PyTorch 对从 Python 列表或 NumPy 浮点数组创建的张量默认为 `torch.float32`，对整数默认为 `torch.int64` (long tensor)。TensorFlow 也通常默认为 `float32` 和 `int32`。不匹配的数据类型是错误的常见来源。您可以使用 `.to(dtype)` 方法或 `.float()`、`.long()`、`.double()` 等特定类型转换方法来转换张量类型。

```python
# PyTorch
float_list_pt = torch.tensor([1.0, 2.5, 3.0])
print("默认浮点数数据类型:", float_list_pt.dtype) # torch.float32

int_list_pt = torch.tensor([1, 2, 3])
print("默认整数数据类型:", int_list_pt.dtype) # torch.int64

# 类型转换
float_to_double_pt = float_list_pt.to(torch.float64)
print("转换为双精度:", float_to_double_pt.dtype) # torch.float64

int_to_float_pt = int_list_pt.float() # 等同于 .to(torch.float32)
print("整数转换为浮点数:", int_to_float_pt.dtype) # torch.float32
```

如您所见，PyTorch 中的许多张量运算在 TensorFlow 中有直接对应，通常具有非常相似的命名约定或运算符用法。PyTorch 的动态特性意味着这些运算会立即执行，这对于调试和交互式开发很有帮助。熟悉这些运算是您转向 PyTorch 的重要一步。

## 参考资料

- [PyTorch Tensors](https://pytorch.org/docs/stable/tensors.html) — PyTorch developers (2024)
  PyTorch 官方文档，提供了对 `torch.Tensor` 对象及其核心数学、索引和重塑操作的基础理解。
- [Tensors](https://www.tensorflow.org/guide/tensor) — TensorFlow developers (2024)
  TensorFlow 官方关于 `tf.Tensor` 对象及其操作的指南，为与 PyTorch 张量功能的比较提供了基础。
- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, and Thomas Viehmann (2020)
  Publisher: Manning Publications
  一本内容全面的书籍，提供了对 PyTorch 基础知识的详细说明，涵盖了张量操作及其在深度学习中的实际应用。
- [Broadcasting semantics](https://pytorch.org/docs/stable/notes/broadcasting.html) — PyTorch developers (2021)
  Publisher: PyTorch Foundation
  官方文档解释了 PyTorch 中广播的规则和行为，这是组合不同形状张量的基础。
- [In-place operations with autograd](https://pytorch.org/docs/stable/notes/autograd.html#in-place-operations-with-autograd) — PyTorch developers (2017)
  Publisher: PyTorch Foundation
  PyTorch 文档中关于原地操作的特定说明，涵盖了它们的性质以及与自动求导引擎的交互。
