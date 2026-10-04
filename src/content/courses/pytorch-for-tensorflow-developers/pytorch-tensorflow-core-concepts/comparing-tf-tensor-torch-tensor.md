---
course: "pytorch-for-tensorflow-developers"
chapter: "pytorch-tensorflow-core-concepts"
lesson: "comparing-tf-tensor-torch-tensor"
sourceId: 5738
sourceUrl: "https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-1-pytorch-tensorflow-core-concepts/comparing-tf-tensor-torch-tensor"
title: "张量比较：tf.Tensor 与 torch.Tensor"
description: "比较 TensorFlow 张量与 PyTorch 张量的异同，包括创建和属性。"
order: 3
plots: []
sourceHash: "a37cb5991d619431a53b31566a34afd57097bdee8246c5c3ebaae69c2a761d33"
sourceCorrections: []
---

TensorFlow 和 PyTorch 的核心都是张量，它是所有计算的基本数据结构。如果你曾使用 TensorFlow，那么你对 `tf.Tensor` 这种可存储数值数据的多维数组应该很熟悉了。PyTorch 的 `torch.Tensor` 也用于完全相同的目的。尽管核心思想一致，但创建、操作和管理这些张量的方式有所不同，这反映了 PyTorch 的动态特性。这些区别将被考察，主要侧重于张量的创建和属性。

## 张量创建：从 TensorFlow 到 PyTorch

两个框架都提供了多种创建张量的方法，无论是从现有 Python 数据结构（如列表或 NumPy 数组），还是通过用特定值初始化它们。

### 从现有数据创建张量

在 TensorFlow 中，你通常使用 `tf.constant()` 或 `tf.convert_to_tensor()` 从 Python 列表或 NumPy 数组创建张量。

```python
# TensorFlow
import tensorflow as tf
import numpy as np

# From a Python list
tf_tensor_list = tf.constant([1, 2, 3])
print(f"TF tensor from list: {tf_tensor_list}, dtype: {tf_tensor_list.dtype}")

# From a NumPy array
np_array = np.array([4.0, 5.0, 6.0])
tf_tensor_numpy = tf.convert_to_tensor(np_array)
print(f"TF tensor from NumPy: {tf_tensor_numpy}, dtype: {tf_tensor_numpy.dtype}")
```

PyTorch 也提供了类似的功能，使用 `torch.tensor()`。需要注意的是，`torch.tensor()` 总是复制数据。如果你想创建一个与 NumPy 数组共享内存的张量（稍后会详细介绍），则可以使用 `torch.from_numpy()` 或 `torch.as_tensor()`。

```python
# PyTorch
import torch
import numpy as np

# From a Python list
pt_tensor_list = torch.tensor([1, 2, 3])
print(f"PyTorch tensor from list: {pt_tensor_list}, dtype: {pt_tensor_list.dtype}")

# From a NumPy array (copies data)
np_array = np.array([4.0, 5.0, 6.0])
pt_tensor_numpy_copy = torch.tensor(np_array)
print(f"PyTorch tensor from NumPy (copy): {pt_tensor_numpy_copy}, dtype: {pt_tensor_numpy_copy.dtype}")

# From a NumPy array (shares memory, if on CPU)
pt_tensor_numpy_share = torch.from_numpy(np_array)
print(f"PyTorch tensor from NumPy (share): {pt_tensor_numpy_share}, dtype: {pt_tensor_numpy_share.dtype}")
```

注意默认整数类型的一个细微差异：TensorFlow 对于从 Python 列表创建的整数通常默认为 `tf.int32`，而 PyTorch 默认为 `torch.int64`（也称为 `torch.long`）。对于浮点数，两者通常都默认为 32 位精度（`tf.float32` 和 `torch.float32`）。

### 用特定值初始化张量

创建填充零、一或随机数的张量是常见的操作。

**TensorFlow：**

```python
# TensorFlow
tf_zeros = tf.zeros(shape=(2, 3), dtype=tf.float32)
tf_ones = tf.ones(shape=(2, 3), dtype=tf.int32)
tf_random = tf.random.normal(shape=(2, 3), mean=0.0, stddev=1.0)
tf_range = tf.range(start=0, limit=5, delta=1)

print(f"TF Zeros:\n{tf_zeros}")
print(f"TF Ones:\n{tf_ones}")
print(f"TF Random Normal:\n{tf_random}")
print(f"TF Range:\n{tf_range}")
```

**PyTorch：**

```python
# PyTorch
pt_zeros = torch.zeros(size=(2, 3), dtype=torch.float32)
pt_ones = torch.ones(size=(2, 3), dtype=torch.int32)
pt_random = torch.randn(size=(2, 3)) # mean 0, std 1 by default
pt_arange = torch.arange(start=0, end=5, step=1) # note 'end' instead of 'limit'

print(f"PyTorch Zeros:\n{pt_zeros}")
print(f"PyTorch Ones:\n{pt_ones}")
print(f"PyTorch Random Normal:\n{pt_random}")
print(f"PyTorch arange:\n{pt_arange}")
```

函数名和参数 (parameter)名通常非常相似（例如，`tf.zeros` 与 `torch.zeros`，`tf.random.normal` 与 `torch.randn`）。PyTorch 的 `torch.arange` 使用 `end` 表示独占上限，而 TensorFlow 的 `tf.range` 使用 `limit`。

两个框架也都提供了 `*_like` 函数（例如，`tf.zeros_like()`，`torch.zeros_like()`），用于创建与现有张量具有相同形状和数据类型的新张量。

## 张量主要属性：比较视图

了解张量的属性非常必要。`tf.Tensor` 和 `torch.Tensor` 都提供了数据类型、形状以及它们所在的设备等属性。

### 数据类型 (`dtype`)

在 TensorFlow 中，数据类型是 `tf.DType` 的实例（例如，`tf.float32`，`tf.int64`）。在 PyTorch 中，它是一个 `torch.dtype` 对象（例如，`torch.float32`，`torch.int64`）。

```python
# TensorFlow
tf_t = tf.constant([1.0, 2.0])
print(f"TF dtype: {tf_t.dtype}") # tf.float32

# PyTorch
pt_t = torch.tensor([1.0, 2.0])
print(f"PyTorch dtype: {pt_t.dtype}") # torch.float32

# Explicitly setting dtype
pt_t_int = torch.tensor([1, 2], dtype=torch.int16)
print(f"PyTorch explicit dtype: {pt_t_int.dtype}") # torch.int16
```

如前所述，请注意默认整数类型。PyTorch 对于从 Python 列表创建的整数通常默认为 `torch.int64`（或 `torch.long`），而 TensorFlow 则倾向于 `tf.int32`。你总可以在创建时明确指定 `dtype`。

要更改张量的数据类型：

- TensorFlow: `tf_new_type = tf.cast(tf_t, dtype=tf.int32)`
- PyTorch: `pt_new_type = pt_t.to(dtype=torch.int32)` 或 `pt_new_type = pt_t.int()`（适用于常见类型）

### 形状

张量的形状描述了它的维度。

- TensorFlow: `shape` 属性是一个 `tf.TensorShape` 对象。你可以使用 `tf_t.shape.as_list()` 将其转换为 Python 列表。
- PyTorch: `shape` 属性是一个 `torch.Size` 对象，其行为类似于元组。你也可以使用 `size()` 方法，它返回相同的 `torch.Size` 对象。

```python
# TensorFlow
tf_t_shape = tf.zeros((2, 3, 4))
print(f"TF shape object: {tf_t_shape.shape}")         # TensorShape([2, 3, 4])
print(f"TF shape as list: {tf_t_shape.shape.as_list()}") # [2, 3, 4]
print(f"TF rank: {tf_t_shape.ndim}")                  # 3

# PyTorch
pt_t_shape = torch.zeros(2, 3, 4)
print(f"PyTorch shape object: {pt_t_shape.shape}")    # torch.Size([2, 3, 4])
print(f"PyTorch size() method: {pt_t_shape.size()}") # torch.Size([2, 3, 4])
print(f"PyTorch individual dim: {pt_t_shape.shape[0]}")# 2
print(f"PyTorch rank: {pt_t_shape.ndim}")             # 3
```

两者也都提供了 `ndim` 属性，用于表示维度数量（秩）。

### 设备

张量可以存在于不同的计算设备上，主要是 CPU 或 GPU。

- TensorFlow: `device` 属性是一个字符串，例如 `'/job:localhost/replica:0/task:0/device:CPU:0'` 或 `'/job:localhost/replica:0/task:0/device:GPU:0'`。
- PyTorch: `device` 属性是一个 `torch.device` 对象，例如 `device(type='cpu')` 或 `device(type='cuda', index=0)`。

```python
# TensorFlow (device placement is more implicit or via tf.device context)
with tf.device('/CPU:0'):
    tf_t_cpu = tf.constant([1, 2])
print(f"TF tensor device: {tf_t_cpu.device}")

# if gpus are available and TensorFlow is configured for GPU
# with tf.device('/GPU:0'):
#    tf_t_gpu = tf.constant([1,2])
# print(f"TF tensor device (GPU): {tf_t_gpu.device}")

# PyTorch
pt_t_cpu = torch.tensor([1, 2]) # Defaults to CPU
print(f"PyTorch tensor device (default): {pt_t_cpu.device}")

if torch.cuda.is_available():
    pt_t_gpu = torch.tensor([1, 2], device='cuda')
    # Or: pt_t_gpu = pt_t_cpu.to('cuda')
    # Or: pt_t_gpu = pt_t_cpu.cuda()
    print(f"PyTorch tensor device (GPU): {pt_t_gpu.device}")
else:
    print("PyTorch: CUDA not available, GPU example skipped.")
```

在 PyTorch 中，使用 `.to(device)` 方法（例如，`my_tensor.to('cuda')` 或 `my_tensor.to('cpu')`）可以明确地在设备之间移动张量。我们将在后面的章节中更详细地介绍设备管理。

## 差异总结

尽管 `tf.Tensor` 和 `torch.Tensor` 在表示多维数据方面有共同目的，但对于从 TensorFlow 过渡的开发者而言，一些差异尤为重要：

1. **可变性**：

   - **TensorFlow 张量**：通常是不可变的。对 `tf.Tensor` 的操作通常会创建并返回新的张量。要拥有可变状态，TensorFlow 使用 `tf.Variable`。
   - **PyTorch 张量**：可变的。许多操作可以就地修改张量（通常通过后缀下划线表示，例如 `tensor.add_()`）。如果小心使用，这可以生成更节省内存的代码，但需要注意避免意外的副作用。
2. **`tf.Variable` 与 `torch.Tensor(requires_grad=True)` 对比**：

   - 在 TensorFlow 中，`tf.Variable` 是一个特殊对象，用于保存可变的张量状值，通常是模型参数 (parameter)，其修改需要被跟踪以进行自动微分。
   - 在 PyTorch 中，任何 `torch.Tensor` 都可以通过将其 `requires_grad` 属性设置为 `True` 来标记 (token)其操作需要被跟踪（例如，`x = torch.randn(3, requires_grad=True)`）。这些仍然是常规的可变张量。`requires_grad` 标志只是告诉 PyTorch 的自动求导系统记录涉及它们的运算以便进行梯度计算。
3. **NumPy 桥接与内存共享**：

   - 两个框架都与 NumPy 良好互通。
   - `tf_tensor.numpy()`: 将 TensorFlow 张量转换为 NumPy 数组。如果张量在 GPU 上，数据会被复制到 CPU。
   - `torch_tensor.numpy()`: 将 PyTorch 张量转换为 NumPy 数组。**尤其值得注意的是，如果 `torch.Tensor` 在 CPU 上，返回的 NumPy 数组会共享相同的底层内存。** 修改其中一个会影响另一个。如果张量在 GPU 上，则会复制一份数据到 CPU 内存。
   - `tf.convert_to_tensor(numpy_array)`: 从 NumPy 数组创建 TensorFlow 张量，通常会复制数据。
   - `torch.from_numpy(numpy_array)`: 从 NumPy 数组创建 PyTorch 张量，如果数组在 CPU 上则共享内存。为确保复制数据，请使用 `torch.tensor(numpy_array)`。

PyTorch 中 CPU 张量和 NumPy 数组的这种内存共享行为是一项性能特点，但需要仔细处理。如果你需要一个独立的副本，PyTorch 到 NumPy 可以使用 `tensor.clone().numpy()`，NumPy 到 PyTorch 则使用 `torch.tensor(numpy_array)`。

下图说明了 PyTorch CPU 张量与 NumPy 数组之间的内存共享特点：

> NumPy 数组与 PyTorch 张量在 CPU 上的交互。`torch.from_numpy()` 和 `.numpy()` 方法（在 CPU 张量上）有助于内存共享，而 `torch.tensor()` 则确保数据复制。

理解 `torch.Tensor` 的这些特点、其创建方法、属性、可变性以及与 NumPy 的交互，是学习的重要一步。随着学习的深入，你将看到这些特性如何影响你在 PyTorch 更命令式风格中构建模型和编写训练循环的方式。下一节将介绍常见的张量操作，进一步强调与 TensorFlow 相比的异同。

## 参考资料

- [Tensors](https://www.tensorflow.org/guide/tensor) — TensorFlow Developers (2024)
  这份官方指南是理解TensorFlow的tf.Tensor对象及其创建、属性和基本操作的主要资料。
- [Tensors](https://pytorch.org/tutorials/beginner/pt_tensor_tutorial.html) — PyTorch Developers (2024)
  Publisher: PyTorch
  这份入门教程概述了torch.Tensor对象，涵盖了PyTorch中的创建、常见属性和基本操作。
- [Tensors](https://d2l.ai/chapter_introduction/tensors.html) — Aston Zhang, Zack C. Lipton, Mu Li, Alex J. Smola (2024)
  Publisher: Cambridge University Press
  《动手学深度学习》教材中的一个章节，在深度学习背景下解释了张量，经常提供PyTorch和TensorFlow两者的代码示例，有助于直接比较。
- [torch.Tensor](https://pytorch.org/docs/stable/tensors.html) — PyTorch Developers (2024)
  torch.Tensor的官方API参考文档，详细说明了其属性、方法和构造函数，包括requires_grad和与NumPy的内存共享信息。
