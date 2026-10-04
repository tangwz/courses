---
course: "getting-started-with-pytorch"
chapter: "pytorch-fundamentals-setup"
lesson: "creating-tensors"
sourceId: 2089
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-1-pytorch-fundamentals-setup/creating-tensors"
title: "创建张量"
description: "从数据、指定形状或随机值创建张量的方法。"
order: 4
plots: []
sourceHash: "78215abdbc48de874b188a576eb49e826114ebdb1027d9834e3534830f1af1dd"
sourceCorrections: []
---

张量是 PyTorch 中的基本数据结构，代表深度学习 (deep learning)模型中使用的多维数组。代码中提供了创建张量的实用方法。PyTorch 提供了一系列灵活的函数来创建张量，无论使用现有数据，还是需要特定形状和初始值。

### 从现有数据创建张量

创建张量最直接的方法通常是使用现有的 Python 数据结构，比如列表或 NumPy 数组。主要用于此目的的函数是 `torch.tensor()`。这个函数会根据输入数据自动判断数据类型 (`dtype`)，但您也可以明确指定。重要的是，`torch.tensor()` 总是复制输入数据。

我们来看看实际操作。首先，请确保已导入 PyTorch：

```python
import torch
import numpy as np
```

现在，从 Python 列表创建一个张量：

```python
# 从 Python 列表创建张量
data_list = [[1, 2], [3, 4]]
tensor_from_list = torch.tensor(data_list)

print("从列表生成的张量:")
print(tensor_from_list)
print(f"数据类型: {tensor_from_list.dtype}")
print(f"形状: {tensor_from_list.shape}")
```

这将输出：

```
从列表生成的张量:
tensor([[1, 2],
        [3, 4]])
数据类型: torch.int64
形状: torch.Size([2, 2])
```

请注意 PyTorch 如何根据输入整数判断数据类型为 `torch.int64`（一个 64 位整数）。

您也可以从 NumPy 数组开始，获得相同的结果：

```python
# 从 NumPy 数组创建张量
data_numpy = np.array([[5.0, 6.0], [7.0, 8.0]])
tensor_from_numpy = torch.tensor(data_numpy)

print("\n从 NumPy 数组生成的张量:")
print(tensor_from_numpy)
print(f"数据类型: {tensor_from_numpy.dtype}")
print(f"形状: {tensor_from_numpy.shape}")
```

输出：

```
从 NumPy 数组生成的张量:
tensor([[5., 6.],
        [7., 8.]], dtype=torch.float64)
数据类型: torch.float64
形状: torch.Size([2, 2])
```

这里，数据类型被判断为 `torch.float64`，因为 NumPy 数组包含浮点数。本章后面将讨论 PyTorch 张量与 NumPy 数组之间的关系，包括更有效的转换方法。

### 创建具有特定形状和值的张量

很多时候，您需要在没有预先数据的情况下创建特定大小的张量，例如用零、一或随机值进行初始化。PyTorch 为这些情况提供了专门的函数。形状通常作为元组或整数序列传递。

**零张量、一张量或未初始化数据张量：**

- `torch.zeros(*size, ...)`: 创建一个填充零的张量。
- `torch.ones(*size, ...)`: 创建一个填充一的张量。
- `torch.empty(*size, ...)`: 创建一个指定大小的张量，但其内容未被设定。其中的值是分配内存时该位置的任意数据。如果您计划在创建后立即填充张量，使用 `torch.empty` 会快一些，但请注意，初始值是不确定的。

```python
# 定义期望的形状
shape = (2, 3) # 2 行, 3 列

# 创建具有特定值的张量
zeros_tensor = torch.zeros(shape)
ones_tensor = torch.ones(shape)
empty_tensor = torch.empty(shape) # 值是任意的

print(f"\n零张量 (形状 {shape}):")
print(zeros_tensor)

print(f"\n一张量 (形状 {shape}):")
print(ones_tensor)

print(f"\n空张量 (形状 {shape}):")
print(empty_tensor)
```

输出（请注意 `empty_tensor` 的值会有所不同）：

```
零张量 (形状 (2, 3)):
tensor([[0., 0., 0.],
        [0., 0., 0.]])

一张量 (形状 (2, 3)):
tensor([[1., 1., 1.],
        [1., 1., 1.]])

空张量 (形状 (2, 3)):
tensor([[2.1321e-38, 5.0837e+24, 3.0763e-41],
        [3.0763e-41, 1.2326e-32, 1.8750e+00]])
```

默认情况下，这些函数创建的数据类型为 `dtype=torch.float32` 的张量。您可以使用 `dtype` 参数 (parameter)指定不同的数据类型：

```python
# 创建一个整数类型的一张量
ones_int_tensor = torch.ones(shape, dtype=torch.int32)
print(f"\n一张量 (dtype=torch.int32):")
print(ones_int_tensor)
```

输出：

```
一张量 (dtype=torch.int32):
tensor([[1, 1, 1],
        [1, 1, 1]], dtype=torch.int32)
```

**带随机值的张量：**

初始化模型参数通常会从随机值开始。PyTorch 为此提供了函数：

- `torch.rand(*size, ...)`: 创建一个从区间 [0, 1) 均匀采样的张量。
- `torch.randn(*size, ...)`: 创建一个从标准正态分布（均值 0，方差 1）采样的张量。

```python
# 创建带随机值的张量
rand_tensor = torch.rand(shape) # 均匀分布 [0, 1)
randn_tensor = torch.randn(shape) # 标准正态分布

print(f"\n随机张量 (均匀分布 [0, 1), 形状 {shape}):")
print(rand_tensor)

print(f"\n随机张量 (标准正态分布, 形状 {shape}):")
print(randn_tensor)
```

输出（每次运行结果会有所不同）：

```
随机张量 (均匀分布 [0, 1), 形状 (2, 3)):
tensor([[0.6580, 0.5089, 0.1642],
        [0.3742, 0.5989, 0.7775]])

随机张量 (标准正态分布, 形状 (2, 3)):
tensor([[-0.2651, -0.3249, -1.0134],
        [ 1.1314,  1.1751, -0.1411]])
```

### 基于其他张量创建张量

有时，您需要创建一个新张量，使其与现有张量具有相同的属性（如形状和 `dtype`）。这对于确保操作兼容性很有帮助。PyTorch 为此提供了 `_like` 变体函数：

- `torch.zeros_like(input_tensor, ...)`
- `torch.ones_like(input_tensor, ...)`
- `torch.rand_like(input_tensor, ...)`
- `torch.randn_like(input_tensor, ...)`

这些函数接受一个现有张量作为输入，并返回一个新张量，其内容为指定的值（零、一、随机），但会与输入张量的形状和 `dtype` 匹配，除非明确覆盖。

```python
# 使用现有张量作为模板
base_tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
print(f"\n基础张量 (形状 {base_tensor.shape}, dtype {base_tensor.dtype}):")
print(base_tensor)

# 创建与基础张量属性匹配的张量
zeros_like_base = torch.zeros_like(base_tensor)
rand_like_base = torch.rand_like(base_tensor)

print("\n类似基础张量的零张量:")
print(zeros_like_base)
print(f"形状: {zeros_like_base.shape}, dtype: {zeros_like_base.dtype}")

print("\n类似基础张量的随机张量:")
print(rand_like_base)
print(f"形状: {rand_like_base.shape}, dtype: {rand_like_base.dtype}")
```

输出：

```
基础张量 (形状 torch.Size([2, 2]), dtype torch.float32):
tensor([[1., 2.],
        [3., 4.]])

类似基础张量的零张量:
tensor([[0., 0.],
        [0., 0.]])
形状: torch.Size([2, 2]), dtype: torch.float32

类似基础张量的随机张量:
tensor([[0.1216, 0.5908],
        [0.3264, 0.9272]])
形状: torch.Size([2, 2]), dtype: torch.float32
```

如您所见，新张量 `zeros_like_base` 和 `rand_like_base` 自动从 `base_tensor` 继承了 `shape` (torch.Size([2, 2])) 和 `dtype` (torch.float32)。

这些方法为您生成所需的张量提供了一套全面的方式，它们支撑着深度学习 (deep learning)模型中的计算。在接下来的章节中，我们将了解如何操作这些张量并进行重要运算。

## 参考资料

- [PyTorch Tensors](https://pytorch.org/docs/stable/tensors.html) — PyTorch Developers (2024)
  Publisher: PyTorch Foundation
  这份官方指南介绍了PyTorch张量、它们的属性和核心操作。
- [\`torch.tensor\`](https://pytorch.org/docs/stable/generated/torch.tensor.html) — PyTorch Developers (2023)
  Publisher: PyTorch Foundation
  提供从Python列表和NumPy数组创建张量的详细API文档，包括参数细节和类型推断。
- [Tensor creation ops](https://pytorch.org/docs/stable/torch.html#creation-ops) — PyTorch Developers (2023)
  Publisher: PyTorch Foundation
  涵盖了`torch.zeros()`、`torch.ones()`、`torch.rand()`、`torch.randn()`等函数，以及用于根据特定形状和值初始化张量的`_like`变体。
- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, and Thomas Viehmann (2020)
  Publisher: Manning Publications
  一本受人尊敬的书籍，解释了在PyTorch中使用张量构建深度学习模型的实际应用和目的。
