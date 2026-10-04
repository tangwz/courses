---
course: "getting-started-with-pytorch"
chapter: "pytorch-fundamentals-setup"
lesson: "relationship-numpy"
sourceId: 2095
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-1-pytorch-fundamentals-setup/relationship-numpy"
title: "与 NumPy 的关联"
description: "了解 PyTorch 张量与 NumPy 数组的异同点，包括它们之间的转换方法。"
order: 6
plots: []
sourceHash: "d26ff5ced74b70dc3abd4746c7c16ed6c2609c121c59f2c99f8e33e610b16c78"
sourceCorrections: []
---

如果你对 Python 中的科学计算有经验，你可能很熟悉 NumPy 及其 `ndarray` 对象。NumPy 提供了一种强大的 N 维数组结构，已成为 Python 中数值操作的标准。PyTorch 认识到这种普遍性，并提供了与 NumPy 数组出色的互操作性。事实上，PyTorch 张量与 NumPy 数组非常相似：它们都是数字多维网格的抽象。

这种密切关联使得两者之间的切换变得简单，让你在 PyTorch 生态系统中工作时，能够运用现有的 NumPy 代码或库。

### 相似结构，不同能力

从本质上看，PyTorch 张量和 NumPy 数组都表示多维的密集数值数据。你可能在 NumPy 数组上执行的许多操作，在 PyTorch 张量中都有直接对应，通常具有相似的命名约定：

- **创建：** 两个库都提供函数来创建填充零、一、随机数或从现有 Python 列表创建数组/张量。
- **数学运算：** 逐元素加法、减法、乘法、除法、求幂、三角函数等，操作方式相似。
- **索引和切片：** 访问和修改元素或子数组/子张量使用可比较的语法。
- **形状操作：** 改变数组/张量的形状、转置和连接遵循相似的原则。

然而，主要由于 PyTorch 对深度学习 (deep learning)的侧重，两者之间存在一些基本区别：

1. **GPU 加速：** PyTorch 张量可以移到图形处理单元 (GPU) 上处理。这使得大规模并行计算成为可能，为深度学习中常见的矩阵乘法及其他操作提供了显著的速度提升。NumPy 数组主要为 CPU 计算设计。
2. **自动微分：** PyTorch 张量通过 `Autograd` 系统（第 3 章会讲到）内置支持自动微分。这种机制会自动跟踪对需要梯度的张量执行的操作，并在反向传播 (backpropagation)期间计算这些梯度，这对训练神经网络 (neural network)来说很重要。NumPy 数组不具备此能力。

### NumPy 数组与张量之间的转换

PyTorch 使得在这两种数据结构之间转换变得简单。

#### NumPy 数组到 PyTorch 张量

你可以使用 `torch.from_numpy()` 函数直接从 NumPy 数组创建 PyTorch 张量。

```python
import numpy as np
import torch

# 创建一个 NumPy 数组
numpy_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
print(f"NumPy 数组:\n{numpy_array}")
print(f"NumPy 数组类型: {numpy_array.dtype}")

# 将 NumPy 数组转换为 PyTorch 张量
pytorch_tensor = torch.from_numpy(numpy_array)
print(f"\nPyTorch 张量:\n{pytorch_tensor}")
print(f"PyTorch 张量类型: {pytorch_tensor.dtype}")
```

**内存共享的重要说明：** 使用 `torch.from_numpy()` 时，生成的 PyTorch 张量和原始 NumPy 数组在 CPU 上共享相同的底层内存位置。这意味着修改一个对象会影响另一个。这种行为很高效，因为它避免了数据复制，但你需要注意这一点。

```python
# 修改 NumPy 数组
numpy_array[0, 0] = 99
print(f"\n修改后的 NumPy 数组:\n{numpy_array}")
print(f"修改 NumPy 数组后的 PyTorch 张量:\n{pytorch_tensor}")

# 修改 PyTorch 张量
pytorch_tensor[1, 1] = -1
print(f"\n修改后的 PyTorch 张量:\n{pytorch_tensor}")
print(f"修改 PyTorch 张量后的 NumPy 数组:\n{numpy_array}")
```

如你所见，更改会反映在两个对象中，因为它们指向内存中的相同数据。

#### PyTorch 张量到 NumPy 数组

反之，你可以使用 `.numpy()` 方法将位于 CPU 上的 PyTorch 张量转换回 NumPy 数组。

```python
# 在 CPU 上创建一个 PyTorch 张量
cpu_tensor = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
print(f"原始 PyTorch 张量 (CPU):\n{cpu_tensor}")

# 将张量转换为 NumPy 数组
numpy_array_converted = cpu_tensor.numpy()
print(f"\n转换后的 NumPy 数组:\n{numpy_array_converted}")
print(f"NumPy 数组类型: {numpy_array_converted.dtype}")
```

同样，生成的 NumPy 数组和原始 CPU 张量共享相同的底层内存。对一个的修改会影响另一个。

```python
# 修改张量
cpu_tensor[0, 1] = 25.0
print(f"\n修改后的 PyTorch 张量:\n{cpu_tensor}")
print(f"修改张量后的 NumPy 数组:\n{numpy_array_converted}")

# 修改 NumPy 数组
numpy_array_converted[1, 0] = 35.0
print(f"\n修改后的 NumPy 数组:\n{numpy_array_converted}")
print(f"修改 NumPy 数组后的张量:\n{cpu_tensor}")
```

**GPU 张量：** `.numpy()` 方法仅适用于存储在 CPU 上的张量。如果你的张量在 GPU 上，你必须先使用 `.cpu()` 方法将其移到 CPU，然后才能将其转换为 NumPy 数组。直接在 GPU 张量上调用 `.numpy()` 会导致错误。

```python
# 假设有 GPU 可用的示例
if torch.cuda.is_available():
    gpu_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    print(f"\nGPU 上的张量:\n{gpu_tensor}")

    # 这将导致错误: numpy_from_gpu = gpu_tensor.numpy()

    # 正确方法: 先移到 CPU
    cpu_tensor_from_gpu = gpu_tensor.cpu()
    numpy_from_gpu = cpu_tensor_from_gpu.numpy()
    print(f"\n转换后的 NumPy 数组 (来自 GPU 张量):\n{numpy_from_gpu}")

    # 注意: numpy_from_gpu 与 cpu_tensor_from_gpu 共享内存，
    # 但不与原始的 gpu_tensor 共享。
else:
    print("\nCUDA 不可用，跳过 GPU 到 NumPy 的示例。")
```

### 结合两者的优势

轻松地在 NumPy 数组和 PyTorch 张量之间转换的能力非常实用。你可能使用熟悉的 NumPy 函数或其他操作 NumPy 数组的库来执行初始数据加载和预处理。然后，当需要构建或训练深度学习 (deep learning)模型时，你可以将数据转换为 PyTorch 张量，以借助于 GPU 加速和自动微分。同样，模型输出（即张量）可以转换回 NumPy 数组，以便使用 Matplotlib 或 Seaborn 等库进行分析或可视化。

理解这种关联和内存共享的含义，让你能够编写高效的代码，有效连接通用科学 Python 生态系统与 PyTorch 提供的专门深度学习能力。

## 参考资料

- [PyTorch Tensors](https://pytorch.org/docs/stable/tensors.html) — PyTorch Documentation (2024)
  PyTorch Tensor 的官方文档，涵盖其属性、操作和设备放置功能。
- [The N-dimensional array (ndarray)](https://numpy.org/doc/stable/reference/arrays.ndarray.html) — NumPy Documentation (2024)
  解释 NumPy 基本 N 维数组对象的官方文档。
- [Introduction to Tensors](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html) — PyTorch Tutorials (2024)
  Publisher: PyTorch
  PyTorch 官方教程，介绍 Tensor 及其操作，以及与 NumPy 数组的互操作性。
- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, and Thomas Viehmann (2020)
  Publisher: Manning Publications
  一本关于学习 PyTorch 的综合性书籍，涵盖 Tensor、自动微分和构建深度学习模型。
