# NumPy 在 PyTorch 和 TensorFlow 中的结合

来源：[原文](https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-1-pytorch-tensorflow-core-concepts/numpy-integration-pytorch-tensorflow)

[返回章节目录](README.md) · [返回课程目录](../README.md)

NumPy 数组是科学 Python 生态系统的重要组成部分，PyTorch 和 TensorFlow 都设计为可与它们良好兼容。这种关联非常有益，因为它让你可以使用大量兼容 NumPy 的库，并简化了核心深度学习 (deep learning)计算之外的数据处理任务。TensorFlow 开发者通常会在 `tf.Tensor` 对象与 NumPy 数组之间进行数据转换。在此，我们将查看 PyTorch 如何处理此事，并指出它们的相似点和不同点。

### TensorFlow 与 NumPy 的衔接

TensorFlow 提供了直接的机制，用于将其张量转换为 NumPy 数组，反之亦然。这对于调试、可视化，或与需要 NumPy 数组的库集成时特别有用。

**从 `tf.Tensor` 到 NumPy 数组**

如果你有一个 `tf.Tensor`，可以使用其 `.numpy()` 方法获取其 NumPy 数组形式。这是一个你可能多次使用过的常见操作。

```python
import tensorflow as tf
import numpy as np

# 创建一个 TensorFlow 张量
tf_tensor_cpu = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)

# 转换为 NumPy 数组
numpy_array_from_tf = tf_tensor_cpu.numpy()
print("NumPy 数组来自 tf.Tensor (CPU):\n", numpy_array_from_tf)
print("类型:", type(numpy_array_from_tf))

# 如果张量在 CPU 上，NumPy 数组会共享内存
# 如果共享内存，修改 NumPy 数组会反映到原始的 CPU 张量中。
# (注意：这种共享行为在 TensorFlow 2.x 的 CPU 张量中很常见)
numpy_array_from_tf[0, 0] = 99.0
print("修改后的 tf.Tensor (CPU) 在 NumPy 数组改变后:\n", tf_tensor_cpu)

# 对于 GPU 张量，.numpy() 涉及从 GPU 到 CPU 内存的复制
if tf.config.list_physical_devices('GPU'):
    with tf.device('/GPU:0'):
        tf_tensor_gpu = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    numpy_array_from_gpu = tf_tensor_gpu.numpy() # 数据从 GPU 复制到 CPU
    print("\nNumPy 数组来自 tf.Tensor (GPU):\n", numpy_array_from_gpu)
    numpy_array_from_gpu[0,0] = 100.0 # 这会修改复制的 NumPy 数组
    print("原始 tf.Tensor (GPU) 在 NumPy 数组改变后 (预计无变化):\n", tf_tensor_gpu)
else:
    print("\n没有可用的 GPU。跳过 GPU 张量到 NumPy 的示例。")
```

当 `tf.Tensor` 位于 CPU 上时，`.numpy()` 方法通常会返回一个共享底层内存的 NumPy 数组。这意味着对 NumPy 数组的修改会影响原始张量，反之亦然。然而，如果张量在 GPU 上，`.numpy()` 将首先把张量的数据复制到 CPU 内存，然后生成的 NumPy 数组将不与原始 GPU 张量共享内存。

**从 NumPy 数组到 `tf.Tensor`**

要从 NumPy 数组创建一个 `tf.Tensor`，可以使用 `tf.convert_to_tensor()` 或 `tf.constant()`。

```python
import tensorflow as tf
import numpy as np

numpy_array = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)

# 将 NumPy 数组转换为 tf.Tensor
tf_tensor_from_numpy = tf.convert_to_tensor(numpy_array)
print("\ntf.Tensor 来自 NumPy 数组:\n", tf_tensor_from_numpy)
print("类型:", type(tf_tensor_from_numpy))
print("设备:", tf_tensor_from_numpy.device)

# tf.convert_to_tensor 通常会复制数据。
# 对原始 NumPy 数组的修改不会影响张量。
numpy_array[0, 0] = 111.0
print("原始 NumPy 数组修改后:\n", numpy_array)
print("tf.Tensor (应保持不变):\n", tf_tensor_from_numpy)
```

通常，`tf.convert_to_tensor()` 和 `tf.constant()` 通过复制 NumPy 数组的数据来创建新张量。这确保了 TensorFlow 张量拥有自己的内存，由 TensorFlow 运行时管理。

### PyTorch 对 NumPy 的紧密连接

PyTorch 与 NumPy 的结合非常直接，尤其是在 CPU 上张量的内存共享方面。这常被认为是 PyTorch 的用户友好特性之一。

**从 `torch.Tensor` 到 NumPy 数组**

与 TensorFlow 类似，你可以使用 `.numpy()` 方法将 PyTorch `torch.Tensor` 转换为 NumPy 数组。

```python
import torch
import numpy as np

# 创建一个 PyTorch 张量（默认为 CPU）
pt_tensor_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

# 转换为 NumPy 数组
numpy_array_from_pt = pt_tensor_cpu.numpy()
print("NumPy 数组来自 torch.Tensor (CPU):\n", numpy_array_from_pt)
print("类型:", type(numpy_array_from_pt))

# 对于 CPU 张量，NumPy 数组与 PyTorch 张量共享内存
numpy_array_from_pt[0, 0] = 99.0
print("修改后的 torch.Tensor (CPU) 在 NumPy 数组改变后:\n", pt_tensor_cpu)

# 如果张量在 GPU 上，.numpy() 首先需要 .cpu()，这是一个复制操作
if torch.cuda.is_available():
    pt_tensor_gpu = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    # 要将 GPU 张量转换为 NumPy，首先将其移至 CPU
    numpy_array_from_gpu_pt = pt_tensor_gpu.cpu().numpy() # .cpu() 将数据复制到 CPU
    print("\nNumPy 数组来自 torch.Tensor (经由 CPU 的 GPU 张量):\n", numpy_array_from_gpu_pt)
    numpy_array_from_gpu_pt[0,0] = 100.0 # 修改 CPU 副本
    print("原始 torch.Tensor (GPU) 在 NumPy 数组改变后 (预计无变化):\n", pt_tensor_gpu)
else:
    print("\nPyTorch 没有可用的 GPU。跳过 GPU 张量到 NumPy 的示例。")
```

PyTorch 的一个特点是，如果 `torch.Tensor` 位于 CPU 上，`.numpy()` 返回的 NumPy 数组会**共享相同的底层内存**。对一个的修改会反映到另一个上。这可以非常高效，但需要留意以避免意外的副作用。如果张量在 GPU 上，你必须首先使用 `.cpu()` 将其移至 CPU，这会执行数据复制。随后的 `.numpy()` 调用将与该张量的 CPU 副本共享内存。

**从 NumPy 数组到 `torch.Tensor`**

PyTorch 提供两种主要方式来从 NumPy 数组创建张量：

1. `torch.from_numpy(numpy_array)`: 此函数创建一个与 NumPy 数组**共享内存**的 `torch.Tensor`。这效率很高，因为它避免了数据复制。当你想要这种共享行为时，这是首选方法。返回的张量和 NumPy 数组将指向相同的内存位置（对于 CPU 数据）。
2. `torch.tensor(numpy_array)`: 此函数更像一个通用张量构造函数。它**总是复制** NumPy 数组中的数据，以创建拥有自己内存的新 `torch.Tensor`。

```python
import torch
import numpy as np

numpy_array_source = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)

# 使用 torch.from_numpy()（共享内存）
pt_tensor_shared = torch.from_numpy(numpy_array_source)
print("\ntorch.Tensor 来自 NumPy（共享内存）:\n", pt_tensor_shared)

numpy_array_source[0, 0] = 111.0 # 修改原始 NumPy 数组
print("原始 NumPy 数组修改后:\n", numpy_array_source)
print("共享的 torch.Tensor（反映变化）:\n", pt_tensor_shared)

pt_tensor_shared[1, 1] = 444.0 # 修改共享的 PyTorch 张量
print("修改后的共享 torch.Tensor:\n", pt_tensor_shared)
print("原始 NumPy 数组（反映变化）:\n", numpy_array_source)

# 为下一个示例重置 numpy_array_source
numpy_array_source = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)

# 使用 torch.tensor()（复制数据）
pt_tensor_copied = torch.tensor(numpy_array_source)
print("\ntorch.Tensor 来自 NumPy（复制数据）:\n", pt_tensor_copied)

numpy_array_source[0, 0] = 222.0 # 修改原始 NumPy 数组
print("原始 NumPy 数组修改后:\n", numpy_array_source)
print("复制的 torch.Tensor（应保持不变）:\n", pt_tensor_copied)
```

`torch.from_numpy()`（共享）和 `torch.tensor()`（复制）之间的这种区别很重要。`torch.from_numpy()` 特别适用于与生成 NumPy 数组的现有数据处理管线集成，允许 PyTorch 在无需昂贵的数据复制操作的情况下使用该数据，前提是数据位于 CPU 上。

### PyTorch 与 TensorFlow 的 NumPy 衔接比较

尽管两个框架都提供 NumPy 互操作性，但 CPU 数据内存共享的默认行为是一个值得注意的区别。

> NumPy 数组与框架特有张量之间的转换路径（CPU 环境）。请注意内存共享默认行为的不同。

**内存管理差异（以 CPU 为主）：**

- **PyTorch `torch.from_numpy()`**: 明确地与 NumPy 数组共享内存。
- **PyTorch `torch.Tensor.numpy()`**: 与 CPU `torch.Tensor` 共享内存。
- **TensorFlow `tf.Tensor.numpy()`**: 与 CPU `tf.Tensor` 共享内存。
- **TensorFlow `tf.convert_to_tensor()` (或 `tf.constant()`)**: 通常复制 NumPy 数组中的数据。

PyTorch 的 `torch.from_numpy()` 为 CPU 数据提供了从 NumPy 到 PyTorch 的直接零拷贝桥梁，这对于数据加载和预处理的性能很有利。TensorFlow 从 NumPy 的转换通常涉及复制，确保数据独立性，除非满足特定的底层 API 或条件。对于转换为 NumPy 的操作，两个框架对于 CPU 张量表现相似，提供内存共享视图。

**数据类型一致性**

两个框架都尝试在转换过程中保留数据类型，但明确指定是良好的实践。NumPy 的默认浮点类型通常是 `float64`（双精度），而深度学习 (deep learning)框架为了效率主要使用 `float32`（单精度）。转换时，请确保你的数据类型符合预期，以防止不易察觉的错误或性能问题。例如：

```python
import numpy as np
import torch
import tensorflow as tf

# NumPy 默认浮点类型为 float64
numpy_float64 = np.array([1.0, 2.0, 3.0])
print(f"NumPy 数组数据类型: {numpy_float64.dtype}")

# PyTorch 转换
pt_tensor_from_f64 = torch.from_numpy(numpy_float64)
print(f"PyTorch 张量数据类型（来自 float64 NumPy）: {pt_tensor_from_f64.dtype}") # torch.float64

# TensorFlow 转换
tf_tensor_from_f64 = tf.convert_to_tensor(numpy_float64)
print(f"TensorFlow 张量数据类型（来自 float64 NumPy）: {tf_tensor_from_f64.dtype}") # tf.float64

# 最佳实践：如有需要请指定数据类型
numpy_float32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
pt_tensor_f32 = torch.from_numpy(numpy_float32)
print(f"PyTorch 张量数据类型（来自 float32 NumPy）: {pt_tensor_f32.dtype}") # torch.float32
```

### 这为何对你的工作流程很重要

理解 PyTorch 和 TensorFlow 如何与 NumPy 衔接不仅仅是学术问题；它具有实际意义：

- **高效数据预处理**: 你可以使用 NumPy 丰富的功能集执行复杂数据操作，然后高效地将这些数组转换为张量进行模型训练，特别是在 PyTorch 中使用 `torch.from_numpy()` 在 CPU 上实现零拷贝。
- **使用外部库**: 许多科学计算、数据分析和可视化库（例如 scikit-learn、Matplotlib、Pandas、OpenCV）都是围绕 NumPy 构建的。顺畅的转换使你可以轻松地将这些工具集成到你的深度学习 (deep learning)管线中。
- **调试和检查**: 将张量转换为 NumPy 数组允许你使用熟悉的 NumPy/Python 工具来检查数值、查看形状，以及对模型中的中间结果执行完整性检查。
- **性能**: 对于 CPU 密集型操作或大型数据集，通过利用共享内存（在适当且安全的情况下）避免不必要的数据复制可以带来明显的性能提升。PyTorch 的 `torch.from_numpy()` 就是一个很好的例子。

当你从 TensorFlow 过渡到 PyTorch 时，你会发现虽然许多高层思想相似，但 NumPy 整合的这些细节，特别是在内存管理方面，代表着不明显但重要的区别，体现在框架的设计和使用方式上。认识到这些不同将帮助你编写更高效、更少错误的 PyTorch 代码。

## 参考资料

- [Tensors](https://www.tensorflow.org/guide/tensor) — TensorFlow Documentation Team (2024)
  Publisher: Google
  详述TensorFlow的张量操作，包括与NumPy数组的转换及内存处理。
- [torch.Tensor](https://pytorch.org/docs/stable/tensors.html) — PyTorch Documentation Team (2024)
  Publisher: PyTorch
  涵盖PyTorch张量从NumPy创建、转换为NumPy以及内存共享机制。
- [The N-dimensional array (ndarray)](https://numpy.org/doc/stable/reference/arrays.ndarray.html) — NumPy Documentation Team (2024)
  关于NumPy ndarray对象的根本性参考，该对象构成数值运算的基础。
- [Deep Learning with PyTorch: A Book for AI Innovators and Researchers](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, Thomas Viehmann (2020)
  Publisher: Manning Publications
  一本关于PyTorch的全面资源，涵盖张量与NumPy互操作性的实践方面。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本关于使用TensorFlow进行机器学习的实践指南，涵盖数据操作和NumPy集成。

---

[上一节](05-%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86%EF%BC%9AGradientTape%20%E4%B8%8E%20Autograd%20%E5%AF%B9%E6%AF%94.md) · [下一节](07-%E8%AE%BE%E5%A4%87%E7%AE%A1%E7%90%86%EF%BC%9ACPU%E5%92%8CGPU%E6%8E%A7%E5%88%B6.md)
