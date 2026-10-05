# PyTorch 与 NumPy 的连接

来源：[原文](https://apxml.com/zh/courses/advanced-pytorch/chapter-6-custom-extensions-interoperability/pytorch-numpy-interfacing)

[返回章节目录](README.md) · [返回课程目录](../README.md)

NumPy 是 Python 科学计算堆栈的根基，它提供强大的 N 维数组对象和丰富的数学函数集。深度学习 (deep learning)工作流程常涉及使用 NumPy 预处理数据、与基于 NumPy 构建的库集成，或使用基于 NumPy 的工具分析模型输出。因此，PyTorch 张量与 NumPy 数组之间的高效连接是高级使用者的一项常见需求。PyTorch 为这些转换提供直接机制，尽可能减少数据拷贝。

### 将 PyTorch 张量转换为 NumPy 数组

获取 PyTorch 张量的 NumPy 表示最直接的方法是调用张量上的 `.numpy()` 方法。

```python
import torch
import numpy as np

# 创建一个 CPU 张量
cpu_tensor = torch.randn(2, 3)
print(f"原始 PyTorch 张量 (CPU):\n{cpu_tensor}")

# 转换为 NumPy 数组
numpy_array = cpu_tensor.numpy()
print(f"转换后的 NumPy 数组:\n{numpy_array}")
print(f"NumPy 数组类型: {type(numpy_array)}")
```

对于驻留在 CPU 上的张量，这种转换的一个重要方面是**内存共享**。生成的 NumPy 数组和原始 PyTorch 张量共享相同的底层内存缓冲区。这使得转换速度极快，因为无需复制数据。然而，这也意味着对其中一个对象的修改会反映到另一个对象上。

```python
# 演示内存共享 (CPU)
print("正在修改 PyTorch 张量...")
cpu_tensor.add_(1) # 原地加法

print(f"修改后的 PyTorch 张量:\n{cpu_tensor}")
print(f"PyTorch 张量修改后的 NumPy 数组:\n{numpy_array}")

print("\n正在修改 NumPy 数组...")
numpy_array[0, 0] = 99.0

print(f"修改后的 NumPy 数组:\n{numpy_array}")
print(f"NumPy 数组修改后的 PyTorch 张量:\n{cpu_tensor}")
```

这种内存共享行为高效但需要谨慎处理，以避免意外的副作用。

**GPU 张量：** NumPy 数组本质上是基于 CPU 的结构。因此，如果您的张量位于 GPU 上，您必须显式地使用 `.cpu()` 方法将其移动到 CPU，*然后*才能调用 `.numpy()`。此操作涉及从 GPU 到 CPU 内存的数据拷贝，并打破了在 CPU 张量中观察到的内存共享特性。

```python
if torch.cuda.is_available():
    # 创建一个 GPU 张量
    gpu_tensor = torch.randn(2, 3, device='cuda')
    print(f"\n原始 PyTorch 张量 (GPU):\n{gpu_tensor}")

    # 直接在 GPU 张量上调用 .numpy() 会引发错误
    # numpy_array_gpu_fail = gpu_tensor.numpy() # 这将导致 TypeError

    # 先移动到 CPU，然后转换
    cpu_tensor_copy = gpu_tensor.cpu()
    numpy_array_from_gpu = cpu_tensor_copy.numpy()
    print(f"转换后的 NumPy 数组 (从 GPU 经由 CPU):\n{numpy_array_from_gpu}")

    # 由于是拷贝，修改是独立的
    print("\n正在修改从 GPU 张量派生出的 NumPy 数组...")
    numpy_array_from_gpu[0, 0] = -50.0
    print(f"修改后的 NumPy 数组:\n{numpy_array_from_gpu}")
    print(f"原始 GPU 张量 (未改变):\n{gpu_tensor}")
    print(f"中间 CPU 张量 (未受 NumPy 修改影响):\n{cpu_tensor_copy}")
else:
    print("\nCUDA 不可用，跳过 GPU 张量转换示例。")
```

**与 Autograd 的交互：** `.numpy()` 转换要求张量不属于需要梯度计算的计算图，或者已明确分离。NumPy 操作不在 PyTorch 自动求导引擎的范围内。如果您尝试在 `requires_grad=True` 的张量上调用 `.numpy()`，PyTorch 将引发 `RuntimeError`。要执行此转换，您必须首先使用 `.detach()` 将张量从图中分离。

```python
# 需要梯度的张量
grad_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

# 直接调用 .numpy() 将失败
# numpy_fail = grad_tensor.numpy() # 引发 RuntimeError

# 先分离，然后转换
detached_tensor = grad_tensor.detach()
numpy_from_grad = detached_tensor.numpy()
print(f"\n从分离张量获得的 NumPy 数组:\n{numpy_from_grad}")

# 原始张量仍需要梯度
print(f"原始张量 requires_grad: {grad_tensor.requires_grad}")
# 分离的张量不需要
print(f"分离的张量 requires_grad: {detached_tensor.requires_grad}")
```

请记住，分离会创建一个新张量，它共享相同的数据，但与梯度历史断开。

### 将 NumPy 数组转换为 PyTorch 张量

要从 NumPy 数组创建 PyTorch 张量，主要函数是 `torch.from_numpy()`。

```python
# 创建一个 NumPy 数组
numpy_array_orig = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
print(f"\n原始 NumPy 数组:\n{numpy_array_orig}")

# 转换为 PyTorch 张量
pytorch_tensor = torch.from_numpy(numpy_array_orig)
print(f"转换后的 PyTorch 张量:\n{pytorch_tensor}")
print(f"PyTorch 张量类型: {pytorch_tensor.dtype}")
```

与 CPU 张量的 `.numpy()` 转换类似，`torch.from_numpy()` 默认情况下与原始 NumPy 数组**共享内存**，前提是数组的数据类型与 PyTorch 兼容。这允许高效的数据交换。

```python
# 演示内存共享 (NumPy -> PyTorch)
print("\n正在修改 NumPy 数组...")
numpy_array_orig[0, 0] = -10.0

print(f"修改后的 NumPy 数组:\n{numpy_array_orig}")
print(f"NumPy 数组修改后的 PyTorch 张量:\n{pytorch_tensor}")

print("\n正在修改 PyTorch 张量...")
pytorch_tensor.add_(5) # 原地加法

print(f"修改后的 PyTorch 张量:\n{pytorch_tensor}")
print(f"PyTorch 张量修改后的 NumPy 数组:\n{numpy_array_orig}")
```

**数据类型注意事项：** PyTorch 从 NumPy 数组的 `dtype` 推断张量的 `dtype`。常见的类型如 `np.float32`、`np.float64`、`np.int32`、`np.int64` 和 `np.uint8` 直接映射到其 PyTorch 等效类型（`torch.float32`、`torch.float64` 等）。请注意，NumPy 中的默认浮点类型通常是 `np.float64`，而 PyTorch 通常默认为 `torch.float32`。如果您在 PyTorch 中需要特定的 `dtype`（例如用于模型输入的 `torch.float32`），请确保您的 NumPy 数组在转换前具有相应的类型 (`np.float32`)，或者使用 `.to(torch.float32)` 转换结果张量。

```python
# float64 示例
numpy_float64 = np.array([1.0, 2.0, 3.0]) # 默认 np.float64
tensor_float64 = torch.from_numpy(numpy_float64)
print(f"\n从 np.float64 数组得到的张量数据类型: {tensor_float64.dtype}") # torch.float64

# 如果需要，转换为 float32
tensor_float32 = tensor_float64.to(torch.float32)
print(f"转换为 float32 后的张量: {tensor_float32.dtype}") # torch.float32
```

**创建副本：** 如果您明确需要 PyTorch 张量是 NumPy 数据的*副本*，而不是共享内存，您可以使用 `torch.tensor()` 或 `torch.as_tensor()` 并传入适当的参数 (parameter)。`torch.tensor()` 总是拷贝数据。

```python
numpy_array_to_copy = np.array([5, 6, 7])
print(f"\n要拷贝的 NumPy 数组: {numpy_array_to_copy}")

# 使用 torch.tensor() 创建一个拷贝
tensor_copy = torch.tensor(numpy_array_to_copy)

# 修改原始 NumPy 数组
numpy_array_to_copy[0] = 500
print(f"修改后的 NumPy 数组: {numpy_array_to_copy}")
print(f"PyTorch 张量副本 (未受影响): {tensor_copy}")
```

当您希望修改张量而不影响原始 NumPy 数组（反之亦然），或者当 NumPy 数组的数据类型不受直接支持并在张量创建时需要转换时，拷贝是必要的。

**移动到 GPU：** 使用 `torch.from_numpy()` 从 NumPy 数组创建张量后，您可以轻松地使用 `.to()` 方法将其移动到 GPU。此操作必然涉及从 CPU 到 GPU 内存的数据拷贝。

```python
numpy_array_for_gpu = np.random.rand(3, 4).astype(np.float32)
tensor_on_cpu = torch.from_numpy(numpy_array_for_gpu)

if torch.cuda.is_available():
    tensor_on_gpu = tensor_on_cpu.to('cuda')
    print(f"\n张量已移动到 GPU:\n{tensor_on_gpu}")
    print(f"张量设备: {tensor_on_gpu.device}")
else:
    print("\nCUDA 不可用，跳过移动到 GPU 示例。")
```

### 性能和最佳实践

内存共享机制使得 CPU PyTorch 张量与 NumPy 数组之间的转换速度极快（零拷贝）。这非常适合数据不需要修改，或者修改应反映在两个对象中的情况。

然而，请记住以下几点：

1. **GPU 数据传输：** 在 CPU 和 GPU 之间移动数据（`.cpu()`，`.to('cuda')`）总是涉及拷贝并产生开销。尽量减少这些传输。如果需要大量计算，请尝试完全在目标设备上的 PyTorch 中进行。
2. **显式拷贝：** 创建拷贝（`torch.tensor()`、`.clone()`、`np.copy()`）所需时间与数据大小成比例。当需要内存独立性时，请有意识地使用拷贝。
3. **数据加载：** 如果初始数据集已经是 NumPy 格式（例如从磁盘加载），`torch.from_numpy()` 对于加载它们非常有效。
4. **后处理/可视化：** 当您需要使用 NumPy 或 Matplotlib/Scikit-learn 等库分析模型输出或中间激活时，使用 `.detach().cpu().numpy()`。
5. **注意共享内存：** 始终注意张量和 NumPy 数组是否共享内存。通过一个对象修改，意外影响到另一个对象可能导致难以追踪的隐蔽错误。如有疑问，请创建拷贝。

熟练掌握 PyTorch 与 NumPy 之间的高效连接对于构建实用的深度学习 (deep learning)流程很重要。理解内存共享、数据拷贝、设备放置以及与自动求导系统的交互等细节，能让您编写出高效且正确的代码，发挥两个库的优点。

## 参考资料

- [torch.Tensor.numpy()](https://pytorch.org/docs/stable/generated/torch.Tensor.numpy.html) — PyTorch Developers (2023)
  Publisher: PyTorch Foundation
  将 PyTorch 张量转换为 NumPy 数组的官方文档，详细说明了内存共享特性和设备要求。
- [torch.from_numpy()](https://pytorch.org/docs/stable/generated/torch.from_numpy.html) — PyTorch Developers (2023)
  Publisher: PyTorch Foundation
  从 NumPy 数组创建 PyTorch 张量的官方文档，解释了内存共享和数据类型兼容性。
- [Automatic differentiation package - torch.autograd](https://pytorch.org/docs/stable/autograd.html) — PyTorch Developers (2023)
  PyTorch 自动微分系统的全面指南，有助于理解与 NumPy 交互时 `requires_grad` 和 `.detach()` 的用法。
- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, and Thomas Viehmann (2020)
  Publisher: Manning Publications
  一本关于 PyTorch 的实用书籍，涵盖了基础数据处理，包括 PyTorch 张量和 NumPy 数组之间的互操作性，提供了实用示例和背景信息。

---

[上一节](03-%E4%BD%BF%E7%94%A8%20ATen%20%E5%BA%93.md) · [下一节](05-%E4%BD%BF%E7%94%A8%E8%87%AA%E5%AE%9A%E4%B9%89%E6%A8%A1%E5%9D%97%E6%89%A9%E5%B1%95%20torch.nn.md)
