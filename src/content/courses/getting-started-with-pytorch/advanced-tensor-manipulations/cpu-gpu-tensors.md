---
course: "getting-started-with-pytorch"
chapter: "advanced-tensor-manipulations"
lesson: "cpu-gpu-tensors"
sourceId: 2121
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-2-advanced-tensor-manipulations/cpu-gpu-tensors"
title: "CPU 与 GPU 张量"
description: "了解如何在 CPU 和 GPU 内存之间移动张量以加速计算。"
order: 6
plots: []
sourceHash: "f84f62dfa4198b79473e9360d21b651934598bea8d8579fd1d9914e79904cc51"
sourceCorrections: []
---

深度学习 (deep learning)计算，特别是涉及大型张量和复杂模型的计算，需要强大的计算能力。中央处理器（CPU）用途广泛，而图形处理器（GPU）提供大规模并行能力，可以大幅加速构成神经网络 (neural network)核心的矩阵和向量 (vector)运算。PyTorch 提供简单直接的机制来管理张量的存放位置和计算发生地。了解如何在 CPU 和 GPU 之间移动张量是高效模型训练和推理 (inference)的必备技能。

### CPU：默认计算中心

默认情况下，当您创建 PyTorch 张量时未指定设备，它会分配在 CPU 上。

```python
import torch

# 默认在 CPU 上创建张量
cpu_tensor = torch.tensor([1.0, 2.0, 3.0])
print(f"默认张量设备：{cpu_tensor.device}")
```

CPU 完全适用于许多任务，包括预处理步骤、较小规模的计算，或在没有兼容 GPU 时运行模型。然而，对于训练大型深度学习 (deep learning)模型，由于 CPU 的顺序处理特性与 GPU 的并行架构相比，仅依赖 CPU 常常会导致训练时间过长，难以承受。

### GPU：使用并行加速深度学习 (deep learning)

GPU 包含数百或数千个核心，旨在同时执行大量计算。这种架构非常适合深度学习中常见的运算类型，如大型矩阵乘法和卷积。PyTorch 通过 CUDA（统一计算设备架构）平台使用 NVIDIA GPU。

要使用 GPU，您需要：

1. 一个兼容 CUDA 的 NVIDIA GPU。
2. 已安装适当的 CUDA 工具包。
3. 安装了 CUDA 支持的 PyTorch 版本。

### 检查 GPU 可用性并设置设备

在尝试使用 GPU 之前，最好检查它是否可用且已为 PyTorch 正确配置。`torch.cuda.is_available()` 函数在 PyTorch 可以访问支持 CUDA 的 GPU 时返回 `True`。

之后我们可以创建一个 `torch.device` 对象来表示我们的目标计算设备（CPU 或 GPU）。这使得代码具有适应性，如果 GPU 可用则自动使用 GPU，否则回退到 CPU。

```python
import torch

# 检查 CUDA 可用性并相应设置设备
if torch.cuda.is_available():
    device = torch.device("cuda") # 使用第一个可用的 CUDA 设备
    print(f"CUDA (GPU) 可用。使用设备：{device}")
    # 您也可以指定特定 GPU，例如 torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print(f"CUDA (GPU) 不可用。使用设备：{device}")

# device 现在包含 torch.device('cuda') 或 torch.device('cpu')
```

### 直接在设备上创建张量

您可以在张量创建期间使用 `device` 参数 (parameter)直接指定目标设备。这通常比在 CPU 上创建然后再移动更高效。

```python
# 直接在选定设备上创建张量
try:
    # 如果 device='cpu'，此张量将在 CPU 上；如果 device='cuda'，则在 GPU 上
    device_tensor = torch.randn(3, 4, device=device)
    print(f"张量创建于：{device_tensor.device}")
except RuntimeError as e:
    print(f"无法直接在 {device} 上创建张量：{e}") # 处理未找到 GPU 等情况
```

### 在 CPU 和 GPU 之间移动张量

通常，您需要在设备之间传输现有张量。例如，从磁盘加载的数据通常位于 CPU 上，但您的模型可能在 GPU 上以加快计算。移动张量的主要方法是 `.to()` 方法。

`.to()` 方法接受 `torch.device` 对象、设备字符串（例如 `'cuda'`、`'cpu'`），甚至另一个张量（在这种情况下，张量会移动到与参数 (parameter)张量相同的设备上）作为输入。它在指定设备上返回一个 *新* 张量。原始张量在其原始设备上保持不变。

```python
# 从 CPU 张量开始
cpu_tensor = torch.ones(2, 2)
print(f"原始张量：{cpu_tensor.device}")

# 将张量移动到选定设备（如果 GPU 可用，则为 GPU；否则为 CPU）
# 请记住，'device' 是根据可用性预先设置的
moved_tensor = cpu_tensor.to(device)
print(f"移动后的张量：{moved_tensor.device}")

# 如果张量在 GPU 上，则显式移回 CPU
if moved_tensor.is_cuda: # 检查张量是否在 CUDA 设备上
    back_to_cpu = moved_tensor.to("cpu")
    print(f"张量移回至：{back_to_cpu.device}")
```

PyTorch 还提供便利方法：`.cpu()` 和 `.cuda()`。它们分别是 `.to('cpu')` 和 `.to('cuda:0')`（或当前默认的 CUDA 设备）的简写。

```python
# 使用便利方法（假设 GPU 可用且 'device' 为 'cuda'）
if device.type == 'cuda':
    # 将 cpu_tensor 移动到 GPU
    gpu_tensor_alt = cpu_tensor.cuda()
    print(f"使用 .cuda()：{gpu_tensor_alt.device}")

    # 将 gpu_tensor_alt 移回 CPU
    cpu_tensor_alt = gpu_tensor_alt.cpu()
    print(f"使用 .cpu()：{cpu_tensor_alt.device}")
```

### 设备管理的重要考量

1. **设备一致性：** 涉及多个张量的操作（例如，加法、矩阵乘法）通常要求所有参与的张量都在 *同一* 设备上。尝试在 CPU 张量和 GPU 张量之间进行操作将导致 `RuntimeError`。在执行操作前，请确保您的数据和模型位于同一设备上。

   ```python
   # 错误示例（假设 device='cuda'）
   cpu_a = torch.randn(2, 2)
   gpu_b = torch.randn(2, 2, device=device)
   try:
       # 如果 device 是 'cuda'，这很可能会失败
       c = cpu_a + gpu_b
   except RuntimeError as e:
       print(f"在不同设备上执行操作时出错：{e}")
   ```
2. **数据传输开销：** 在 CPU 和 GPU 内存之间移动数据并非瞬间完成。虽然 GPU 计算速度快，但如果管理不当，数据的来回传输可能成为瓶颈。为了最佳性能，请尝试在 GPU 上尽可能多地执行操作，然后在需要时（例如，保存到磁盘或转换为 NumPy）再将最终结果移回 CPU。
3. **模型放置：** 与张量一样，使用 `torch.nn.Module` 定义的神经网络 (neural network)模型也需要使用 `.to(device)` 方法移动到适当的设备。这确保模型的参数 (parameter)（本身也是张量）位于目标设备上进行计算。在讨论模型构建时，会更详细地介绍这一点。

掌握使用 `device` 和 `.to()` 方法进行张量放置是发挥 GPU 计算能力和编写高效、硬件感知型 PyTorch 代码不可或缺的要素。请记住始终检查设备一致性并注意数据传输成本。

## 参考资料

- [PyTorch Tensors](https://pytorch.org/docs/stable/tensors.html) — PyTorch Core Team (2024)
  Publisher: PyTorch Foundation
  描述了`torch.Tensor`对象的基础特性，例如`device`属性和基本操作，这对于理解张量创建和放置很有用。
- [CUDA semantics](https://pytorch.org/docs/stable/cuda.html) — PyTorch Core Team (2024)
  Publisher: PyTorch Foundation
  解释了PyTorch如何利用NVIDIA GPU与CUDA进行交互，内容涵盖设备管理、可用性检查以及在CPU和GPU之间移动张量。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本全面的教科书，阐述了深度学习背后的数学运算，阐明了GPU并行处理能力为何有利于高效训练。
- [Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html) — Szymon Migacz (2020)
  提供了优化PyTorch代码性能的实用建议和最佳实践，包括高效的设备利用和管理数据传输开销。
