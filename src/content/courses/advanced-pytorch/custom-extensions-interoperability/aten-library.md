---
course: "advanced-pytorch"
chapter: "custom-extensions-interoperability"
lesson: "aten-library"
sourceId: 2782
sourceUrl: "https://apxml.com/zh/courses/advanced-pytorch/chapter-6-custom-extensions-interoperability/aten-library"
title: "使用 ATen 库"
description: "与 PyTorch 底层 C++ 张量库 (ATen) 交互，进行底层操作。"
order: 3
plots: []
sourceHash: "94c5bda81977c691e09191c837c50721dbe33c50eb6e031b2c31b9306576abe1"
sourceCorrections: []
---

PyTorch Python 接口的底层是 ATen，这是一个驱动其张量计算的 C++ 基本库。虽然 Python API 提供了便利性和灵活性，但在开发高性能定制 C++ 或 CUDA 扩展时，或者需要对 Python 中未完全公开的操作进行更细致的控制时，直接与 ATen 库交互就变得有必要。理解 ATen 有助于阐明 PyTorch 内部如何执行操作，并提供构建真正优化的底层组件的工具。

ATen 是 PyTorch 中主要的张量库。它在 C++ 中定义了 `Tensor` 对象，并实现了数百个作用于这些张量的数学操作。可以将其看作是为 `torch.add`、`torch.matmul` 或复杂神经网络 (neural network)层等函数执行实际数值工作的驱动力。你通常调用的 Python 函数通常作为封装器，最终调度到 ATen 的 C++ 实现。

### 为何与 ATen 交互？

在你的 C++ 代码中直接使用 ATen 函数具有以下优点：

1. **性能：** 对于计算密集型操作，特别是那些在 Python 中速度较慢的涉及循环或复杂索引模式的操作，使用 ATen 的 C++ 实现可以通过避免 Python 解释器开销来带来显著的加速。
2. **访问底层操作：** ATen 提供的函数和张量操作的范围比公共 Python API 更广。这对于实现新颖算法或高度优化的核函数可能很有用。
3. **定制扩展的支撑：** 在构建定制 C++ 或 CUDA 操作符时（如本章其他部分所述），你必然会用到 ATen 张量和函数。ATen 作为 PyTorch 体系与你的定制 C++/CUDA 代码之间的连接点。

### ATen 的结构与调度

ATen 采用复杂的调度机制，将张量操作路由到合适的后端实现（CPU、CUDA，可能还有其他）。当你调用 `at::add(tensor1, tensor2)` 这样的操作时，ATen 会检查输入张量的属性，主要是它们的设备（CPU 或 CUDA）和数据类型（`float`、`int` 等）。根据这些属性，它会动态选择并执行正确的底层核函数。

> 操作从 Python 经由 ATen 调度器到后端特定核函数的流程。

这种机制使得 PyTorch 能够在利用硬件特定优化的同时，保持一致的 API。在编写定制扩展时，你通常会为特定后端（如 CUDA 核函数）实现函数，并将它们注册到调度器，让 ATen 在合适的时候找到并使用你的定制代码。

### 在 C++ 中使用 ATen

要在 C++ 代码中使用 ATen，你主要需要包含 ATen 的主头文件：

```cpp
#include <ATen/ATen.h>
```

这个头文件引入了张量和函数所需的定义。在 C++ 中，PyTorch 张量由 `at::Tensor` 类表示。你可以像在 Python 中一样创建和操作这些张量：

```cpp
// 示例：在 C++ 中创建和使用 ATen 张量

// 在 CPU 上创建一个 2x3 的全一张量
at::Tensor tensor_a = at::ones({2, 3}, at::kFloat);

// 在 CPU 上创建一个 2x3 的随机数张量
at::Tensor tensor_b = at::randn({2, 3}, at::kFloat);

// 使用 ATen 函数执行加法
at::Tensor tensor_c = at::add(tensor_a, tensor_b);

// 执行元素级乘法
at::Tensor tensor_d = at::mul(tensor_c, 2.0); // 乘以一个标量

// 打印张量属性（需要 <iostream>）
std::cout << "张量 D:\n" << tensor_d << std::endl;
std::cout << "张量 D 数据类型: " << tensor_d.scalar_type() << std::endl;
std::cout << "张量 D 设备: " << tensor_d.device() << std::endl;
```

ATen 提供了与大多数 PyTorch 操作对应的函数，名称通常相似（例如，`at::matmul`、`at::relu`、`at::sigmoid`）。这些函数直接操作 `at::Tensor` 对象。

### 与 PyTorch C++ 扩展的集成

ATen 是 PyTorch C++ 扩展的支撑。当你使用 `pybind11` 定义一个要绑定到 Python 的 C++ 函数时（通常使用 `torch/extension.h` 完成），你在 C++ 中接收到的 `torch::Tensor` 参数 (parameter)本质上是 `at::Tensor` 的封装器。你可以直接将它们与 ATen 函数一起使用。

考虑一个用于定制扩展的简单 C++ 函数：

```cpp
#include <torch/extension.h>
#include <ATen/ATen.h>

// 一个使用 ATen 的简单 C++ 函数
torch::Tensor scaled_add(torch::Tensor x, torch::Tensor y, float scale_factor) {
  // x 和 y 是 torch::Tensor，但与 at:: 函数兼容
  // 使用 ATen 函数执行操作
  at::Tensor scaled_x = at::mul(x, scale_factor);
  at::Tensor result = at::add(scaled_x, y);
  return result; // 返回类型为 torch::Tensor
}

// 绑定代码（通常在单独的文件或代码块中）
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scaled_add", &scaled_add, "执行缩放加法: scale_factor * x + y");
}
```

在这个例子中，`x` 和 `y` 是从 Python 接收的 `torch::Tensor` 对象。它们可以直接传递给 `at::mul` 和 `at::add` 等 ATen 函数。结果在内部是 `at::Tensor`，但作为 `torch::Tensor` 返回，PyTorch 会自动处理以供 Python 使用。`torch::Tensor` 作为面向用户的 C++ API 张量类型，与底层的 ATen 实现结合。

### 需注意的事项

- **命名空间：** 你会经常遇到 `at::` 和 `torch::` 这两种命名空间。`at::` 专门指 ATen 库组件（底层张量操作）。`torch::` 通常指更广泛的 PyTorch C++ API，包含自动微分功能、模块（`torch::nn`）以及 `torch::Tensor` 封装器本身。对于定制核函数中使用的基本张量操作，你将主要使用 `at::`。
- **张量类型和设备：** 在 C++ 中工作时，你需要明确指定张量数据类型（`at::ScalarType`，例如 `at::kFloat`、`at::kHalf`、`at::kInt`）和设备（`at::Device`，例如 `at::kCPU`、`at::kCUDA`）。ATen 函数通常要求输入张量在同一设备上，并且可能具有特定的数据类型要求。你可以使用 `tensor.to(at::kCUDA)` 或 `tensor.to(at::kFloat)` 等方法检查和转换张量。
- **内存管理：** `at::Tensor` 和 `torch::Tensor` 都使用引用计数进行内存管理，类似于 Python 张量。当张量在 Python 和 C++ 之间传递，或在 C++ 函数中使用时，它们的引用计数会自动管理。除非执行了显式复制操作，内存通常是共享的（而不是复制）。
- **自动微分：** 需要注意的是，如上所示直接调用 ATen 函数通常**不会**在 PyTorch 计算图中记录操作。如果你的涉及 ATen 调用的定制 C++ 操作需要支持自动微分，你必须将其封装在 `torch::autograd::Function` 中，定义定制的 `forward` 和 `backward` 方法。这在第 1 章（“PyTorch 内部机制与自动微分”）中有介绍，在将定制计算核函数集成到可训练模型中时不可或缺。ATen 提供了计算构件，而 `torch::autograd::Function` 则将其与梯度跟踪系统结合。

直接使用 ATen 赋予你强大的功能，用于性能优化和扩展 PyTorch 的主要功能。它是 PyTorch 张量计算发生的层面，在构建提升性能和能力上限的定制 C++ 和 CUDA 扩展时，理解它很有帮助。虽然它需要仔细处理类型、设备和自动微分系统，但掌握 ATen 的交互使用可以充分发挥 PyTorch 在高级深度学习 (deep learning)工程中的全部潜能。

## 参考资料

- [PyTorch C++ Frontend and API Reference](https://pytorch.org/cppdocs/) — PyTorch Contributors (2024)
  Publisher: PyTorch Foundation
  提供PyTorch C++前端的全面文档，涵盖`torch::Tensor`、自定义扩展以及与ATen的交互。
- [PyTorch: An Imperative Style, High-Performance Deep Learning Library](http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf) — Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, Soumith Chintala (2019)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 32; Pages: 8024-8035; DOI: [10.48550/arXiv.1912.01703](https://doi.org/10.48550/arXiv.1912.01703)
  介绍了PyTorch框架，概述了其命令式设计、动态计算图以及用于张量操作的高效C++后端。
- [Custom Autograd Functions in C++](https://pytorch.org/cppdocs/notes/autograd_and_custom_function.html) — PyTorch Contributors (2024)
  解释了如何在C++中使用`torch::autograd::Function`实现自定义自动求导函数，从而为涉及ATen张量的自定义操作启用自动微分。
