---
course: "advanced-pytorch"
chapter: "custom-extensions-interoperability"
lesson: "custom-cuda-extensions"
sourceId: 2780
sourceUrl: "https://apxml.com/zh/courses/advanced-pytorch/chapter-6-custom-extensions-interoperability/custom-cuda-extensions"
title: "构建自定义 CUDA 扩展"
description: "用 CUDA C++ 编写自定义 GPU 内核，并将其整合到 PyTorch 工作流中。"
order: 2
plots: []
sourceHash: "6e77d2026e2abfd6897767b50efc0dac5986896e5335003e74f5ce3657680d43"
sourceCorrections: []
---

虽然自定义 C++ 扩展提供了将受 CPU 约束的逻辑或外部 C++ 库整合进去的途径，但深度学习 (deep learning)中的性能瓶颈通常存在于 GPU 上。当标准 PyTorch 操作不足以满足需求，或者您需要以最高的 GPU 效率实现一种新颖的算法时，编写自定义 CUDA 内核就变得必须。学习如何创建、构建和直接将自定义 CUDA C++ 代码并入您的 PyTorch 工作流中。

CUDA 扩展的主要目的是性能。您可能有一个特定的数学运算、一个数据处理程序，或者一个来自现有 CUDA 研究代码的内核，希望直接在 GPU 上执行它们，而无需数据在 CPU 之间来回传输的开销，也无需依赖可能并非最优的标准 PyTorch 操作序列。

### CUDA 扩展的工作流程

整合自定义 CUDA 代码涉及多个步骤，类似于 C++ 扩展，但增加了 GPU 编程的复杂性：

1. **编写 CUDA 内核：** 用 CUDA C++（`.cu` 文件）实现您的核心逻辑。这涉及编写函数（`__global__` 用于从主机启动的内核，`__device__` 用于从 GPU 调用的函数），这些函数在 GPU 内存中的数据上进行操作。
2. **编写 C++ 包装器：** 创建一个 C++ 函数（`.cpp` 文件），它作为 PyTorch 和您的 CUDA 内核之间的接口。该包装器将：
   - 接收 PyTorch 张量作为输入。
   - 检查张量属性（设备、数据类型、连续性）。
   - 获取原始数据指针（`tensor.data_ptr()`），以便访问 GPU 内存。
   - 计算 CUDA 内核启动参数 (parameter)（网格大小、块大小）。
   - 启动 CUDA 内核，传递数据指针和任何其他所需的参数。
   - 将生成的张量返回给 PyTorch。
3. **创建 Python 绑定：** 使用 PyTorch 的 C++ 扩展实用工具（`torch.utils.cpp_extension`）编译 CUDA 内核和 C++ 包装器，并使包装函数可以在 Python 中调用。这可以通过即时 (JIT) 编译或通过 `setup.py` 脚本完成。

### 示例：自定义 CUDA 向量 (vector)加法

我们用一个简单的向量加法内核来阐明这一点。

**1. CUDA 内核 (`vector_add_kernel.cu`)**

```cpp
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h> // 如果内核调试需要 printf

// 用于元素级向量加法的 CUDA 内核
__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = index; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// 一个调用内核的简单函数（可以更复杂）
// 注意：为简洁起见省略了错误检查 (cudaGetLastError)，但在生产环境中很重要。
void vector_add_cuda_launcher(const float* a, const float* b, float* c, int n) {
    int threadsPerBlock = 256;
    // 使用整数向上取整除法
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // 启动内核
    vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, n);

    // 可选：如果需要立即同步设备，在内核启动后同步
    // cudaDeviceSynchronize(); // 请注意对性能的影响
}
```

**2. C++ 包装器 (`vector_add.cpp`)**

此文件连接 PyTorch 和我们的 CUDA 启动函数。

```cpp
#include <torch/extension.h>
#include <vector>

// CUDA 前向声明（假设 vector_add_cuda_launcher 在其他地方定义，例如在 .cu 文件或头文件中）
void vector_add_cuda_launcher(const float* a, const float* b, float* c, int n);

// 符合 PyTorch C++ API 的 C++ 接口函数
// 注意：AT_ASSERT 宏确保张量在正确的设备上并具有预期的类型/形状。
torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    // 输入验证
    TORCH_CHECK(a.device().is_cuda(), "Input tensor a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "Input tensor b must be a CUDA tensor");
    TORCH_CHECK(a.is_contiguous(), "Input tensor a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "Input tensor b must be contiguous");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "Input tensor a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "Input tensor b must be float32");
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same shape");

    // 在与输入相同的设备上创建输出张量
    torch::Tensor c = torch::empty_like(a);
    int n = a.numel(); // 元素总数

    // 调用 CUDA 启动函数
    vector_add_cuda_launcher(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        n);

    return c;
}

// 绑定函数：将 'vector_add' C++ 函数作为 'vector_add_cuda' 暴露给 Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &vector_add, "CUDA vector addition forward");
  // 如果有反向传播，您也应在此处进行绑定。
}
```

**3. Python 绑定和构建（使用 JIT）**

对于简单情况，最方便的编译和加载方式是使用 `torch.utils.cpp_extension.load`。

```python
import torch
import time
from torch.utils.cpp_extension import load

# 加载 CUDA 扩展，如果需要则进行 JIT 编译
# 'verbose=True' 会显示编译命令
vector_add_module = load(
    name='vector_add_cuda',
    sources=['vector_add.cpp', 'vector_add_kernel.cu'],
    verbose=True
)

# 在 GPU 上准备输入张量
device = torch.device('cuda')
size = 10000000  # 大向量大小
a = torch.randn(size, device=device, dtype=torch.float32)
b = torch.randn(size, device=device, dtype=torch.float32)

# --- 使用 PyTorch 默认加法 ---
start_time = time.time()
c_pytorch = a + b
torch.cuda.synchronize() # 等待 GPU 操作完成
pytorch_time = time.time() - start_time
print(f"PyTorch default add time: {pytorch_time:.6f} seconds")

# --- 使用自定义 CUDA 扩展 ---
start_time = time.time()
c_cuda = vector_add_module.forward(a, b)
torch.cuda.synchronize() # 等待 GPU 操作完成
cuda_time = time.time() - start_time
print(f"Custom CUDA add time: {cuda_time:.6f} seconds")

# 验证结果（允许小的浮点差异）
diff = torch.abs(c_pytorch - c_cuda).max()
print(f"Maximum difference between PyTorch and CUDA results: {diff.item()}")

assert torch.allclose(c_pytorch, c_cuda, atol=1e-6), "Results differ significantly!"
print("CUDA extension test passed!")
```

本示例使用了 JIT 编译器。对于较大项目或分发，您通常会使用 `setup.py` 文件：

**`setup.py` 示例**

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='vector_add_cuda',
    ext_modules=[
        CUDAExtension('vector_add_cuda', [ # 模块名称必须与 PYBIND11_MODULE 匹配
            'vector_add.cpp',
            'vector_add_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
```

然后，您可以使用 `python setup.py install` 来构建它。安装后，您可以像常规 Python 模块一样导入它：`import vector_add_cuda`。

> PyTorch CUDA 扩展的构建过程。Python 工具通过系统编译器（如 g++ 和 NVCC）协调 C++ 和 CUDA 代码的编译，生成可加载的共享库。

### CUDA 扩展的注意事项

- **数据类型：** 内核通常专门用于特定的数据类型（例如 `float`、`half`）。C++ 包装器必须处理类型检查，并可能分派到不同的内核版本或执行类型转换。在 C++ 和 CUDA 中使用模板有助于处理此问题。
- **张量连续性：** CUDA 内核通常期望连续的内存块。`TORCH_CHECK(tensor.is_contiguous(), ...)` 断言很有用。如果张量不连续，您可能需要在 Python 代码中调用 `.contiguous()`，或者在内核中谨慎处理非连续内存访问（这通常效率会低得多）。
- **设备管理：** 确保张量在目标 CUDA 设备上（`tensor.device().is_cuda()`），并且输出张量也在同一设备上创建。
- **内核启动参数 (parameter)：** 选择最佳的 `gridDim`（块数）和 `blockDim`（每块线程数）会显著影响性能，并取决于特定的 GPU 架构和内核逻辑。这通常需要进行实验。
- **同步：** CUDA 内核启动是异步的。CPU 代码在启动内核后会继续执行。如果后续操作（在 CPU 或 GPU 上）依赖于内核的结果，则需要同步。`torch.cuda.synchronize()` 会等待当前流上所有先前的 CUDA 操作完成。但是，过度使用它会损害性能。通常，当数据复制回 CPU 或被另一个 PyTorch CUDA 操作使用时，同步是隐式处理的。
- **错误处理：** 代码需要进行错误检查。在内核启动和其他 CUDA API 调用之后，在您的 C++/CUDA 代码中使用 `cudaGetLastError()` 来捕获运行时错误。这些错误可以作为异常传播回 Python。
- **构建系统：** 尽管 JIT 加载便于开发，但 `setup.py` 为复杂构建、链接外部库和分发提供了更好的控制。
- **自动求导整合：** 上述示例只实现了前向传播。为了使您的自定义 CUDA 操作可微分，您需要实现一个相应的 `backward` 函数（通常需要另一个自定义 CUDA 内核），并使用 `torch::autograd::Function` 进行绑定，类似于第 1 章中讨论的自定义 C++ 自动求导函数，但由 CUDA 内核来处理计算。

构建自定义 CUDA 扩展需要熟悉 CUDA C++ 编程以及 PyTorch 的 C++ API。然而，它提供了对 GPU 执行的终极控制，为您的深度学习 (deep learning)模型中专门的、计算密集型操作实现了显著的性能提升。

## 参考资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) — NVIDIA (2024)
  Publisher: NVIDIA
  CUDA C++编程模型的权威指南，涵盖了内核执行、内存管理、同步以及编写自定义CUDA内核所必需的性能优化技术。
- [Professional CUDA C Programming](https://www.wiley.com/en-us/Professional+CUDA+C+Programming-p-9781118739327) — John Cheng, Max Grossman, and Ty McKercher (2014)
  Publisher: Wiley
  一本关于CUDA架构、编程模型和优化GPU应用程序高级技术的书籍，直接适用于编写高效的自定义CUDA扩展。
