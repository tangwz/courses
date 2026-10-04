---
course: "advanced-pytorch"
chapter: "custom-extensions-interoperability"
lesson: "practice-cuda-extension"
sourceId: 2788
sourceUrl: "https://apxml.com/zh/courses/advanced-pytorch/chapter-6-custom-extensions-interoperability/practice-cuda-extension"
title: "实践：构建一个简单的 CUDA 扩展"
description: "在 PyTorch Python 脚本中创建、编译和使用一个基本的自定义 CUDA 核函数。"
order: 8
plots: []
sourceHash: "f6448ce5bf207577c3cce9034d02bc5c66b2ee23d56cb682584d6dfdc24bfcc8"
sourceCorrections: []
---

构建一个简单的 CUDA 扩展涉及创建用于基本操作（标量向量 (vector)加法）的自定义 CUDA 核函数，编写必要的 C++ 绑定，编译它，并最终使用 PyTorch 张量从 Python 调用它。这个过程展示了在 GPU 上加速特定计算的基本步骤。

### 目标

我们的目标是实现一个函数 `scaled_add(alpha, x, y)`，计算 $z = \alpha * x + y$，其中 $\alpha$ 是一个标量，$x, y, z$ 是向量 (vector)（一维张量）。我们将把核心计算写成 CUDA 核函数，并将其作为 PyTorch C++ 扩展进行集成。

### 前提条件

请确保您已安装并配置以下各项：

1. **PyTorch：** 已安装并支持 CUDA（`torch.cuda.is_available()` 应该返回 `True`）。
2. **CUDA 工具包：** 与 PyTorch 编译版本相同或兼容的版本。`nvcc` 编译器必须在您系统的 PATH 环境变量中。
3. **C++ 编译器：** 兼容的 C++ 编译器（如 g++ 或 MSVC）。PyTorch 的 C++ 扩展工具通常会处理编译器的查找。

### 文件结构

让我们组织代码。创建一个这样的目录结构：

```
simple_cuda_extension/
├── setup.py
└── src/
    ├── scaled_add.cpp
    └── scaled_add_kernel.cu
```

### 步骤 1：编写 CUDA 核函数 (`scaled_add_kernel.cu`)

此文件包含实际的 GPU 代码。我们定义一个 CUDA 核函数，逐元素执行标量加法。

```cuda
// src/scaled_add_kernel.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h> // 如果需要，用于 CUDA 数学函数，但本例中并非严格需要

// CUDA 核函数定义
// 为每个元素计算 z = alpha * x + y
__global__ void scaled_add_kernel(const float* x, const float* y, float* z, float alpha, int N) {
    // 计算全局线程索引
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x; // 网格中的线程总数

    // 使用网格步进循环确保所有元素都被处理
    // 即使 N 大于启动的线程数。
    for (int i = index; i < N; i += stride) {
        z[i] = alpha * x[i] + y[i];
    }
}

// C++ 包装函数（可选但推荐）
// 这可以从主要的 C++ 绑定代码中调用。
// 它设置核函数的启动配置。
void scaled_add_cuda_launcher(const float* x, const float* y, float* z, float alpha, int N) {
    // 定义块和网格维度
    // 通常，选择块大小为 32（warp 大小）的倍数
    // 常见选择有 128, 256, 512, 1024
    int blockSize = 256;
    // 计算覆盖所有 N 个元素所需的网格大小
    // 等同于 ceil(N / blockSize)
    int gridSize = (N + blockSize - 1) / blockSize;

    // 启动核函数
    scaled_add_kernel<<<gridSize, blockSize>>>(x, y, z, alpha, N);

    // 可选：检查核函数启动错误（对调试有用）
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
        // 在实际应用中，可以考虑在此处抛出异常
    }
    // 可选：如果立即需要，同步设备（等待核函数完成）
    // cudaDeviceSynchronize(); // 如果后续操作使用相同的流，通常不需要
}
```

**说明：**

- `__global__ void scaled_add_kernel(...)`：定义一个在 GPU 上运行的函数。
- `blockIdx.x`、`blockDim.x`、`threadIdx.x`、`gridDim.x`：内置的 CUDA 变量，为每个线程在其启动的线程网格中提供唯一的 ID 和上下文 (context)。
- `index = blockIdx.x * blockDim.x + threadIdx.x`：计算每个线程的唯一全局索引。
- `stride = gridDim.x * blockDim.x`：网格中的线程总数。
- **网格步进循环：** `for` 循环 (`for (int i = index; i < N; i += stride)`) 非常重要。它允许固定数量的线程（可能少于 `N`）通过让每个线程处理多个间隔 `stride` 的元素来处理所有 `N` 个元素。这比假设 `N` 可以被块大小完美整除，或者网格大小与 `N / blockSize` 精确匹配要准确。
- `scaled_add_cuda_launcher`：一个辅助 C++ 函数，用于配置并启动核函数。它根据输入大小 `N` 和选择的块大小 (`blockSize`) 计算所需的块数量 (`gridSize`)。`<<<gridSize, blockSize>>>` 是启动核函数的 CUDA 语法。

### 步骤 2：编写 C++ 绑定 (`scaled_add.cpp`)

此文件将 CUDA 代码与 PyTorch 连接起来。它定义了一个可从 Python 调用的函数，处理张量数据访问，并调用 CUDA 启动器。

```cpp
// src/scaled_add.cpp
#include <torch/extension.h>
#include <vector>

// 从 scaled_add_kernel.cu 转发声明 CUDA 启动函数
void scaled_add_cuda_launcher(const float* x, const float* y, float* z, float alpha, int N);

// 将绑定到 Python 的 C++ 接口函数
// 它接受 PyTorch 张量作为输入
torch::Tensor scaled_add(torch::Tensor x, torch::Tensor y, float alpha) {
    // 输入验证：确保张量在 GPU 上且具有相同的形状/数据类型
    TORCH_CHECK(x.device().is_cuda(), "输入张量 x 必须是 CUDA 张量");
    TORCH_CHECK(y.device().is_cuda(), "输入张量 y 必须是 CUDA 张量");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "输入张量 x 必须是 float32");
    TORCH_CHECK(y.scalar_type() == torch::kFloat32, "输入张量 y 必须是 float32");
    TORCH_CHECK(x.is_contiguous(), "输入张量 x 必须是连续的");
    TORCH_CHECK(y.is_contiguous(), "输入张量 y 必须是连续的");
    TORCH_CHECK(x.sizes() == y.sizes(), "输入张量 x 和 y 必须具有相同的形状");
    TORCH_CHECK(x.dim() == 1, "输入张量 x 必须是 1D"); // 本例的简单检查
    TORCH_CHECK(y.dim() == 1, "输入张量 y 必须是 1D"); // 本例的简单检查

    // 获取元素数量
    int N = x.numel();

    // 创建输出张量（与输入在同一设备上）
    auto z = torch::empty_like(x); // 创建具有相同形状、数据类型、设备的张量

    // 获取原始数据指针
    // .data_ptr<float>() 允许访问底层 C++ float* 数据
    const float* x_ptr = x.data_ptr<float>();
    const float* y_ptr = y.data_ptr<float>();
    float* z_ptr = z.data_ptr<float>();

    // 调用 .cu 文件中定义的 CUDA 核函数启动器函数
    scaled_add_cuda_launcher(x_ptr, y_ptr, z_ptr, alpha, N);

    return z;
}

// 使用 PYBIND11_MODULE 宏的绑定代码
// 这将创建名为 'simple_cuda_extension_cpp' 的 Python 模块
// 第二个参数 'm' 是模块对象
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // 将 C++ 'scaled_add' 函数公开为 Python 中的 'scaled_add'
    m.def("scaled_add", &scaled_add, "在 CUDA 上计算的标量向量加法 (alpha * x + y)");
}
```

**说明：**

- `#include <torch/extension.h>`：PyTorch C++ 扩展的主要头文件。
- `前向声明`：我们声明 `scaled_add_cuda_launcher`，以便编译器在使用它之前知道它。实际实现位于 `.cu` 文件中，并将在稍后链接。
- `scaled_add(torch::Tensor x, torch::Tensor y, float alpha)`：向 Python 公开的函数。它接受 PyTorch 张量和一个浮点数。
- `TORCH_CHECK(...)`：PyTorch 的断言宏。它检查条件并在失败时抛出描述性的 C++ 异常（这些异常会被转换为 Python 异常）。我们验证设备、数据类型、连续性、形状和维度。连续性很重要，因为 CUDA 核函数通常假设数据在内存中是连续排列的。
- `torch::empty_like(x)`：创建与 `x` 具有相同属性（大小、数据类型、设备）的输出张量 `z`，但不初始化内存内容。
- `.data_ptr<float>()`：获取指向张量底层数据缓冲区的原始 C 风格指针。这是传递给 CUDA 核函数所必需的。
- `scaled_add_cuda_launcher(...)`：调用在我们的 `.cu` 文件中定义的函数。
- `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)`：这个宏（由 `torch/extension.h` 提供，它包含 pybind11）为 Python 模块创建入口点。`TORCH_EXTENSION_NAME` 是一个占位符，它将被 `setup.py` 中指定的模块名称替换。
- `m.def("scaled_add", ...)`：将 C++ 函数 `scaled_add` 绑定到模块 `m` 中 Python 名称 `scaled_add`。该字符串是 Python 函数的文档字符串。

### 步骤 3：创建构建脚本 (`setup.py`)

此脚本使用 Python 的 `setuptools` 和 PyTorch 的工具来将 C++ 和 CUDA 代码编译成一个 Python 扩展模块。

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='simple_cuda_extension_cpp',  # 包名，可以是任意名称
    ext_modules=[
        CUDAExtension(
            name='simple_cuda_extension_cpp', # 用户将导入的 Python 模块名称
            sources=[
                'src/scaled_add.cpp',
                'src/scaled_add_kernel.cu',
            ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
```

**说明：**

- `from torch.utils.cpp_extension import BuildExtension, CUDAExtension`：从 PyTorch 导入必要的构建工具。
- `CUDAExtension(...)`：指定我们正在构建一个包含 CUDA 代码的扩展。
  - `name`：生成的 Python 模块的名称（例如，`import simple_cuda_extension_cpp`）。这**必须**与 `PYBIND11_MODULE` 宏内部使用的 `TORCH_EXTENSION_NAME` 占位符匹配。
  - `sources`：扩展所需的所有源文件（`.cpp` 和 `.cu`）列表。
- `cmdclass={'build_ext': BuildExtension}`：告知 `setuptools` 使用 PyTorch 的自定义构建命令，该命令知道如何处理 CUDA 编译（`nvcc`）和与 PyTorch 库链接。

### 步骤 4：编译扩展

在您的终端中导航到 `simple_cuda_extension` 目录（包含 `setup.py` 的目录），然后运行构建命令：

```bash
# 选项 1：构建并安装到您的 Python 环境中
python setup.py install

# 选项 2：就地构建（在当前目录创建 .so 或 .pyd 文件）
# 对开发有用
python setup.py build_ext --inplace
```

如果成功，此命令将调用 C++ 编译器和 `nvcc` 来编译您的代码，并将其链接到 PyTorch 库，生成一个共享对象文件（例如，在 Linux 上为 `simple_cuda_extension_cpp.cpython-39-x86_64-linux-gnu.so`，在 Windows 上为 `simple_cuda_extension_cpp.pyd`），可供 Python 导入。

### 步骤 5：在 Python 中使用扩展

现在您可以像使用任何其他 Python 模块一样导入和使用您的自定义 CUDA 函数了。

```python
# test_extension.py（放置在 simple_cuda_extension 目录之外，或安装后放置）
import torch
import time

# 尝试导入已编译的扩展
try:
    import simple_cuda_extension_cpp
    print("成功导入 CUDA 扩展。")
except ImportError:
    print("导入 CUDA 扩展时出错。您是否成功编译了它？")
    print("运行：python setup.py build_ext --inplace（在扩展目录中）")
    exit()

# 首先在 CPU 上定义输入张量
N = 1024 * 1024  # 向量大小
alpha = 2.5
x_cpu = torch.randn(N, dtype=torch.float32)
y_cpu = torch.randn(N, dtype=torch.float32)

# 将张量移动到 GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    x_gpu = x_cpu.to(device)
    y_gpu = y_cpu.to(device)
    print(f"使用设备: {device}")
else:
    print("CUDA 不可用。退出。")
    exit()

# 确保输入是连续的（对 .data_ptr() 很重要）
x_gpu = x_gpu.contiguous()
y_gpu = y_gpu.contiguous()

# --- 使用自定义 CUDA 扩展 ---
print("\n测试自定义 CUDA 扩展：")
# GPU 预热
_ = simple_cuda_extension_cpp.scaled_add(alpha, x_gpu, y_gpu)
torch.cuda.synchronize() # 等待预热完成

start_time = time.time()
z_gpu_custom = simple_cuda_extension_cpp.scaled_add(alpha, x_gpu, y_gpu)
torch.cuda.synchronize() # 等待核函数完成再停止计时器
end_time = time.time()
print(f"自定义 CUDA 扩展时间：{(end_time - start_time)*1000:.4f} 毫秒")

# --- 使用标准 PyTorch 操作进行验证 ---
print("\n测试标准 PyTorch 操作：")
# GPU 预热
_ = alpha * x_gpu + y_gpu
torch.cuda.synchronize()

start_time = time.time()
z_gpu_pytorch = alpha * x_gpu + y_gpu
torch.cuda.synchronize()
end_time = time.time()
print(f"标准 PyTorch 时间：{(end_time - start_time)*1000:.4f} 毫秒")

# --- 验证 ---
# 检查结果是否接近（允许浮点差异）
difference = torch.abs(z_gpu_custom - z_gpu_pytorch).mean()
print(f"\n自定义和 PyTorch 结果之间的平均绝对差异：{difference.item()}")

if torch.allclose(z_gpu_custom, z_gpu_pytorch, atol=1e-6):
    print("结果匹配！")
else:
    print("结果不匹配！")

# 示例：如果需要，打印前几个元素
# print("自定义输出（前 10 个）:", z_gpu_custom[:10])
# print("PyTorch 输出（前 10 个）:", z_gpu_pytorch[:10])
```

**运行测试：**
保存上述 Python 代码（例如，保存为 `test_extension.py`），然后运行它：`python test_extension.py`。

您应该会看到输出，显示导入是否成功、自定义核函数和标准 PyTorch 操作的执行时间，以及确认结果在数值上非常接近的检查。对于现代 GPU 上的这种简单操作，标准 PyTorch 操作是高度优化的，因此如果 PyTorch 版本更快或相似，请不要惊讶。自定义扩展的好处在复杂、非标准的操作或可以融合到单个核函数中的操作序列中会变得更明显。

### 总结

此实践练习展示了创建 PyTorch CUDA 扩展的端到端过程：

1. 在 CUDA 核函数中实现核心逻辑（`.cu`）。
2. 编写一个 C++ 函数来与 PyTorch 张量连接并启动核函数（`.cpp`）。
3. 使用 `pybind11` 将 C++ 函数绑定到 Python（`.cpp`）。
4. 使用 `setuptools` 和 `torch.utils.cpp_extension` 编译 CUDA 和 C++ 代码（`setup.py`）。
5. 在 Python 中导入和使用已编译的扩展。

"虽然这个例子很简单，但它建立了基本的工作流程。扩展通常涉及更复杂的核函数，可能处理不同的数据类型、多个维度，如果需要自动求导支持，则需要定义自定义反向传播 (backpropagation)（请参考第 1 章关于自定义自动求导函数的内容）。构建扩展需要仔细注意内存管理、数据类型、设备放置和同步，但它提供了一种有效的方法来优化 PyTorch 模型中性能关键的部分。"

## 参考资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) — NVIDIA (2023)
  Publisher: NVIDIA
  涵盖 CUDA 内核开发、执行模型、内存层次结构和 GPU 编程中优化技术的官方文档。
- [PyTorch C++ Extensions](https://pytorch.org/docs/stable/cpp_extension.html) — PyTorch Developers (2023)
  Publisher: PyTorch Foundation
  关于为 PyTorch 创建自定义 C++ 和 CUDA 扩展的官方指南，包括构建系统和 Python 绑定。
- [Programming Massively Parallel Processors: A Hands-on Approach](https://www.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-32-391231-0) — Wen-mei W. Hwu, David B. Kirk, Izzat El Hajj (2022)
  Publisher: Morgan Kaufmann; DOI: [10.1016/C2020-0-03823-3](https://doi.org/10.1016/C2020-0-03823-3)
  一本关于 GPU 架构和编程的教科书，专注于 CUDA，有助于超越基础示例的学习。
