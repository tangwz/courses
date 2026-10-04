---
course: "advanced-pytorch"
chapter: "custom-extensions-interoperability"
lesson: "foreign-function-interfaces"
sourceId: 2787
sourceUrl: "https://apxml.com/zh/courses/advanced-pytorch/chapter-6-custom-extensions-interoperability/foreign-function-interfaces"
title: "外部函数接口 (FFI)"
description: "使用 FFI 技术将现有 C 库与 PyTorch 集成。"
order: 7
plots: []
sourceHash: "f1176347ac604946fe6bba8052c25e38fe0bc200dc8c7625396d22f202aac585"
sourceCorrections: []
---

虽然构建自定义 C++ 或 CUDA 扩展能提供与 PyTorch 最紧密的结合，尤其是对于需要自动求导支持的操作，但在某些情况下，您需要与现有 C 库对接，而无需将它们重写为完整的 PyTorch 扩展。这时，外部函数接口 (FFI) 就派上用场了。FFI 允许 Python 代码调用用其他语言（最常见的是 C 或 C++）编写并编译成共享库的函数。

Python 的标准库包含 `ctypes` 模块，它是为此目的设计的一个强大工具。它能够加载共享库（Linux/macOS 上的 `.so` 文件，Windows 上的 `.dll` 文件），并直接从 Python 调用其中的函数。这种方法在以下情况中特别有用：

1. **运用现有代码：** 您有一个经过良好测试和优化的 C/C++ 库，用于执行特定任务（例如，专用信号处理、物理模拟、自定义数据解析器），您想将其整合到 PyTorch 数据管道或模型中。
2. **硬件交互：** 与硬件供应商 SDK 对接通常需要调用 C 函数。
3. **性能瓶颈：** 某些纯 Python 操作可能过慢，而通过 FFI 调用的有针对性的 C 实现可以提供显著的加速，同时避免完整 C++/CUDA 扩展的复杂性（特别是在该特定部分不需要自动求导时）。

### 使用 `ctypes` 进行 C 集成

`ctypes` 的核心流程包含以下步骤：

1. **编译 C 代码：** 您的 C 代码必须编译成共享库（例如，使用 GCC 或 Clang 并带上 `-shared` 和 `-fPIC` 标志）。
2. **在 Python 中加载库：** 使用 `ctypes.CDLL` 或 `ctypes.PyDLL` 将编译好的共享库加载到您的 Python 进程中。
3. **定义函数签名：** 为您打算调用的 C 函数指定参数 (parameter)类型 (`argtypes`) 和返回类型 (`restype`)。这对于 `ctypes` 在 Python 和 C 之间正确编组数据非常重要。`ctypes` 提供与 C 类型对应的类型（例如，`c_int`、`c_float`、`c_double`、`c_void_p`）。
4. **准备数据：** 将 Python 数据转换为与 C 函数兼容的格式。对于涉及 PyTorch 张量的数值操作，这通常意味着获取指向张量底层数据的原始指针。
5. **调用 C 函数：** 使用已加载的库对象从 Python 调用 C 函数。

### 与 PyTorch 张量对接

将 C 库与 PyTorch 集成时最常见的需求是传递张量数据。由于 C 函数操作原始内存缓冲区，您需要提供指向张量数据的指针。PyTorch 张量为此提供了 `data_ptr()` 方法。

```python
import torch
import ctypes

# 假设 'mylib.so' 或 'mylib.dll' 包含一个函数：
# void process_data(float* data_ptr, int size);

# 加载共享库
try:
    # 根据需要调整路径/名称
    lib = ctypes.CDLL('./mylib.so')
except OSError as e:
    print(f"加载共享库错误: {e}")
    # 适当处理错误，例如在 Windows 上尝试 .dll 等。
    exit()

# 定义函数签名
try:
    process_data_func = lib.process_data
    process_data_func.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    process_data_func.restype = None # void 返回类型
except AttributeError as e:
    print(f"查找函数或设置签名错误: {e}")
    # 处理错误：库中可能不存在该函数
    exit()

# 创建一个 PyTorch 张量
tensor = torch.randn(100, dtype=torch.float32)

# --- 重要部分：确保张量数据布局兼容 ---
# 许多 C 函数期望连续的 C 风格数组。
if not tensor.is_contiguous():
    tensor = tensor.contiguous()

# 获取数据指针（作为 void 指针，然后进行类型转换）
data_ptr_void = tensor.data_ptr()
# 将 void 指针转换为 C 函数期望的特定类型
data_ptr_c = ctypes.cast(data_ptr_void, ctypes.POINTER(ctypes.c_float))

# 调用 C 函数
size = tensor.numel()
try:
    process_data_func(data_ptr_c, ctypes.c_int(size))
    print("成功调用 C 函数。")
    # 'tensor' 的数据可能被 C 函数原地修改
    # print(tensor)
except Exception as e:
    print(f"C 函数执行错误: {e}")
```

**重要注意事项：**

- **内存管理：** 当您将 `tensor.data_ptr()` 传递给 C 时，您是在共享内存。C 代码直接读取或写入由 PyTorch 管理的内存。您必须确保 PyTorch 张量在 C 函数使用其指针的整个过程中保持分配和有效。在传递指针后在 Python 中修改张量的大小或存储可能导致崩溃或数据损坏。
- **数据连续性：** C 函数通常期望数据数组在内存中是连续的（就像标准的 C 数组一样）。某些操作（例如切片、转置）产生的 PyTorch 张量可能不连续。在为需要连续数据的 C 函数获取数据指针之前，始终使用 `tensor.is_contiguous()` 进行检查，并在必要时调用 `tensor.contiguous()`。
- **类型匹配：** 精确匹配 C 函数的参数 (parameter)类型与 `ctypes` 定义（`c_float`、`c_int`、`POINTER(...)` 等），并确保 PyTorch 张量的 `dtype` 与使用的指针类型一致（例如，`torch.float32` 对应 `ctypes.POINTER(ctypes.c_float)`）。不匹配会导致未定义行为。
- **错误处理：** C 函数可能不会引发 Python 异常。您可能需要设计您的 C 函数返回错误代码或状态指示器，并在您的 Python 包装代码中显式检查这些信息。
- **全局解释器锁 (GIL)：** 通过 `ctypes` 调用 C 函数可以释放 Python 的 GIL，这意味着 C 代码可以与其他 Python 线程（如果有的话）并发执行。如果 C 代码是 CPU 密集型且耗时较长，这可以提供并行性。然而，如果 C 代码频繁回调 Python C API，它可能会重新获取 GIL，从而限制并发优势。
- **复杂性：** 尽管 `ctypes` 很强大，但为具有数据结构、回调或错误处理的复杂 C API 创建绑定可能变得有难度。

### 图示：FFI 数据流

> 流程图说明了如何通过 `ctypes` 获取 PyTorch 张量的内存地址并传递给编译好的 C 共享库中的函数。C 函数直接操作张量的内存。

### 替代方案：CFFI

Python 中另一个常用的 FFI 库是 CFFI（C 外部函数接口）。CFFI 通常要求您提供 C 函数声明（例如，在 Python 字符串中使用 C 语法），并处理大部分类型转换和接口生成。对于复杂 API 而言，它有时可能更易于使用，并且在某些情况下可能比 `ctypes` 提供更好的性能。然而，`ctypes` 是标准库的一部分，无需额外安装。

### 何时使用 FFI 与自定义扩展

使用 `ctypes` 或 CFFI 的 FFI 通常最适合集成现有、自包含的 C/C++ 库，且这些 C 函数不需要自动求导支持。如果您需要与 PyTorch 的自动求导引擎紧密结合，需要为您的 C/C++ 代码实现自定义梯度计算，或者专门为 PyTorch 构建性能关键组件，那么编写原生 C++ 或 CUDA 扩展（如章节“构建自定义 C++ 扩展”和“构建自定义 CUDA 扩展”中所述）是更合适、功能更强的方法。当运用外部预编译代码是主要目标时，FFI 可作为实用的桥梁。

## 参考资料

- [\`ctypes\` - A foreign function library for Python](https://docs.python.org/3/library/ctypes.html) — Python Software Foundation (2024)
  Python标准库模块的官方参考资料，用于调用共享库中的函数。
- [CFFI Documentation](https://cffi.readthedocs.io/en/latest/) — Armin Rigo, Maciej Fijalkowski (2024)
  CFFI的官方文档，它是Python的另一种外部函数接口。
- [Extending PyTorch with C++ and CUDA](https://pytorch.org/docs/stable/cpp_extension.html) — PyTorch Authors (2025)
  Publisher: PyTorch
  提供构建自定义PyTorch扩展的指南，这是本节中讨论的FFI的替代方案。
- [High Performance Python: Practical Performant Programming for Humans](https://www.oreilly.com/library/view/high-performance-python/9781492055013/) — Micha Gorelick, Ian Ozsvald, Mikael Vejdemo-Johansson, Jon Roes, Reuven M. Lerner, Tarek Ziadé, David Beazley (2020)
  Publisher: O'Reilly Media; Pages: Chapter 7
  包含提高Python性能的实用策略，包括对`ctypes`和CFFI的详细解释。
