---
course: "advanced-jax"
chapter: "jax-interoperability-custom-operations"
lesson: "practice-integrating-cpp-function"
sourceId: 2925
sourceUrl: "https://apxml.com/zh/courses/advanced-jax/chapter-5-jax-interoperability-custom-operations/practice-integrating-cpp-function"
title: "实践：整合 C++ 函数"
description: "构建一个简单示例，通过回调函数或自定义原语将外部 C++ 函数整合到 JAX 计算中。"
order: 10
plots: []
sourceHash: "e19eb3bc7b697733ad34880a010b797a0670c9c32bdb0531069959dbdcdcee2c"
sourceCorrections: []
---

将一个简单的 C++ 函数整合到 JAX 工作流程中，这是一个实际的应用场景。虽然 JAX 和 XLA 已经高度优化，但你可能会遇到需要使用现有 C++ 库的情况，或直接在更低层级的语言中实现自定义的、对性能要求高的操作。

我们将介绍两种主要方法来实现此目的：一种是使用回调函数进行更简单的集成，另一种是概述创建完整自定义原语的步骤，以实现更紧密的集成。对于这个动手示例，我们将专注于使用 `ctypes` 和 `jax.pure_callback` 来实现回调方法。

### 场景：一个自定义的逐元素操作

假设我们想执行一个特定的逐元素操作，其由函数 $f(x) = x^2 + 10$ 定义。虽然这在 JAX 中可以直接轻松实现，但我们将假装它是一个在 C++ 中实现的复杂遗留计算，我们希望调用它。

### 步骤 1：C++ 实现

首先，我们来编写这个简单的 C++ 函数。我们需要确保它具有 C 语言链接 (`extern "C"`) 以防止 C++ 名称修饰，从而使其能够通过 `ctypes` 从 Python 轻松调用。我们将对 double 类型的数组进行操作。

```cpp
// custom_op.cpp
#include <vector>
#include <cmath> // 如果需要std::pow，尽管x*x更简单

// 使用 extern "C" 防止 C++ 名称修饰
extern "C" {
    // 函数接受输入数组、输出数组和大小
    void custom_elementwise_func(const double* input, double* output, int size) {
        for (int i = 0; i < size; ++i) {
            output[i] = input[i] * input[i] + 10.0;
        }
    }
}
```

### 步骤 2：将 C++ 代码编译为共享库

现在，我们将此 C++ 代码编译为共享库（在 Linux/macOS 上是 `.so`，在 Windows 上是 `.dll`）。具体命令可能会因您的编译器 (g++ 或 clang) 和操作系统而略有不同。

**在 Linux 上：**

```bash
g++ -shared -fPIC -o custom_op.so custom_op.cpp
```

**在 macOS 上：**

```bash
g++ -shared -o custom_op.dylib custom_op.cpp
```

**在 Windows 上（使用 MinGW/MSVC）：** （命令可能不同）

```bash
g++ -shared -o custom_op.dll custom_op.cpp -Wl,--out-implib,libcustom_op.a
```

请确保编译好的库（例如 `custom_op.so`）位于 Python 可以找到的位置，对于本示例而言，通常是当前工作目录。

### 步骤 3：使用 `ctypes` 的 Python 封装

我们将使用 Python 内置的 `ctypes` 库来加载共享库并定义 `custom_elementwise_func` 的函数签名。

```python
import ctypes
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import host_callback as hcb # 为简洁起见使用别名
from jax.experimental import pure_callback # 纯函数首选

# 加载共享库
try:
    # 根据您的操作系统和编译调整路径/名称
    lib = ctypes.CDLL('./custom_op.so') # Linux 示例
    # lib = ctypes.CDLL('./custom_op.dylib') # macOS 示例
    # lib = ctypes.CDLL('./custom_op.dll') # Windows 示例
except OSError as e:
    print(f"Error loading shared library: {e}")
    print("确保 C++ 代码已编译且库位于正确路径。")
    # 适当退出或处理错误
    exit()

# 定义 C 函数的参数类型和返回类型
lib.custom_elementwise_func.argtypes = [
    ctypes.POINTER(ctypes.c_double), # const double* input
    ctypes.POINTER(ctypes.c_double), # double* output
    ctypes.c_int                     # int size
]
lib.custom_elementwise_func.restype = None # void 返回类型

# 创建一个处理 NumPy 数组转换的 Python 封装函数
def custom_op_numpy(x_np: np.ndarray) -> np.ndarray:
    """使用 NumPy 数组调用 C++ 函数。"""
    if x_np.dtype != np.float64:
        # 确保数据是 C++ 期望的双精度浮点数
        x_np = x_np.astype(np.float64)

    # 确保输入在内存中是连续的
    x_np = np.ascontiguousarray(x_np)

    # 创建一个与输入形状和类型相同的输出数组
    output_np = np.empty_like(x_np)
    size = x_np.size

    # 获取数据缓冲区的指针
    input_ptr = x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    output_ptr = output_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # 调用 C 函数
    lib.custom_elementwise_func(input_ptr, output_ptr, size)

    return output_np

# 直接测试 NumPy 封装（可选）
test_input_np = np.array([1.0, 2.0, 3.0], dtype=np.float64)
result_np = custom_op_numpy(test_input_np)
print(f"NumPy 封装测试：输入={test_input_np}，输出={result_np}")
# 预期输出：[11. 14. 19.]
```

这个 Python 封装 `custom_op_numpy` 接收一个 NumPy 数组，确保其类型正确 (`float64`) 且内存连续，准备一个输出数组，使用 `ctypes` 获取内存指针，调用 C 函数，并以 NumPy 数组形式返回结果。

### 步骤 4：使用 `pure_callback` 与 JAX 集成

现在，我们将这个基于 NumPy 的函数整合到 JAX 中。由于我们的 C++ 函数在数学上是纯粹的（没有副作用，输出只取决于输入），因此 `jax.pure_callback` 是合适的工具。它允许 JAX 追踪函数的形状/数据类型行为，并将其整合到 JIT 编译的计算中，尽管 C++ 代码本身不会被 XLA 优化。

```python
def custom_op_jax_via_callback(x: jax.Array) -> jax.Array:
    """通过 pure_callback 调用 C++ 代码的 JAX 函数。"""
    # 定义预期输出的形状和数据类型
    # 对于此逐元素操作，它与输入相同
    result_shape_dtype = jax.ShapeDtypeStruct(x.shape, x.dtype)

    # 使用 pure_callback
    # 参数：
    # 1. 回调函数（接受 NumPy 数组，返回 NumPy 数组）
    # 2. 结果的形状/数据类型结构
    # 3. 输入 JAX 数组
    # vectorized=True 告诉 JAX 如果底层 C/Python 函数设计支持（我们的隐式支持），它可以自动处理批量维度。
    result = pure_callback(
        custom_op_numpy, result_shape_dtype, x, vectorized=True
    )
    return result

# 测试 JAX 函数
x_jax = jnp.arange(1.0, 5.0, dtype=jnp.float64)
y_jax = custom_op_jax_via_callback(x_jax)

print(f"JAX 回调测试（即时执行）：输入={x_jax}，输出={y_jax}")
# 预期输出：[11. 14. 19. 26.]

# 验证它在 JIT 编译下是否有效
custom_op_jax_jit = jax.jit(custom_op_jax_via_callback)
y_jax_jit = custom_op_jax_jit(x_jax)

# 确保计算在打印前完成
y_jax_jit.block_until_ready()

print(f"JAX 回调测试（JIT）：输入={x_jax}，输出={y_jax_jit}")
# 预期输出：[11. 14. 19. 26.]

# 验证求导（没有自定义规则会失败！）
try:
    grad_func = jax.grad(lambda x: jnp.sum(custom_op_jax_via_callback(x)))
    g = grad_func(x_jax)
    print(f"梯度计算：{g}")
except Exception as e:
    print(f"\n梯度计算如预期般失败：{e}")
    print("像 pure_callback 这样的回调函数不会自动求导。")
```

如示例所示，`pure_callback` 允许从 JIT 编译的 JAX 代码中调用 C++ 函数（在 Python 中封装）。然而，请注意一个值得注意的限制：JAX 无法通过回调自动求导。C++ 代码对 JAX 的自动微分系统来说是不透明的。

### 替代方案：自定义原语（概述）

如果您需要完全集成，包括自动微分和针对*调用*本身的潜在 XLA 优化（尽管不是 C++ 内部实现），您将需要定义一个自定义 JAX 原语。这是一个更复杂的过程：

1. **定义原语：** 创建一个 `jax.core.Primitive` 实例。

   ```python
   # 示例结构 - 需要更多导入和细节
   # from jax import core
   # custom_op_p = core.Primitive("custom_op")
   ```
2. **实现抽象求值：** 定义一个函数，该函数根据输入形状和数据类型告诉 JAX 输出的形状和数据类型。这在追踪过程中会用到。

   ```python
   # def custom_op_abstract_eval(x_abstract):
   #     # 对于逐元素操作，输出形状/数据类型与输入相同
   #     return jax.core.ShapedArray(x_abstract.shape, x_abstract.dtype)
   # custom_op_p.def_abstract_eval(custom_op_abstract_eval)
   ```
3. **实现下沉规则：** 这是最复杂的一步。您需要告诉 XLA 如何在每个后端（CPU、GPU、TPU）上执行您的原语。这通常涉及使用 XLA 的 HLO（高级操作）指令编写代码，或者使用诸如 `xla_client.ops.CustomCall` 之类的机制从 XLA 生成的代码中调用您预编译的 C++ 函数。

   ```python
   # from jax.interpreters import xla
   # def custom_op_xla_translation(ctx, x_operand, **params):
   #      # 生成调用 C++ 函数的 XLA HLO 代码
   #      # 这可能涉及使用 XLA 的 ExternalCall 或类似机制
   #      # ... 高度依赖于后端和 XLA 细节 ...
   #      pass
   # xla.register_translation(custom_op_p, custom_op_xla_translation)
   ```
4. **实现求导规则：** 为您的原语定义 JVP（前向模式）和/或 VJP（反向模式）规则。对于我们的示例 $f(x) = x^2 + 10$，导数是 $f'(x) = 2x$。您将实现计算 JVP $(v \mapsto 2x \odot v)$ 和 VJP $(v \mapsto 2x \odot v)$ 的规则，其中 $\odot$ 是逐元素乘法。

   ```python
   # from jax.interpreters import ad
   # def custom_op_jvp_rule(primals, tangents):\
   #     (x,) = primals
   #     (x_dot,) = tangents
   #     y = custom_op_p.bind(x) # 为原始输出调用原语
   #     # 导数是 2*x，所以 JVP 是 (2*x) * x_dot
   #     y_dot = (2 * x) * x_dot
   #     return y, y_dot
   # ad.primitive_jvps[custom_op_p] = custom_op_jvp_rule
   #
   # # 类似地，用于 VJP 规则 (jax.grad 所需)
   # def custom_op_vjp_rule(cotangent, x):\
   #    # 对于逐元素 * 标量函数，VJP 在数学上等同于 JVP
   #    # vjp = lambda v: (2*x) * v
   #    # return vjp(cotangent)
   #    return (2 * x) * cotangent
   # ad.primitive_transposes[custom_op_p] = custom_op_vjp_rule
   ```
5. **创建绑定函数：** 创建一个面向用户的 Python 函数，该函数使用 `primitive.bind()` 调用您的原语。

   ```python
   # def custom_op_jax_via_primitive(x):
   #    return custom_op_p.bind(x)
   ```

创建自定义原语可提供最紧密的集成，但需要理解 JAX 的内部机制以及潜在的 XLA 知识。

### 总结

在此实践中，我们成功地使用 `ctypes` 和 `jax.pure_callback` 将一个简单的 C++ 函数整合到 JAX 中。当您需要从 JIT 编译的代码中调用外部纯函数，但不需要通过外部代码进行自动微分时，这种方法是有效的。

- **回调函数 (`pure_callback`, `host_callback`)**：对于现有代码更容易实现，适用于不可微分的部分或与具有副作用的系统（`host_callback`）进行接口。它们在 JAX 计算图中作为不透明调用。对于功能上纯粹的外部代码，推荐使用 `pure_callback`。
- **自定义原语**：提供完全集成，包括对调用本身的潜在后端优化，并在提供规则时支持自动微分。此方法复杂得多，需要定义抽象求值、后端下沉和微分规则。

根据您在性能、微分以及您愿意管理的复杂性方面的具体需求选择方法。对于许多涉及调用外部库而无需通过它们获取梯度的用例，回调函数提供了一个实用方案。

## 参考资料

- [JAX and C++ Interoperability](https://jax.readthedocs.io/en/latest/jax.ffi.html) — JAX core contributors (2023)
  Publisher: Google
  JAX官方文档，提供了使用主机回调（jax.pure_callback、jax.experimental.host_callback）集成外部C++代码的指南。
- [How JAX primitives work](https://jax.readthedocs.io/en/latest/how_jax_primitives_work.html) — JAX core contributors (2023)
  Publisher: Google
  JAX官方文档，详细介绍了JAX原语的内部机制，包括抽象评估、XLA编译以及微分规则，这是自定义操作的基础。
- [ctypes - A foreign function library for Python](https://docs.python.org/3/library/ctypes.html) — Python Software Foundation (2023)
  Publisher: Python Software Foundation
  Python `ctypes`模块的官方文档，对于Python与C共享库进行接口操作至关重要。
- [XLA: Optimizing Compilations for Machine Learning](https://doi.org/10.1145/3382098.3403160) — Alex Aiken, Chris Leary, Roy Frostig, Peter Hawkins, Matthew Johnson, Skye Mendel, et al. (2020)
  Journal: Proceedings of the 3rd MLSys Conference (MLSys '20); Publisher: ACM; DOI: [10.1145/3382098.3403160](https://doi.org/10.1145/3382098.3403160)
  描述XLA编译器及其架构的论文，解释了它如何优化机器学习模型，为JAX自定义原语的底层编译提供了更深入的背景。
