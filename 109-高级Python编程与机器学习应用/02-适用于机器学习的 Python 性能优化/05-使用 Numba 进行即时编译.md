# 使用 Numba 进行即时编译

来源：[原文](https://apxml.com/zh/courses/advanced-python-programming-ml/chapter-2-python-performance-optimization-ml/using-numba-jit)

[返回章节目录](README.md) · [返回课程目录](../README.md)

使用 NumPy 和 Pandas 进行向量 (vector)化是一种在 Python 机器学习 (machine learning)编程中实现显著加速的常用方法。然而，有些数值算法包含循环或逻辑，它们难以或不便进行向量化 (quantization)。此外，直接将某些数学操作转换为高效的 NumPy 代码也并非总是简单直接。对于这些情况，特别是涉及对数值数据进行计算密集的循环时，**Numba** 提供了一种优雅且通常非常高效的办法。

Numba 是一个开源的即时 (JIT) 编译器，它能将部分 Python 和 NumPy 代码编译成快速机器码。它通过使用 LLVM 编译器架构实现这一点。使用 Numba 最常见的方式是通过其函数装饰器，这些装饰器会在被装饰的函数首次被调用时指示 Numba 进行编译。

### `@jit` 装饰器：你进入 Numba 的途径

Numba 的主要接口是 `@numba.jit` 装饰器（通常作为 `from numba import jit` 导入）。让我们考虑一个简单的函数，它在一个大型数组上逐元素执行计算，这种模式有时会出现在自定义激活函数 (activation function)或损失计算中。

```python
import numpy as np
import numba
import time

# 纯 Python/NumPy 函数（已有一定优化）
def calculate_logistic_numpy(x):
    return 1.0 / (1.0 + np.exp(-x))

# Numba 加速版本
@numba.jit
def calculate_logistic_numba(x):
    # Numba 能识别基本的 NumPy 函数和循环
    result = np.empty_like(x)
    for i in range(x.shape[0]):
        result[i] = 1.0 / (1.0 + np.exp(-x[i]))
    return result

# 创建一些数据
large_array = np.random.rand(10_000_000) * 10 - 5 # 大型数组

# --- 计时 NumPy 版本 ---
start_time = time.time()
result_numpy = calculate_logistic_numpy(large_array)
numpy_time = time.time() - start_time
print(f"NumPy 版本时间：{numpy_time:.4f} 秒")

# --- 计时 Numba 版本（包含首次调用编译时间） ---
start_time = time.time()
result_numba_first = calculate_logistic_numba(large_array)
numba_compile_time = time.time() - start_time
print(f"Numba 版本时间（首次调用，含编译）：{numba_compile_time:.4f} 秒")

# --- 再次计时 Numba 版本（已编译代码） ---
start_time = time.time()
result_numba_second = calculate_logistic_numba(large_array)
numba_run_time = time.time() - start_time
print(f"Numba 版本时间（第二次调用，已缓存）：{numba_run_time:.4f} 秒")

# 验证结果是否接近
assert np.allclose(result_numpy, result_numba_second), "结果不一致！"
```

运行这段代码通常会显示，尽管首次调用 Numba 函数会产生编译开销，但后续调用会明显快于纯 NumPy 版本，尤其当底层操作未能完美向量 (vector)化或涉及 Numba 可以在循环内优化的复杂控制流时。Numba 擅长优化操作 NumPy 数组的 Python 循环。

### Nopython 模式：达到最高性能

Numba 有两种主要的编译模式：**nopython 模式**和**object 模式**。

- \*\*Nopython 模式：\*\*这是性能方面的首选模式。Numba 会尝试完全编译函数，而不回退到 Python C API。这意味着所有变量和操作都必须能被 Numba 推断类型。如果 Numba 无法在 nopython 模式下编译（例如，由于不支持的 Python 特性），并且指定了 `nopython=True`，它将默认引发错误。此模式会带来最明显的加速，因为它生成专门的机器码。
- \*\*Object 模式：\*\*如果 *未* 指定 `nopython=True`，并且 Numba 遇到无法优化的代码（例如对 Python 列表的操作或不支持的函数），它可能会退回到 object 模式。在此模式下，Numba 基本上只编译它*能*优化的循环，并通过回调 Python 解释器来处理其余部分。这提供的性能提升少得多，有时甚至因为开销而比纯 Python 更慢。

对于对性能要求高的代码，您应该总是通过显式使用 `@jit(nopython=True)` 来力求使用 nopython 模式：

```python
@numba.jit(nopython=True) # 显式请求 nopython 模式
def pairwise_distance_numba(X, Y):
    """ 计算成对欧几里得距离 """
    M = X.shape[0]
    N = Y.shape[0]
    D = np.empty((M, N), dtype=np.float64)
    for i in range(M):
        for j in range(N):
            # 计算欧几里得距离的平方
            sum_sq_diff = 0.0
            for k in range(X.shape[1]): # 假设 X, Y 具有相同数量的特征
                diff = X[i, k] - Y[j, k]
                sum_sq_diff += diff * diff
            D[i, j] = np.sqrt(sum_sq_diff)
    return D

# 使用示例：
X_data = np.random.rand(100, 10) # 100 个点，10 个特征
Y_data = np.random.rand(150, 10) # 150 个点，10 个特征

# 首次调用时编译
distances = pairwise_distance_numba(X_data, Y_data)

# 后续调用会很快
# distances_again = pairwise_distance_numba(X_data, Y_data)
```

如果 `pairwise_distance_numba` 包含 Numba 在 nopython 模式下无法处理的操作（例如，在没有特定 Numba 支持的情况下在内部循环中打印到控制台，或使用不支持的数据类型），使用 `@jit(nopython=True)` 将会引发 `TypingError` 错误，迫使您重构代码，使其与 Numba 兼容以获得最佳性能。

### 指定类型签名

虽然 Numba 通常会自动推断函数参数 (parameter)的类型，但您可以提供显式类型签名。这有时可以帮助 Numba 生成更专门的代码，或允许进行预先 (AOT) 编译，尽管带有类型推断的 JIT 编译更为常见。签名定义了参数类型和返回类型。

```python
from numba import float64, int32

# 签名：接受两个一维 float64 数组，返回一个 float64 标量
@numba.jit(float64(float64[:], float64[:]), nopython=True)
def dot_product(a, b):
    result = 0.0
    for i in range(a.shape[0]):
        result += a[i] * b[i]
    return result

vec1 = np.arange(5, dtype=np.float64)
vec2 = np.arange(5, dtype=np.float64) * 2
dp = dot_product(vec1, vec2) # 对 float64 数组使用已编译版本
```

这里，`float64[:]` 表示一个由 64 位浮点数组成的一维数组。通常，只有在特定的 AOT 编译场景或不同输入类型需要多个签名时才需要指定签名。对于大多数 JIT 使用情况，让 Numba 推断类型就足够且更灵活。

### 编译缓存

如前所述，首次调用 Numba JIT 编译的函数会产生编译开销。为了避免您的脚本每次运行时都产生此开销，您可以指示 Numba 缓存已编译的机器码：

```python
@numba.jit(nopython=True, cache=True)
def some_fast_function(x):
    # ... 大量计算 ...
    return x * x
```

设置 `cache=True` 后，Numba 会将为遇到的特定输入类型编译的代码写入文件系统缓存（`__pycache__` 子目录）。在后续运行中，如果函数源代码未更改且输入类型与缓存版本匹配，Numba 会加载缓存代码，大大地加快了后续脚本执行中的“首次”调用。

### 何时在机器学习 (machine learning)中使用 Numba

Numba 在以下情况中表现出色：

1. \*\*自定义数值算法：\*\*实现优化算法（如自定义梯度下降 (gradient descent)步骤）、距离计算（如成对距离示例）或涉及对数值数据的显式循环的模拟组件。
2. \*\*特征工程：\*\*复杂的特征转换，需要按行或按元素进行操作，而这些操作难以纯粹用向量 (vector)化的 NumPy/Pandas 函数来表达。
3. \*\*加速推理 (inference)的部分：\*\*加速自定义模型预测逻辑中特定、计算受限的部分，特别是如果它们涉及循环。
4. \*\*绕过 NumPy 的限制：\*\*当一个操作很简单但没有直接、高效的 NumPy 等价实现时，用循环实现它并用 Numba 进行 JIT 编译可能比复杂的向量化 (quantization)尝试更快。

### 限制与考量

- \*\*兼容性：\*\*Numba 支持 Python 和 NumPy 的很大一部分，但并非所有功能。它最适用于标准数值类型（整数、浮点数）、NumPy 数组和简单的控制流（循环、条件语句）。对 Python 类的支持存在但有局限性，并且 Numba 编译的代码中通常不支持直接操作 Pandas DataFrame（您通常需要提取底层的 NumPy 数组）。
- \*\*编译开销：\*\*JIT 编译时间可能会很明显，特别是对于复杂的函数。缓存有助于在多次脚本运行中缓解此问题，但不适用于首次执行。
- \*\*调试：\*\*调试 Numba 编译的代码可能比纯 Python 更具挑战性。错误可能源于编译阶段或已编译的机器码本身，有时提供的追踪信息不够直观。像 `pdb` 这样的标准 Python 调试器无法有效地进入 `nopython` 编译的代码。
- \*\*重点：\*\*Numba 主要为数值计算而设计，不适用于 I/O 密集型任务或字符串操作，在这些方面其益处微乎其微或根本没有。

### 示例：性能比较

让我们量化 (quantization)成对距离示例的潜在加速。

```python
import numpy as np
import numba
import time
from math import sqrt # 纯 Python 版本使用标准 math 库的 sqrt

def pairwise_distance_python(X, Y):
    """ 使用标准 math 库的纯 Python 版本 """
    M = X.shape[0]
    N = Y.shape[0]
    D = np.empty((M, N), dtype=np.float64)
    for i in range(M):
        for j in range(N):
            sum_sq_diff = 0.0
            for k in range(X.shape[1]):
                diff = X[i, k] - Y[j, k]
                sum_sq_diff += diff * diff
            D[i, j] = sqrt(sum_sq_diff) # 使用 math.sqrt
    return D

@numba.jit(nopython=True, cache=True) # Numba 版本
def pairwise_distance_numba(X, Y):
    M = X.shape[0]
    N = Y.shape[0]
    D = np.empty((M, N), dtype=np.float64)
    for i in range(M):
        for j in range(N):
            sum_sq_diff = 0.0
            for k in range(X.shape[1]):
                diff = X[i, k] - Y[j, k]
                sum_sq_diff += diff * diff
            # Numba 可以高效地优化标量上的 np.sqrt
            D[i, j] = np.sqrt(sum_sq_diff)
    return D

# 数据设置（根据您的机器调整大小）
X_data = np.random.rand(200, 50)
Y_data = np.random.rand(300, 50)

# --- 计时纯 Python 版本 ---
start = time.time()
dist_py = pairwise_distance_python(X_data, Y_data)
time_py = time.time() - start

# --- 计时 Numba 版本（首次调用） ---
start = time.time()
dist_nb1 = pairwise_distance_numba(X_data, Y_data)
time_nb1 = time.time() - start

# --- 计时 Numba 版本（第二次调用） ---
start = time.time()
dist_nb2 = pairwise_distance_numba(X_data, Y_data)
time_nb2 = time.time() - start

print(f"纯 Python 时间：{time_py:.4f} 秒")
print(f"Numba 时间（首次调用）：{time_nb1:.4f} 秒")
print(f"Numba 时间（第二次调用）：{time_nb2:.4f} 秒")
assert np.allclose(dist_py, dist_nb2)

# 可视化
plotly_bar_data = {
    "layout": {
        "title": "执行时间比较：成对距离",
        "yaxis": {"title": "时间（秒）", "type": "log"}, # Log scale often needed
        "xaxis": {"title": "实现方式"},
         "template": "plotly_white",
         "width": 600,
         "height": 400
    },
    "data": [
        {
            "type": "bar",
            "x": ["纯 Python", "Numba（首次调用）", "Numba（已缓存）"],
            "y": [time_py, time_nb1, time_nb2],
            "marker": {
                "color": ["#ff6b6b", "#ffc078", "#38d9a9"], # red, orange, teal
            }
        }
    ]
}
```



[交互图表：执行时间比较：成对距离](https://apxml.com/zh/courses/advanced-python-programming-ml/chapter-2-python-performance-optimization-ml/using-numba-jit#plot-qg0sya)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "执行时间比较：成对距离",
    "yaxis": {
      "title": "时间（秒）",
      "type": "log"
    },
    "xaxis": {
      "title": "实现方式"
    },
    "template": "plotly_white",
    "width": 600,
    "height": 400
  },
  "data": [
    {
      "type": "bar",
      "x": [
        "纯 Python",
        "Numba（首次调用）",
        "Numba（已缓存）"
      ],
      "y": [
        2.5123,
        0.2156,
        0.0189
      ],
      "marker": {
        "color": [
          "#ff6b6b",
          "#ffc078",
          "#38d9a9"
        ]
      }
    }
  ]
}
```

</details>



> 计算两组点（200x50 和 300x50）之间成对欧几里得距离的执行时间比较。请注意 y 轴上的对数刻度，它突出了缓存的 Numba 版本与纯 Python 相比所达到的明显加速。（实际时间会因硬件而异）。

如图所示，即使首次调用有编译开销，Numba 仍提供了明显的提升。后续调用会利用缓存的优化机器码，从而使执行时间比等效的纯 Python 循环实现快几个数量级，对于这类数值任务常常接近 C 语言级别的性能。

与 Cython（在上一节中讨论）相比，Numba 通常为加速现有 Python 函数提供了更低的入门门槛，主要只需添加一个装饰器。Cython 提供了更精细的控制，更好地支持复杂的 Python 对象，并且更容易与外部 C/C++ 代码集成，但它涉及单独的编译步骤，并且通常需要添加静态类型声明。两者之间的选择取决于具体的性能瓶颈和开发工作流程偏好。

Numba 是 Python 机器学习 (machine learning)实践者优化工具箱中一个强大的工具，对于加速那些难以直接向量 (vector)化的、循环密集的数值计算特别有效。通过了解如何应用 `@jit(nopython=True)` 并考虑其优点和局限性，您可以大幅减少机器学习流程中主要代码段的运行时间。

## 参考资料

- [Numba Documentation](https://numba.readthedocs.io) — The Numba Developers (2024)
  Numba特性、用法和API最官方、最全面的来源，包括JIT编译模式、装饰器和类型规范的详细信息。
- [High Performance Python: Practical Performant Programming for Humans with Computers](https://www.oreilly.com/library/view/high-performance-python/9781492055013/) — Micha Gorelick and Ian Ozsvald (2020)
  Publisher: O'Reilly Media
  一本优化Python代码的实用指南，涵盖Numba、NumPy、Cython和性能分析等主题。它提供了加速数值计算的策略。
- [Effective Computation in Physics: Field Guide to Research with Python](https://www.oreilly.com/library/view/effective-computation-in/9781491901564/) — Anthony Scopatz, Kathryn D. Huff (2015)
  Publisher: O'Reilly Media
  为科学研究人员提供使用Python编写高效且可维护的科学代码的指导，包括数值库和Numba等性能工具的讨论。

---

[上一节](04-Cython%E7%AE%80%E4%BB%8B%EF%BC%9A%E5%8A%A0%E9%80%9FPython%E4%BB%A3%E7%A0%81.md) · [下一节](06-%E7%90%86%E8%A7%A3Python%E7%9A%84%E5%85%A8%E5%B1%80%E8%A7%A3%E9%87%8A%E5%99%A8%E9%94%81%EF%BC%88GIL%EF%BC%89.md)
