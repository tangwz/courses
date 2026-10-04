---
course: "advanced-jax"
chapter: "optimizing-jax-code-performance"
lesson: "asynchronous-dispatch"
sourceId: 2832
sourceUrl: "https://apxml.com/zh/courses/advanced-jax/chapter-2-optimizing-jax-code-performance/asynchronous-dispatch"
title: "异步调度"
description: "了解 JAX 的异步调度机制，以及如何使用 `block_until_ready()` 正确测试代码性能。"
order: 7
plots: []
sourceHash: "9f033e9d601047766ef20a4229efcb4f85f7635a865cec52eb1e171811c4faf4"
sourceCorrections: []
---

在优化 JAX 代码时，尤其是在 GPU 或 TPU 等加速器上，您可能会遇到代码计时结果出人意料地快，甚至快于预期的情况。这通常指向 JAX 执行模式的一个主要特性：**异步调度**。了解此机制对于准确衡量性能和构建高效流程非常重要。

## 异步调度的特性

与通常按顺序运行操作并等待每个操作完成的标准 Python 执行不同，JAX 在与加速器交互时通常异步运行。当您执行一个针对 GPU 或 TPU 的 JAX 函数（尤其是经过 JIT 编译的函数）时，JAX 会执行以下步骤：

1. **追踪并编译（如果需要）：** Python 函数被追踪以生成 `jaxpr`，XLA 将其编译为优化的设备代码。这发生在首次使用特定输入形状/类型调用时，或在触发重新编译时。
2. **调度：** JAX 将已编译的内核（实际计算）放入队列，以便在加速器（GPU/TPU）上执行。
3. **返回控制：** 非常重要的一点是，JAX *立即*将控制权返回给 Python 解释器，*而不*等待设备上的计算完成。

加速器在后台处理已调度的计算，而您的 Python 程序则继续执行后续的代码行。这种解耦允许 CPU (Python) 上运行的控制逻辑与加速器上运行的繁重数值计算之间存在潜在的并行性。

例如，当 GPU 忙于处理一批数据时，CPU 已经可以开始准备下一批数据（加载数据、预处理）。这种重叠可以显著提高应用程序的整体吞吐量 (throughput)。

## 对基准测试的影响

异步调度的主要结果是，标准的 Python 计时机制，例如 `time.time()` 或 `time.perf_counter()`，在衡量 JAX 计算在加速器上的实际执行时间时变得不可靠。

考虑这种简单的计时方法：

```python
import jax
import jax.numpy as jnp
import time

# 假设在 GPU/TPU 上运行
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (4096, 4096))

@jax.jit
def compute_heavy(m):
  return jnp.dot(m, m.T)

# 简单计时 - 仅测量调度时间
start_time = time.perf_counter()
result = compute_heavy(x)
# 控制权几乎立即返回此处！
end_time = time.perf_counter()

print(f"简单计时: {end_time - start_time:.6f} 秒")
# 这可能会打印一个非常小的数字，
# 不代表实际的矩阵乘法时间。
```

此处测量的 `end_time - start_time` 主要捕获 JAX 将 `jnp.dot` 操作调度到加速器所花费的时间，而不是加速器执行矩阵乘法可能花费的更长时间。

## 使用 `block_until_ready()` 进行准确计时

为了正确测量 JAX 异步操作的执行时间，您需要明确告知 Python 程序等待加速器上的计算实际完成。JAX 为此提供了 `block_until_ready()` 方法。

您可以在任何 JAX 数组 (`jax.Array`) 上调用此方法。这样做会阻塞 Python 解释器，直到在设备上生成该特定数组的计算完成为止。

这是对先前示例进行基准测试的修正方法：

```python
import jax
import jax.numpy as jnp
import time

# 假设在 GPU/TPU 上运行
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (4096, 4096))

@jax.jit
def compute_heavy(m):
  return jnp.dot(m, m.T)

# 提前强制编译（可选，但对于计时是好做法）
result_compiled = compute_heavy(x).block_until_ready()

# 正确计时 - 测量实际执行时间
start_time = time.perf_counter()
result = compute_heavy(x)
result.block_until_ready() # 等待计算完成
end_time = time.perf_counter()

print(f"正确计时: {end_time - start_time:.6f} 秒")
# 这将打印反映实际 GPU/TPU 执行时长的结果。
```

通过添加 `result.block_until_ready()`，我们确保 `end_time` 仅在 `jnp.dot` 操作在加速器上完成后才被记录。

或者，您可以使用 `jax.block_until_ready(result)`。如果您在没有参数 (parameter)的情况下调用 `jax.block_until_ready()`，或者在包含 JAX 数组的结构（例如 PyTree）上调用，它会等待所有设备上*所有*未完成的异步计算完成。

**何时使用 `block_until_ready()`：**

- **基准测试：** 在测量旨在加速器上运行的 JAX 代码部分的性能时，请务必使用它。
- **调试：** 如果怀疑存在与计时相关的问题，有时有助于确保操作按特定顺序完成。
- **同步点：** 当后续的非 JAX 代码依赖于完全计算出的结果时（例如，将结果保存到磁盘，或明确传输到 NumPy）。

## 隐式同步

请记住，某些操作会隐式强制同步，这意味着它们将自动等待必要的计算完成：

1. **复制到主机：** 通过 `np.array(jax_array)` 或使用 `.item()` 将 JAX 数组转换为 NumPy 数组，需要值在主机 CPU 上可用，因此会阻塞直到计算完成。
2. **打印：** 打印 JAX 数组通常会触发复制到主机以进行格式化，从而导致阻塞。

虽然存在这些隐式阻塞，但依赖它们进行基准测试不够明确，并可能造成混淆。使用 `block_until_ready()` 使同步点明确且有意。

了解异步调度对于正确解释 JAX 中的性能测量结果非常重要。在对 GPU/TPU 计算进行计时时，请务必使用 `block_until_ready()`，以确保您测量的是实际执行时间，而不仅仅是调度开销。这些信息还允许您可能调整代码结构，以便善用 CPU 和加速器工作之间的重叠，从而获得更好的整体性能。

## 参考资料

- [Asynchronous Dispatch](https://jax.readthedocs.io/en/latest/async_dispatch.html) — JAX developers (2024)
  JAX官方文档，解释了其异步执行模型以及如何在加速器上使用`block_until_ready()`进行准确计时。
- [XLA Overview](https://www.tensorflow.org/xla/) — Google Developers (2024)
  提供XLA的背景信息。XLA是JAX用于为加速器生成优化设备代码的领域特定编译器，支持JAX的异步调度。
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) — NVIDIA (2023)
  Publisher: NVIDIA
  提供了GPU编程中异步执行的详细信息，特别是通过CUDA流，这与JAX用于调度计算的底层机制相似。
