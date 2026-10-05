# 使用 jax.experimental.host_callback 调用外部 CPU/GPU 代码

来源：[原文](https://apxml.com/zh/courses/advanced-jax/chapter-5-jax-interoperability-custom-operations/calling-external-code-host-callback)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管 JAX 旨在通过 XLA 编译来高效执行数值程序，但有些情况您可能需要暂时脱离这个编译环境，并在 JAX 计算执行期间在宿主 CPU 上运行任意 Python 代码。这可能用于调试、记录中间值、与外部硬件交互或调用没有 JAX 对应项的库。

`jax.experimental.host_callback` 模块为此提供了函数，主要是 `id_tap` 和 `call`。然而，正如 `experimental` 命名空间所示，这些工具伴随着重要的注意事项，应谨慎使用。它们实质上打破了 JAX/XLA 的编译和优化界限。

### 运作方式：脱离 XLA

当包含 `host_callback.id_tap` 或 `host_callback.call` 的 JAX 函数被执行时（而不仅仅是追踪时），会发生以下情况：

1. **暂停执行：** 当达到回调指令时，加速器（GPU/TPU）或编译的 CPU 代码的执行会暂停。
2. **数据传输（设备 -> 宿主）：** 为回调指定的 JAX 数组参数 (parameter)会从设备内存传输到宿主 CPU 的内存。这涉及序列化和可能开销大的数据移动。
3. **宿主 Python 执行：** 指定的 Python 函数在宿主 CPU 上执行。它接收传输的数据，通常是 NumPy 数组。
4. **数据传输（宿主 -> 设备）：** 如果使用 `host_callback.call`，Python 函数返回的结果会从宿主 CPU 传回设备内存，这涉及反序列化。`id_tap` 不以同样的方式传回结果，因为它主要用于副作用。
5. **恢复执行：** 加速器上的执行会使用原始数据或返回的数据继续进行。

设备和宿主之间的这种往返会引入同步点和数据传输开销，可能抵消 JAX 和 XLA 的许多性能优势。

### 使用 `host_callback.id_tap` 实现副作用

`id_tap` 函数主要设计用于执行 Python 代码以产生副作用，例如打印或日志记录，而不会改变 JAX 计算的数据流。它在 JAX 计算图中充当一个“恒等”函数，这意味着它返回其输入 JAX 参数 (parameter)不变，但会触发宿主端的 Python 执行。

```python
import jax
import jax.numpy as jnp
from jax.experimental import host_callback
import numpy as np

# 这个函数在 Python 宿主上运行
def log_intermediate_data(arg, transform_info):
  # arg 在这里预期是一个 NumPy 数组
  print("\n--- 宿主回调 ---")
  print(f"接收到数据类型: {type(arg)}")
  print(f"数据形状: {arg.shape}")
  print(f"数据内容（示例）: {arg.flatten()[:5]}")
  print(f"JAX 变换上下文: {transform_info}")
  print("--- 宿主回调结束 ---\n")
  # id_tap 不使用显式返回值

@jax.jit
def process_data(x):
  y = jnp.sin(x) * 2.0
  # 在计算 y 后插入到计算流程中
  # tap_with_transform=True 提供关于 jit/vmap 等的信息。
  host_callback.id_tap(log_intermediate_data, y, tap_with_transform=True)
  # 下面使用的 'y' 的值是原始的 jnp.sin(x) * 2.0
  z = jnp.mean(jnp.cos(y))
  return z

# 示例用法
key = jax.random.PRNGKey(42)
input_data = jax.random.uniform(key, (8, 8))

print("正在运行带有 host_callback 的 JIT 编译函数...")
result = process_data(input_data)

# 重要提示：回调默认是异步执行的。
# 我们需要阻塞直到计算完成才能看到打印输出。
result.block_until_ready()

print(f"最终计算结果: {result}")
```

当您运行这段代码时，您会看到在 JIT 编译的 `process_data` 函数执行期间，`log_intermediate_data` 的输出打印到您的控制台。请注意 `result.block_until_ready()` 的使用。没有它，Python 脚本可能会在异步 JAX 计算完成并执行回调之前结束，因此您将无法可靠地看到打印输出。

一个简化版本 `id_print` 专门用于从宿主打印 JAX 数组：

```python
@jax.jit
def simple_print_example(x):
  y = x + 5.0
  # 直接打印 y（从宿主）
  host_callback.id_print(y, what="中间值 y")
  return y * 2.0

data = jnp.arange(3.0)
output = simple_print_example(data)
output.block_until_ready() # 需要此行才能看到打印输出
```

### 使用 `host_callback.call` 返回值

如果您的外部 Python 函数需要计算一个值，然后将其用于 JAX 计算中，则需要 `host_callback.call`。与 `id_tap` 不同，`call` 获取宿主函数的返回值，将其传回设备，并注入到 JAX 数据流中。

因为 JAX 在追踪期间（在函数实际运行之前）需要知道返回值的形状和数据类型以构建计算图，您**必须**提供 `result_shape_dtypes` 参数 (parameter)。

```python
import jax
import jax.numpy as jnp
from jax.experimental import host_callback
import numpy as np

# 模拟 JAX 中不可用的外部库函数
def external_cpu_calculation(data_np, parameter):
    # 这个函数在 Python 宿主上运行
    print(f"\n--- 宿主回调 (call) ---")
    print(f"接收到数据形状: {data_np.shape}")
    print(f"接收到参数: {parameter}")
    # 执行一些计算，可能使用 SciPy/OpenCV 等库
    result_np = (np.tanh(data_np) + parameter).astype(data_np.dtype)
    print(f"返回结果形状: {result_np.shape}")
    print(f"--- 宿主回调 (call) 结束 ---\n")
    return result_np

@jax.jit
def jax_workflow(x, ext_param):
    intermediate = jnp.log1p(jnp.abs(x))

    # 调用外部函数
    external_result = host_callback.call(
        external_cpu_calculation,         # 要在宿主上调用的 Python 函数
        (intermediate, ext_param),        # 参数（JAX 数组自动转换）
        # 重要提示：指定 *返回值* 的预期形状和数据类型
        result_shape_dtypes=intermediate  # 在这里，结果与 'intermediate' 具有相同的形状/数据类型
        # 如果不同，请使用：result_shape_dtypes=jax.ShapeDtypeStruct(shape=(...), dtype=jnp.float32)
    )

    # 在后续的 JAX 操作中使用宿主回调的结果
    final_output = external_result / (1.0 + jnp.mean(intermediate))
    return final_output

# 示例用法
input_array = jnp.linspace(-2.0, 2.0, 5, dtype=jnp.float32)
parameter_value = 0.5 # 作为参数传递的静态 Python 值

print("正在运行带有 host_callback.call 的 JIT 编译函数...")
final_result = jax_workflow(input_array, parameter_value)
final_result.block_until_ready() # 确保宿主执行完成

print(f"最终 JAX 结果: {final_result}")
```

### 重要注意事项和性能影响

使用 `host_callback` 存在显著的缺点：

1. **性能开销：** 数据传输（序列化、设备-宿主-设备往返）和同步引入了显著的开销。这很容易成为瓶颈，尤其是在对性能敏感的循环中使用时。
2. **优化屏障：** `host_callback` 对 XLA 编译器而言是一个不透明的屏障。XLA 无法跨回调融合操作，也无法执行依赖于分析整个计算图的优化。
3. **异步执行：** 回调遵循 JAX 的异步调度。Python 回调函数在 Python 代码中遇到该行时可能不会立即执行。执行通常发生在需要结果时（例如，通过 `block_until_ready()` 或数据传回 Python）。这在调试时可能令人困惑。
4. **副作用：** 宿主函数可以执行任意 I/O 或修改 Python 全局状态，这打破了使 JAX 程序更易于理解和转换的函数纯度。
5. **不可微分性：** 默认情况下，通过回调使用 `host_callback.call` 的函数**不可微分**。尝试通过 `call` 计算梯度将引发错误，除非您手动定义自定义微分规则（一个稍后介绍的复杂主题）或使用 `jax.lax.stop_gradient` 显式停止梯度。`id_tap` 通常放置在不需要梯度的地方，但仍需谨慎。
6. **变换交互：**
   - `jit`：有效，但会产生上述性能开销。
   - `vmap`：行为可能复杂。默认情况下，宿主函数接收*整个批次*的数据。您可能需要在宿主回调中添加 `vmap` 特定的逻辑或采用其他方法。将 `tap_with_transform=True` 与 `id_tap` 结合使用可以帮助查看 `vmap` 如何影响被“轻触”的参数 (parameter)。
   - `pmap`：回调在与每个 JAX 设备进程关联的宿主 Python 进程上执行。在单机多设备设置中，这可能意味着回调在*同一个*宿主上运行多次。在多宿主设置（如 TPU Pods）中，它在每个参与的宿主上运行。管理副作用（如写入同一文件）需要仔细协调。

### 为什么是 `jax.experimental`？

`experimental` 状态表明该 API 可能会发生变化，并且其用例有些小众或存在问题。它表明您正在放弃一些标准的 JAX 保证（例如端到端 XLA 优化和易于微分性）。

### 何时使用 `host_callback`？

考虑到这些缺点，`host_callback` 通常应被视为最后的手段，或用于特定的、对性能不敏感的任务：

- **调试：** 即使有性能开销，暂时在 `jit` 或其他变换中打印中间值也是非常有用的。`id_print` 或 `id_tap` 在此情况下有用。
- **日志记录：** 在训练期间将指标或摘要发送到外部日志系统。
- **遗留代码集成：** 当在 JAX 中重写现有 Python/NumPy/SciPy 函数不可行且性能影响可接受时，调用它们。
- **硬件交互：** 通过标准 Python 库与专用硬件进行接口。

在诉诸 `host_callback` 之前，请考虑替代方案：

- 是否可以使用 `jax.numpy` 和 `jax.lax` 直接在 JAX 中重新实现该逻辑？
- 对于无副作用的外部 Python 调用，`jax.pure_callback`（接下来介绍）是否合适？它提供了一个稍微更简洁的接口，但仍然有性能开销。
- 对于集成高性能 C++ 或 CUDA 代码，定义自定义 JAX 原语（本章稍后介绍）是最有效和高效的解决方案，尽管它涉及更多工作。

总而言之，`jax.experimental.host_callback` 提供了一个桥梁，以便在 JAX 计算中执行宿主端的 Python 代码。尽管它对调试和特定的集成情况有用，但其显著的性能开销以及与 JAX 变换的复杂交互意味着应谨慎使用它，并清楚了解其影响。

## 参考资料

- [jax.experimental.host_callback](https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html) — JAX developers (2024)
  Publisher: Google
  `host_callback`模块的官方文档，详细介绍了其API和用法。
- [JAX Quickstart](https://jax.readthedocs.io/en/latest/jax-101/quickstart.html) — JAX developers (2024)
  Publisher: Google LLC
  JAX核心功能（如JIT编译、自动微分和XLA集成）的通用介绍，为`host_callback`与JAX系统的交互提供背景。
- [XLA: Accelerated Linear Algebra](https://www.tensorflow.org/xla) — Google Brain Team (2024)
  XLA的描述，这是JAX使用的领域特定编译器，有助于理解使用`host_callback`跳出其编译边界的性能影响。

---

[上一节](02-%E4%BD%BF%E7%94%A8%20DLPack%20%E5%AE%9E%E7%8E%B0%E9%9B%B6%E6%8B%B7%E8%B4%9D%E6%95%B0%E6%8D%AE%E5%85%B1%E4%BA%AB.md) · [下一节](04-%E4%BD%BF%E7%94%A8%20jax.pure_callback%20%E8%BF%9B%E8%A1%8C%E6%97%A0%E5%89%AF%E4%BD%9C%E7%94%A8%E8%B0%83%E7%94%A8.md)
