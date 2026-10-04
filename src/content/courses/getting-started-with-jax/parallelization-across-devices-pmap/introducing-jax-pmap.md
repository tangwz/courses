---
course: "getting-started-with-jax"
chapter: "parallelization-across-devices-pmap"
lesson: "introducing-jax-pmap"
sourceId: 1774
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-jax/chapter-5-parallelization-across-devices-pmap/introducing-jax-pmap"
title: "介绍 `jax.pmap`"
description: "`jax.pmap` 转换的基本用法，用于跨设备并行执行。"
order: 2
plots: []
sourceHash: "85b2e2d27de7e31efcb681bcddac9ebb2edae1dc40a551eee1777940af01cc4c"
sourceCorrections: []
---

`jax.jit` 为单个加速器上的高效运行编译代码，而 `jax.vmap` 提供自动向量 (vector)化功能。但是，将计算分配到*多个*设备（例如 GPU 或 TPU）则需要另一种工具：`jax.pmap`。

`pmap` 代表“并行映射”。它是一种函数转换，专门为单一程序、多重数据（SPMD）模型设计。在该模型中，您编写一次程序（您的函数），`pmap` 会安排其在多个设备上同时运行。每个设备接收并处理输入数据的不同部分。这是一种实现数据并行、通过分割工作负载来加速计算的常见且有效的方法。

### 基本使用方法

本质上，`pmap` 的使用方式在语法上与 `jit` 或 `vmap` 相似。您可以将其作为装饰器应用，或者直接在函数上调用。让我们从一个简单例子开始。

首先，让我们查看 JAX 能检测到哪些设备。这很重要，因为 `pmap` 需要多个设备来分配工作。

```python
import jax
import jax.numpy as jnp

# 检查可用设备
print(f"Available JAX devices: {jax.devices()}")
num_devices = len(jax.devices())
print(f"Number of devices: {num_devices}")
```

如果您在多 GPU 机器或 TPU pod 上运行此代码，`jax.devices()` 将列出这些设备。如果您只有一个 CPU 或单个 GPU，`pmap` 仍然有效，但它会在该单个设备上运行所有计算，实质上模仿了 `vmap`（尽管对此目的而言效率较低）。当 `num_devices > 1` 时，真正的优势才会显现。

现在，让我们定义一个简单函数并应用 `pmap`：

```python
# 一个应用 pmap 的简单函数
def simple_computation(x):
  return x * 2

# 应用 pmap
pmapped_computation = jax.pmap(simple_computation)

# 准备输入数据：需要一个与设备数量匹配的前导轴
try:
    # 创建跨设备分片的数据
    # 示例：如果设备数量为 4，则创建形状为 (4, ...) 的数组
    input_data = jnp.arange(num_devices * 3).reshape((num_devices, 3))
    print(f"Input data shape: {input_data.shape}")

    # 执行 pmap 化的函数
    result = pmapped_computation(input_data)

    print(f"Output result:\n{result}")
    print(f"Output type: {type(result)}")
    print(f"Output shape: {result.shape}")
    print(f"Output devices: {result.devices()}")

except Exception as e:
    print(f"Error during pmap execution: {e}")
    print("Note: pmap typically requires the size of the mapped axis")
    print("to be equal to the number of available devices.")
    print("If running with only 1 device, this example might not show parallelism.")
```

您会在这里看到几件重要的事情：

1. **输入数据分片：** 输入数据 `input_data` 需要一个前导维度，其大小等于 `num_devices`。这里，我们创建了一个形状为 `(num_devices, 3)` 的数组。`pmap` 自动沿第一个轴（默认是轴 0）分割此数组，并将每个切片（在本例中形状为 `(3,)`）发送到不同的设备。
2. **执行：** 每个设备并发地在其自己的数据切片上执行 `simple_computation`。
3. **输出：** 输出 `result` 通常是 `ShardedDeviceArray`（或类似的分布式数组类型）。这表明结果数据也物理上分布在设备上。它的形状与输入形状 `(num_devices, 3)` 相同，您可以使用 `result.devices()` 确认其分布。

### `pmap` 的工作方式

可以把 `pmap` 看作在做以下事情：

1. **编译：** 像 `jit` 一样，`pmap` 首先将您的 Python 函数（使用 XLA）编译成一个优化过的、适用于目标硬件（GPU/TPU）的可执行程序。
2. **复制：** 它将此编译好的程序复制到每个可用设备上。
3. **数据分配：** 当您调用 `pmap` 化的函数时，JAX 获取输入数组并沿指定轴（默认为轴 0）分割它们。每个数据块被发送到一个设备。
4. **并行执行：** 所有设备同时执行相同的编译程序，但每个设备都在其独特的数据块上操作（SPMD）。
5. **结果收集：** 从每个设备获得的结果被收集起来。默认情况下，它们保持分布在设备上，形成一个 `ShardedDeviceArray`。

这是一个简化的视觉表示：

> 一张图显示了 `pmap` 如何沿第一个轴分割输入数据，将每个切片发送到不同设备以并行执行相同的程序，并将结果收集到分布式数组中。

### `pmap` 与 `vmap` 对比

将 `pmap` 与 `vmap` 进行对比很有帮助：

- **`vmap`（向量 (vector)化）：** 将处理单个示例的函数转换为在单个计算图内处理批次示例的函数。它通过在*单个*设备上的向量化 (quantization)指令来实现并行。它本身不将工作分配到多个物理设备上。
- **`pmap`（并行化）：** 明确地将计算复制到*多个*物理设备上。它在不同加速器上并行执行*相同*的函数，但处理不同的数据切片。

尽管两者都可以处理批次数据，但 `pmap` 是通过使用单个机器上的多个可用加速器来扩展计算的工具。

在接下来的部分中，我们将查看如何控制映射哪些轴，如何处理多个参数 (parameter)，以及如何在 `pmap` 化的函数中使用集合操作进行设备间通信。

## 参考资料

- [jax.pmap](https://jax.readthedocs.io/en/latest/_autosummary/jax.pmap.html) — JAX developers (2024)
  提供 `jax.pmap` 详细 API 参考和使用示例的官方文档。
- [Distributed Arrays and the JAX Device Mesh](https://jax.readthedocs.io/en/latest/jep/9407-jax-distributed-arrays.html) — JAX developers (2024)
  解释 JAX 分布式数组设计的文档，这是 `pmap` 和分布式计算中数据处理的基础。
