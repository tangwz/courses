# 使用 pmap 进行单一程序多数据 (SPMD) 运算

来源：[原文](https://apxml.com/zh/courses/advanced-jax/chapter-3-distributed-computing-jax/spmd-with-pmap)

[返回章节目录](README.md) · [返回课程目录](../README.md)

单一程序多数据 (SPMD) 模型是并行计算中一种普遍使用的方法，特别适合 GPU 和 TPU 等加速器。它的基本思路直接明了：你编写一个程序，该程序在多个处理器或设备上同时运行。然而，程序的每个实例操作的是总体数据的一个*不同子集*。这与其他模型（如多程序多数据 (MPMD)，其中不同程序可能在不同处理器上运行）形成对比。对于许多机器学习 (machine learning)任务，特别是数据并行，SPMD 是一个很恰当的选择。

在 JAX 中，在多个设备上实现 SPMD 风格并行运算的主要工具是 `jax.pmap`（并行映射）。`pmap` 将为单个设备编写的 Python 函数转换为可在多个设备（例如 JAX 进程可用的 GPU 或 TPU 核心）上并行执行的函数。它自动管理计算的复制和数据的分发（分片）。

可以把 `pmap` 看作类似于 Python 内置的 `map` 函数，但 `pmap` 不是按顺序将函数映射到列表元素上，而是并行地将函数映射到设备上。每个设备执行*相同*的编译函数，但接收输入数据的独特切片。

### `pmap` 如何实现 SPMD

让我们用 `pmap` 来展示 SPMD 的原理。假设你有 4 个 TPU 核心和一批要处理的数据。

1. **编写单设备代码：** 你首先编写 JAX 函数，就好像它在单个设备上运行一样。这个函数可以定义神经网络 (neural network)的一层、损失计算或任何其他计算。
2. **使用 `pmap` 转换：** 将 `jax.pmap` 应用于此函数。
3. **数据分片：** 你准备输入数据，使其主轴对应于设备数量。例如，如果你的数据数组形状为 `(128, 50)`（批大小 128，特征大小 50）且有 4 个设备，你通常会重塑或确保数据加载将其提供为 `(4, 32, 50)`。主维度（大小 4）表示设备轴。
4. **执行：** 当你使用此分片数据调用经 `pmap` 转换的函数时，JAX 会执行以下操作：
   - 它使用 XLA 编译原始函数（如果尚未通过 `jit` 编译）。
   - 它将编译后的 XLA 计算复制到所有 4 个指定设备上。
   - 它将输入数据的对应切片发送到每个设备（设备 0 获取数据切片 0，设备 1 获取数据切片 1，依此类推）。
   - 所有设备在其本地数据切片上同时执行计算。
   - 每个设备的结果被收集并沿着输出中的新主轴堆叠。

下图展示了这一过程：

> 每个设备执行的编译代码一致，但操作的是其被分配的输入数据切片。输出结果通常会被收集回来，并沿新的设备轴堆叠。

### 基本用法示例

让我们看一个实例。我们将定义一个简单函数并使用 `pmap` 在多个设备上应用它。首先，请确认 JAX 可以识别你可用的设备。

```python
import jax
import jax.numpy as jnp

# 检查可用设备（CPU、GPU 或 TPU 核心）
num_devices = jax.local_device_count()
print(f"可用设备数量: {num_devices}")

# 示例：如果可用，使用 4 个设备，否则使用实际数量
if num_devices >= 4:
  num_devices_to_use = 4
else:
  num_devices_to_use = num_devices

print(f"pmap 将使用 {num_devices_to_use} 个设备。")

# 创建一些示例数据，在设备维度上进行分片
# 总批处理大小 = 设备数量 * 每个设备的批处理大小
per_device_batch_size = 8
feature_size = 16
global_batch_size = num_devices_to_use * per_device_batch_size

# 形状: (设备数量, 每个设备的批处理大小, 特征大小)
sharded_data = jnp.arange(global_batch_size * feature_size).reshape(
    (num_devices_to_use, per_device_batch_size, feature_size)
)

print(f"分片输入数据形状: {sharded_data.shape}")

# 定义一个简单的函数，用于每个设备的运算
def simple_computation(x):
  # 示例：缩放并加上一个常量
  return x * 2.0 + 1.0

# 对函数应用 pmap
# 默认情况下，pmap 假设输入的第一个轴（轴 0）
# 应该映射到设备上。
parallel_computation = jax.pmap(simple_computation)

# 执行并行计算
# JAX 将 sharded_data 的主轴分布到设备上
result = parallel_computation(sharded_data)

# 输出也沿主轴分片
print(f"输出形状: {result.shape}")

# 验证一个设备输出的值（例如，设备 0 上的第一个元素）
# 原始值为 0。计算结果是 0 * 2.0 + 1.0 = 1.0
print(f"Result[0, 0, 0]: {result[0, 0, 0]}")
```

在这个例子中：

1. 我们确定 JAX 可以使用的设备数量。
2. 我们生成输入数据 `sharded_data`，其中第一个维度与我们计划使用的设备数量相符。每个切片 `sharded_data[i]` 将发送到设备 `i`。
3. 我们定义 `simple_computation`，它处理单个数据切片。
4. `jax.pmap(simple_computation)` 生成 `parallel_computation`，这是一个可用于 SPMD 执行的新函数。
5. 调用 `parallel_computation(sharded_data)` 启动并行执行。每个设备在其对应的数据切片 `sharded_data[i]` 上运行 `simple_computation`。
6. 输出 `result` 与输入形状一致，主轴表示设备。`result[i]` 包含设备 `i` 计算出的结果。

这展示了 `pmap` 搭配 SPMD 的主要特点：定义每个设备的逻辑，让 `pmap` 通过映射输入数组的主轴来处理设备间的复制和并行执行。底层的 XLA 编译使得核心计算针对目标硬件进行了优化。在接下来的章节中，我们将讨论如何使用 `in_axes` 管理复制的数据（如模型参数 (parameter)），以及如何通过集体操作来实现设备间的数据交换。

## 参考资料

- [pmap and distributed JAX](https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html) — Vladimir Mikulik, Roman Ring (2024)
  Publisher: JAX project (Google)
  官方 JAX 文档，解释了 `jax.pmap` 在并行执行和数据分片中的应用。

---

[上一节](02-JAX%20%E4%B8%AD%E7%9A%84%E8%AE%BE%E5%A4%87%E7%AE%A1%E7%90%86.md) · [下一节](04-%E4%BD%BF%E7%94%A8%20pmap%20%E5%AE%9E%E7%8E%B0%E6%95%B0%E6%8D%AE%E5%B9%B6%E8%A1%8C.md)
