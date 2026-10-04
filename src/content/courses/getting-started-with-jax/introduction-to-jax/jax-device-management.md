---
course: "getting-started-with-jax"
chapter: "introduction-to-jax"
lesson: "jax-device-management"
sourceId: 1687
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-jax/chapter-1-introduction-to-jax/jax-device-management"
title: "设备管理：CPU、GPU、TPU"
description: "了解JAX如何与不同硬件加速器交互。"
order: 6
plots: []
sourceHash: "71ea9f9f251dbc4c6ba5b1be6ca5c1e2926299eb0318769e887c9b9cd3080890"
sourceCorrections: []
---

JAX 的一个主要优点是它能够在不同类型的硬件加速器（如图形处理器GPU和张量处理器TPU）以及传统中央处理器CPU上运行相同的代码。了解 JAX 如何管理这些设备对于编写高效代码很重要。

JAX 使用一个名为 XLA（加速线性代数）的加速器后端来编译并运行你类似 NumPy 的代码。这意味着你通常使用 `jax.numpy` 和 JAX 变换编写一次代码，JAX 会处理在可用硬件上的执行，无需你编写设备特定代码。

### 默认设备选择

默认情况下，JAX 会尝试使用它在你的系统上检测到的功能最强的硬件。通常的优先顺序是 TPU > GPU > CPU。如果你有可用的 TPU 并已配置，JAX 将使用它。如果没有，它会寻找兼容的 GPU。如果未找到或未正确配置任何加速器，JAX 会退回使用 CPU。

这种自动选择简化了开始使用流程。你通常可以在标准 CPU 设置上编写和测试代码，然后将完全相同的代码运行在配备 GPU 或 TPU 的机器上，以获得大幅性能提升，尤其是在使用 `jax.jit` 等变换时。

### 识别可用设备

你可以使用 `jax.devices()` 函数查看 JAX 识别的设备。此函数返回当前 JAX 进程可用的设备对象列表。

```python
import jax

# 列出JAX可见的所有设备
available_devices = jax.devices()
print(f"可用设备: {available_devices}")

# 获取JAX将使用的默认设备
default_device = jax.default_backend()
print(f"默认后端: {default_device}")
```

输出结果会因你的硬件和 JAX 安装而异：

- **仅CPU：**

  ```
  Available devices: [CpuDevice(id=0)]
  Default backend: cpu
  ```
- **CPU + GPU：**（输出可能因GPU型号和驱动而异）

  ```
  Available devices: [cuda(id=0)] # 或有时是 [GpuDevice(id=0)] 或类似
  Default backend: gpu
  ```
- **CPU + TPU：**（输出可能因TPU类型和连接而异）

  ```
  Available devices: [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0), TpuDevice(id=1, process_index=0, coords=(0,0,0), core_on_chip=1), ...] # TPU通常显示为多个设备
  Default backend: tpu
  ```

知道哪些设备可用是管理计算放置的第一步。

### 数据放置与传输

JAX 数组与始终驻留在 CPU 内存（RAM）中的标准 NumPy 数组不同，它们存在于特定的计算设备（CPU、GPU 或 TPU）上。当你创建一个 JAX 数组时，JAX 通常会将其放置在默认设备上。

```python
import jax
import jax.numpy as jnp

# x通常会在默认设备上创建（例如，如果GPU可用）
x = jnp.arange(10.0)
print(f"数组 x 位于设备: {x.device()}")
```

涉及同一设备上数组的操作通常高效。然而，涉及*不同*设备上数组的计算（例如，将一个 CPU 数组添加到 GPU 数组）可能需要隐式数据传输，这会引入性能开销。JAX 会自动处理这些传输，但注意数据局部性对优化有好处。

你可以使用 `jax.device_put()` 显式控制设备放置。此函数接受一个 NumPy 数组或一个 JAX 数组，并返回一个放置在指定设备上的新 JAX 数组。

```python
import jax
import jax.numpy as jnp
import numpy as np

# 创建一个NumPy数组（位于主机CPU内存中）
numpy_array = np.array([1.0, 2.0, 3.0])

# 获取可用设备列表
devices = jax.devices()

if devices:
    # 将数组放置在第一个可用的JAX设备上
    jax_array_on_device0 = jax.device_put(numpy_array, devices[0])
    print(f"数组放置在: {jax_array_on_device0.device()}")

    # 如果有多个设备可用（例如，多个GPU或TPU核心）
    if len(devices) > 1:
        # 尝试放置在不同的设备上
        jax_array_on_device1 = jax.device_put(numpy_array, devices[1])
        print(f"数组放置在: {jax_array_on_device1.device()}")
    else:
        # 如果是唯一设备，则显式放置在CPU上
        cpu_device = jax.devices('cpu')[0]
        jax_array_on_cpu = jax.device_put(numpy_array, cpu_device)
        print(f"数组显式放置在: {jax_array_on_cpu.device()}")

else:
    print("未找到JAX设备。")

# 直接创建JAX数组通常会将其放置在默认设备上
default_device_array = jnp.ones(5)
print(f"默认数组位于: {default_device_array.device()}")
```

虽然显式放置是可行的，但在典型工作流程中，其必要性通常低于理解设备放置的*影响*。例如，在使用 `jax.jit` 时，JIT 编译过程会为将运行计算的特定设备优化函数。输入数组可能会在编译函数执行前自动移动到目标设备。

### 主机与设备

区分“主机”（通常是控制 Python 进程的 CPU）和“设备”（主要进行计算的加速器，如 GPU 或 TPU）很有帮助。

- **主机内存（RAM）：** 标准 Python 对象和 NumPy 数组存储在此处。
- **设备内存（例如，GPU显存 (VRAM)）：** 用于加速计算的 JAX 数组存储在此处。

在主机内存和设备内存之间传输数据需要时间。为了获得最佳性能，尤其是在训练机器学习 (machine learning)模型等迭代算法中，目标是：

1. 首次将初始数据移动到设备（例如，使用 `jax.device_put`）。
2. 使用 JAX 操作和变换（`jit`、`vmap`、`grad`）尽可能多地直接在设备上执行计算。
3. 仅在必要时将结果传回主机（例如，用于保存、日志记录或可视化）。

JAX 的抽象层处理了大部分，但记住主机与设备的区别有助于诊断性能瓶颈或理解内存使用情况。

后续关于 `pmap` 的章节将讨论如何同时管理*多个*设备上的计算，届时显式设备感知将变得更重要。目前，请理解 JAX 提供了一个简化在加速器上运行代码的层次，它自动选择设备并管理数据放置，同时提供 `jax.devices()` 和 `jax.device_put()` 等工具，以便在需要时进行查看和控制。

## 参考资料

- [JAX: Device and Shard Arrays](https://jax.readthedocs.io/en/latest/distributed_arrays_and_automatic_parallelization.html) — JAX documentation team (2024)
  Publisher: JAX (Google)
  解释JAX如何与各种硬件加速器（CPU、GPU、TPU）交互，详细说明设备选择、数据放置及相关API。
- [JAX 101: JIT compilation](https://jax.readthedocs.io/en/latest/jax-101/02_jitting.html) — JAX documentation team (2024)
  描述了JAX中的JIT编译，其性能优势，以及与XLA结合在加速器上执行代码的方式。
- [XLA: Accelerated Linear Algebra Overview](https://openxla.org/) — Google (2024)
  Publisher: Google
  介绍了XLA（JAX使用的编译器后端），阐明了其在优化和运行各种硬件上计算方面的作用。
