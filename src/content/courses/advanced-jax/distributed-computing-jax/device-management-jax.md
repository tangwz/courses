---
course: "advanced-jax"
chapter: "distributed-computing-jax"
lesson: "device-management-jax"
sourceId: 2840
sourceUrl: "https://apxml.com/zh/courses/advanced-jax/chapter-3-distributed-computing-jax/device-management-jax"
title: "JAX 中的设备管理"
description: "查询可用设备（CPU、GPU、TPU）并明确放置计算。"
order: 2
plots: []
sourceHash: "d8a344af73c052573d3f6bd11e750db4d7acf4c2687c0081ba0dfe9c4f9598e7"
sourceCorrections: []
---

在使用 `pmap` 等原语将计算分布到多个设备之前，了解 JAX 如何识别和管理可用的硬件加速器是很重要的。JAX 会自动检测连接到您系统（或在分布式环境中分配）的 CPU、GPU 和 TPU，并提供工具来查看和操作它们。

### 查看可用设备

JAX 提供一些函数来查询它可以访问的计算资源。最基本的是 `jax.devices()`。它返回 JAX 可以在所有参与主机（在多主机设置中）或仅在本地主机（如果独立运行）上使用的所有设备列表。

```python
import jax

# 列出 JAX 可以全局看到的所有设备（在单主机中可能与 local_devices 相同）
all_devices = jax.devices()
print(f"All available devices: {all_devices}")

# 获取全局设备总数
num_devices = jax.device_count()
print(f"Total number of devices: {num_devices}")

# 仅列出当前进程/主机上的本地设备
local_devices_list = jax.local_devices()
print(f"Local devices: {local_devices_list}")

# 获取本地设备数量
num_local_devices = jax.local_device_count()
print(f"Number of local devices: {num_local_devices}")

# 示例输出（可能因硬件而异）
# All available devices: [CpuDevice(id=0)] # 如果只有 CPU 可用
# All available devices: [TpuDevice(id=0), TpuDevice(id=1), ...] # 在 TPU Pod 切片上
# All available devices: [GpuDevice(id=0), GpuDevice(id=1)] # 在双 GPU 机器上

# Total number of devices: 1
# Local devices: [CpuDevice(id=0)]
# Number of local devices: 1
```

`jax.devices()` 或 `jax.local_devices()` 返回列表中每个元素都是一个 `Device` 对象。这些对象包含特定硬件的信息，例如其平台（`'cpu'`、`'gpu'`、`'tpu'`）、在该平台内的唯一 ID，以及在多主机情况下可能包含进程索引等其他属性。

```python
if local_devices_list:
    device = local_devices_list[0]
    print(f"First local device: ID={device.id}, Platform={device.platform}")
    # 示例输出：First local device: ID=0, Platform=gpu
```

区分 `jax.devices()` 和 `jax.local_devices()` 的差别在多主机 TPU 环境中特别重要。`jax.devices()` 提供所有连接到 TPU Pod 切片的主机上的全局视图，而 `jax.local_devices()` 仅显示直接连接到当前 Python 进程的设备。对于单主机 GPU 或 CPU 设置，这两个函数通常返回相同的列表。

### 默认设备放置

默认情况下，JAX 操作和数组创建以 `jax.local_devices()` 列出的第一个设备为目标，这通常是 `cpu:0`、`gpu:0` 或 `tpu:0`，具体取决于您的设置和后端配置。

```python
import jax.numpy as jnp

# 在默认设备上创建数组（通常是设备 0）
x = jnp.ones((3, 3))
print(f"Default device for x: {x.device()}")

# 示例输出：Default device for x: GpuDevice(id=0)
```

这种默认行为对于单设备工作流很方便，但对于分布式计算，您通常需要更明确的控制。

### 使用 `jax.device_put` 进行明确设备放置

`jax.device_put()` 函数允许您明确地将 NumPy 数组或 Python 标量放置到特定的 JAX 设备上，返回一个位于该设备上的 `DeviceArray`（JAX 的数组类型）。

其签名是 `jax.device_put(x, device=None)`。

- `x`: 要放置的数据（例如，NumPy 数组、Python 标量/列表）。
- `device`: 目标 `Device` 对象（从 `jax.devices()` 或 `jax.local_devices()` 获取）。如果为 `None`，则使用默认设备。

```python
import numpy as np

if num_local_devices > 1:
    # 在主机 CPU 上创建 NumPy 数组
    host_array = np.random.rand(2, 2)
    print(f"NumPy array type: {type(host_array)}")

    # 明确地将其放置到第二个可用的 JAX 设备上
    target_device = jax.local_devices()[1]
    device_array = jax.device_put(host_array, device=target_device)

    print(f"Placed array type: {type(device_array)}")
    print(f"Array is now on device: {device_array.device()}")

    # 示例输出（在多 GPU 系统上）：
    # NumPy array type: <class 'numpy.ndarray'>
    # Placed array type: <class 'jaxlib.xla_extension.DeviceArray'>
    # Array is now on device: GpuDevice(id=1)

elif num_local_devices == 1:
    print("只有一个本地设备可用，跳过明确放置示例。")

else:
    print("未找到 JAX 设备。")
```

`jax.device_put()` 之所以重要，是因为：

1. **数据传输：** 它是将数据从主机（CPU 内存，NumPy 数组所在的位置）移动到加速器设备（GPU/TPU 内存）的主要方式。
2. **为 `pmap` 做准备：** 使用 `pmap` 时，您通常在主机上准备数据分片，然后使用 `jax.device_put`（或依赖 `pmap` 的隐式放置）来确保数据在并行计算开始前位于正确的设备上。尽管 `pmap` 可以隐式处理输入参数 (parameter)的分布，但明确放置初始模型参数或状态有时对于清晰度或特定初始化模式来说是必要的。
3. **控制计算位置：** 尽管 `jit` 编译通常决定操作运行的位置，但 `jax.device_put` 可以影响初始数据的位置，这对于性能很重要，可以避免后续不必要的传输。

### 设备数据驻留

一旦数组被放置到设备上（无论是通过 `jax.device_put` 明确放置还是隐式放置），JAX 会尝试将涉及该数组的计算保留在同一设备上，以减少数据传输。涉及不同设备上数组的操作可能会触发数据移动或导致错误，如果该操作未跨设备定义。

### 设备可见性环境变量

您可以在导入 JAX *之前*使用环境变量来影响 JAX 可以看到的设备，特别是 GPU。最常用的是 `CUDA_VISIBLE_DEVICES`。

```bash
# 示例：使 JAX（和其他 CUDA 应用程序）仅能看到 GPU 1
export CUDA_VISIBLE_DEVICES=1
python my_jax_script.py
```

在 `my_jax_script.py` 中，`jax.local_devices()` 可能只会列出 `GpuDevice(id=0)`（因为 JAX 会从 0 开始重新编号可见设备），这实际对应于物理 GPU 1。设置 `CUDA_VISIBLE_DEVICES=""` 通常会隐藏所有 GPU，强制 JAX 使用 CPU。

JAX 也遵守 `JAX_PLATFORMS` 环境变量。如果设置了它，JAX 将仅初始化指定的平台后端（例如，`JAX_PLATFORMS=cpu` 或 `JAX_PLATFORMS=gpu`）。当有多个后端可用时，这有助于强制使用特定后端。

掌握设备管理是分布式计算的根本。了解哪些设备可用、数据位于何处（`x.device()`）以及如何控制放置（`jax.device_put`）是有效使用 `pmap` 并扩展您的计算的前提条件。在接下来的章节中，我们将在此根本上实现并行执行模式。

## 参考资料

- [JAX Documentation: Working with multiple devices](https://jax.readthedocs.io/en/latest/multi_controller_intro.html) — The JAX Authors (2024)
  Publisher: The JAX Authors
  JAX多设备功能的官方指南，涵盖设备发现、放置以及跨各种硬件加速器的数据处理。
- [XLA: Optimizing Compilers for Machine Learning](https://www.tensorflow.org/xla) — The XLA Authors (2024)
  Publisher: Google
  描述了XLA，这是JAX使用的领域特定编译器，用于为CPU、GPU和TPU生成高效机器代码，是JAX设备管理的基础。
