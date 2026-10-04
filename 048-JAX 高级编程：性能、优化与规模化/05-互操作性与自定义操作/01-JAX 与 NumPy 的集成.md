# JAX 与 NumPy 的集成

来源：[原文](https://apxml.com/zh/courses/advanced-jax/chapter-5-jax-interoperability-custom-operations/integrating-jax-numpy)

[返回章节目录](README.md) · [返回课程目录](../README.md)

将 JAX 计算与现有工具和库结合是常见需求。鉴于 NumPy 在科学 Python 生态系统中的核心作用，理解 JAX 数组和 NumPy 数组之间如何高效地交互非常重要。JAX 被有意设计为具有类似 NumPy 的 API（`jax.numpy`），这简化了此过程，但有一些重要的区别和性能考量需要记住，特别是在处理硬件加速器时。

### 理解 JAX 数组与 NumPy 数组的区别

本质上，一个标准的 `numpy.ndarray` 存在于宿主主内存（CPU RAM）中。对 NumPy 数组的操作由 CPU 执行。

相反，`jax.Array` 代表由 JAX 管理的数据。尽管它*可以*存在于 CPU 上，但其主要优势在于它能够存在于 GPU 或 TPU 等加速器上。此外，`jax.Array` 对象是 JAX 变换（`jit`、`grad`、`vmap` 等）和编译计算的操作数。

这种潜在驻留位置（CPU 与加速器）和预期用途（标准计算与变换/编译计算）上的不同是为何需要仔细管理转换的主要原因。

### 将 NumPy 数组转换为 JAX 数组

你通常会从使用 NumPy 加载或生成的数据开始（例如，加载数据集、初始参数 (parameter)），并需要将其移入 JAX 生态系统，可能移至加速器上。

最直接的方法是使用 `jax.numpy.array()`：

```python
import numpy as np
import jax
import jax.numpy as jnp

# 在宿主 CPU 上创建一个 NumPy 数组
numpy_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
print(f"Original NumPy array type: {type(numpy_arr)}")
print(f"Original NumPy array device: (Implicitly Host CPU)")

# 转换为 JAX 数组
jax_arr = jnp.array(numpy_arr)
print(f"Converted JAX array type: {type(jax_arr)}")

# 检查 JAX 数组的设备（将是 JAX 的默认后端）
print(f"Converted JAX array device: {jax_arr.device()}")
```

当你调用 `jnp.array(numpy_arr)` 时，JAX 会从 NumPy 数组（在宿主内存中）获取数据，并可能将其复制到默认 JAX 设备（如果可用，可能是 GPU 或 TPU，否则是 CPU 后端）。

为了更明确地控制 JAX 数组应放置在哪个设备上，请使用 `jax.device_put()`：

```python
# 假设你有多个设备，例如 GPU
available_devices = jax.devices()
print(f"Available JAX devices: {available_devices}")

if len(available_devices) > 1:
    target_device = available_devices[1] # 示例：放置在第二个设备上
else:
    target_device = available_devices[0] # 回退到第一个/唯一设备

# 明确地将 NumPy 数据放置到目标 JAX 设备上
jax_arr_explicit = jax.device_put(numpy_arr, device=target_device)
print(f"Explicitly placed JAX array device: {jax_arr_explicit.device()}")
```

`jax.device_put` 在管理分布式计算或确保特定加速器硬件的数据局部性时很有用。

### 将 JAX 数组转换为 NumPy 数组

反之，你可能需要将 JAX 计算的结果带回宿主 CPU，以便保存、使用 Matplotlib 等库绘图，或使用标准 Python/NumPy 工具进一步处理。

做到这一点的一种标准方法是使用 `numpy.array()` 构造函数或函数：

```python
# 假设 jax_arr 是 JAX 计算的结果
# 例如:
key = jax.random.PRNGKey(0)
jax_arr = jax.random.normal(key, (3,)) * 2.0 + 1.0
print(f"JAX array: {jax_arr}")
print(f"JAX array device: {jax_arr.device()}")

# 将 JAX 数组转换回 NumPy 数组
numpy_result = np.array(jax_arr)

print(f"Converted NumPy array type: {type(numpy_result)}")
print(f"Converted NumPy array value: {numpy_result}")
# 对于 NumPy 数组，设备隐含为宿主 CPU
```

**重要的性能考量：** 将 `jax.Array`（可能位于 GPU/TPU 上）转换为 `numpy.ndarray` 需要 JAX 执行以下操作：

1. 确保生成 `jax_arr` 的计算已在设备上完成。
2. 将数据从设备内存（GPU/TPU）传输回宿主 CPU 内存。

因为 JAX 操作默认异步执行（见第 2 章，“异步调度”），这种转换充当一个同步点。调用 `np.array(jax_arr)` 的 Python 代码将阻塞，直到数据在宿主上可用。因此，在对性能要求高的循环中，频繁地从 JAX 转换回 NumPy 会通过停滞 Python 解释器并强制进行不必要的设备到宿主数据传输和同步来严重降低性能。

### NumPy 互操作性的优良做法

1. **在内部循环中尽量减少转换：** 避免在频繁执行的代码路径中在 JAX 和 NumPy 数组之间进行转换，特别是在使用 `@jax.jit` 装饰的函数中。每次转换都涉及开销（潜在的数据传输、同步）。
2. **在边界处转换：** 主要在计算工作流的开始和结束时执行转换。使用 NumPy 加载数据，使用 `jnp.array` 或 `jax.device_put` 将其转换为 JAX 数组，运行你的核心 JAX 计算，并且仅在需要保存、可视化或与非 JAX 库交互时，才使用 `np.array()` 将最终结果转换回 NumPy。
3. **为明确性使用 `jax.device_put`：** 当将数据移入 JAX 时，特别是在多设备场景中，使用 `jax.device_put` 可以使你的设备放置意图更加明确。
4. **注意同步：** 请记住，`np.array(jax_array)` 会隐式调用 `jax_array.block_until_ready()`。如果你只需要触发同步而不需要数据传输（例如，为了精确计时），请直接在 JAX 数组上使用 `.block_until_ready()`。
5. **为保持一致性使用 `jax.numpy`：** 在你的 JAX 代码中，尽可能优先使用 `jax.numpy` 函数而不是 `numpy` 函数。`jnp` 函数操作 `jax.Array` 对象，并与 JAX 变换和编译集成。将 NumPy 数组直接传递给 `jnp` 函数通常会因为隐式转换而起作用，但依赖这种做法有时会掩盖性能成本。

通过理解 JAX 和 NumPy 数组的特性以及它们之间转换的含义，你可以确保 JAX 与更广泛的科学 Python 生态系统之间的数据高效流动。将转换，特别是 JAX 到 NumPy 的转换，视为可能开销大的操作，并将其战略性地放置在你的程序架构中。

## 参考资料

- [JAX NumPy](https://jax.readthedocs.io/en/latest/jax.numpy.html) — JAX developers (current)
  解释了 jax.numpy API、其与 NumPy 的相似性以及底层 jax.Array 类型。
- [Device and memory considerations](https://jax.readthedocs.io/en/latest/device_and_memory_considerations.html) — JAX developers (current)
  详细说明了 jax.Array 设备放置、jax.device_put、异步调度和同步，与性能考量直接相关。
- [JAX: Composable transformations of Python+NumPy programs](https://github.com/google/jax) — James Bradbury, Roy Frostig, Peter Hawkins, Matthew J. Johnson, Chris Leary, Skye Wanderman-Milne, and Adam Paszke (2018)
  描述 JAX 架构的奠基性技术报告，包括其类 NumPy API 和加速器集成机制。

---

[上一节](../04-%E8%BF%9B%E9%98%B6%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86%E6%96%B9%E6%B3%95/10-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%E8%87%AA%E5%AE%9A%E4%B9%89%E6%A2%AF%E5%BA%A6.md) · [下一节](02-%E4%BD%BF%E7%94%A8%20DLPack%20%E5%AE%9E%E7%8E%B0%E9%9B%B6%E6%8B%B7%E8%B4%9D%E6%95%B0%E6%8D%AE%E5%85%B1%E4%BA%AB.md)
