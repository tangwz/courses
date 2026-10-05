# 在 CPU、GPU 和 TPU 上分析 JAX 代码

来源：[原文](https://apxml.com/zh/courses/advanced-jax/chapter-2-optimizing-jax-code-performance/profiling-jax-code)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然 `jax.jit` 为加速代码提供了一个有力的起点，但要达到最佳性能，需要一种更具分析性的方法。由于架构差异、内存带宽限制以及 XLA 编译器执行的特定硬件优化，在 CPU 上运行速度快的代码在 GPU 或 TPU 上表现可能不同。正因如此，性能分析变得必不可少。性能分析让您可以了解 JAX 执行的内部工作方式，精确定位时间花费的位置，并发现针对目标硬件的优化机会。

有效性能分析有助于回答以下重要问题：

- 哪些特定操作消耗了最多的执行时间？
- 程序受限于计算（CPU/GPU/TPU 限制）还是数据传输（内存限制）？
- 加速器是否得到了有效利用，或者存在大量的空闲时间？
- 是否存在由宿主设备数据传输引起的意外延迟？
- 程序是否遭受过多的重新编译开销？

通过理解这些方面，您可以避免猜测，并应用有针对性的优化。

### 使用 `jax.profiler` 收集轨迹

JAX 包含一个内置分析器 `jax.profiler`，用于捕获详细的执行轨迹，可与 TensorBoard 等标准可视化工具兼容。它允许您记录在宿主（CPU）和加速器设备（GPU/TPU）上执行的操作的顺序和持续时间。

主要函数是 `jax.profiler.start_trace` 和 `jax.profiler.stop_trace`。您将要分析的代码段包裹在这些调用中。

```python
import jax
import jax.numpy as jnp
import jax.profiler
import time

# 确保 JAX 使用所需的后端（例如 GPU）
# jax.config.update('jax_platform_name', 'gpu')

@jax.jit
def complex_computation(x, y):
  # 一系列 JAX 操作
  z = jnp.dot(x, y)
  z = jnp.sin(z)
  z = jnp.mean(z * x + y)
  return z

# 准备一些数据
key = jax.random.PRNGKey(0)
size = 2000
x = jax.random.normal(key, (size, size))
y = jax.random.normal(key, (size, size))

# 确保数据在分析前已在设备上
x = jax.device_put(x)
y = jax.device_put(y)
_ = complex_computation(x, y).block_until_ready() # 预热编译

# 开始分析
log_dir = "/tmp/jax_profiling"
jax.profiler.start_trace(log_dir)

# 多次运行计算以获得代表性轨迹
for _ in range(5):
    result = complex_computation(x, y)
    # 重要：阻塞以确保计算在停止分析前完成
    result.block_until_ready()

# 停止分析
jax.profiler.stop_trace()

print(f"Profiling trace saved to: {log_dir}")
# 您现在可以使用 TensorBoard 查看此轨迹：
# tensorboard --logdir /tmp/jax_profiling
```

此代码片段展示了基本流程：

1. 导入必要的库。
2. 定义要分析的 JIT 编译函数。
3. 准备输入数据，并使用 `jax.device_put` 明确放置到目标设备上。
4. 执行一次预热运行，以便在启动分析器*之前*触发编译。
5. 使用 `jax.profiler.start_trace(log_dir)` 开始记录。`log_dir` 指定轨迹文件将保存的位置。
6. 执行感兴趣的函数。在分析上下文 (context)中多次运行代码通常很有用。
7. 请务必在循环内的计算之后（或者如果分析一个序列，则在整个循环之后）使用 `.block_until_ready()`。JAX 的异步调度意味着 Python 函数可能在加速器完成之前就返回。阻塞操作确保分析器捕获完整的执行。
8. 使用 `jax.profiler.stop_trace()` 来完成并保存轨迹文件。

### 使用 TensorBoard 可视化轨迹

`jax.profiler` 生成的轨迹文件旨在用于 TensorBoard 可视化。通过将 TensorBoard 指向您保存轨迹的目录来启动它：

```bash
tensorboard --logdir /tmp/jax_profiling
```

导航到 TensorBoard 网页界面中的“Profile”选项卡。您会发现几个有用的工具：

1. \*\*轨迹视图 (Trace Viewer)：\*\*这通常是最具信息量的视图。它显示不同处理单元上操作随时间执行情况的时间轴图表。

   - \*\*宿主线程 (Host Threads)：\*\*显示 CPU 上的活动，包括 Python 函数调用、JAX 调度开销，以及如果您的代码混合了 NumPy 操作，还可能包括一些 NumPy 操作。
   - \*\*设备流 (Device Streams)（例如 `/GPU:0/stream:all`、`/TPU:0/stream:all`）：\*\*显示在加速器上执行的内核。不同的流可能处理计算、内存复制（例如 `HtoD` 表示宿主到设备，`DtoH` 表示设备到宿主）或通信。
   - \*\*分析 (Analysis)：\*\*寻找设备流上长时间运行的内核（潜在的计算瓶颈）、指示空闲时间的大间隙，或在 `HtoD`/`DtoH` 复制操作上花费的大量时间（数据传输瓶颈）。

digraph G {
rankdir=LR;
node [shape=record];
subgraph cluster\_0 {
label = "分析工作流程";
style=filled;
color="#e9ecef";
JaxCode [label="JAX Python 代码"];
Profiler [label="jax.profiler.\nstart/stop\_trace()"];
TraceFile [label="轨迹文件\n(profile.pb 等)"];
TensorBoard [label="TensorBoard"];
Visualization [label="轨迹视图,\n操作视图等"];
JaxCode -> Profiler -> TraceFile -> TensorBoard -> Visualization;
}
}
\`\`\`
> JAX 性能分析轨迹的生成与查看过程。

````
```plotly
````

{"data":[{"type":"scatter","mode":"lines","x":[0,1,1,3,3,5,5,6],"y":[1,1,0,0,1,1,0,0],"name":"CPU 宿主活动","line":{"color":"#4263eb"}},{"type":"scatter","mode":"lines","x":[1,2,2.5,2.5,4,4.5,4.5,5.5],"y":[2,2,1.5,1.5,2,2,1.5,1.5],"name":"GPU 计算流","line":{"color":"#12b886"}},{"type":"scatter","mode":"lines","x":[1,1.5,1.5,3.5,4,4,5,5.5],"y":[2.5,2.5,2.2,2.2,2.5,2.5,2.2,2.2],"name":"GPU 复制流","line":{"color":"#f76707"}}],"layout":{"title":{"text":"简化版分析器轨迹示例"},"xaxis":{"title":{"text":"时间 (毫秒)"},"range":[0,6]},"yaxis":{"title":{"text":"执行单元"},"tickvals":[1,2,2.5],"ticktext":["CPU","GPU 计算","GPU 复制"],"range":[0,3]},"showlegend":true,"legend":{"x":0.05,"y":0.95},"margin":{"l":80,"r":20,"t":40,"b":40}}}
\`\`\`

> 这是一个类似于 TensorBoard 轨迹视图的简化时间轴视图，显示了 CPU 和 GPU 流上的并发活动。请注意 GPU 计算流在 2.5 毫秒和 4 毫秒之间的间隙，这可能表示空闲时间或等待数据。

2. \*\*操作视图 (Ops View)：\*\*提供每种执行操作类型（例如 `dot_general`、`sin`、`reduce_mean`）的聚合统计信息。它显示总耗时、平均耗时以及调用次数。这有助于快速发现代码中计算成本最高的 JAX 原语。
3. \*\*内存视图 (Memory Viewer)：\*\*有助于通过显示随时间变化的内存分配模式来诊断内存相关问题。高峰内存使用或频繁的分配/解除分配循环可能表示存在问题。（可用性和详细程度可能有所不同）。

### 设备特有考量

根据目标硬件，性能分析需要进行一些调整：

- \*\*CPU：\*\*尽管 `jax.profiler` 会捕获 CPU 上 JIT 编译的工作，但像 `cProfile` 这样的标准 Python 分析器对于分析代码中以纯 Python 解释方式运行的部分（例如数据加载、JIT 外部的预处理循环、整体脚本逻辑）仍然有用。将 JIT 编译函数与大量 Python 逻辑交织在一起，可能会引入在 CPU 分析中容易看到的开销。
- \*\*GPU：\*\*TensorBoard 的轨迹视图明确区分 GPU 计算流和内存复制流（PCIe 传输）。请密切关注 `HtoD` 和 `DtoH` 操作的持续时间。长时间的复制操作表明您的数据传输策略需要优化。您是否正在移动不必要的数据？数据是否可以更长时间地保留在 GPU 上？对于极其详细的内核分析，NVIDIA Nsight Systems 和 Nsight Compute 等工具可以提供对单个 CUDA 内核性能的更深入了解，但对于 JAX 级别的优化来说，TensorBoard 通常就足够了。
- \*\*TPU：\*\*TensorBoard 中的 TPU 分析通常显示专用硬件单元（如用于矩阵乘法的 MXU）的高利用率。请留意针对 TPU 性能的指标，例如 MXU 利用率百分比。操作可能会为了适应 TPU 的硬件要求（例如，MXU 的维度可被 128 整除）而进行填充，这有时可能表现为开销。低利用率可能表明存在其他瓶颈（例如，输入管线、宿主 CPU），或者计算不太适合 TPU 架构。对于 TPU VM，Google Cloud Profiler 可以提供系统层面的洞察。

### 发现常见瓶颈

性能分析通常会显示重复出现的性能模式：

1. \*\*宿主-设备数据传输：\*\*在轨迹视图的复制流中显示为长的 `HtoD`/`DtoH` 条。通过将数据一次性移动到设备并尽可能长时间地保留在那里来最大程度减少这些传输。在设备上批量处理数据，而不是逐个元素处理。
2. \*\*小内核 / 启动延迟：\*\*设备流上大量非常短的条。每次内核启动都有一定的开销。JAX 的 JIT 编译（和 XLA）尝试融合操作以减轻此问题，但过度碎片化仍然可能发生。尽可能争取更大、融合的内核。
3. \*\*设备空闲时间：\*\*设备计算流时间轴上的明显间隙。这可能意味着宿主 CPU 提供数据不够快，数据传输阻塞了计算，或者计算本身存在导致等待的同步点。
4. \*\*计算限制操作：\*\*一条或几条非常长的条目支配着设备计算流。这表明特定的 JAX 操作是主要的时间消耗者。优化可能涉及算法更改、使用更高效的原语，或检查混合精度训练（稍后介绍）是否适用。

性能分析是一个迭代过程。使用从 TensorBoard 获得的洞察来假设瓶颈所在，在代码中实现更改（例如优化数据移动、调整 JIT 编译策略或修改算法），然后再次分析以衡量影响。请记住，在对 JAX 代码进行计时或分析时，使用 `block_until_ready()` 来考虑到异步执行，并获得实际加速器工作的准确测量结果。

## 参考资料

- [Profiling JAX programs](https://jax.readthedocs.io/en/latest/profiling.html) — JAX documentation team (2024)
  关于使用`jax.profiler`收集执行轨迹、通过TensorBoard可视化并解释结果以进行JAX特定优化的官方指南。
- [Profile your ML model performance](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras) — TensorFlow documentation team (2024)
  解释如何使用TensorBoard的Profiler插件分析性能轨迹，包括与JAX分析输出兼容的Trace Viewer和Ops View。
- [NVIDIA Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/index.html) — NVIDIA Corporation (2025)
  Publisher: NVIDIA Corporation
  提供NVIDIA Nsight Systems的全面文档，这是一个用于CUDA应用的低级分析器，对JAX内置分析器无法提供的深度GPU内核分析很有用。

---

[上一节](../01-%E9%AB%98%E7%BA%A7%20JAX%20%E5%8F%98%E6%8D%A2%E4%B8%8E%E6%8E%A7%E5%88%B6%E6%B5%81/08-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%E5%A4%8D%E6%9D%82%E7%9A%84%E5%BE%AA%E7%8E%AF%E9%80%BB%E8%BE%91.md) · [下一节](02-%E7%90%86%E8%A7%A3%20JAX%20%E8%AE%A1%E7%AE%97%E5%9B%BE%20%28jaxpr%29.md)
