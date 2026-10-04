# 剖析GPU和CPU使用情况

来源：[原文](https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-5-strategies-for-performance-optimization/profiling-gpu-cpu-usage)

[返回章节目录](README.md) · [返回课程目录](../README.md)

为了优化您的系统，首先必须了解其行为。简单地运行一个训练作业并等待其完成，无法知道其耗时原因。剖析是衡量应用程序不同部分的资源消耗和执行时间的过程。它使您能够从性能猜测转变为基于数据的决策。GPU未充分利用是一个常见问题，通常由受CPU限制的数据管道或低效操作引起。通过使用正确的工具，您可以准确找出系统时间花费的位置，并识别要解决的主要瓶颈。

### 高级系统监控

在研究复杂的剖析库之前，对系统资源进行简单、实时的检查可以提供即时线索。这些命令行工具是任何性能检查的第一步。

#### 使用 `htop` 监控CPU和内存

`htop` 工具是一个用于类Unix系统的交互式进程查看器，比传统的 `top` 命令有明显改进。它为您提供CPU核心、内存使用情况以及按资源消耗排序的运行进程列表的彩色实时视图。

当您运行训练脚本时，打开另一个终端并运行 `htop`。观察顶部的CPU仪表。如果看到一个或多个CPU核心持续保持在100%而GPU处于空闲状态，则很可能存在CPU瓶颈。这通常发生在数据预处理期间，例如图像增强或文本分词 (tokenization)等任务在数据发送到GPU之前在CPU上运行。

#### 使用 `nvidia-smi` 监控GPU

对于任何配备NVIDIA GPU的系统，NVIDIA系统管理界面（`nvidia-smi`）都是一个必不可少的工具。它提供所有可用GPU状态的详细摘要。要获取持续更新的视图，请使用 `watch` 命令运行它。

```bash
watch -n 1 nvidia-smi
```

此命令每秒刷新输出。以下是如何解读这些重要字段：

```text
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.65.01    Driver Version: 515.65.01    CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A100-SXM...  On   | 00000000:00:04.0 Off |                    0 |
| N/A   38C    P0    59W / 400W |   1540MiB / 40960MiB |     95%      Default |
+-------------------------------+----------------------+----------------------+
```

- **GPU利用率 (GPU-Util):** 这是训练中最重要的指标。它显示了在最后一秒内一个或多个内核在GPU上执行的时间百分比。对于优化良好的深度学习 (deep learning)工作负载，此数值应持续较高，理想情况下高于90%。如果它剧烈波动或保持较低水平，则表示您的GPU正在等待数据。
- **内存使用量 (Memory-Usage):** 这显示了GPU内存（VRAM）的使用量。如果此值接近GPU的容量，您可能会遇到内存不足（OOM）错误。这是另一种问题，通常通过减小批次大小或使用混合精度训练等技术来解决。
- **功耗 (Pwr:Usage/Cap):** 这表示当前功耗相对于GPU最大容量的比例。高功耗是GPU正在努力工作的另一个迹象。

如果 `nvidia-smi` 显示 `GPU利用率` 较低，您的检查应立即转向数据管道和受CPU限制的操作。

### 使用框架剖析器进行细粒度分析

尽管 `nvidia-smi` 能告诉您GPU是否繁忙，但它不会告诉您GPU正在做什么。要了解模型中各个操作的性能，您需要使用机器学习 (machine learning)框架内置的剖析器。这些工具可以分解每个函数、内核启动和内存复制的执行时间。

#### PyTorch中的剖析

PyTorch包含一个内置的剖析器 `torch.profiler`，它易于集成到您的训练脚本中。您只需将要分析的代码（通常是训练循环）包装在上下文 (context)管理器中。该剖析器可以跟踪CPU和GPU活动。

以下是其基本使用示例：

```python
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

model = models.resnet18().cuda()
inputs = torch.randn(16, 3, 224, 224).cuda()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True
) as prof:
    with record_function("model_inference"):
        model(inputs)

# 打印总结到控制台
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# 如需更详细的视图，可导出到TensorBoard
# prof.export_chrome_trace("trace.json")
```

`prof.key_averages().table()` 的输出为您提供操作列表，显示每个操作在CPU和GPU上花费的时间。这对于找出哪些特定层或函数耗时最多非常有用。为了获得更强大的视图，您可以将结果导出为Chrome trace文件或在TensorBoard中查看。trace视图显示了操作的时间线，使得容易发现GPU空闲的间隙。

#### TensorFlow中的剖析

TensorFlow的剖析器与TensorBoard紧密集成。捕获剖析的最简单方法是在使用 `tf.keras` 进行模型训练期间使用 `TensorBoard` 回调。您可以将其配置为剖析特定批次。

```python
import tensorflow as tf

# ... 模型和数据设置 ...

# 创建一个TensorBoard回调
log_dir = "logs/profile/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    profile_batch='10,20' # 剖析批次10到20
)

model.fit(
    train_dataset,
    epochs=5,
    callbacks=[tensorboard_callback]
)
```

运行此代码后，启动TensorBoard并导航到“剖析”选项卡。您会发现多种工具：

- **概览页 (Overview Page):** 这为您提供高级摘要和建议。它通常会直接告诉您程序是否受输入限制。
- **跟踪查看器 (Trace Viewer):** 这是最有价值的工具。与PyTorch跟踪类似，它提供了CPU和GPU上所有操作的时间线。
- **输入管道分析器 (Input Pipeline Analyzer):** 此工具专门分析您的 `tf.data` 管道，以找出数据加载和预处理中的瓶颈。

### 可视化性能时间线

PyTorch和TensorFlow剖析器中提供的跟踪查看器对于理解CPU和GPU之间的相互影响非常重要。一个常见的性能问题是数据加载管道无法跟上GPU的速度。跟踪查看器使这个问题变得明显。

> 一个显示受GPU限制工作负载的性能跟踪。GPU完成工作后进入空闲状态，等待CPU加载、预处理并传输下一批数据。GPU时间线中的这些间隙代表了浪费的计算能力。

在此图中，GPU快速执行一批工作，然后处于空闲状态。在此空闲期间，CPU正忙于准备下一批数据。优化的目标是流水化这些活动，使得CPU总是在准备未来的批次，而GPU正在处理当前批次，从而消除GPU时间线上的“空闲”间隙。通过分析剖析器的真实跟踪，您可以测量这些间隙的持续时间，并确认输入管道确实是您的瓶颈。

## 参考资料

- [PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html) — PyTorch Authors (2025)
  Publisher: PyTorch
  详细介绍了 `torch.profiler` API 的使用方法以及如何与 TensorBoard 集成，用于分析 PyTorch 应用中的 CPU 和 GPU 活动。
- [NVIDIA System Management Interface](https://developer.nvidia.com/nvidia-system-management-interface) — NVIDIA Corporation (2025)
  Publisher: NVIDIA
  关于 `nvidia-smi` 的官方文档，详细说明了其命令、输出解读以及如何用于监控 NVIDIA GPU 的健康和性能。
- [Deep Learning Performance Documentation](https://docs.nvidia.com/deeplearning/performance/index.html) — NVIDIA Corporation (2023)
  Publisher: NVIDIA
  NVIDIA 提供的指南和实践集，用于优化 NVIDIA GPU 上的深度学习工作负载，涵盖了多种性能策略和识别瓶颈的工具。

---

[上一节](05-%E4%BC%98%E5%8C%96%E6%95%B0%E6%8D%AE%E5%8A%A0%E8%BD%BD%E5%92%8C%E9%A2%84%E5%A4%84%E7%90%86%E7%AE%A1%E9%81%93.md) · [下一节](07-%E5%AE%9E%E6%93%8D%E7%BB%83%E4%B9%A0%EF%BC%9A%E5%BA%94%E7%94%A8%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6%E8%AE%AD%E7%BB%83.md)
