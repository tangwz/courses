---
course: "planning-optimizing-ai-infrastructure"
chapter: "strategies-for-performance-optimization"
lesson: "identifying-performance-bottlenecks"
sourceId: 7022
sourceUrl: "https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-5-strategies-for-performance-optimization/identifying-performance-bottlenecks"
title: "识别性能瓶颈"
description: "学习使用性能分析工具识别AI工作流中的瓶颈，无论是CPU、GPU、I/O还是网络方面。"
order: 1
plots: ["plots/7022-0.json", "plots/7022-1.json"]
sourceHash: "42b502ce1321d41e5045022dba9e5f4c1352e8e09345906846e24f8e99623b17"
sourceCorrections: []
---

在解决性能问题之前，必须先找到它。一个常见错误是以为简单增加更强大的GPU就能神奇地加速缓慢的训练任务。尽管计算能力很重要，但真正的瓶颈可能隐藏在数据加载管线、网络通信或低效的CPU操作中。这种测量和定位的过程称为性能分析。性能分析将你从猜测转向数据驱动的优化，确保你的精力集中在实际限制性能的系统部分。

### 瓶颈的构成

在AI系统中，性能受限于链条中最慢的组件。可以将其类比为工厂的装配线；整条线只能以最慢工位的速度运行。你将遇到四种主要类型的瓶颈。

- **CPU受限：** 工作负载受限于中央处理器（CPU）的速度。这在数据预处理和增强期间很常见，因为复杂转换在数据发送到GPU之前在CPU上完成。如果你的GPU经常空闲，而CPU核心运行在100%，那么你遇到了CPU瓶颈。你的高并行加速器正在等待数据。
- **GPU受限：** 工作负载受限于图形处理器（GPU）的计算能力。在深度神经网络 (neural network)训练期间，这通常是理想的状态。这意味着你的数据管线效率高，GPU正在执行其设计目的的繁重计算。然而，即使在GPU受限的状态下，仍有优化空间，例如使用混合精度或更高效的模型架构，我们将在后续章节中介绍。
- **I/O受限：** 工作负载受限于存储子系统的速度。当你的模型等待从硬盘驱动器（HDD）、固态硬盘（SSD）或网络文件系统读取数据时，就会发生这种情况。在由数百万小文件组成的大型数据集上进行训练是导致I/O瓶颈的典型原因。即使最快的GPU，如果大部分时间都在等待数据到达，也毫无用处。
- **网络受限：** 在分布式训练设置中，工作负载可能受限于连接不同机器的网络带宽或延迟。在每个训练步骤中，梯度必须在所有节点之间同步。如果网络太慢无法处理此流量，你的GPU将闲置等待来自其他节点的更新，从而严重削弱分布式计算的益处。

### 系统的检查方法

为了有效找出这些问题，你需要的不仅仅是一个计时器。一种系统化的方法涉及使用一系列工具，从高级系统监视器到细粒度代码性能分析器。

#### 步骤1：高级系统监控

你的第一步应该始终是观察训练运行期间的系统表现。这些工具简单易用，随处可得，并提供快速的诊断概览。

用于GPU监控的最重要工具是NVIDIA系统管理接口，即 `nvidia-smi`。以循环方式运行它可提供GPU状态的实时视图。

```bash
watch -n 1 nvidia-smi
```

查看 `GPU-Util` 百分比。如果它持续很高（例如，>90%），你很可能是GPU受限。如果它很低或剧烈波动，瓶颈可能在别处。

接下来，使用像 `htop` 这样的CPU监视器来观察CPU使用率。如果 `GPU-Util` 很低，而一个或多个CPU核心达到100%，则有力证据表明存在CPU瓶颈。这通常意味着你的Python数据加载器难以跟上。

这种初步分析可以快速指引你的检查方向。

> 一个简单的诊断流程图，用于使用系统监控工具识别性能瓶颈的类型。

#### 步骤2：使用框架工具进行详细性能分析

一旦高级监控指明了方向，就该使用更专业的工具了。PyTorch和TensorFlow都内置了强大的性能分析器，可以追踪代码在CPU和GPU上的执行情况，帮助你精确找出哪些操作耗时最多。

**PyTorch 性能分析器**

`torch.profiler` 模块提供了一个上下文 (context)管理器，使得对代码片段进行性能分析变得简单直接。它跟踪每个操作的持续时间，并将所用时间归因于原始Python源代码。

你可以这样用性能分析器包裹你的训练循环：

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

# ... 模型、数据加载器、优化器设置 ...

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_training"):
        for step, batch in enumerate(data_loader):
            inputs, labels = batch
            inputs = inputs.to("cuda")
            labels = labels.to("cuda")

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step >= 10: # 分析几个步骤
                break

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

性能分析器的输出将提供一个详细的表格，显示CPU和GPU（CUDA设备）上最耗时的操作。

**TensorFlow 性能分析器**

TensorFlow的性能分析器与TensorBoard紧密集成，提供丰富、交互式的用户界面，以便查看性能数据。你可以使用 `tf.profiler.experimental.Profile` 上下文管理器来启用它。

```python
import tensorflow as tf

# ... 模型、数据集、优化器设置 ...

logdir = "logs/profile/"
with tf.profiler.experimental.Profile(logdir):
    for step, (x_batch, y_batch) in enumerate(dataset.take(10)):
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            loss_value = loss_fn(y_batch, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

# 要查看分析结果，启动 TensorBoard：
# tensorboard --logdir logs/profile/
```

运行此代码后，启动TensorBoard将显示一个“Profile”标签页，你可以在其中查看操作时间线、性能统计信息，甚至一个“Trace Viewer”来可视化CPU和GPU上的执行情况。

### 读取迹象：解读性能分析器输出

这些性能分析器的输出，特别是时间线或跟踪视图，能提供非常有用的信息。你正在寻找规律。

一个**理想的性能分析结果**显示GPU充满计算核（代表CUDA操作的彩色块），中间几乎没有间隙。同时，CPU在GPU工作开始之前就活跃起来，准备下一批数据。这表明一个健康的、GPU受限的工作负载。



![健康GPU受限工作负载的性能分析](plots/7022-0.json)



> GPU忙于执行核，同时CPU并行准备下一批数据。GPU时间线上的间隙极小。

然而，一个**CPU受限的性能分析结果**讲述了不同的情况。你会看到GPU时间线上有大的间隙。在这些间隙期间，跟踪将显示CPU上有大量活动，通常与你的 `DataLoader` 或数据增强函数有关。这是一个清晰的信号，表明GPU正在等待CPU完成工作。



![CPU受限工作负载的性能分析](plots/7022-1.json)



> GPU时间线显示有明显的空闲间隙，而CPU则完全被长时间的数据加载步骤占用。

如果CPU和GPU利用率都较低，瓶颈很可能是I/O。性能分析器可能显示CPU处于空闲或等待状态。这需要将你的检查转移到监控磁盘活动的工具（如 `iostat` 或 `dstat`），以确认系统正在等待从存储读取数据。

通过从高级观察转向详细的性能分析，你可以精确地找出性能问题的根本原因。找到瓶颈后，你就可以应用我们将在后续章节中讨论的特定优化策略了。

## 参考资料

- [Profiler](https://pytorch.org/docs/stable/profiler.html) — PyTorch Developers (2020)
  PyTorch官方内置性能分析工具文档，详细说明其用于识别CPU和GPU瓶颈的用法。
- [Profile Keras and TensorFlow models](https://www.tensorflow.org/guide/profiler) — TensorFlow Authors (2024)
  TensorFlow性能分析工具官方指南，包含与TensorBoard集成以可视化执行轨迹。
- [High-Performance Computing for Deep Learning: From Learning Algorithms to Hardware Architectures](https://books.google.com/books/about/High_Performance_Computing_for_Deep_Lear.html?id=lI3eDwAAQBAJ) — Moshe Ben-Nun, Torsten Hoefler (2020)
  Publisher: The MIT Press
  一本关于优化深度学习系统原理和实践的综合性书籍，其中包含对各类性能瓶颈的讨论。
