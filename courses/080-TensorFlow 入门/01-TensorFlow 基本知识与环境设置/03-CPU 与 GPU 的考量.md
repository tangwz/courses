# CPU 与 GPU 的考量

来源：[原文](https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-1-tensorflow-fundamentals-setup/cpu-vs-gpu)

[返回章节目录](README.md) · [返回课程目录](../README.md)

TensorFlow 计算可以在不同类型的硬件处理器上运行，主要包括中央处理器 (CPU) 和图形处理器 (GPU)。了解它们之间的区别以及何时使用哪种处理器，有助于实现高效的模型开发和训练。TensorFlow 安装后默认即可在 CPU 上运行，但对于许多机器学习 (machine learning)任务，特别是深度学习 (deep learning)，使用 GPU 可以大幅加快训练速度。

### CPU：通用处理器

计算机的 CPU 是一种多功能处理器，专为处理各种任务而设计，擅长顺序操作和复杂逻辑处理。TensorFlow 可以只使用 CPU 来执行所有操作。

- **工作方式：** TensorFlow 操作，包括矩阵乘法或计算梯度等数学运算，都由 CPU 核心执行。
- **优点：**
  - **普遍性：** 每台计算机都有 CPU，使其成为默认的执行设备。使用标准的 TensorFlow 安装无需特殊的硬件或驱动。
  - **简单性：** 与 GPU 配置相比，设置更简单，出现配置问题的可能性更小。
  - **适用任务：** 适用于小型模型、以复杂控制流而非大规模并行计算为主的任务、初始原型开发以及在没有独立 GPU 的设备上运行推理 (inference)。
- **缺点：**
  - **性能瓶颈：** CPU 通常只有少量强大的核心（例如 4、8、16 个）。深度学习 (deep learning)主要依赖于大型矩阵和张量运算，这些运算涉及许多可同时进行的独立计算。CPU 在这种大规模并行工作方面相对较慢，导致复杂模型的训练时间很长。

### GPU：并行计算的强者

GPU 最初是为图形渲染而设计的，这项任务需要大量的并行计算。这种架构使其非常适合深度学习 (deep learning)中的核心数学运算。

- **工作方式：** 与 CPU 相比，GPU 包含数百或数千个更简单的核心。它们经过优化，可以同时对大量数据执行相同的操作（单指令多数据，即 SIMD）。这种并行性直接对应神经网络 (neural network)中张量操作（如矩阵乘法和卷积）的需求。
- **优点：**
  - **大幅加速：** 对于典型的深度学习工作负载，在兼容 GPU 上训练的速度可以比单独使用 CPU 训练快 10 到 100 倍（甚至更多）。在处理大型数据集和复杂模型时，这种差异会变得很明显。
  - **并行任务的高效性：** 非常适合深度学习中常见的数值计算类型。
- **缺点：**
  - **硬件要求：** 您需要特定类型的 GPU，通常是支持 CUDA（Compute Unified Device Architecture）的 NVIDIA GPU。
  - **软件设置：** 需要安装 NVIDIA 驱动、CUDA Toolkit 和 cuDNN 库。确保 TensorFlow、CUDA 和 cuDNN 之间的版本兼容性有时可能会有难度（请参考“设置开发环境”部分获取安装指南）。
  - **成本和功耗：** GPU 可能很昂贵，并且比 CPU 消耗更多的电力。

让我们看一下训练时间的潜在差异。下面的图表显示了训练一个中等复杂度的图像分类模型的比较。



[交互图表：训练时间比较（相对单位）](https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-1-tensorflow-fundamentals-setup/cpu-vs-gpu#plot-2x94ox)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "x": [
        "CPU",
        "GPU"
      ],
      "y": [
        100,
        5
      ],
      "type": "bar",
      "marker": {
        "color": [
          "#4263eb",
          "#12b886"
        ]
      }
    }
  ],
  "layout": {
    "title": "训练时间比较（相对单位）",
    "xaxis": {
      "title": "处理器类型"
    },
    "yaxis": {
      "title": "相对时间（CPU = 100 单位）",
      "range": [
        0,
        110
      ]
    },
    "bargap": 0.4
  }
}
```

</details>



> 针对一个示例深度学习任务的相对训练时间。实际加速效果很大程度上取决于模型、数据和具体硬件。

### TensorFlow 如何使用 GPU

如果您已正确安装了支持 GPU 的 TensorFlow 以及必要的 NVIDIA 软件（CUDA Toolkit 和 cuDNN），TensorFlow 通常会自动优先使用可用的 GPU 来执行大多数操作。它负责将计算任务分配给 GPU，以提升性能。

您通常不需要显式地告诉 TensorFlow 每次操作都使用 GPU。不过，在安装后验证 TensorFlow 是否能检测到您的 GPU 是个好习惯。您可以使用 TensorFlow 提供的函数来完成此操作，这将在“验证安装”部分介绍。TensorFlow 还提供了更精细的设备放置控制机制（例如，即使有 GPU 可用，也可以在 CPU 上运行特定操作），但自动放置对于大多数常见情况都适用。

### 如何选择

- **从 CPU 开始：** 对于学习 TensorFlow 基础知识、尝试小型模型或不涉及大量数值计算的任务，CPU 完全足够且更易于管理。
- **重要训练转向 GPU：** 如果您计划训练深度神经网络 (neural network)、处理大型数据集（如图像或序列），或者发现 CPU 训练时间变得不切实际地长，那么投入时间（和潜在的资金）进行 GPU 设置非常值得推荐。
- **考虑云平台选项：** 如果由于成本或设置复杂性导致本地 GPU 不可行，Google Colab（通常提供免费 GPU 访问）、Google Cloud AI Platform、AWS SageMaker 或 Azure Machine Learning 等云平台可按需提供强大的 GPU（甚至 TPU）资源。

对于本课程，虽然许多示例在现代 CPU 上运行速度合理，但如果您拥有支持 GPU 的环境，在后面涉及模型训练的章节中将看到显著的性能优势。前一节的设置说明提供了仅限 CPU 和 GPU 加速安装的指南。

## 参考资料

- [Install TensorFlow with GPU support](https://www.tensorflow.org/install/gpu) — TensorFlow Authors (2024)
  TensorFlow GPU加速的官方设置指南，涵盖硬件、驱动、CUDA和cuDNN要求。
- [Programming Massively Parallel Processors: A Hands-on Approach](https://www.elsevier.com/books/programming-massively-parallel-processors/kirk/978-0-12-811986-0) — David B. Kirk and Wen-mei W. Hwu (2016)
  Publisher: Morgan Kaufmann
  一本关于GPU架构和CUDA并行编程的综合性教材，对于理解深度学习加速原理至关重要。
- [CS231n: Convolutional Neural Networks for Visual Recognition - Course Materials](https://cs231n.github.io/) — Stanford University (2024)
  Publisher: Stanford University
  来自一门著名深度学习课程的讲义，讨论了硬件（特别是GPU）在加速神经网络训练中的作用。

---

[上一节](02-%E9%85%8D%E7%BD%AE%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83.md) · [下一节](04-%E9%AA%8C%E8%AF%81%E6%82%A8%E7%9A%84%E5%AE%89%E8%A3%85.md)
