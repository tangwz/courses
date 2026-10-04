# 图形处理器 (GPU)

来源：[原文](https://apxml.com/zh/courses/llm-model-sizes-hardware/chapter-2-essential-hardware-components/gpu-role)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管中央处理器 (CPU) 扮演着电脑的总管角色，内存 (RAM) 提供工作存储，但许多现代计算任务，尤其是在人工智能 (AI) 应用中，需要一种不同的算力 (compute)。这正是图形处理器 (GPU) 发挥作用的地方。

GPU 最初是为了处理视频游戏和视觉应用中渲染三维图形所需的复杂计算而设计的，但开发者们发现其独特的架构使其非常适合科学计算和人工智能中常见的计算类型。

### 并行处理的实力

GPU 与 CPU 有何不同？主要区别在于它们的设计理念。CPU 通常拥有少量功能强大的核心，擅长顺序执行复杂任务或同时处理少量任务。可以把它们想象成高技能的专家，能非常快速地一个接一个地处理复杂的指令。

另一方面，GPU 通常包含数百甚至数千个更简单的核心。这些核心的单个性能不如 CPU 核心，但它们擅长同时对许多不同的数据进行*相同*的操作。这种能力被称为**并行处理**。

想象一下，你有一千个简单的加法问题需要解决。CPU（拥有几个快速核心）会迅速处理它们，但主要是按顺序一个接一个地处理。GPU（拥有数千个更简单的核心）可以为每个核心分配一个加法问题，几乎同时解决所有问题。

> CPU 和 GPU 架构及其擅长处理的任务类型比较的简化图。

### GPU 与大型语言模型背后的数学

大型语言模型，如同其他深度学习 (deep learning)模型一样，大量依赖于对排列成矩阵和向量 (vector)的大量数字集合进行数学运算。一个核心操作是矩阵乘法。训练和运行大型语言模型涉及无数次这样的乘法。

思考两个大型矩阵相乘。这涉及到许多独立的乘法和加法。GPU 的并行架构非常适合此任务。它可以同时计算许多这些小的运算，相比于 CPU 执行相同任务，能大幅加快整个过程。执行这些任务的硬件性能通常以 **FLOPS** (每秒浮点运算次数) 衡量，现代 GPU 尤其对于并行工作负载，能实现比 CPU 高得多的 FLOPS 值。更高的 FLOPS 值通常表示更强的计算能力。



[交互图表：任务速度：CPU 对比 GPU](https://apxml.com/zh/courses/llm-model-sizes-hardware/chapter-2-essential-hardware-components/gpu-role#plot-ww5j7u)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "任务速度：CPU 对比 GPU",
    "xaxis": {
      "title": "处理器类型"
    },
    "yaxis": {
      "title": "时间 (任意单位)",
      "range": [
        0,
        110
      ]
    },
    "legend": {
      "traceorder": "reversed"
    },
    "plot_bgcolor": "#f8f9fa",
    "paper_bgcolor": "white",
    "font": {
      "family": "sans-serif"
    }
  },
  "data": [
    {
      "type": "bar",
      "name": "GPU",
      "x": [
        "GPU"
      ],
      "y": [
        5
      ],
      "marker": {
        "color": "#12b886"
      }
    },
    {
      "type": "bar",
      "name": "CPU",
      "x": [
        "CPU"
      ],
      "y": [
        100
      ],
      "marker": {
        "color": "#4263eb"
      }
    }
  ]
}
```

</details>



> 示例比较显示了 GPU 完成大规模并行任务（如大型矩阵运算）可能比 CPU 快多少。实际加速效果会因具体任务和硬件而有很大差异。

### 这对运行大型语言模型为何重要

当你与大型语言模型交互时（一个被称为推理 (inference)的过程），模型的参数 (parameter)（数百万或数十亿个数字）被用于复杂的计算，以理解你的输入并生成回复。这些计算本质上是并行的。使用 GPU 可以使这些操作快得多，从而缩短大型语言模型的响应时间。对于非常大的模型来说，使用 GPU 往往不仅能加快速度，而且是在合理时间内获得回复的实际必需。

总之，虽然 CPU 管理整个系统，但 GPU 充当专门的加速器，处理人工智能任务（如运行大型语言模型）所需的大规模并行计算。它同时执行无数简单操作的能力，对现代人工智能应用的性能非常重要。我们将在下一节中看到，GPU 自己的专用内存 VRAM 也非常重要。

## 参考资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) — NVIDIA (2024)
  Publisher: NVIDIA
  这份官方指南详细介绍了CUDA架构及其并行编程模型，这对于理解GPU如何同时执行众多简单操作至关重要。
- [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) — Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton (2012)
  Journal: Advances in Neural Information Processing Systems (NIPS 2012); Publisher: Curran Associates, Inc.; Volume: 25; Pages: 1097-1105
  一篇里程碑式的论文，展示了GPU在加速深度学习模型方面的有效性，对GPU在AI研究和应用中的普及做出了重大贡献。
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) — Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer (2022)
  Journal: NeurIPS 2022; DOI: [10.48550/arXiv.2208.07339](https://doi.org/10.48550/arXiv.2208.07339)
  这篇研究论文讨论了在GPU上运行大型语言模型的实际挑战，并探讨了8位矩阵乘法等技术以提高效率，这与LLM的GPU利用率直接相关。
- [Computer Architecture: A Quantitative Approach](https://www.elsevier.com/books/computer-architecture/hennessy/978-0-12-811905-1) — John L. Hennessy and David A. Patterson (2017)
  Publisher: Elsevier
  一本备受推崇的教科书，涵盖了计算机体系结构的基本原理，包括CPU和GPU设计的详细比较，以及并行处理对于不同工作负载的优势。

---

[上一节](02-%E9%9A%8F%E6%9C%BA%E5%AD%98%E5%8F%96%E5%AD%98%E5%82%A8%E5%99%A8%20%28RAM%29.md) · [下一节](04-%E8%A7%86%E9%A2%91%E5%86%85%E5%AD%98%20%28VRAM%29.md)
