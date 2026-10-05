# 计算需求 (FLOPS)

来源：[原文](https://apxml.com/zh/courses/llm-model-sizes-hardware/chapter-3-model-size-hardware-connection/compute-requirements-flops)

[返回章节目录](README.md) · [返回课程目录](../README.md)

我们已经看到，大型语言模型中参数 (parameter)的庞大数量决定了仅仅存储模型需要多少内存（特别是显存 (VRAM)）。但存储模型只是问题的一部分。要实际 *使用* 模型，无论是生成文本还是回答问题（这个过程称为推理 (inference)），计算机都需要执行大量的计算。模型越大，涉及的计算就越多。

### LLM 执行哪些计算？

本质上，LLM 通过多层人工神经元处理输入数据（例如您的提示）。这包含许多数学运算，主要是矩阵乘法和加法。可以将其想象成一个庞大而复杂的计算链，每一步的计算结果都会输入到下一步。模型中的每个参数 (parameter)都在这些计算中起作用。当您要求 LLM 生成文本时，它实质上是在执行数百万或数十亿次涉及小数（浮点数）的算术运算。

### 测量计算速度：FLOPS

我们如何衡量这些任务所需的计算能力？一个标准衡量单位是 **FLOPS**，它代表每秒**浮点运算**次数（**FL**oating point **O**perations **P**er **S**econd）。

- **浮点运算：** 这是一种对可以有小数点的数字（例如 3.14159 或 -0.002）执行的单一算术运算（如加法、减法、乘法或除法）。LLM 严重依赖这类计算。
- **每秒：** 这部分告诉我们处理器在一秒内可以执行多少次此类运算。FLOPS 数值越高，意味着硬件每秒可以执行更多计算，表明计算能力更强。

我们经常看到像吉浮点运算（GFLOPS，十亿次浮点运算）、太浮点运算（TFLOPS，万亿次浮点运算），甚至是拍浮点运算（PFLOPS，千万亿次浮点运算）这样的前缀，因为现代硬件，特别是 GPU，能够非常快速地执行数量惊人的此类计算。

### GPU 为何擅长 FLOPS

CPU 设计用于通用任务，并顺序执行指令或一次执行少量指令。另一方面，GPU 设计有数千个更简单的核心，它们并行工作。这种并行架构使得它们在同时处理大量数据时，执行相同类型的计算（例如 LLM 中矩阵乘法所需的计算）效率极高。

这意味着 GPU 在对深度学习 (deep learning)和 LLM 至关重要的各类运算方面，通常具有比 CPU 高得多的 FLOPS 评级。这是 GPU 成为运行这些模型的首选硬件的主要原因。

### 模型大小与 FLOPS 需求的关系

LLM 的大小（参数 (parameter)数量）与有效运行它所需的计算能力（FLOPS）之间存在直接关系：

1. **更多参数 = 更多计算：** 与拥有数百万参数的模型相比，拥有数十亿参数的模型在每个处理步骤中需要明显更多的算术运算。
2. **更多计算 = 需要更高 FLOPS：** 为了在合理的时间内（低延迟）从 LLM 获得响应，硬件需要能够快速执行这些大量计算。这意味着需要更高的 FLOPS 能力。

可以将其想象成组装产品。一个零件很少的简单产品（小型模型）即使使用基本工具（较低 FLOPS）也能快速组装。一个拥有数千个零件的复杂产品（大型模型）需要一条精密的组装线，配备许多并行工作的机械臂（更高 FLOPS）才能高效完成。如果为复杂产品使用基本工具，那将花费不切实际的长时间。



[交互图表：计算需求与模型大小](https://apxml.com/zh/courses/llm-model-sizes-hardware/chapter-3-model-size-hardware-connection/compute-requirements-flops#plot-1sncubb)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "type": "bar",
      "x": [
        "小型 LLM（例如，<10亿）",
        "中型 LLM（例如，70亿）",
        "大型 LLM（例如，700亿+）"
      ],
      "y": [
        1,
        7,
        70
      ],
      "marker": {
        "color": [
          "#96f2d7",
          "#4dabf7",
          "#f76707"
        ]
      }
    }
  ],
  "layout": {
    "title": {
      "text": "计算需求与模型大小"
    },
    "xaxis": {
      "title": {
        "text": "模型大小（参数）"
      }
    },
    "yaxis": {
      "title": {
        "text": "相对计算需求（FLOPS）"
      },
      "rangemode": "tozero",
      "tickvals": [],
      "showgrid": false
    },
    "bargap": 0.3,
    "margin": {
      "l": 50,
      "r": 20,
      "t": 50,
      "b": 50
    }
  }
}
```

</details>



> 这张图表是一种示意。虽然在实践中并非精确的线性比例，但大型模型从根本上需要明显更多的计算（更高 FLOPS）才能在推理 (inference)过程中高效运行。

### 推理 (inference)与训练的 FLOPS

需要注意的是（我们将在下一章更详细地讨论这一点），从头开始 *训练* LLM 的计算需求比仅仅 *运行* 推理（使用预训练 (pre-training)模型）高出几个数量级。训练涉及反复处理大量数据集并调整所有模型参数 (parameter)，需要长时间维持巨大的 FLOPS。

对于大多数用户的重点——推理，目标通常是获得足够快的响应以进行交互。虽然仍然是计算密集型任务，特别是对于大型模型，但所需的 FLOPS 远低于训练。然而，对于大型模型而言，拥有足够 FLOPS 能力的 GPU 对于流畅的用户体验仍然必不可少。

总之，除了内存（显存 (VRAM)）之外，以 FLOPS 衡量的计算吞吐量 (throughput)是将模型大小与硬件需求关联起来的一个重要因素。大型模型需要更多计算，从而要求具有更高 FLOPS 评级的硬件（通常是 GPU）才能以可接受的速度提供结果。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  涵盖神经网络数学基础的教材，包括LLM计算核心的矩阵运算。
- [Programming Massively Parallel Processors: A Hands-on Approach](https://www.elsevier.com/books/programming-massively-parallel-processors/kirk/978-0-12-811986-0) — David B. Kirk and Wen-mei W. Hwu (2016)
  Publisher: Elsevier; DOI: [10.1016/C2016-0-03708-1](https://doi.org/10.1016/C2016-0-03708-1)
  一本解释GPU架构和并行计算原理的权威资源，阐明了GPU为何擅长深度学习所需的浮点运算。
- [NVIDIA Ampere Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/) — Ronny Krashinsky, Olivier Giroux, Stephen Jones, Nick Stam, Sridhar Ramaswamy (2020)
  Journal: NVIDIA Technical Blog; Publisher: NVIDIA
  一份官方技术概述，介绍专为AI设计的现代GPU架构，阐述其对深度学习工作负载高FLOPS吞吐量的支持能力。

---

[上一节](03-%E9%87%8F%E5%8C%96%E7%AE%80%E4%BB%8B.md) · [下一节](05-%E5%86%85%E5%AD%98%E5%B8%A6%E5%AE%BD%E7%9A%84%E9%87%8D%E8%A6%81%E6%80%A7.md)
