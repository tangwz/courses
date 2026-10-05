# LLM 部署的硬件

来源：[原文](https://apxml.com/zh/courses/llm-compression-acceleration/chapter-1-foundations-llm-efficiency-challenges/hardware-landscape-llm-deployment)

[返回章节目录](README.md) · [返回课程目录](../README.md)

了解 LLM 的计算需求和瓶颈，自然引出问题：这些模型实际运行在哪里？硬件的选择不仅仅是实现细节；它从根本上决定了部署大型语言模型的可能性、性能和成本。不同的硬件平台在计算能力、内存特性、能效和可编程性方面提供不同的特点。因此，选择合适的硬件，或针对特定目标优化模型，是 LLM 效率的一个基本构成部分。本文将审视与 LLM 部署相关的几种主要硬件类型。

### 中央处理器 (CPUs)

CPU 是无处不在的通用处理器，从笔记本电脑到服务器等各种设备中都能找到。它们旨在提供灵活性，能够顺序处理或以有限并行度处理各种任务。

- **特点：** 拥有数量较少但性能强大的核心，针对单个任务的低延迟进行了优化。它们擅长控制流、整数运算和需要复杂逻辑的任务。现代服务器级 CPU 通常包含矢量指令集（如 AVX，即高级矢量扩展），通过同时对多个数据点执行操作，可以为数值计算提供一定加速。
- **与 LLM 的关联：** 尽管 CPU *可以*运行 LLM，但它们通常难以应对 Transformer 模型固有的海量并行性。矩阵乘法和注意力机制 (attention mechanism)涉及大量的独立计算，这些更适合并行度更高的架构。CPU 上的内存带宽也通常远低于专用加速器，这往往成为 LLM 推理 (inference)的主要瓶颈，特别是随着模型规模的增加。
- **使用场景：** 最适合用于非常小的模型、开发和调试（因为易于使用且调试工具完善），或在推理延迟不是绝对优先事项且专用硬件不可用或成本过高的情况下。即使推理在加速器上运行，它们也经常处理编排和数据预处理任务。

### 图形处理器 (GPUs)

GPU 最初设计用于图形渲染，但由于其大规模并行架构，已成为深度学习 (deep learning)的主力。

- **特点：** 与 CPU 相比，包含数千个更简单的核心，针对可并行化工作负载的高吞吐量 (throughput)进行了优化。它们拥有显著更高的内存带宽，通常使用如 HBM（高带宽内存）之类的专用内存，这对于为众多核心提供数据以及处理 LLM 的大量参数 (parameter)集和激活数据非常重要。现代 GPU（如 NVIDIA 的 A100、H100 或 AMD 的 MI 系列）包含专用单元，如 Tensor Core，旨在加速混合精度矩阵乘法（例如 FP16、BF16、INT8），直接提升深度学习性能。
- **与 LLM 的关联：** GPU 在性能、可编程性（通过 CUDA 和 ROCm 等框架）以及相对成熟的软件生态系统（PyTorch、TensorFlow、cuDNN、cuBLAS、TensorRT 等优化库）之间取得了引人注目的平衡。它们大幅加速了大多数 LLM 的训练和推理 (inference)，解决了前面指出的计算和内存带宽瓶颈问题。
- **使用场景：** 在数据中心和云环境中训练大型模型以及部署对性能敏感的 LLM 推理服务的标准选择。不同等级的 GPU 可用性允许在性能和成本之间进行权衡。

### 张量处理器 (TPUs)

TPU 是由 Google 开发的专用集成电路（ASIC），专门为加速神经网络 (neural network)计算而开发。

- **特点：** 采用脉动阵列架构，针对大型矩阵乘法进行了高度优化。它们旨在在特定张量操作上提供高吞吐量 (throughput)和高能效。TPU 在处理 `bfloat16` 等特定数据类型时表现出色，并且能有效处理非常大的批次大小。TPU 之间的互连也旨在支持大规模分布式训练。
- **与 LLM 的关联：** 为大规模 LLM 训练和推理 (inference)提供了显著的性能，特别是在模型和工作负载与 TPU 的优势高度匹配时。它们是 Google 云 AI 基础设施的核心组成部分。
- **使用场景：** 主要在 Google Cloud Platform 生态系统中使用，用于训练核心模型和大规模推理。尽管性能很高，但与 GPU 相比，它们的可用性有限，并且编程模型（通常涉及 XLA - 加速线性代数）可能需要特定调整。

### 其他加速器：NPU、FPGA 和定制 ASIC

各种专用硬件正在不断涌现 (emergence)：

- **神经网络 (neural network)处理器 (NPUs)：** 通常集成到手机、边缘设备和笔记本电脑的片上系统（SoC）中。它们旨在直接在设备上对中小型模型进行高能效推理 (inference)，从而支持实时翻译或内容生成等应用，无需依赖云端。它们的能力差异很大。
- **现场可编程门阵列 (FPGAs)：** 提供硬件可重构性，允许开发人员创建高度定制的数据路径，以针对特定模型架构或操作进行优化。这可以带来高性能和高效率，但需要大量的硬件设计专业知识（使用 Verilog 或 VHDL 等硬件描述语言）和更长的开发周期。
- **定制 ASIC：** Google 等公司也开发针对 AI/ML 工作负载量身定制的 ASIC。这些旨在为特定应用实现峰值性能或效率，但缺乏 CPU 的通用性或 GPU 的可编程灵活性。例子包括来自初创公司和老牌硬件供应商的专用推理芯片。

### 比较概览

选择合适的硬件需要平衡性能需求、预算、功耗限制以及所需的软件生态系统灵活性。以下图表提供了定性比较：



[交互图表：用于 LLM 部署的硬件定性比较（值越高表示越好/越多）](https://apxml.com/zh/courses/llm-compression-acceleration/chapter-1-foundations-llm-efficiency-challenges/hardware-landscape-llm-deployment#plot-1npzudv)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "type": "bar",
      "x": [
        4,
        5,
        3,
        2
      ],
      "y": [
        "CPU",
        "GPU",
        "TPU",
        "其他 (NPU/FPGA/ASIC)"
      ],
      "orientation": "h",
      "name": "并行性",
      "marker": {
        "color": "#4c6ef5"
      }
    },
    {
      "type": "bar",
      "x": [
        2,
        5,
        4,
        3
      ],
      "y": [
        "CPU",
        "GPU",
        "TPU",
        "其他 (NPU/FPGA/ASIC)"
      ],
      "orientation": "h",
      "name": "内存带宽",
      "marker": {
        "color": "#228be6"
      }
    },
    {
      "type": "bar",
      "x": [
        3,
        2,
        4,
        5
      ],
      "y": [
        "CPU",
        "GPU",
        "TPU",
        "其他 (NPU/FPGA/ASIC)"
      ],
      "orientation": "h",
      "name": "能效（推理）",
      "marker": {
        "color": "#37b24d"
      }
    },
    {
      "type": "bar",
      "x": [
        5,
        4,
        3,
        2
      ],
      "y": [
        "CPU",
        "GPU",
        "TPU",
        "其他 (NPU/FPGA/ASIC)"
      ],
      "orientation": "h",
      "name": "可编程性/灵活性",
      "marker": {
        "color": "#f59f00"
      }
    },
    {
      "type": "bar",
      "x": [
        5,
        3,
        2,
        2
      ],
      "y": [
        "CPU",
        "GPU",
        "TPU",
        "其他 (NPU/FPGA/ASIC)"
      ],
      "orientation": "h",
      "name": "可用性/成本",
      "marker": {
        "color": "#868e96"
      }
    }
  ],
  "layout": {
    "title": {
      "text": "用于 LLM 部署的硬件定性比较（值越高表示越好/越多）"
    },
    "barmode": "group",
    "yaxis": {
      "title": {
        "text": "硬件类型"
      }
    },
    "xaxis": {
      "tickvals": [
        1,
        2,
        3,
        4,
        5
      ],
      "ticktext": [
        "很低",
        "低",
        "中等",
        "高",
        "很高"
      ],
      "range": [
        0,
        5.5
      ]
    },
    "margin": {
      "l": 150,
      "r": 20,
      "t": 50,
      "b": 40
    },
    "legend": {
      "traceorder": "normal"
    }
  }
}
```

</details>



> 与 LLM 相关的硬件特性相对比较。“其他”代表一个范围宽泛且差异很大的类别（例如，边缘 NPU 侧重效率，高端 ASIC 侧重性能）。可用性/成本反映了普遍的可得性和典型价格点，在不同类别（特别是 GPU 和“其他”）内部可能差异很大。

### 系统级考虑

对于参数 (parameter)量超过数十亿甚至数万亿的最大型 LLM，单个加速器设备（GPU 或 TPU）通常因内存容量限制而不足。部署这些模型需要由多个互连加速器组成的分布式系统。高速互连（如 NVIDIA 的 NVLink 或 InfiniBand）对于在多 GPU 设置中高效传输激活和梯度变得非常重要。因此，有效的部署不仅需要考虑单个加速器，还需要考虑整个系统架构，包括网络、主机 CPU 能力和存储。

了解这种硬件背景非常重要，因为不同压缩和加速技术（量化 (quantization)、剪枝、蒸馏等）的有效性通常很大程度上取决于目标硬件的能力。例如，某些量化格式可能具有直接的硬件支持，而非结构化剪枝的益处则取决于对稀疏计算的运行时支持。在本课程中，我们将持续把优化技术与它们所部署的底层硬件平台联系起来。

## 参考资料

- [In-Datacenter Performance Analysis of a Tensor Processing Unit](https://dl.acm.org/doi/10.1145/3079737.3079803) — Norman P. Jouppi, Cliff Young, Nishant Agrawal, Mike Baker, Gaurav Bates, Kelly Cao, Raymond M. Chiu, George Chou, Jeremy Clark, Brad Conrad, John N. Cook, Phoebe Coplon, Pat Costello, Anna Cuningham, Nathan Eifrig, Jeremy Kaiser, Paul Kallman, Alan Lee, Jason Li, Alex Lukefahr, David Mullis, Alex Nagurney, LaMDA Tran, Trevor Norris, Grant Ortega, Lawrence Ortega, Rahul Pandit, Daniel Smith, Kevin Tarolli, Greg Tassa, Anant Thazhuthaveetil, Rajat Verma, Dean Way, David Welch, Jennifer Wen, Paul N. Williams, William Wolf, Scott Wong, Tim Xu, and David Zhabel (2017)
  Journal: Proceedings of the 44th Annual International Symposium on Computer Architecture (ISCA '17); Publisher: ACM; Pages: 1-13; DOI: [10.1145/3079737.3079803](https://doi.org/10.1145/3079737.3079803)
  这篇基础性论文介绍了谷歌的张量处理单元（TPU），并详细阐述了其架构和在神经网络工作负载中的性能特点。
- [NVIDIA H100 Tensor Core GPU Architecture](https://riscv.org/wp-content/uploads/2022/03/NVIDIA-Hopper-Architecture-In-Depth.pdf) — NVIDIA (2022)
  Publisher: NVIDIA
  这份官方白皮书描述了英伟达H100 GPU的架构，包括其Tensor Cores、内存层次结构（HBM3）和互连（NVLink），所有这些都对LLM部署至关重要。
- [A Survey of Deep Learning Accelerators: Architectural Innovations and Open Challenges](https://dl.acm.org/doi/10.1145/3472017) — Vahid Esmaeilzadeh, Babak Falsafi, and Hadi Esmaeilzadeh (2021)
  Journal: ACM Computing Surveys; Publisher: Association for Computing Machinery (ACM); Volume: 54; Pages: 1-36; DOI: [10.1145/3472017](https://doi.org/10.1145/3472017)
  提供了各种深度学习硬件加速器的全面概述，涵盖了CPU、GPU、FPGA和ASIC，并讨论了它们的架构创新和面临的挑战。

---

[上一节](04-%E8%AF%84%E4%BC%B0LLM%E5%8E%8B%E7%BC%A9%E4%B8%8E%E5%BB%B6%E8%BF%9F%E7%9A%84%E8%A1%A1%E9%87%8F%E6%A0%87%E5%87%86.md) · [下一节](06-%E5%8E%8B%E7%BC%A9%E4%B8%8E%E5%8A%A0%E9%80%9F%E7%9A%84%E7%90%86%E8%AE%BA%E8%BE%B9%E7%95%8C.md)
