---
course: "planning-optimizing-ai-infrastructure"
chapter: "designing-on-premise-ai-infrastructure"
lesson: "assessing-workload-requirements"
sourceId: 6978
sourceUrl: "https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-2-designing-on-premise-ai-infrastructure/assessing-workload-requirements"
title: "评估工作负载需求"
description: "学习如何分析您的机器学习工作负载，以确定本地部署所需的具体硬件配置。"
order: 1
plots: []
sourceHash: "8e4df002e879cb34e4336c946e7f6ba8022f54d666fdace057999b3ae8713dd1"
sourceCorrections: []
---

在为本地AI服务器选择任何组件之前，您必须首先制定一份需求蓝图。这个过程不是简单地浏览硬件目录，而是对您的系统预期执行的任务进行系统性分析。一台为训练大型语言模型优化的服务器，与一台为移动应用提供实时预测的服务器将大不相同。试图构建一台万能机器，只会导致资金浪费和性能瓶颈。

这项分析始于对您特定AI工作负载的详细审视。尽管我们在上一章中对工作负载进行了分类，但现在我们将从硬件要求的角度来审视它们。

### 细分您的工作负载

您的基础设施需求主要由三项活动决定：数据预处理、模型训练和模型推理 (inference)。每项活动都有独特的资源配置。

- **数据预处理：** 这是机器学习 (machine learning)流程中常被忽视的重要环节。数据清洗、转换、分词 (tokenization)和数据增强等任务通常受CPU限制。如果数据管道无法足够快地向加速器传输数据，昂贵的GPU就会闲置。这表明需要强大的多核CPU、充足的系统内存来存储数据，以及快速的存储I/O以高效读取原始数据集。
- **模型训练：** 这是计算要求最高的阶段。训练是一个迭代过程，包括通过模型输入数据、计算损失并通过反向传播 (backpropagation)更新权重 (weight)。其特点是运行时间长、大规模并行计算以及对数据的大量需求。主要要求是配备尽可能多VRAM的强大GPU、用于分布式工作负载的GPU之间的高速互连，以及能维持高吞吐量 (throughput)的数据管道。
- **模型推理：** 模型训练完成后，为预测提供服务则面临不同的挑战。目标通常是实现低延迟（单个查询的快速响应）或高吞吐量（每秒大量查询，即QPS）。单个推理过程比训练迭代的要求低得多，但系统必须可靠地处理并发请求。这可能导致不同的硬件选择，例如配备足够VRAM的经济型GPU，甚至适用于不易并行化的模型的强大CPU。

### 从工作负载到具体要求：量化 (quantization)指标

为了将这些通用配置转化为具体的采购清单，您需要回答关于模型和数据的具体量化问题。

#### 模型大小与复杂性

模型本身是决定您GPU和内存选择的最重要因素。

- **VRAM需求：** 模型的参数 (parameter)、梯度和优化器状态都必须适合GPU的显存（VRAM）。使用标准Adam优化器进行训练时，模型VRAM占用量的简单估算是一个很好的起点。对于使用32位精度（FP32）的模型，每个参数需要4字节。梯度也需要每个参数4字节，而Adam优化器通常存储两种状态，需要额外8字节每参数。

  
  $$
  VRAM_{min} \approx (参数 \times 4) + (参数 \times 4) + (参数 \times 8)
  $$
  

  对于一个70亿参数的模型，这相当于$7B \times 16$字节，或大约112 GB的VRAM，这甚至没有计算数据批次和中间激活。这立即表明，单个配备24 GB或48 GB VRAM的GPU是不够的，您需要转向多GPU解决方案。
- **计算负载 (FLOPs)：** 前向和反向传播 (backpropagation)所需的浮点运算次数决定了给定GPU完成一次迭代的速度。虽然您可能无法精确计算这一点，但了解您的模型是计算密集型（如大型Transformer）还是相对轻量级（如ResNet-50）有助于选择合适的GPU级别。

#### 数据集和I/O模式

如果GPU缺乏数据输入，它们就毫无用处。数据集的特性决定了您的存储和网络需求。

- **存储容量：** 这是最直接的指标。您需要存储几TB的数据，还是在PB级别运行？这决定了您所需硬盘的数量和大小。
- **I/O吞吐量 (throughput)：** 您能多快地从磁盘读取数据并将其传输到GPU？这以千兆字节每秒（GB/s）衡量。在高清视频上进行训练需要高顺序读取速度，这是RAID配置中NVMe SSD的优势。
- **每秒I/O操作数 (IOPS)：** 如果您的数据集包含数百万个小文件（如单个图像），则存储系统每秒处理多个独立读取请求（IOPS）的能力将成为瓶颈，而不是原始吞吐量。在这种情况下，高IOPS的NVMe硬盘非常重要。

#### 性能与扩展目标

最后，定义成功在时间和规模上的表现。

- **训练时间：** 可接受的“解决方案时间”是多少？如果单个GPU需要一个月才能训练您的旗舰模型，这很可能无法接受。如果您需要将其缩短到两天，就可以开始计算所需的GPU数量，并考虑公式$T_{总} = T_{计算} + T_{通信}$中的通信开销。
- **推理 (inference)目标：** 对于推理，您是在优化单个用户的最低可能延迟，还是为大量用户群优化最高的QPS？大型模型的低延迟需求可能需要强大的专用GPU。而较小模型的高QPS目标可能通过在多个经济型GPU甚至CPU核心上部署多个模型实例来更好地满足。

### 创建您的工作负载配置

通过提出这些问题，您可以建立一个工作负载配置，作为您硬件决策的蓝图。这份配置将您从模糊的需求转向一套具体的技规范。

> 将工作负载特征转化为高层级硬件需求的一个决策流程图。

有了这份详细的配置，您现在已准备好从分析转向行动。以下章节将指导您选择特定的服务器组件，从主板和GPU到存储和网络，以满足您刚刚定义的需求。

## 参考资料

- [Designing Machine Learning Systems: An Iterative Process for Production-Ready AI](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  一本关于设计和构建机器学习系统的综合指南，涵盖从数据到部署的整个生命周期，重点关注训练和推理的硬件考量。
- [Memory Saving Techniques for Training Large Language Models](https://developer.nvidia.com/blog/memory-saving-techniques-for-training-large-language-models/) — Paulius Micikevicius, NVIDIA Corporation (2023)
  Journal: NVIDIA Developer Blog; Publisher: NVIDIA
  讨论了优化GPU内存使用的先进技术和最佳实践，特别是用于训练大型语言模型，直接解决VRAM限制和计算方法。
