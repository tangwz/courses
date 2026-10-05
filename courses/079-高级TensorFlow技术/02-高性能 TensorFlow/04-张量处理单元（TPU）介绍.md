# 张量处理单元（TPU）介绍

来源：[原文](https://apxml.com/zh/courses/advanced-tensorflow/chapter-2-high-performance-tensorflow/introduction-tpus)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然GPU通过并行处理为许多机器学习 (machine learning)任务提供了显著的速度提升，但谷歌开发了一种名为张量处理单元（TPU）的专用硬件，它专门用于加速神经网络 (neural network)计算。TPU是应用专用集成电路（ASIC），从底层构建，旨在处理深度学习 (deep learning)模型中常见的大规模矩阵乘法及其他运算。

### TPU架构基础

TPU性能的核心部件是其矩阵乘法单元（MXU）。与CPU等通用处理器甚至GPU（它们平衡了并行处理和图形处理能力）不同，TPU将其硅片上的很大一部分区域专门用于极快且高效地执行矩阵运算。MXU通常包含数千个乘法器和累加器，它们以脉动阵列架构排列。这种设计使得数据能够流经阵列，在芯片内部以高吞吐量 (throughput)和最小的数据移动开销同时执行大量计算。

> 此图展示了输入数据和权重 (weight)如何流入TPU的矩阵乘法单元（MXU），通过其专用脉动阵列设计，实现快速计算。

TPU还配备了可由MXU直接访问的大量高带宽内存（HBM）。这最大限度地减少了获取模型参数 (parameter)和激活相关的延迟，这在处理超大型模型时，在其他架构中通常是一个瓶颈。

### TPU代次与规模

谷歌对TPU设计进行了迭代，形成了多个代次（v2、v3、v4），每个代次都提供了更高的性能和效率。单个TPU设备包含多个TPU核心。此外，TPU旨在实现大规模可扩展性。多个TPU设备可以通过高速网络互连，形成“TPU Pod”，其中可以包含数百甚至数千个TPU核心共同工作。这使得在海量数据集上训练极其庞大的模型成为可能，而这些任务即使在多GPU设置下也可能不切实际或速度极慢。

### 何时考虑TPU

TPU在特定情况下表现出色：

1. **大型模型：** 拥有海量参数 (parameter)的模型，特别是那些以密集矩阵乘法为主（如Transformer）的模型，将从TPU架构中受益良多。
2. **大型数据集：** 当训练需要处理大量数据集时，TPU的计算吞吐量 (throughput)可以显著缩短训练时间。
3. **计算密集型工作负载：** 如果您的性能分析表明训练受限于原始计算速度而非数据输入或其他因素，TPU是加速的有力选择。
4. **大规模场景下的成本效益：** 对于非常大的训练任务，TPU Pod的总计算成本有时可能低于在所需训练持续时间内使用同等数量的GPU，尽管定价模式有所不同。

### 与TensorFlow集成

在TensorFlow中使用TPU通常通过`tf.distribute` API进行管理，具体而言，是使用`tf.distribute.TPUStrategy`。此策略处理了将计算图和数据分配到可用TPU核心的复杂性。虽然大部分底层硬件交互已被抽象化，但了解TPU架构有助于优化此硬件的输入管道和模型设计。例如，当批量大小较大且输入数据维度固定并填充到与硬件兼容的倍数（通常是矩阵维度的128的倍数）时，TPU通常表现最佳。

### 优势与考量

**优势：**

- **卓越性能：** 对矩阵密集型机器学习 (machine learning)任务提供显著加速。
- **可扩展性：** 能够将训练扩展到大规模Pod配置。
- **能源效率：** 优化设计通常会带来比GPU在特定任务中更好的每瓦性能。

**考量因素：**

- **可用性：** 主要通过Google Cloud Platform (GCP) 和特定的研究项目（如TRC）访问。
- **成本：** 定价可能复杂，涉及承诺使用或按需使用费用。
- **代码调整：** 尽管`TPUStrategy`简化了使用，但与GPU训练相比，最佳性能可能需要调整数据管道（例如，固定形状、更大批量）以及有时调整模型架构。
- **内核支持：** 尽管TPU支持大多数标准TensorFlow操作，但高度定制或不常见的操作可能没有优化的TPU内核。

总而言之，TPU代表了一种专门为深度学习 (deep learning)定制的强大硬件加速选择。当面临大型模型、庞大数据集和计算密集型训练时，通过TensorFlow的分发策略来理解和使用TPU，可以带来显著的性能改进，使以前难以处理的训练任务变得可行。下一章将探讨`tf.distribute.Strategy`，包括`TPUStrategy`，如何实现在不同硬件配置下的扩展。

## 参考资料

- [In-Datacenter Performance Analysis of a Tensor Processing Unit](https://dl.acm.org/doi/10.1145/3079894.3079895) — Norman P. Jouppi, Cliff Young, Nishant Agrawal, Gul Khan, Anna Li, Raymond A. Tarjan, and Lake Wen (2017)
  Journal: Proceedings of the 44th Annual International Symposium on Computer Architecture (ISCA); Publisher: ACM; Pages: 1-12; DOI: [10.1145/3079894.3079895](https://doi.org/10.1145/3079894.3079895)
  这是描述Google第一代TPU架构和性能的原始论文，介绍了用于神经网络的ASIC概念。
- [A Domain-Specific Architecture for Training Deep Neural Networks](https://dl.acm.org/doi/10.1145/3363380) — Norman P. Jouppi, Cliff Young, David Patil, David Lake, Nir Shavit, Raymond A. Tarjan, David Patterson, and Monica S. Lam (2020)
  Journal: Communications of the ACM; Publisher: ACM; Volume: 63; Pages: 50-59; DOI: [10.1145/3363380](https://doi.org/10.1145/3363380)
  本文讨论了TPU架构的演变，重点介绍第二代和第三代TPU的设计与性能，包括它们的扩展性和能效。
- [What are TPUs?](https://cloud.google.com/tpu/docs/tpus) — Google Cloud Documentation (2024)
  Publisher: Google Cloud
  提供Google Cloud TPU的官方概述，包括其功能、可用代次以及如何在GCP生态系统中使用它们。
- [Use TPUs](https://www.tensorflow.org/guide/tpu) — TensorFlow Documentation (2023)
  Publisher: TensorFlow
  说明如何在TensorFlow中使用`tf.distribute.TPUStrategy`在TPU上训练模型，包括设置、代码调整和最佳实践。

---

[上一节](03-%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6%E8%AE%AD%E7%BB%83%E6%8A%80%E6%9C%AF.md) · [下一节](05-XLA%EF%BC%88%E5%8A%A0%E9%80%9F%E7%BA%BF%E6%80%A7%E4%BB%A3%E6%95%B0%EF%BC%89%E7%BC%96%E8%AF%91.md)
