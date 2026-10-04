---
course: "advanced-ai-infrastructure-design-optimization"
chapter: "high-performance-model-inference"
lesson: "model-quantization-techniques"
sourceId: 7005
sourceUrl: "https://apxml.com/zh/courses/advanced-ai-infrastructure-design-optimization/chapter-4-high-performance-model-inference/model-quantization-techniques"
title: "模型量化技术：INT8和FP8"
description: "通过训练后和量化感知训练方法减少模型大小并提升性能。"
order: 3
plots: ["plots/7005-0.json"]
sourceHash: "e4e77fc197ba2a71d3400f63e644c07a45b14f6f2bc3897344c0ca91ffbc960a"
sourceCorrections: []
---

量化 (quantization)重新组织了模型中的数字，这是一个与通过模型编译优化计算图不同的过程。它降低了模型权重 (weight)（有时也包括激活值）的数值精度。通过将32位浮点数（$FP32$）转换为8位整数（$INT8$）或8位浮点数（$FP8$）等低精度格式，可以获得明显的性能提升。主要有三方面好处：

1. **内存占用减少：** 将$FP32$模型量化到$INT8$后，模型大小会缩小四倍。这能减轻服务主机的内存压力，并可能是在内存受限边缘设备上部署时的决定性因素。
2. **计算速度更快：** 现代处理器，特别是配备NVIDIA Tensor Cores等专用硬件的GPU，能够以远高于浮点运算的速度执行整数算术运算。这直接带来更低的推理 (inference)延迟和更高的吞吐量 (throughput)。
3. **功耗更低：** 整数运算能耗较低，这对于大规模数据中心和电池供电设备来说是一个重要的考量。

### 量化 (quantization)映射

本质上，量化是将高精度浮点值范围映射到较小范围的低精度整数值。这通过使用比例因子和（可选的）零点来实现。基本仿射变换如下：


$$
真实值 = 比例 \times (量化值 - 零点)
$$


这里，`比例`是一个浮点数，定义了量化的步长；`零点`是一个整数，确保真实值零能正确映射到量化值。

这种变换可以是以下两种：对称或非对称：

- **非对称量化：** 使用比例因子和零点。这使得它能够将浮点数的任意范围（例如，$[min, max]$）映射到完整的整数范围（例如，`UINT8`的$[0, 255]$）。这常用于激活值，特别是在ReLU函数之后，因为所有值都是非负数。
- **对称量化：** 将`零点`设为0，简化了映射。浮点数范围以零为中心（例如，$[-abs_{max}, +abs_{max}]$），并映射到整数范围（例如，`INT8`的$[-127, 127]$）。这经常用于模型权重 (weight)，这些权重通常以零为中心呈正态分布。

> 浮点数范围到整数范围的映射，适用于非对称量化和对称量化。

### 量化 (quantization)策略

确定`比例`和`零点`参数 (parameter)的方法定义了量化策略。两种主要方法是训练后量化和量化感知训练。

#### 训练后量化（PTQ）

PTQ是最直接的方法。它涉及在模型已完成$FP32$训练*之后*对其进行量化。该过程需要一个“校准”步骤，其中您将一小部分具有代表性的验证数据通过模型运行。在此过程中，量化框架会记录每层激活值的动态范围（最小值和最大值）。然后，这些观察到的范围用于计算量化激活值的最佳`比例`和`零点`参数。权重 (weight)直接从训练好的检查点进行量化。

由于PTQ不涉及重新训练，它快速且易于实现。然而，对于某些模型，精度损失可能导致无法接受的准确率下降。

#### 量化感知训练（QAT）

当PTQ导致模型性能不佳时，QAT是解决方案。QAT在训练过程中模拟量化效果。它通过在模型的图中插入“伪”或“模拟”量化操作来实现。这些操作接受$FP32$输入，模拟转换为`INT8`等低精度格式时的舍入和钳制效果，然后将结果转换回$FP32$以供后续层使用。

这个过程迫使模型的训练算法（例如，SGD）学习对量化导致的信息损失具有韧性的权重。模型学习调整其权重以最小化量化误差。虽然QAT更复杂且需要完整的重新训练周期，但它几乎总是比PTQ获得更高的准确率，通常接近原始$FP32$模型的性能。

> 训练后量化（PTQ）和量化感知训练（QAT）的工作流程比较。PTQ是一个后处理步骤，而QAT将量化模拟集成到训练循环中。

### 高级精度格式：INT8的扩展

虽然`INT8`是量化 (quantization)的主力，但新格式正在出现，以应对现代模型，特别是大型语言模型（LLM）的独特需求。

#### INT8

如前所述，`INT8`能使模型大小减少四倍，并在配备专用整数加速单元的硬件上带来明显的加速。NVIDIA A100等GPU提供的Tensor Cores为`INT8`矩阵乘法高度优化，相较于`FP16`和`FP32`，提供了大幅的性能提升。性能差异不容小觑；它代表吞吐量 (throughput)上的阶梯式提升。



![NVIDIA A100 GPU的理论吞吐量](plots/7005-0.json)



> NVIDIA A100 GPU上不同数值格式的理论峰值性能。转向INT8对推理 (inference)吞吐量有明显的提升。

#### FP8（8位浮点）

对于像Transformer这样的大型模型，它们的激活值可能具有很大的动态范围，`INT8`的定点表示有时会过于受限并导致准确率下降。这就是8位浮点数或`FP8`发挥作用的地方。

`FP8`不是整数格式。它保留了浮点数的结构，包括符号位、指数位和尾数位，只是位数更少。这使得它能够表示比`INT8`更广的数值范围，但代价是牺牲了这些值之间的精度。NVIDIA的Hopper Transformer Engine定义了两种主要的`FP8`变体：

- **E4M3：** 4位用于指数，3位用于尾数。它提供更高的精度，但动态范围较小。它非常适合前向传播中的权重 (weight)和激活值。
- **E5M2：** 5位用于指数，2位用于尾数。它具有更广的动态范围，使其非常适合在训练过程中表示梯度，因为梯度可能具有极端值。对于推理，其广阔范围对于显示大异常值的激活值也有益。

`FP8`是一种需要硬件（例如，NVIDIA H100 GPU）和软件框架支持的新技术。它代表了模型优化的前沿，在浮点数的动态范围和8位数据类型的计算效率之间取得了平衡。

### 实际应用和工具

在实现量化 (quantization)时，您很少亲自进行低级别数值转换。相反，您会使用将这些技术集成到部署工作流程中的高级工具。

- **TensorRT和ONNX Runtime：** 这些推理 (inference)运行时是应用PTQ的主要工具。您提供一个训练好的`FP32`模型和一个校准数据集，工具会自动生成一个高度优化、量化后的引擎。
- **PyTorch和TensorFlow：** 这两个框架都内置支持QAT。PyTorch提供了`torch.ao.quantization` API，用于插入量化桩并微调 (fine-tuning)模型。TensorFlow在其`tf.quantization`模块和TFLite Converter中也集成了类似的能力。

一种常见策略是因其简单性而从PTQ开始。如果准确率下降无法接受（例如，低于产品定义的阈值），则投入额外的工程精力来实现QAT。对于最新硬件上的非常大型模型，尝试`FP8`可以提供额外的性能优势。您还可以应用**混合精度量化**，其中对准确率损失贡献最大的敏感层保留在`FP16`或`FP32`中，而模型的其余部分则转换为`INT8`。

现在模型不仅在结构上得到优化，而且在数值上也高效，我们已准备好提供服务。下一步是使用生产级推理服务器部署此产物，该服务器可以处理并发请求、管理多个模型，并通过动态批处理等技术进一步提升性能。

## 参考资料

- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877) — Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, and Dmitry Kalenichenko (2017)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); DOI: [10.48550/arXiv.1712.05877](https://doi.org/10.48550/arXiv.1712.05877)
  一篇介绍深度神经网络后训练量化和量化感知训练核心概念的基础论文。
- [Quantization for PyTorch Models](https://pytorch.org/docs/stable/quantization.html) — PyTorch Authors (2019)
  Publisher: PyTorch Foundation
  PyTorch的官方文档，提供了实现后训练量化和量化感知训练的实践指南和API。
