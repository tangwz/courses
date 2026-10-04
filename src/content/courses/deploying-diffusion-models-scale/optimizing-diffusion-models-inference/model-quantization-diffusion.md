---
course: "deploying-diffusion-models-scale"
chapter: "optimizing-diffusion-models-inference"
lesson: "model-quantization-diffusion"
sourceId: 5381
sourceUrl: "https://apxml.com/zh/courses/deploying-diffusion-models-scale/chapter-2-optimizing-diffusion-models-inference/model-quantization-diffusion"
title: "模型量化技术 (INT8, FP16)"
description: "应用量化方法以减小模型尺寸、提升推理速度，同时保持极小的质量损失。"
order: 2
plots: ["plots/5381-0.json"]
sourceHash: "974395f7aad5d12b8a022168df227ae0818739a1edba57dbe350b30d0056b4bd"
sourceCorrections: []
---

扩散模型的核心迭代去噪过程包含多次通过大型神经网络 (neural network)（通常是U-Net架构）的运算。每一步都需要大量的计算和内存带宽，这主要是因为对大型张量（权重 (weight)和激活）进行浮点运算。模型量化 (quantization)通过降低这些张量所用数值精度，直接应对这一制约。

量化的基本原理是将模型的权重（有时也包括激活）从标准的32位浮点格式（FP32）转换为低精度格式，如16位浮点（FP16、BF16）或8位整数（INT8）。这种精度降低为部署带来了多项显著好处：

1. **模型尺寸减小：** 较低的精度表示每个参数 (parameter)占用的内存更少。FP16模型的大小约为FP32模型的一半，而INT8模型约为四分之一。这大幅减少了存储需求和推理 (inference)时的内存（RAM和VRAM）占用。
2. **计算速度加快：** 现代硬件，特别是GPU和TPU等专用加速器，通常包含专用执行单元，使用低精度格式能更快地执行计算。INT8整数运算可以比FP32操作快很多。FP16也能提供显著的加速。
3. **内存带宽占用降低：** 在内存和计算单元之间移动数据通常是主要的制约。由于低精度数据更小，消耗的带宽更少，这进一步有助于加快推理速度。
4. **功耗降低：** 更快的计算和更少的数据移动通常会使每次推理的能耗降低。

### 常见量化 (quantization)格式

我们来看一下量化最常用的格式：

- **FP16 (半精度浮点数)：** 使用16位（1位符号、5位指数、10位尾数）。与FP32（1位符号、8位指数、23位尾数）相比，它在精度降低和数值范围之间取得了良好的平衡。在支持的硬件（大多数现代GPU）上，它提供了显著的加速和内存减少。然而，其缩小的指数范围使得它在数值范围极端时更容易出现数值下溢或上溢，尽管在推理 (inference)实践中这通常可以应对。
- **BF16 (脑浮点数)：** 另一种16位格式（1位符号、8位指数、7位尾数）。它保持与FP32相同的指数范围，但尾数比FP16小。与FP16相比，这种更宽的动态范围可以使其更稳定，更不容易出现溢出/下溢问题，有时在训练或微调 (fine-tuning)时更受青睐。较新的GPU和TPU普遍支持此硬件。
- **INT8 (8位整数)：** 使用8位整数表示值，通常范围从-128到127（有符号）或0到255（无符号）。这种格式在模型尺寸减小（相对于FP32约4倍）和计算加速方面提供了最大的可能性，因为整数运算在兼容硬件上非常快。然而，将浮点值的连续范围映射到有限的256个整数值集合需要仔细校准，如果处理不当，可能会导致模型准确性出现更明显的下降。

### 量化 (quantization)策略

应用量化主要有两种方法：

#### 1. 训练后量化 (PTQ)

PTQ在模型已经完成FP32训练*之后*应用。该过程包括将预训练 (pre-training)模型的权重 (weight)转换为目标低精度格式（例如FP16或INT8）。

- **权重量化：** 最简单的形式只涉及量化模型权重。激活可能仍然在FP32或FP16中计算。
- **激活量化（校准）：** 对于INT8量化，层之间传递的激活通常也会被量化。这需要一个“校准”步骤。将一个少量、有代表性的数据集（几百个样本可能就足够）通过FP32模型，以观察每层激活的典型范围（最小值和最大值）。然后，这些范围信息用于计算缩放因子，以将FP32激活分布有效地映射到INT8范围。
- **优点：** 实施相对简单，因为它不需要更改训练流程或访问完整的训练数据集。应用速度比QAT快。
- **缺点：** 有时可能导致准确性大幅下降，特别是对于INT8量化，因为模型在训练时没有考虑量化效应。校准数据需要能代表真实的推理 (inference)数据。

#### 2. 量化感知训练 (QAT)

QAT在模型训练或微调 (fine-tuning)过程中*融入*量化效应。它通过在训练期间将“伪量化”操作插入到模型图中来实现这一点。这些操作在前向传播中模拟量化的信息损失（将值四舍五入到较低精度），同时在反向传播 (backpropagation)中允许梯度不变地流过（或使用直通估计器等技术）。

- **过程：** 模型学习调整其权重，使其更能适应伪量化节点模拟的精度降低。
- **优点：** 通常情况下，对于相同的目标精度（特别是INT8），与PTQ相比，它能带来更好的准确性，通常能接近原始FP32模型的准确性。
- **缺点：** 实施更复杂，需要修改训练代码、访问训练数据以及额外的训练/微调时间。

### 将量化 (quantization)应用于扩散模型

考虑到扩散模型的规模和迭代特性，它们（特别是U-Net组件）是量化的理想选择。

- **U-Net 量化：** U-Net中的卷积层和注意力层构成了大部分计算和参数 (parameter)，使它们成为量化的主要目标。
- **混合精度：** 使用混合精度方法可能更有利。模型的一些部分，如初始嵌入 (embedding)层或重要的归一化 (normalization)层，可能对精度损失更敏感，可以保留在FP32或FP16中，而U-Net的大部分则量化为INT8。
- **采样器影响：** 尽管量化会略微改变数值，但应用经过良好校准的PTQ或QAT通常能以很小的图像质量下降保留生成图像的感知质量。在许多采样步骤后，其对最终输出的影响需要仔细验证。

### 工具和考量

PyTorch（及其`torch.quantization`模块）、TensorFlow（通过TensorFlow Lite）等框架以及NVIDIA TensorRT等推理 (inference)优化引擎提供了工具和API，以便利PTQ和QAT的工作流程。例如，TensorRT可以自动应用PTQ（包括校准）或接收使用QAT训练的模型，从而为NVIDIA GPU生成高度优化的INT8推理引擎。



![模型相对尺寸和潜在加速](plots/5381-0.json)



> 不同量化 (quantization)格式相对于标准FP32的模型尺寸减小和潜在推理加速的比较。实际加速效果很大程度上取决于硬件支持和具体的模型架构。

选择合适的量化策略需要权衡。FP16/BF16提供了一个不错的起点，复杂性适中，且效果良好。INT8能带来最大的潜在好处，但需要更细致地应用（通常是QAT或经过良好校准的PTQ）以及全面的验证，以确保可接受的质量。正如我们将在基准测试部分讨论的，在应用任何量化技术后，严格测量延迟、吞吐量 (throughput)*和*输出质量是非常重要的。

## 参考资料

- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf) — Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, Dmitry Kalenichenko (2018)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Publisher: IEEE; Pages: 2704-2713; DOI: [10.1109/CVPR.2018.00696](https://doi.org/10.1109/CVPR.2018.00696)
  一篇基础论文，介绍了 INT8 量化技术，包括训练后量化 (PTQ) 和量化感知训练 (QAT)，被 TensorFlow Lite 广泛采用。
- [NVIDIA TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html) — NVIDIA (2024)
  Publisher: NVIDIA
  NVIDIA TensorRT 的官方文档，提供了如何使用其模型优化功能（包括用于 GPU 部署的 INT8 和 FP16 量化）的详细信息。
- [Quantization with PyTorch](https://pytorch.org/docs/stable/quantization.html) — PyTorch Documentation (2024)
  Publisher: PyTorch
  PyTorch 官方指南，介绍如何使用 `torch.quantization` 模块实现量化技术（PTQ、QAT、FX 图模式），展示了特定于框架的工作流程。
