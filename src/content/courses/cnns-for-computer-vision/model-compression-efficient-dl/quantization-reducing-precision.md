---
course: "cnns-for-computer-vision"
chapter: "model-compression-efficient-dl"
lesson: "quantization-reducing-precision"
sourceId: 2634
sourceUrl: "https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-8-model-compression-efficient-dl/quantization-reducing-precision"
title: "量化：降低模型精度"
description: "了解降低模型权重和激活数值精度（例如FP16，INT8）以提高效率的方法。"
order: 4
plots: ["plots/2634-0.json"]
sourceHash: "f89e577a35297b7fc40ae19f86a761c976f77c47330de1226c32a2e75a0e8b0c"
sourceCorrections: []
---

量化 (quantization)是深度学习 (deep learning)模型中用于减少单个参数 (parameter)和激活值大小的方法。这与修剪等技术不同，修剪技术侧重于减少参数的总数量。标准深度学习模型通常使用32位浮点数（FP32）进行计算。量化是将这些FP32值转换为低精度表示的过程，例如16位浮点数（FP16），或者为了显著提高效率，更常见的是8位整数（INT8）。

这种数值精度的降低直接带来了显著优势，这在资源受限的环境中部署模型时尤其重要：

1. **减小模型大小：** 每值使用更少的比特位数极大地缩小了模型的内存占用。例如，一个INT8模型可以比其FP32对应模型小约4倍，使其更容易存储和传输。
2. **更快的推理 (inference)：** 在许多硬件平台上，使用低精度数字（尤其是整数）的计算速度明显更快。CPU、GPU（特别是带有Tensor Cores的）、以及TPU或NPU等专用加速器通常具有专门且高度优化的INT8执行单元。这会带来更低的延迟和更高的吞吐量 (throughput)。
3. **更低的功耗：** 更快的计算和减少的内存访问通常会降低能耗，这对于电池供电的移动和边缘设备是重要因素。

### 量化 (quantization)如何工作：值映射

主要问题是将FP32值的宽范围和高精度映射到更有限的低精度值集合，同时尽量减少信息损失。

对于**整数量化**（如INT8），这通常涉及由**比例因子** ($S$) 和**零点** ($Z$) 定义的仿射映射。比例因子是一个正浮点数，它决定了量化级别之间的步长；零点是一个整数，对应于真实值0.0。

从真实值 $x$ (FP32) 到其量化整数表示 $x_q$ (例如，INT8) 的映射是：
$x_q = \text{截断}(\text{取整}(x / S) + Z, Q_{min}, Q_{max})$
这里，$\text{取整}$ 四舍五入到最近的整数，$Z$ 移动结果以使真实值0.0正确映射，而 $\text{截断}$ 确保值保持在目标整数类型可表示的范围内（例如，对于有符号INT8，范围是$[-128, 127]$，所以$Q_{min}=-128$，$Q_{max}=127$）。

为了进行后续操作或分析而反向转换（反量化），可以使用以下方法恢复近似真实值$x'$：
$x' = S (x_q - Z)$

这种映射会引入**量化误差**，即原始值$x$与反量化值$x'$之间的差异。量化方法的目的是仔细选择$S$和$Z$，以尽量减少网络中权重 (weight)和激活分布上的这种误差。

映射范围有两种常见选择：

- **非对称量化：** 使用比例$S$和零点$Z$将FP32范围$[min_{fp32}, max_{fp32}]$映射到完整的整数范围$[Q_{min}, Q_{max}]$。真实值0.0可能无法精确映射到整数0。
- **对称量化：** 将零点$Z$设置为0，并将零附近的对称范围$[-abs_{max}, abs_{max}]$映射到整数范围。这会略微简化计算，因为$Z$项消失了，但如果原始FP32分布偏斜，则可能无法有效利用整数范围。

此外，比例$S$和零点$Z$可以按以下方式确定：

- **逐张量：** 对整个张量使用单个$S$和$Z$（例如，层中的所有权重，或特征图中的所有激活值）。
- **逐通道（或逐轴）：** 对每个通道使用单独的$S$和$Z$值（通常是权重的输出通道维度，或激活的通道维度）。这更细致，通常能得到更好的精度，特别是对于卷积层，但会增加少量簿记复杂性。

### 量化 (quantization)策略

应用量化有两种主要方法：

1. **训练后量化 (PTQ)：** 这是更简单的方法。您从一个完全训练好的FP32模型开始，然后将其权重 (weight)转换为目标低精度（例如INT8）。激活通常在推理 (inference)期间动态量化。为了确定激活的适当比例因子($S$)和零点($Z$)，PTQ需要一个**校准步骤**。这涉及到通过FP32模型输入少量有代表性的数据集（几百个样本），以收集网络中各个点激活分布的统计数据（例如，最小值/最大值范围）。这些统计数据指导$S$和$Z$的选择。PTQ很方便，因为它不需要重新训练，但有时可能会导致精度明显下降，特别是对于更激进的量化（如INT8）或敏感模型。
2. **量化感知训练 (QAT)：** 这种方法将量化过程*融入*训练循环。它通过插入“假”量化节点来模拟前向传播期间的量化效果，这些节点模仿INT8推理的舍入和截断行为。在反向传播 (backpropagation)期间，梯度通常直接通过这些节点（使用Straight-Through Estimator技术）。通过在模拟量化到位的情况下微调 (fine-tuning)模型，网络学会对精度降低更具鲁棒性。QAT通常能恢复PTQ所损失的大部分甚至全部精度，但它需要访问原始训练流程、数据和额外的训练计算资源。

### 常见精度格式及权衡

- **FP32（单精度）：** 标准基准。提供高动态范围和精度。大小：4字节/值。
- **FP16（半精度）：** 与FP32相比，内存使用量减少2倍。在原生支持FP16的硬件（如NVIDIA Tensor Cores）上提供显著的加速。它保持浮点表示，这可能比整数更容易处理，但动态范围比FP32小得多，如果处理不当（例如，在训练期间使用损失缩放），可能导致溢出/下溢问题。大小：2字节/值。
- **Bfloat16（脑浮点）：** 也使用16位，但比特分配与FP16不同：它保留FP32的8个指数位（保持动态范围），但减少了尾数位（降低精度）。在训练稳定性方面通常比FP16更受青睐。大小：2字节/值。
- **INT8（8位整数）：** 提供最显著的压缩（是FP32的4倍）和在兼容硬件上最大的推理 (inference)加速潜力。需要仔细校准（PTQ）或量化 (quantization)感知训练（QAT）来减轻因精度和范围大幅降低而导致的精度损失。大小：1字节/值。
- **更低位宽（例如INT4，二进制）：** 代表更激进的量化，提供更大的理论效率增益。然而，保持精度变得明显更具挑战性，通常需要专门的技术或网络架构。这些是活跃的研究方向，在一般应用中部署较少。



![量化格式比较（示意图）](plots/2634-0.json)



> 常见量化格式在每参数 (parameter)大小和潜在相对推理加速方面的示意性比较。实际加速效果很大程度上取决于模型、任务和目标硬件能力。

### 考量因素与框架支持

尽管量化 (quantization)提供了显著优势，但它并非万能。

- **精度敏感性：** 某些模型或任务本身对精度降低比其他模型或任务更敏感。对于INT8，QAT通常是保持精度所必需的。
- **硬件依赖性：** 实际实现的加速完全取决于目标硬件对低精度算术的支持。如果硬件缺乏高效的INT8计算单元，量化到INT8带来的速度提升就很小。
- **工作流程复杂性：** QAT增加了训练过程的复杂性。PTQ需要仔细选择校准数据集。调试量化问题也可能不简单。

主流深度学习 (deep learning)框架提供了方便量化的工具：

- **TensorFlow：** TensorFlow Lite为PTQ（包括全整数量化）和QAT提供了全面的工具。
- **PyTorch：** 在`torch.quantization`下提供模块，支持PTQ（静态和动态）和QAT工作流程。对不同后端（例如，x86的FBGEMM，ARM的QNNPACK）的支持旨在实现高效执行。
- **专用库/编译器：** 像NVIDIA的TensorRT这样的工具执行激进的层合并和优化，通常包括为NVIDIA GPU定制的FP16或INT8量化。

量化是一种优化深度学习模型的强大且广泛使用的技术。通过降低权重 (weight)和激活的数值精度，它显著减小了模型大小，加快了推理 (inference)速度，并降低了功耗，使得复杂CNN模型可以在更广泛的设备上部署，尤其是在边缘设备上。在PTQ和QAT之间选择，选择正确的精度格式（FP16，INT8），以及理解目标硬件能力，是成功应用量化的重要步骤。

## 参考资料

- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf) — Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, Dmitry Kalenichenko (2018)
  Journal: 2018 IEEE Conference on Computer Vision and Pattern Recognition; Pages: 2704-2713; DOI: [10.1109/CVPR.2018.00551](https://doi.org/10.1109/CVPR.2018.00551)
  详细介绍了深度学习中整数专用量化的基础知识，包括仿射映射和零点概念。
- [Quantization for Deep Learning Models](https://pytorch.org/docs/stable/quantization.html) — PyTorch Contributors (2019)
  PyTorch中实现量化的官方指南，涵盖了后训练量化和量化感知训练。
- [TensorFlow Lite Post-training quantization](https://www.tensorflow.org/lite/performance/post_training_quantization) — Google (2024)
  TensorFlow Lite中实现训练后量化的官方指南。
- [Learned Step Size Quantization](https://doi.org/10.48550/arXiv.1902.08153) — Steven K. Esser, Jeffrey L. McKinstry, Deepika Bablani, Rathinakumar Appuswamy, Dharmendra S. Modha (2020)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1902.08153](https://doi.org/10.48550/arXiv.1902.08153)
  介绍了一种先进的量化感知训练方法，用于动态学习量化参数。
