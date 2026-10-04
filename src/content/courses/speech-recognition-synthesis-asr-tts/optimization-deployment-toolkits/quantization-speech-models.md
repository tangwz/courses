---
course: "speech-recognition-synthesis-asr-tts"
chapter: "optimization-deployment-toolkits"
lesson: "quantization-speech-models"
sourceId: 3216
sourceUrl: "https://apxml.com/zh/courses/speech-recognition-synthesis-asr-tts/chapter-6-optimization-deployment-toolkits/quantization-speech-models"
title: "语音模型量化"
description: "应用训练后量化和量化感知训练等方法。"
order: 1
plots: ["plots/3216-0.json"]
sourceHash: "29c395598827caa7003673d82433f71f5e278a6c7c88926e61b045d4c3a1544d"
sourceCorrections: []
---

ASR和TTS的大型深度学习 (deep learning)模型通常需要大量计算资源和内存。虽然在开发过程中功能强大，但其尺寸和处理需求可能在资源受限的设备（如手机、嵌入 (embedding)式系统）上部署时，或在服务器上实现低延迟和高吞吐量 (throughput)时，令人却步。模型量化 (quantization)是应对这些问题的主要方法。它指的是降低表示模型参数 (parameter)（权重 (weight)）以及可能在计算过程中表示其激活值的数值精度。

基本理念是从高精度浮点表示（通常是32位浮点数，即`float32`）转换为低精度格式，最常见的是8位整数（`int8`）。这种转换带来以下几项好处：

1. **模型尺寸减小：** 使用`int8`代替`float32`立即使存储模型权重所需的内存减少约4倍。这对于存储有限的设备端部署非常重要。
2. **推理 (inference)速度加快：** 许多现代CPU、GPU和专用加速器（如NPU或TPU）都针对整数运算有高度优化的指令。与`float32`操作相比，在`int8`中执行计算可显著提升速度。
3. **功耗降低：** 整数运算通常消耗的电量少于浮点运算，这对于电池供电设备很重要。

当然，这种精度降低并非没有代价。用更少的位数表示值本身会限制数值范围和精细度，可能引入近似误差，影响模型准确性。量化技术的目的是在最大限度提高效率的同时，尽量减少这种准确性下降。

### 量化 (quantization)策略

应用量化有两种主要策略：

1. **训练后量化（PTQ）：** 这通常是最简单的做法。您从一个预训练 (pre-training)的`float32`模型开始，然后将其权重 (weight)转换为低精度格式，如`int8`。激活值可以通过两种方式处理：

   - **动态量化：** 权重离线量化，但在推理 (inference)过程中激活值动态（实时）量化。这需要观察每次计算的激活值范围。应用相对简单，但由于动态范围计算和量化，可能在推理过程中引入计算开销。
   - **静态量化：** 权重和激活值均离线量化。这需要一个校准步骤，即您在小型、有代表性的数据集（校准数据集）上运行`float32`模型，以收集每层激活值典型范围的统计信息。然后，这些范围用于确定推理过程中将`float32`激活值映射到`int8`所需的固定缩放因子。静态量化通常比动态量化带来更快的推理速度，因为缩放因子是预先计算的。

   PTQ具有吸引力，因为它无需重新训练模型，也无需访问原始训练流程和数据集（静态PTQ需要一个小型校准集）。然而，它有时可能导致准确性显著下降，特别是对于对精度变化敏感的模型。
2. **量化感知训练（QAT）：** 此方法将量化过程整合到模型训练循环中。在训练的前向传播过程中，模型图中会插入“伪”量化操作。这些操作模拟`int8`精度（舍入、截断）对权重和激活值的影响，同时确保在反向传播 (backpropagation)过程中梯度仍能流动（通常使用直通估计器，即STE等方法）。

   通过在训练期间模拟量化，模型学习调整其权重，以更好地适应精度降低。与PTQ相比，QAT通常能使量化模型获得更高的准确性，通常能与原始`float32`模型的性能非常接近。主要缺点是训练复杂度增加、需要修改训练代码以及需要访问原始训练数据。

### 量化 (quantization)方案和参数 (parameter)

进行量化时，有几个选项定义了如何将浮点值映射到整数：

- **精度：** 尽管`int8`最常见，但研究人员也在关注`int4`甚至二元/三元表示以实现进一步压缩，尽管通常会带来更大的准确性挑战。半精度浮点格式如`float16`或`bfloat16`提供了一种折衷方案，将尺寸减小2倍，同时对准确性的影响通常小于`int8`，但可能无法像`int8`那样有效利用整数专用的硬件加速。
- **映射：** 量化的核心是将浮点范围映射到整数范围。常见的仿射映射为：
  $真实值 \approx (整数值 - 零点) \times 缩放因子$
  其中`scale`是决定步长的正浮点数，`zero_point`是一个将浮点零点与整数值对齐 (alignment)的整数。
- **对称性：**
  - **非对称量化：** 将观察到的浮点范围`[min_val, max_val]`映射到完整的整数范围（例如，`uint8`的`[0, 255]`或`int8`的`[-128, 127]`）。这需要`scale`和`zero_point`两者。它通常适用于ReLU之后的激活值，它们是非负的。
  - **对称量化：** 将中心化的范围`[-abs_max, +abs_max]`映射到整数范围（例如，`int8`的`[-127, 127]`，会留下一个值未使用或特殊处理`-128`）。`zero_point`对于有符号整数通常固定为0。这常用于权重 (weight)，因为它们往往集中在零附近。
- **粒度：**
  - **逐张量：** 对整个权重张量或激活张量使用单个`scale`和`zero_point`。实现最简单。
  - **逐通道（或逐轴）：** 对张量的不同切片使用不同的`scale`和`zero_point`值，通常是沿着卷积层或线性层权重的输出通道维度。这提供更精细的控制，并且通常比逐张量量化获得显著更高的准确性，特别适用于通道间权重分布不同的层。

### 量化 (quantization)在实践中的应用

对于大型语音模型（如Transformer或深度RNN），量化通常应用于计算密集型层：卷积层、线性/全连接层和循环单元。需要谨慎，因为某些操作或层可能比其他层对量化更敏感。例如，归一化 (normalization)层或注意力分数计算有时可能保持更高的精度（例如，`float16`或`float32`）以保持准确性，从而形成混合精度模型。

PyTorch（`torch.quantization`）、TensorFlow（通过TensorFlow Lite）和ONNX Runtime等框架提供工具和API来促进PTQ和QAT。这些工具通常自动化了部分过程，例如插入量化/反量化节点或执行校准。



![典型量化权衡（示意性）](plots/3216-0.json)



> 从32位浮点数（FP32）到更低精度时的示意性权衡。准确性通常在INT8时略有下降，但在INT4时下降可能更明显。模型尺寸与位宽成比例减小。推理 (inference)速度的提升在很大程度上依赖于硬件对低精度运算的支持。

在部署量化模型之前，严格的评估是必要的。不仅要衡量诸如ASR的词错率（WER）或TTS的平均意见得分（MOS）等标准指标，还要评估模型在测试数据中具有挑战性的子集上的性能（例如，嘈杂音频、特定口音、复杂句子），以确保鲁棒性没有受到过度影响。目的是为您的特定应用找到计算效率和可接受性能之间的最佳平衡点。

## 参考资料

- [Quantizing Deep Convolutional Networks for Accelerated Inference](https://doi.org/10.48550/arXiv.1806.08342) — Raghuraman Krishnamoorthi (2018)
  Journal: arXiv preprint arXiv:1806.08342; DOI: [10.48550/arXiv.1806.08342](https://doi.org/10.48550/arXiv.1806.08342)
  一份来自Google的实践指南，涵盖了训练后量化（PTQ）和量化感知训练（QAT）等技术，有助于部署高效的深度学习模型。
- [Introduction to Quantization on PyTorch](https://pytorch.org/docs/stable/quantization.html) — PyTorch Contributors (2019)
  Publisher: PyTorch Foundation
  PyTorch官方文档，提供了在PyTorch中应用量化技术（PTQ、QAT）的实践示例和API。
- [Quantization with TensorFlow Lite](https://www.tensorflow.org/lite/performance/quantization_spec) — TensorFlow Authors (2024)
  Publisher: Google AI
  TensorFlow Lite的官方指南，详细介绍了针对设备和边缘部署的量化策略和实现。
