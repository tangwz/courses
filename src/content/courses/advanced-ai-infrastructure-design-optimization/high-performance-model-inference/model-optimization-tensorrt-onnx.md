---
course: "advanced-ai-infrastructure-design-optimization"
chapter: "high-performance-model-inference"
lesson: "model-optimization-tensorrt-onnx"
sourceId: 7003
sourceUrl: "https://apxml.com/zh/courses/advanced-ai-infrastructure-design-optimization/chapter-4-high-performance-model-inference/model-optimization-tensorrt-onnx"
title: "使用 TensorRT 和 ONNX Runtime 进行模型优化"
description: "应用图优化、内核融合和精度降低以加速模型执行。"
order: 2
plots: ["plots/7003-0.json"]
sourceHash: "fd1f9ae55388eba3c2a216acefeeac48cbb0b20724a7c9afedd52f68830b25a9"
sourceCorrections: []
---

训练后保存的模型产物，例如 PyTorch 的 `.pt` 文件或 TensorFlow SavedModel 文件，主要表示模型的架构和学习到的权重 (weight)。然而，它并非是为高性能推理 (inference)而优化的可执行文件。为了让训练好的模型能够满足严格的延迟和吞吐量 (throughput)服务水平目标（SLOs），并成为可用于生产的服务，我们必须进行一系列优化步骤。这些步骤将模型的计算图转换为能在目标硬件上高效运行的格式。

在这方面，两个基础工具是开放神经网络 (neural network)交换（ONNX）格式及其关联的运行时，以及 NVIDIA 的 TensorRT。与专用的 LLM 服务引擎相结合，它们共同提供了一个强大的流水线，用于将特定框架的模型转换为高度加速的推理系统。

### ONNX 的解耦能力

许多优化流程中的第一步是将模型从其原始训练框架导出为 ONNX 格式。ONNX 是一种开放标准格式，用于表示机器学习 (machine learning)模型。可以将其视为一种中间表示（IR），它将模型的架构与其创建所在的框架解耦。这种可移植性是其主要益处。一旦模型转换为 ONNX 格式，它就可以由任何兼容的运行时或编译器运行，使您摆脱供应商锁定，并提供一致的部署目标。

> ONNX 生态系统。模型从各种训练框架导出到通用的 ONNX 格式，然后可以被多个推理 (inference)引擎和编译器使用。

导出模型通常是一个简单的过程。例如，在 PyTorch 中，您可以使用 `torch.onnx.export()` 函数：

```python
import torch
import torchvision

# 加载预训练模型
model = torchvision.models.resnet50(pretrained=True)
model.eval()

# 创建具有正确尺寸的虚拟输入
dummy_input = torch.randn(1, 3, 224, 224)

# 将模型导出为 ONNX 格式
torch.onnx.export(model,
                  dummy_input,
                  "resnet50.onnx",
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
```

`dynamic_axes` 参数 (parameter)对于推理服务器尤为重要，因为它允许模型接受不同大小的批次，这一特性对于动态批处理等技术来说必不可少。

### 使用 ONNX Runtime 进行加速

尽管 ONNX 定义了格式，**ONNX Runtime** 是一个为执行这些模型而构建的高性能推理 (inference)引擎。它不仅仅是一个简单的解释器。加载 `.onnx` 文件后，ONNX Runtime 会应用一系列与硬件无关的图优化，包括：

- **常量折叠：** 预先计算图中仅依赖于常量输入的部分。
- **节点消除：** 移除冗余节点，例如恒等操作或 Dropout（推理时不需要）。
- **节点融合：** 将多个简单操作合并为一个更复杂的操作。例如，将 `MatMul` 操作与随后的 `Add`（用于偏置 (bias)）融合为一个 `FusedMatMul`。

ONNX Runtime 的一个重要特性是它使用**执行提供程序（EPs）**。EP 是一种后端，允许 ONNX Runtime 将图执行委托给专门的硬件库。您可以使用默认的 EP 在 CPU 上运行相同的 ONNX 模型，或者通过指定 CUDA EP 或 TensorRT EP 在 NVIDIA GPU 上对其进行加速，所有这些都无需更改模型文件。这种架构在性能和跨平台兼容性之间取得了很好的平衡。

### 使用 NVIDIA TensorRT 实现最高性能

对于部署在 NVIDIA GPU 上的工作负载，TensorRT 作为深度学习 (deep learning)编译器和运行时，能提供最高水平的性能。尽管带有 CUDA EP 的 ONNX Runtime 在 GPU 上执行操作的通用版本，但 TensorRT 走得更远。它接受模型（通常是 ONNX 格式），并执行硬件特定的优化，以生成序列化的“引擎”文件。这个过程计算成本高且需预先完成，但生成的引擎是为特定的 GPU 架构（例如 Ampere A100、Hopper H100）和特定的精度量身定制的。

TensorRT 的主要优化包括：

#### 1. 图与层融合

TensorRT 积极地融合层，以最大程度地减少内存带宽使用和内核启动开销。它可以将卷积、偏置 (bias)加法和 ReLU 激活等连续操作组合成一个“CBR”内核。这意味着这些层之间的中间数据永远不需要写入和读取全局 GPU 内存，从而显著降低了延迟。

> TensorRT 的层融合通过将多个节点组合成一个优化的内核，减少了内存操作。

#### 2. 内核自动调优

NVIDIA GPU 包含数千个核心，并且通常有许多不同的算法（内核）来实现像卷积这样的单一操作。TensorRT 维护着这些内核的库，并在引擎构建过程中，它会对模型中的层测试多种实现方案。然后它会为特定的张量维度和目标 GPU 选择最快的内核，从而为您的模型高效创建定制的计算路径。

#### 3. 精度校准

TensorRT 是执行训练后量化 (quantization)（PTQ）的主要工具。它可以将 32 位浮点（FP32）模型转换为使用 8 位整数（INT8）。这不仅将模型的内存占用减少了 4 倍，还在现代 NVIDIA GPU 上使用了专门的 Tensor Cores，以显著提升性能。为了在不大幅降低准确性的情况下实现这一点，TensorRT 使用校准过程，在该过程中，它会在一小部分代表性数据样本上运行 FP32 模型，以测量激活值的分布。然后它使用这些信息来确定将浮点范围转换为有限的 INT8 范围的最佳缩放因子。这种转换基于仿射映射：


$$
v_{float} \approx S \cdot (v_{quant} - Z)
$$


$v_{float}$ 是实际值，$v_{quant}$ 是量化后的整数值，$S$ 是缩放因子，而 $Z$ 是零点。TensorRT 的校准过程旨在找到最小化信息损失的最佳 $S$ 和 $Z$。

### 专用的现代服务运行时

虽然 ONNX Runtime 和 TensorRT 等通用运行时针对的是传统的 CNN、Vision Transformer 和经典序列模型，但现代生成式 AI (Generative AI) 和大语言模型（LLM）需要专门针对动态序列长度、KV 缓存和连续批处理进行优化的执行引擎：

- **TensorRT-LLM：** 基于 TensorRT 构建的开源库，提供专门的内核（例如 FlashAttention 和分页 KV 缓存）、多 GPU 张量并行以及专为 NVIDIA 平台优化的 In-flight batching。
- **vLLM：** 一个高吞吐量 (throughput)、内存高效的 LLM 服务引擎，采用 PagedAttention 技术，通过虚拟分页管理 KV 缓存内存以消除内存碎片。
- **SGLang：** 一种专为复杂多调用 LLM 工作流设计的结构化生成与服务框架，提供跨多个请求的基于基数树（radix-tree）的 KV 缓存复用和高效的受限解码。
- **LMDeploy：** 由 OpenMMLab 开发的生产级部署工具包，提供自定义 C++ 推理 (inference)后端（TurboMind）、自动化量化 (quantization)（AWQ、W8A8、W4A16）以及动态批处理。
- **Modular MAX (Mojo Accelerated Execution)：** 一个统一且高度可扩展的 AI 引擎与编译器工具链，旨在跨不同硬件架构（CPU、GPU 和自定义加速器）提供统一性能，且无需锁定特定框架。

### 选择您的优化策略

优化运行时的选择取决于您的具体性能要求、架构类型和部署限制。

- **在以下情况使用 ONNX Runtime：**

  - 您需要在广泛的硬件上实现可移植性，包括 CPU 和非 NVIDIA GPU。
  - “足够好”的 GPU 性能可以接受，且开发简易性是优先考虑的。
  - 您想要一个可以在多个环境中部署的单一模型产物。
- **在以下情况使用 TensorRT：**

  - 您在 NVIDIA GPU 上部署视觉、音频或标准 Transformer 架构，并需要最低的延迟。
  - 您可以承担离线构建时间来生成硬件特定的引擎。
- **在以下情况使用专用的 LLM 引擎（TensorRT-LLM、vLLM、SGLang、LMDeploy、Modular MAX）：**

  - 您正在提供自回归 (autoregressive)大语言模型 (LLM)服务，其中内存带宽、动态序列长度、KV 缓存效率和连续批处理决定了吞吐量 (throughput)。



![GPU 推理延迟比较](plots/7003-0.json)



> 常见模型在不同执行后端下的示例性延迟比较。注意对数刻度。优化带来的性能提升显著，其中 INT8 精度的 TensorRT 提供了最低的延迟。

## 参考资料

- [Open Neural Network Exchange (ONNX)](https://onnx.ai/) — ONNX Community (Continuously updated)
  Publisher: LFAI
  提供关于ONNX格式的信息，包括其规范、模型表示以及用于不同机器学习框架之间互操作的工具。
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/) — Microsoft (Continuously updated)
  Publisher: Microsoft
  ONNX Runtime功能的全面指南，包括其架构、执行提供程序、图优化以及在各种平台上的部署说明。
- [NVIDIA TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html) — NVIDIA (Continuously updated)
  Publisher: NVIDIA
  详细说明TensorRT的功能，涵盖层融合、内核自动调优和精度校准等模型优化技术，用于加速NVIDIA GPU上的深度学习推理。
