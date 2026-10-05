# 编译器优化 (TensorRT, OpenVINO)

来源：[原文](https://apxml.com/zh/courses/deploying-diffusion-models-scale/chapter-2-optimizing-diffusion-models-inference/compiler-optimization-diffusion)

[返回章节目录](README.md) · [返回课程目录](../README.md)

深度学习 (deep learning)编译器通过在计算图层面工作来优化模型。这种方法不同于量化 (quantization)和采样器优化等直接修改模型或采样过程的技术。深度学习编译器为特定的目标硬件优化代表模型的计算图，将操作序列转换为更高效的可执行格式，而不会从根本上改变模型的数学定义（可能伴随精度变化）。可将其视为神经网络 (neural network)的专用编译器，如同C++编译器优化CPU代码的方式。

对于扩散模型，它们涉及U-Net等复杂网络架构在多个时间步上的重复执行，优化底层计算图可以带来显著的性能提升。此类工具包的两个主要例子是NVIDIA TensorRT和Intel的OpenVINO。

### NVIDIA TensorRT

NVIDIA TensorRT是专为最大化NVIDIA GPU上的吞吐量 (throughput)和最小化延迟而设计的高性能深度学习 (deep learning)推理 (inference)优化器和运行时库。它接收在PyTorch或TensorFlow等框架中训练的模型，并应用一系列优化措施，然后生成一个运行时引擎。

优化技术:

1. **层和张量融合:** TensorRT识别模型图中（例如，卷积后跟偏置 (bias)加法和ReLU激活）的操作序列，并将它们合并为一个单一的核。这减少了启动多个GPU核相关的开销，并最大程度地减少了操作之间的内存传输。在扩散模型的U-Net中，这可以有效地将许多小操作组合成更少、更高效的操作。
2. **精度校准:** TensorRT支持使用较低的数值精度进行推理，例如 $FP16$（半精度浮点）和 $INT8$（8位整数）。使用较低精度可以减少内存带宽需求，降低内存占用，并使用专用硬件单元（如NVIDIA Tensor Cores）进行更快计算。TensorRT包含校准工具，用于确定 $INT8$ 量化 (quantization)的适当缩放因子，旨在最小化相比原始 $FP32$ 模型的精度下降。这与前面讨论的量化技术配合良好，但它是在编译器/运行时级别应用的。
3. **核自动调优:** NVIDIA GPU的架构和能力各不相同。TensorRT对其库中针对模型中特定操作和参数 (parameter)的预实现核进行基准测试并选择最优的（或生成专用核），并根据目标GPU架构（例如Ampere、Hopper）进行调整。
4. **动态张量内存:** TensorRT优化图中中间张量的内存分配，减少总体内存占用并提高内存复用。

典型的工作流程包括:

1. 将训练好的模型从其原始框架（例如PyTorch）导出为ONNX（Open Neural Network Exchange）等中间格式。
2. 使用TensorRT构建器（`trtexec`命令行工具或Python/C++ API）解析ONNX图，根据所选的精度模式（$FP32$、$FP16$、$INT8$）和目标GPU应用优化，并生成一个序列化的、优化的推理引擎（`.plan`或`.engine`文件）。
3. 将此引擎加载到TensorRT运行时，以便在推理期间高效执行。

对于扩散模型，TensorRT通常通过优化在每个扩散步骤中执行的计算密集型U-Net组件来提供显著的加速。

### Intel OpenVINO 工具包

OpenVINO (Open Visual Inference & Neural Network Optimization) 是Intel开发的一个工具包，旨在优化和部署深度学习 (deep learning)模型到各种Intel硬件上，包括CPU、集成显卡 (iGPU) 和视觉处理单元 (VPU)。尽管扩散模型通常在强大的独立GPU上运行，但OpenVINO为在其他平台上高效推理 (inference)提供了一条途径，这对于成本优化、边缘部署或没有NVIDIA GPU的系统可能非常重要。

组件和优化:

1. **模型优化器:** 这个命令行工具将来自各种框架（TensorFlow、通过ONNX的PyTorch等）的模型转换为OpenVINO的中间表示 (IR) 格式，包含`.xml`（拓扑）和`.bin`（权重 (weight)）文件。在转换过程中，它执行平台无关的优化，如图剪枝和算子融合。
2. **推理引擎:** 这个运行时库加载IR文件并在目标Intel硬件上高效执行模型。它应用进一步的硬件特定优化，借助如用于CPU的Intel深度神经网络 (neural network)数学核心库 (MKL-DNN，现为oneDNN) 或用于iGPU的专用核。
3. **硬件特定优化:** OpenVINO自动检测可用的Intel硬件并应用优化，例如在CPU上借助高级向量 (vector)扩展 (AVX2, AVX-512)，利用iGPU上的并行执行单元，或针对特定的VPU架构进行优化。它也支持精度优化，特别是对于 $FP16$ 和 $INT8$，通常使用训练后优化工具进行校准。

工作流程类似于TensorRT:

1. 使用模型优化器转换训练好的模型（通常通过ONNX），以生成`.xml`和`.bin` IR文件。
2. 使用OpenVINO推理引擎API（Python, C++）加载IR并在所选的Intel设备（CPU、GPU等）上运行推理。

虽然大规模扩散模型部署通常偏爱TensorRT表现出色的高端GPU，但OpenVINO使在CPU或集成GPU上推理足够或必要的场景成为可能。它对于成本或特定硬件可用性是主要限制的应用可能特别适用。

### 编译器优化工作流程示例

下面的图示说明了编译器如何通过融合来优化一小段操作序列。

> 说明算子融合的图示。编译器将多个顺序操作（卷积、批量归一化 (normalization)、ReLU）组合成一个单一的优化核，从而减少开销。

### 集成与考虑

编译器优化通常在训练和初始模型格式转换（例如转换为ONNX）后，集成到模型部署流水线中。优化后的引擎文件（例如TensorRT `.engine` 或 OpenVINO `.bin`/`.xml`）成为推理 (inference)服务器加载的部署成果。

重要考虑因素包括:

- **算子支持:** 确保编译器支持特定扩散模型架构中使用的所有操作。不受支持的操作可能需要自定义实现或回退到原始框架，这可能会限制性能提升。
- **准确性:** 当使用较低精度（$FP16$、$INT8$）时，严格验证优化模型的准确性和输出质量与原始模型相比。通常需要校准步骤以保持可接受的保真度。
- **构建时间:** 生成优化引擎，特别是包含自动调优和校准的，可能耗时。此构建步骤需要集成到模型更新的CI/CD流水线中。
- **硬件绑定:** TensorRT针对NVIDIA GPU，而OpenVINO主要针对Intel硬件。选择编译器可能会隐式地将部署与特定硬件供应商绑定。

通过使用深度学习 (deep learning)编译器如TensorRT和OpenVINO，您可以显著减少扩散模型推理的计算开销，补充其他优化技术，并使大规模部署在延迟、吞吐量 (throughput)和成本方面更可行。它们将高级模型定义转换为高度优化的、硬件特定的指令，从而实现通常难以手动达到的性能。

## 参考资料

- [NVIDIA TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html) — NVIDIA Corporation (2023)
  Publisher: NVIDIA Corporation
  NVIDIA TensorRT 的官方开发者指南。它详细介绍了其架构、优化技术（如层融合、精度校准、内核自动调整）和部署工作流程。
- [OpenVINO Documentation](https://docs.openvino.ai/) — Intel Corporation (2023)
  Publisher: Intel Corporation
  Intel OpenVINO 工具包的官方文档。它涵盖了模型优化器、推理引擎和针对 Intel 平台的硬件特定优化。
- [The Deep Learning Compiler: A Unified Solution for Hardware Acceleration](https://dl.acm.org/doi/10.1145/3409383) — Hao Wang, Dit-Yan Yeung (2020)
  Journal: ACM Computing Surveys; Publisher: Association for Computing Machinery; Volume: 53; Pages: 1-37; DOI: [10.1145/3409383](https://doi.org/10.1145/3409383)
  一项综合性调查，介绍了深度学习编译器的架构、优化技术和挑战，包括图优化和硬件特定代码生成。

---

[上一节](05-%E7%A1%AC%E4%BB%B6%E5%8A%A0%E9%80%9F%EF%BC%88GPU%E3%80%81TPU%EF%BC%89.md) · [下一节](07-%E6%8E%A8%E7%90%86%E6%80%A7%E8%83%BD%E5%9F%BA%E5%87%86%E6%B5%8B%E8%AF%95.md)
