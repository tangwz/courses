# TensorFlow Lite (TF Lite) 简介

来源：[原文](https://apxml.com/zh/courses/advanced-tensorflow/chapter-6-model-deployment-optimization/introduction-tensorflow-lite)

[返回章节目录](README.md) · [返回课程目录](../README.md)

使用 TensorFlow Serving 部署模型解决了服务器端推理 (inference)的需求，但许多应用需要在用户设备或嵌入 (embedding)式系统上直接进行机器学习 (machine learning)处理。在手机、微控制器或边缘设备上运行完整的 TensorFlow 模型，面临着算力 (compute)、内存、电池续航和网络连接方面的显著制约。TensorFlow Lite (TF Lite) 是谷歌专门为此打造的框架，旨在解决这一问题，从而实现高效的设备端机器学习推理。

可以将 TF Lite 视为一个全面的工具集和运行时环境，它不仅是一个库，更是为资源受限平台优化的。它让开发者能将用标准 TensorFlow 训练好的模型，转换为一种特殊格式，这种格式能以低延迟和小的二进制体积高效执行。这种能力对于要求实时响应、离线运行、加强隐私保护（因为数据无需离开设备）以及更低功耗的应用来说非常重要。

### TensorFlow Lite 生态系统的主要组成部分

TF Lite 生态系统主要围绕两个部分：转换器和解释器。

1. **TensorFlow Lite 转换器：** 这个工具负责将标准 TensorFlow 模型（SavedModels、Keras 模型或具体函数）转换成优化的 TensorFlow Lite 格式（`.tflite`）。在转换过程中，它会进行多种优化，例如操作符融合（将多个操作合并为一个以加快执行速度）和量化 (quantization)（降低模型参数 (parameter)的精度，通常从 32 位浮点数到 8 位整数），这些都显著减少模型大小并加快推理 (inference)速度。其输出是基于 FlatBuffers 的序列化模型表示，FlatBuffers 是一个高效的跨平台序列化库，它允许模型在没有复杂解析步骤的情况下加载和执行，从而最大限度地缩短加载时间和减少内存使用。
2. **TensorFlow Lite 解释器：** 这是执行 `.tflite` 模型的核心运行时引擎。它被设计为轻量且快速，具有最小的二进制体积（通常小于几百千字节，具体取决于所包含的操作符）和很少的依赖项。解释器加载 `.tflite` 模型，并使用为各种硬件平台精心优化的内核实现来执行计算图。值得一提的是，解释器通过**委托**支持硬件加速。委托是一种机制，它允许解释器将模型图的特定部分（或全部）的执行工作交给设备上可用的专用硬件加速器，例如 GPU、数字信号处理器（DSP）或专用神经网络 (neural network)处理单元（NPU）。常见例子包括 GPU 委托、NNAPI 委托（用于 Android 神经网络 API）和 Core ML 委托（用于苹果设备）。使用委托可以带来比纯 CPU 执行显著的性能提升。

> 使用 TensorFlow Lite 转换和部署 TensorFlow 模型的高层工作流程。

### 为何使用 TensorFlow Lite？

使用 TF Lite 部署模型具有多项优势，尤其是在边缘计算场景下：

- **降低延迟：** 推理 (inference)直接在设备上运行，消除了基于服务器的方法固有的网络通信延迟。这对于实时应用，例如实时视频分析或增强现实，非常重要。
- **加强隐私保护：** 敏感用户数据可以在本地处理，无需发送到服务器，从而提升用户隐私和安全性。
- **离线能力：** 即使设备没有互联网连接，模型也能运行，确保在网络不稳定或无连接环境下的可靠性。
- **更低功耗：** 优化的运行时和模型格式旨在提高效率，有助于节省移动设备的电池续航。
- **更小的模型体积：** 量化 (quantization)等技术显著减少模型所需的存储空间，这对于移动应用和存储有限的设备非常重要。

TF Lite 提供所需的工具和运行时，将复杂的机器学习 (machine learning)模型部署到过去此类技术无法触及的各类设备上。以下章节将介绍将 TensorFlow 模型转换为 `.tflite` 格式并进一步优化其设备端性能的实际步骤。

## 参考资料

- [TensorFlow Lite Developer Guide](https://www.tensorflow.org/lite/guide) — Google (2024)
  Publisher: Google
  官方和综合文档，提供了TensorFlow Lite框架的详细信息，包括模型转换、优化和部署过程。
- [TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers](https://www.oreilly.com/library/view/tinyml/9781492052036/) — Pete Warden and Daniel Situnayake (2019)
  Publisher: O'Reilly Media
  一本实用指南，侧重于使用TensorFlow Lite在资源受限的嵌入式系统和微控制器上部署机器学习模型。

---

[上一节](04-%E6%A8%A1%E5%9E%8B%E4%BC%98%E5%8C%96%E6%96%B9%E6%B3%95.md) · [下一节](06-%E4%B8%BA%20TF%20Lite%20%E8%BD%AC%E6%8D%A2%E6%A8%A1%E5%9E%8B.md)
