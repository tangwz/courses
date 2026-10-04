---
course: "speech-recognition-synthesis-asr-tts"
chapter: "optimization-deployment-toolkits"
lesson: "optimized-inference-engines"
sourceId: 3225
sourceUrl: "https://apxml.com/zh/courses/speech-recognition-synthesis-asr-tts/chapter-6-optimization-deployment-toolkits/optimized-inference-engines"
title: "优化推理引擎（ONNX Runtime, TensorRT）"
description: "部署使用针对特定硬件（CPU、GPU、边缘设备）优化的运行时模型。"
order: 4
plots: ["plots/3225-0.json"]
sourceHash: "07a051e547d7ff5e7044e71f57c4bf85ccb4d9caf2eabd8628b4e9f6b83cbabd"
sourceCorrections: []
---

ASR或TTS模型经过训练并可能使用量化 (quantization)或剪枝等技术进行优化后，如何在目标应用中高效地进行推理 (inference)成为一个主要关注点。直接使用原始训练框架（如PyTorch或TensorFlow）进行部署通常不够理想。训练框架为灵活性和研究而设计，带有纯推理不需要的额外开销。这就是优化推理引擎发挥作用的地方。它们是专门的运行时库，旨在特定硬件目标上以最高性能和最少资源消耗来运行训练好的神经网络 (neural network)模型。

### 什么是推理 (inference)引擎？

把推理引擎想象成一个高度优化的解释器或虚拟机，它专门用于执行机器学习 (machine learning)模型。与需要支持反向传播 (backpropagation)、自动微分和一系列实验操作的训练框架不同，推理引擎只专注于前向传播（从输入数据生成预测的过程）。

特点包括：

- **硬件适配性：** 它们通常包含针对特定硬件平台（如CPU，会使用Intel MKL或OpenBLAS等库；NVIDIA GPU，会使用cuDNN、Tensor Cores；ARM处理器；或专用AI加速器，例如TPU、NPU）高度调整的代码（内核）。
- **图优化：** 引擎会分析模型的计算图，并应用精细的优化，例如算子融合（将多个操作合并为一个）、常量折叠、布局优化（例如NCHW与NHWC）以及内存规划。
- **低精度支持：** 许多引擎支持使用FP16（半精度浮点）或INT8（8位整数）等低精度格式运行模型，这可以在兼容硬件上大大加快计算速度并大幅减少内存使用。
- **更小的体积：** 与完整的深度学习 (deep learning)框架相比，它们通常具有更少的依赖项和更小的二进制文件大小，使其更适合在资源受限的环境中部署。

### ONNX Runtime (ORT)

开放神经网络 (neural network)交换（ONNX）是一种开放标准格式，用于表示深度学习 (deep learning)模型。ONNX的目的是实现互操作性；你可以在一个框架（例如PyTorch、TensorFlow、scikit-learn）中训练模型，并将其导出为ONNX格式。一旦模型采用ONNX格式，就可以使用支持该标准的各种工具和运行时来运行它。

ONNX Runtime（ORT）是由微软开发的高性能推理 (inference)引擎，用于运行ONNX格式的模型。它被设计为跨平台且用途广泛。

**ONNX Runtime的特点：**

- **广泛兼容性：** 可在Windows、Linux、macOS、Android、iOS甚至网页浏览器（通过WebAssembly）上运行。
- **硬件加速：** 通过其可插拔的\*\*执行提供者（EPs）\*\*架构，可使用各种硬件加速器。你可以配置ORT来优先使用特定的硬件后端，例如CUDA（用于NVIDIA GPU）、TensorRT（也来自NVIDIA，集成了TensorRT优化）、DirectML（用于Windows GPU加速）、OpenVINO（用于Intel硬件）、CoreML（用于Apple设备）、CPU等。ORT将自动尝试使用配置中最佳的可用提供者。
- **可扩展性：** 允许添加自定义算子和集成新的执行提供者。
- **优化：** 在加载模型时自动执行图优化。

**使用ONNX Runtime：**

典型的工作流程包括：

1. 将你训练好的语音模型（例如，从PyTorch）导出为`.onnx`格式。框架通常会为此提供内置函数（例如`torch.onnx.export`）。
2. 在你的部署应用中，使用ONNX Runtime库加载`.onnx`文件并创建推理会话。
3. 按照模型期望的格式准备输入数据（例如，作为NumPy数组的音频特征）。
4. 使用会话的`run`方法运行推理。

这是一个Python代码片段：

```python
import onnxruntime as ort
import numpy as np

# 定义所需的执行提供者（顺序很重要）
providers = [
    ('TensorRTExecutionProvider', { # 可选：如果使用TensorRT EP
        'trt_fp16_enable': True,    # 示例选项
        'trt_int8_enable': False
    }),
    'CUDAExecutionProvider',       # 如果可用，使用CUDA
    'CPUExecutionProvider'         # 回退到CPU
]

# 加载ONNX模型并创建推理会话
try:
    session = ort.InferenceSession("path/to/your_tts_model.onnx", providers=providers)
    print(f"Using EP: {session.get_providers()}")
except Exception as e:
    print(f"Error loading model or setting providers: {e}")
    # 回退或适当处理错误
    session = ort.InferenceSession("path/to/your_tts_model.onnx", providers=['CPUExecutionProvider'])

# 获取输入/输出名称（有助于清晰理解）
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name # 假设只有一个输出

# 准备虚拟输入数据（例如，TTS的音素ID）
# 形状取决于你的具体模型（批量大小，序列长度）
dummy_input = np.random.randint(0, 100, size=(1, 50), dtype=np.int64)

# 运行推理
# 第一个参数'None'表示获取所有输出
results = session.run([output_name], {input_name: dummy_input})

# 处理输出（例如，TTS的梅尔谱图）
mel_output = results[0]
print("Inference output shape:", mel_output.shape)
```

ONNX Runtime在性能、易用性和广泛的硬件/平台支持之间提供了良好平衡，使其成为部署许多语音模型的常用选择。

### NVIDIA TensorRT

NVIDIA TensorRT™ 是一个软件开发工具包（SDK），专门为在NVIDIA GPU上进行高性能深度学习 (deep learning)推理 (inference)而设计。它包含一个优化器和一个运行时引擎。TensorRT致力于在推理过程中实现最低的延迟和最高的吞吐量 (throughput)。

**TensorRT的特点：**

- **激进优化：** TensorRT针对NVIDIA硬件执行多项优化，这当中包含：
  - **层和张量融合：** 将多个层或操作合并到一个内核中，以减少内存传输和内核启动开销。
  - **内核自动调优：** 根据目标GPU架构和参数 (parameter)选择最佳的预实现GPU内核。
  - **精度校准：** 支持以低精度（FP16或INT8）运行模型，同时最大限度地减少精度损失。INT8优化通常涉及使用代表性数据集进行校准步骤。
  - **多流执行：** 实现多个输入流的并行处理。
- **GPU侧重：** 主要面向NVIDIA GPU，从嵌入 (embedding)式Jetson平台到大型数据中心GPU。
- **集成：** 可以直接使用，或通常作为执行提供者集成到TensorFlow或ONNX Runtime等框架中。

**使用TensorRT：**

TensorRT的工作流程在部署*之前*包含一个明确的优化步骤：

1. 获取你的模型，通常为ONNX格式（TensorRT有ONNX、TensorFlow和Caffe的解析器，但ONNX通常更受青睐）。
2. 使用TensorRT构建器API（C++或Python）解析模型并配置优化过程。这包括设置目标精度（FP32、FP16、INT8）、指定工作空间大小，并可能为INT8提供校准数据。
3. TensorRT优化图并生成一个序列化的“计划”文件（即优化后的引擎）。此步骤可能需要几分钟或更长时间，具体取决于模型的复杂度和目标精度。
4. 在你的部署应用中，使用TensorRT运行时API加载此计划文件并创建推理执行上下文 (context)。
5. 执行推理，方式与其他引擎相似。

> 使用NVIDIA TensorRT准备和部署模型的工作流程。

TensorRT通常能在NVIDIA GPU上实现最高的推理性能，特别是对于延迟敏感的ASR/TTS应用，但其代价是NVIDIA专用性，并且可能需要耗时的离线优化步骤。

### 比较ONNX Runtime和TensorRT

| 特性 | ONNX Runtime | NVIDIA TensorRT |
| --- | --- | --- |
| **主要目的** | 跨平台，兼容性 | NVIDIA GPU上的峰值性能 |
| **平台支持** | 广泛（CPU、GPU、移动、Web） | NVIDIA GPU（Linux、Windows） |
| **优化** | 良好的通用优化，EP特定 | 激进的GPU特定优化 |
| **精度** | FP32、FP16（通过EPs）、INT8（通过EPs） | FP32、FP16、INT8（带校准） |
| **工作流程** | 加载ONNX -> 推理 (inference) | 转换/解析 -> 优化 -> 推理 |
| **易用性** | 通常初始设置更简单 | 需要明确的构建步骤 |
| **厂商绑定** | 低（使用开放ONNX标准） | 高（NVIDIA专用运行时） |



![语音模型推理加速](plots/3225-0.json)



> 不同的推理引擎实现的相对加速示例。实际效果在很大程度上取决于模型、硬件和所施加的具体优化。CPU性能设置为基准（1倍）。

### 选择合适的引擎

ONNX Runtime、TensorRT或其他推理 (inference)引擎的选择取决于你的项目具体需求：

- **目标硬件：** 如果仅在NVIDIA GPU上部署并要求最高性能，TensorRT是一个强有力的选择。如果面向多个平台（CPU、不同GPU厂商、移动设备），ONNX Runtime提供更大的灵活性。
- **性能需求：** 对于NVIDIA硬件上的绝对最低延迟或最高吞吐量 (throughput)，TensorRT通常具有优势。ONNX Runtime比原生框架提供显著加速，并且通常足以满足需求。
- **开发复杂度：** ONNX Runtime在模型导出后通常涉及更直接的部署路径。TensorRT则增加了引擎构建步骤。
- **模型动态性：** 如果你需要频繁运行具有动态输入形状的模型，请确保所选引擎及其优化路径能够高效处理它们（支持情况可能不同）。

“使用ONNX Runtime和TensorRT等优化推理引擎，是部署像高级ASR和TTS系统中使用的那些要求高的深度学习 (deep learning)模型的常用做法。它们连接了优化后的模型成果与生产环境中的高效执行之间的差距，保证通过模型优化技术获得的计算收益转化为性能提升。请记住，还存在其他引擎，例如Intel的OpenVINO（针对Intel硬件优化）和TensorFlow Lite（侧重于移动和边缘设备），它们可能根据你的具体部署场景而具有相关性。”

## 参考资料

- [NVIDIA TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html) — NVIDIA Corporation (2024)
  Publisher: NVIDIA Corporation
  NVIDIA TensorRT的官方开发者指南，解释了其深度学习推理优化、API使用和在NVIDIA GPU上的部署工作流。
- [Open Neural Network Exchange (ONNX)](https://onnx.ai/) — ONNX Community (2024)
  Publisher: LFAI (Linux Foundation AI & Data)
  ONNX标准的官方网站，定义了一种开放格式用于表示机器学习模型，以实现互操作性。
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/) — Microsoft (2024)
  ONNX Runtime的全面官方文档，涵盖了在各种平台上的安装、API使用、执行提供程序和部署。
