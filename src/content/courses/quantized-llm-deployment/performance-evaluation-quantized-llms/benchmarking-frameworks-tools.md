---
course: "quantized-llm-deployment"
chapter: "performance-evaluation-quantized-llms"
lesson: "benchmarking-frameworks-tools"
sourceId: 4638
sourceUrl: "https://apxml.com/zh/courses/quantized-llm-deployment/chapter-3-performance-evaluation-quantized-llms/benchmarking-frameworks-tools"
title: "基准测试框架与工具"
description: "对旨在进行量化模型性能基准测试的工具和框架进行概述。"
order: 5
plots: ["plots/4638-0.json"]
sourceHash: "af9be0c9fd65fb418fa00f3eca05ed385deb64456f153ded2c992cf95704d652"
sourceCorrections: []
---

准确衡量量化 (quantization)带来的性能提升和潜在精度变化，需要进行系统化的基准测试。尽管手动计时或基础性能分析能提供初步估算，但专用框架和工具提供了标准化流程、全面指标，并能更好地处理评估大型语言模型（尤其是在专用硬件上）所固有的复杂性。依靠已有工具可确保结果的可复现性和可比性。

对大型语言模型（特别是量化后的模型）进行基准测试，带来一些特有的挑战：

- **变动性：** 延迟和吞吐量 (throughput)等性能指标对硬件（CPU 类型、GPU 型号、内存带宽）、批处理大小、输入序列长度、输出序列长度以及服务框架应用的具体推理 (inference)优化非常敏感。
- **指标复杂性：** 评估大型语言模型不仅仅是原始速度。评估精度需要标准化数据集和任务（例如用于困惑度计算或下游任务评估的数据集和任务），而性能评估则涉及测量不同负载条件下的延迟（例如首个令牌生成时间与单令牌延迟）以及整体吞吐量。
- **硬件交互：** 量化模型通常依赖特定的底层硬件指令或优化过的内核（例如，GPU 上的 INT4 矩阵乘法）。标准性能分析工具在未经正确配置的情况下，可能无法准确捕获这些专用操作的性能特点。

为应对这些挑战，出现了一些框架和工具，包括通用性能分析工具和大型语言模型专用评估套件。

### 通用和框架集成工具

标准深度学习 (deep learning)框架提供了内置的性能分析功能，可以作为起点：

- **PyTorch Profiler：** 提供 PyTorch 模型中各个操作符的执行时间和内存使用详细信息。它可以帮助识别模型执行图中的瓶颈，包括 CPU 和 GPU 操作中花费的时间。它与 TensorBoard 集成以进行可视化。虽然对调试特定层有用，但在真实负载条件下设置其进行端到端推理 (inference)基准测试可能需要大量工作。
- **TensorFlow Profiler：** 与 PyTorch Profiler 类似，它允许捕获 TensorFlow 模型的执行时间线、操作符统计信息和内存使用情况。它也与 TensorBoard 集成，并提供工具用于分析输入流水线性能和 GPU 利用率。

对于 Python 脚本中的基本计时，`timeit` 等模块可以测量代码片段的执行时间。然而，它们通常缺乏全面性能分析所需的细粒度，并且不易考虑异步操作，尤其是在 GPU 上。`cProfile` 可以分析 Python 代码，但在理解深度学习框架使用的底层 C++/CUDA 内核性能方面效果较差。

这些通用工具对初步检查或调试特定模型组件很有用，但在全面、标准化的大型语言模型评估方面往往不足，尤其是在比较不同量化 (quantization)方法或部署策略时。

### 大型语言模型专用基准测试框架

更专用工具专门设计用于评估语言模型：

- **LM Evaluation Framework：** 这个框架（`lm-eval`）是一个广泛采用的标准，用于在广泛的下游任务上评估生成式语言模型的精度。它提供了数百个基准测试的实现，让您能够评估量化 (quantization)如何影响零样本、少样本和微调 (fine-tuning)后模型在多样化能力（推理 (inference)、常识、阅读理解等）上的表现。虽然它主要侧重于精度，但运行其评估本身会提供特定任务的执行时间测量，尽管它并非设计为专用性能基准测试工具。在量化前后使用 `lm-eval` 可以比较精度变化。

```python
# lm-eval 使用示例（实际用法可能有所不同）
# pip install lm-eval
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM

# 加载您的原始 FP16 模型
model_fp16 = HFLM(pretrained="your_model_name_fp16")
# 加载您的量化模型（例如，使用 AutoGPTQ, bitsandbytes）
# 确保量化模型包装器兼容或调整 HFLM
model_quantized = HFLM(pretrained="your_model_name_quantized", ...)

# 在 'hellaswag' 等任务上评估 FP16 模型
results_fp16 = simple_evaluate(
    model=model_fp16,
    tasks=['hellaswag'],
    num_fewshot=0,
    batch_size='auto'
)
print("FP16 结果：", results_fp16['results']['hellaswag'])

# 在相同任务上评估量化模型
results_quantized = simple_evaluate(
    model=model_quantized,
    tasks=['hellaswag'],
    num_fewshot=0,
    batch_size='auto' # 使用相同设置以进行公平比较
)
print("量化结果：", results_quantized['results']['hellaswag'])
```

- **Hugging Face Evaluate & Optimum：** Hugging Face 生态系统提供了与评估相关的工具。
  - `evaluate`：一个专注于简化各种机器学习 (machine learning)指标计算的库，包括 BLEU、ROUGE 和困惑度等自然语言处理特定指标。它简化了为您量化模型计算与精度相关的分数的流程。
  - `optimum`：此库将 Transformers 与 ONNX Runtime、OpenVINO 和 TensorRT 等硬件加速库连接起来。它通常包含用于基准测试通过这些后端优化模型的性能（延迟、吞吐量 (throughput)）的实用工具或示例，提供了一种更直接的方式来衡量量化与运行时优化相结合所带来的速度提升。

### 硬件专用性能分析工具

当特定硬件（尤其是 GPU）上的性能重要时，需要更详细的性能分析工具：

- **NVIDIA Nsight Systems：** 提供系统范围的性能视图，捕获 CPU 和 GPU 之间的交互。它有助于识别与数据传输、内核启动延迟或 CPU 密集型操作相关的瓶颈。它对理解整体推理 (inference)流水线性能非常有价值。
- **NVIDIA Nsight Compute：** 专门侧重于分析在 GPU 上运行的 CUDA 内核。它提供各个内核的详细性能指标，例如占用率、指令吞吐量 (throughput)和内存带宽利用率。这对分析量化 (quantization)模型执行的底层计算效率很有用，尤其是在使用像 TensorRT-LLM 这样生成高度优化内核的库时。

这些工具提供最详细的见解，但通常学习曲线较高，并在需要细粒度优化时使用，通常与下一章讨论的 TensorRT-LLM 等部署框架结合使用。

### 推理 (inference)服务器中的基准测试工具

许多专用大型语言模型推理服务器都配备了各自的基准测试工具或脚本，旨在测量真实服务条件下的性能：

- **vLLM：** 通常包含脚本（例如 `benchmark_throughput.py`）来测量各种模型配置、序列长度和请求速率下的延迟和吞吐量 (throughput)，凭借其 PagedAttention 机制。
- **Text Generation Inference (TGI)：** 提供 Prometheus 端点以导出请求延迟、吞吐量和 GPU 利用率等指标，这些指标可以使用标准监控工具抓取和分析。它也可能提供特定的基准测试客户端或功能。
- **TensorRT-LLM：** 通常包含 C++ 和 Python 示例，演示如何构建优化引擎并直接对其性能进行基准测试，从而允许对推理参数 (parameter)进行细粒度控制。

使用目标部署框架的内置工具通常是评估生产环境中预期性能最直接的方法。

> 对量化 (quantization)大型语言模型进行基准测试的典型工作流程包括：选择模型、选择合适的工具、定义评估场景、运行基准测试，以及分析结果指标。

### 选择合适的工具

最佳工具取决于您的具体目标：

- **精度评估：** `lm-eval` 是综合基于任务的精度评估的标准。Hugging Face `evaluate` 对困惑度等特定指标很有用。
- **通用性能瓶颈：** PyTorch/TensorFlow Profiler 是识别模型图中慢速操作符的良好起点。
- **端到端推理 (inference)速度（延迟/吞吐量 (throughput)）：** 推理服务器（vLLM、TGI、TensorRT-LLM）提供的基准测试工具或 `optimum` 中的基准测试脚本最相关。
- **GPU 内核详细分析：** NVIDIA Nsight Compute 是理解各个 CUDA 内核性能的工具。
- **系统级交互（CPU/GPU）：** NVIDIA Nsight Systems 有助于可视化整个流水线。



![延迟比较（单次请求）](plots/4638-0.json)



> 比较示例，显示通过基准测试工具测量到的 INT4 量化 (quantization)实现的潜在延迟降低。

建立一致的基准测试协议很有必要。始终使用*相同工具、硬件、软件环境和评估场景*（数据集、批处理大小、序列长度）将量化模型与其浮点基线进行比较，以确保公平评估效率和预测质量之间的权衡。

## 参考资料

- [NVIDIA Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/) — NVIDIA Corporation (2024)
  Publisher: NVIDIA Corporation
  NVIDIA Nsight Systems的官方文档，这是一款系统级性能分析器，用于识别深度学习工作负载中CPU和GPU交互的性能瓶颈。
- [EleutherAI/lm-evaluation-harness GitHub Repository](https://github.com/EleutherAI/lm-evaluation-harness) — EleutherAI (2024)
  LM Evaluation Harness的主要来源和文档，这是一个用于在众多基准测试中评估语言模型准确性的标准框架。
- [Hugging Face Optimum Library Documentation](https://huggingface.co/docs/optimum/index) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face Optimum的官方文档，详细介绍了优化Transformer模型以适应各种硬件后端的方法和工具，包括性能基准测试。
- [PyTorch Profiler Documentation](https://pytorch.org/docs/stable/profiler.html) — PyTorch (2020)
  Publisher: PyTorch Foundation
  PyTorch Profiler的官方指南，提供如何分析PyTorch模型执行时间与内存消耗的信息。
