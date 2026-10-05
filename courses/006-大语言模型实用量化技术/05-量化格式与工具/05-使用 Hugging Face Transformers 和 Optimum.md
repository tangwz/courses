# 使用 Hugging Face Transformers 和 Optimum

来源：[原文](https://apxml.com/zh/courses/practical-llm-quantization/chapter-5-quantization-formats-tooling/huggingface-optimum-quantization)

[返回章节目录](README.md) · [返回课程目录](../README.md)

对大型语言模型进行量化 (quantization)不只在于应用某个算法；它还需要实用的工具，才能有效保存、加载和运行这些优化后的模型。尽管GGUF等格式以及GPTQ/AWQ模型的特定约定处理了存储，但Hugging Face生态系统提供了强大的库，即`transformers`和`optimum`，以使量化处理和运行由此产生的模型更加高效。

Hugging Face `transformers` 核心库本身提供了一些直接的集成，用于加载仅权重 (weight)量化的模型，这主要是在其内部使用了`bitsandbytes`库。您可能在`from_pretrained`方法中遇到过`load_in_8bit=True`或`load_in_4bit=True`这样的参数 (parameter)。这些标志提供了一种方便的方式，能将模型直接加载到GPU等硬件上，使用降低精度后的权重，大幅减少推理 (inference)时的内存占用。

```python
# 使用 transformers 原生 bitsandbytes 集成的示例
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mistral-7B-v0.1" # 示例模型

# 加载已启用4位量化的模型
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    device_map="auto" # 自动将层映射到可用设备（CPU/GPU）
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 现在 'model' 使用4位权重进行推理
# ... 使用模型和分词器继续生成
```

这种直接方法非常适合快速部署和节省内存，尤其是在消费级GPU上。然而，它主要侧重于通过`bitsandbytes`进行推理时的权重量化。对于更多元的量化方案，包括权重和激活的静态量化、与不同硬件加速器的兼容性以及标准化的导出格式，我们会转向Hugging Face Optimum。

## 认识 Hugging Face Optimum

Optimum是`transformers`的一个扩展，其专门用于连接`transformers`模型与各种硬件加速技术及优化方法，这包括量化 (quantization)。可以把它看作是一个工具集，它提供了一个标准化的接口，用于应用如训练后量化（PTQ）这样的优化，并将模型导出到ONNX（开放神经网络 (neural network)交换）等有效率的运行时格式。

Optimum 的主要目标包括：

1. **统一API：** 提供一致的方法，以便在不同后端（如ONNX Runtime、PyTorch XLA、Intel OpenVINO等）应用优化。
2. **硬件加速：** 使使用专用硬件功能变得更简单，以实现更快的推理 (inference)（例如，CPU/GPU上的INT8支持）。
3. **性能优化：** 实施诸如图融合和优化内核使用等技术，这些通常由所选后端提供支持。

## Optimum 的量化 (quantization)工作流程

Optimum 主要通过与各种优化后端的集成来支持训练后量化（PTQ）。使用Optimum进行PTQ的典型工作流程包括以下步骤：

1. **加载基础模型：** 从一个标准的`transformers`模型开始。
2. **定义量化配置：** 指定所需的量化参数 (parameter)，例如目标精度（如INT8），是量化激活（静态）还是仅量化权重 (weight)（动态），校准数据集以及目标后端。Optimum 为此提供了配置类（例如`QuantizationConfig`、`AutoQuantizationConfig`）。
3. **实例化量化器：** 从Optimum创建一个量化器对象，并将模型和配置关联起来（例如`OptimumORTOptimizer`、`OptimumIntelQuantizer`）。
4. **运行量化：** 执行量化过程。如果执行静态量化，这通常涉及使用提供的GSE数据集校准模型，以确定最佳量化范围（缩放因子、零点）。
5. **保存量化模型：** 导出优化后的模型，通常以与所选后端兼容的格式（例如，带有量化算子的ONNX图）。

让我们演示如何使用 Optimum 与 ONNX Runtime 后端进行静态 INT8 量化：

```python
from optimum.onnxruntime import ORTQuantizer, ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import QuantizationConfig, AutoQuantizationConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# 假设“calibration_dataset”已准备好（例如，训练数据的子集）

model_id = "distilbert-base-uncased-finetuned-sst-2-english"
onnx_output_dir = "./quantized_onnx_model"

# 1. 加载基础模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

# 2. 定义量化配置（AVX512 CPU 的静态 INT8）
# 针对 AVX512 CPU 静态量化的示例
qconfig = AutoQuantizationConfig.avx512_vnni(is_static=True, per_channel=False)

# 3. 实例化量化器
quantizer = ORTQuantizer.from_pretrained(model_id, feature=model.config.task_specific_params.get("feature", "sequence-classification"))

# 4. 运行量化（校准在此进行）
quantizer.quantize(
    save_dir=onnx_output_dir,
    quantization_config=qconfig,
    calibration_dataset=calibration_dataset # 提供校准数据
)

# 5. 量化后的 ONNX 模型保存在 'onnx_output_dir' 中
print(f"Quantized ONNX model saved to: {onnx_output_dir}")

# 通过 Optimum 的 ORTModel 类加载并使用量化模型
quantized_model = ORTModelForSequenceClassification.from_pretrained(onnx_output_dir)
# 现在您可以使用“quantized_model”和“tokenizer”进行推理
```

> 这个代码片段演示了使用 Optimum 和 ONNX Runtime 的典型 PTQ 流程。实际操作需要以预期格式准备一个`calibration_dataset`。

Optimum 可以屏蔽底层后端（例如 ONNX Runtime 的量化工具或 Intel 的 Neural Compressor）的许多细节。它提供了一个更高级别的接口，使这个过程更易于使用。

## 加载并运行 Optimum 量化 (quantization)模型

一旦模型使用Optimum进行量化并导出（通常为ONNX格式），您通常会使用Optimum的专用模型类来加载和运行它。

这些类（如`ORTModelForCausalLM`、`ORTModelForSequenceClassification`）旨在与特定的运行时后端（例如ONNX Runtime）配合使用。

```python
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

quantized_model_dir = "./quantized_onnx_model" # 量化模型保存的路径
tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir) # 也加载分词器

# 加载为 ONNX Runtime 优化的量化模型
ort_model = ORTModelForCausalLM.from_pretrained(quantized_model_dir, use_io_binding=True) # IO 绑定可以提升性能

# 使用优化后的模型进行推理
# API 通常与 transformers API 相似
inputs = tokenizer("Generate text with the quantized model:", return_tensors="pt")
outputs = ort_model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

使用这些Optimum类可确保模型使用预期的加速运行时执行，并发挥了先前进行的量化作用。

## 支持的后端和硬件

Optimum 的一个重要优势在于它支持多种执行后端，让您可以针对不同的硬件目标优化模型：

- **ONNX Runtime：** 优秀的跨平台支持（CPU、通过CUDA/TensorRT的GPU、ARM）。通常是默认或用途最广的选择。支持静态/动态量化 (quantization)。
- **Intel OpenVINO：** 为Intel硬件（CPU、集成显卡、VPU）优化。提供强大的PTQ能力。
- **TensorRT：** NVIDIA GPU 的高性能推理 (inference)优化器和运行时。Optimum 可以协助导出与TensorRT兼容的模型。
- **其他：** 对IPEX（PyTorch的Intel扩展）、TensorFlow Lite等后端的支持也属于Optimum生态系统的一部分，具体取决于任务和模型。

量化技术以及由此产生的性能提升会因所选后端和目标硬件的能力（例如原生INT8支持）而有所不同。

> 使用 Hugging Face Optimum 进行训练后量化（PTQ）、导出为 ONNX 格式并使用 Optimum 运行时类运行的典型工作流程。

总而言之，Hugging Face Transformers 提供了量化的基本途径，主要通过 `bitsandbytes` 进行推理时的权重 (weight)优化。Hugging Face Optimum 大幅扩展了这些能力，提供了一个标准化的框架，用于应用各种训练后量化技术，针对 ONNX Runtime 和 OpenVINO 等多个硬件后端，并管理优化后的模型以实现高效部署。对于希望系统地应用和部署量化`transformers`模型到不同硬件平台的开发者而言，它是一个必要的工具。

## 参考资料

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index) — Hugging Face (2024)
  `transformers` 库的官方文档，详细介绍了如何加载 4 位和 8 位量化模型。
- [Hugging Face Optimum Documentation](https://huggingface.co/docs/optimum/index) — Hugging Face (2024)
  `optimum` 库的官方文档，支持模型优化、量化以及在各种硬件上部署。
- [\`bitsandbytes\`: 8-bit Optimizers and Quantization Functions for PyTorch](https://github.com/bitsandbytes-foundation/bitsandbytes) — Tim Dettmers (2023)
  `bitsandbytes` 库的 GitHub 仓库，为 PyTorch 模型提供核心的 4 位和 8 位量化功能。
- [Quantize ONNX Models with ONNX Runtime](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) — Microsoft (2024)
  ONNX Runtime 的官方指南，说明如何对 ONNX 模型进行量化，这是 Hugging Face Optimum 的一个后端选项。

---

[上一节](04-AWQ%20%E6%A0%BC%E5%BC%8F%E8%AF%B4%E6%98%8E.md) · [下一节](06-%E4%BD%BF%E7%94%A8bitsandbytes%E8%BF%9B%E8%A1%8C%E9%87%8F%E5%8C%96.md)
