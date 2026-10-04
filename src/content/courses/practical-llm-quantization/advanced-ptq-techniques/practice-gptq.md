---
course: "practical-llm-quantization"
chapter: "advanced-ptq-techniques"
lesson: "practice-gptq"
sourceId: 4635
sourceUrl: "https://apxml.com/zh/courses/practical-llm-quantization/chapter-3-advanced-ptq-techniques/practice-gptq"
title: "动手实践：使用 GPTQ 进行量化"
description: "一个应用 GPTQ 算法来量化大型语言模型的实操环节。"
order: 7
plots: ["plots/4635-0.json"]
sourceHash: "90f95e332f0923745309980f41eb96045f45b47badf2e64fdcd796ef9830e785"
sourceCorrections: []
---

应用 GPTQ 算法量化 (quantization)大型语言模型 (LLM)。此过程演示了如何在减小模型尺寸的同时，尽可能比基本 PTQ 方法更好地保持准确性。常用库可促进此应用，使其易于操作。

我们假定您已有一个可用的 Python 环境，并熟悉使用 `pip` 安装软件包。您还应该对使用 Hugging Face `transformers` 库加载模型和分词 (tokenization)器 (tokenizer)有初步认识。

### 环境搭建

首先，我们需要安装必要的库。Hugging Face 的 `optimum` 库为多种优化技术（包括 GPTQ 集成）提供了便利的封装。我们还需要 `auto-gptq` 用于核心 GPTQ 实现，以及 `transformers`、`datasets` 用于处理校准数据，和 `accelerate` 用于高效的模型加载和运行。

```bash
pip install torch transformers datasets accelerate optimum[exporters]
pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/ # 根据您的 CUDA 版本调整 cu118
```

*注意：请确保您已安装兼容的 PyTorch 版本，最好支持 CUDA 以进行 GPU 加速，因为 GPTQ 计算量较大。*

### 加载预训练 (pre-training)模型

我们将从加载一个预训练模型及其对应的分词 (tokenization)器 (tokenizer)开始。在此示例中，我们使用一个较小的模型，如 `facebook/opt-125m`，以使过程易于控制。在实际情况中，您会将其应用于量化 (quantization)优势更明显的大型模型。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 从 Hugging Face Hub 定义模型 ID
model_id = "facebook/opt-125m"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 加载模型
# 使用 device_map="auto" 来实现 accelerate 库的高效设备分配
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16, # 使用 float16 以加快加载速度并初始占用更少内存
    device_map="auto"
)

print("模型加载成功！")
print(f"模型内存占用: {model.get_memory_footprint() / 1e9:.2f} GB")
```

### 准备校准数据集

GPTQ 需要一个小型数据集，称为校准数据集，用于分析权重 (weight)和激活。该数据集有助于算法在逐层量化 (quantization)过程中做出更优的决定，从而最大限度地减少引入的误差。校准数据理想情况下应代表模型在推理 (inference)时将遇到的文本。

我们将使用流行的 C4（Colossal Clean Crawled Corpus）数据集的一个小部分进行校准。

```python
from datasets import load_dataset

# 加载 C4 数据集的一小部分（例如 128 个样本）
# 对于大型数据集，使用流式方法可以节省内存
calibration_dataset = load_dataset("c4", "en", split="train", streaming=True)

# 选择用于校准的样本数量
n_calibration_samples = 128
calibration_data = []

# 迭代数据流并预处理数据
max_length = 512 # 定义校准样本的最大序列长度
count = 0
for sample in calibration_dataset:
    if count >= n_calibration_samples:
        break
    # 对文本样本进行分词
    tokenized_sample = tokenizer(sample['text'], return_tensors="pt", max_length=max_length, truncation=True)
    # 此示例中仅保留 input_ids，舍弃 attention_mask
    calibration_data.append({"input_ids": tokenized_sample.input_ids, "attention_mask": tokenized_sample.attention_mask})
    count += 1

print(f"为校准准备了 {len(calibration_data)} 个样本。")

# 示例：检查一个校准样本的结构
# print(calibration_data[0])
```

*注意：校准数据集的选择和大小会影响最终量化模型的性能。通常，128-256 个序列长度为 512-2048 的样本就已足够。*

### 应用 GPTQ 算法

现在，我们使用 `optimum` 库与 `auto-gptq` 的接口来执行量化 (quantization)。我们需要定义 `GPTQConfig`，指定诸如目标 `bits`（例如，INT4 为 4）、用于校准的 `dataset` 以及可选的 `group_size` 等参数 (parameter)。分组将层内的权重 (weight)划分为更小的块，允许每个块使用单独的量化参数，这通常能提高准确性，但代价是与逐张量量化相比模型尺寸会略大。常见的 `group_size` 是 128。

```python
from optimum.gptq import GPTQQuantizer, GPTQConfig

# 定义 GPTQ 配置
gptq_config = GPTQConfig(
    bits=4,                  # 目标位宽（例如，4 位）
    dataset=calibration_data,# 之前准备的校准数据集
    tokenizer=tokenizer,     # 与模型关联的分词器
    group_size=128,          # 用于细粒度量化的组大小（可选）
    desc_act=False,          # 激活顺序（对于 GPTQ，False=降序是常见设置）
    model_seqlen=max_length  # 用于校准的序列长度
)

# 初始化量化器
quantizer = GPTQQuantizer.from_config(gptq_config)

# 运行量化过程
print("开始 GPTQ 量化...")
quantizer.quantize_model(model, tokenizer)
print("量化完成！")

# 定义保存量化模型的路径
quantized_model_dir = "opt-125m-gptq-4bit"

# 保存量化模型和分词器
quantizer.save(model, quantized_model_dir)
print(f"量化模型已保存到 {quantized_model_dir}")

# （可选）重新加载分词器并与模型一起保存
tokenizer.save_pretrained(quantized_model_dir)
```

此步骤执行核心 GPTQ 算法：遍历模型层，使用校准数据计算近似的 Hessian 信息，并逐块量化权重以最小化平方误差。此过程可能需要大量时间，特别是对于大型模型和较小的 `group_size` 值。

### 加载和使用量化 (quantization)模型

保存后，GPTQ 量化模型可以使用 `AutoModelForCausalLM.from_pretrained` 方法重新加载，就像标准的 Hugging Face 模型一样。必要的量化参数 (parameter)和配置与模型权重 (weight)一同存储。诸如 `auto-gptq` 之类的库在推理 (inference)期间在后台处理反量化或低精度计算。

```python
# 如果需要，清除内存（尤其是在资源受限的环境中）
# import gc
# del model
# torch.cuda.empty_cache()
# gc.collect()

# 加载量化模型
# 注意：确保 device_map="auto" 和 torch_dtype=torch.float16 以实现最佳加载效果
quantized_model = AutoModelForCausalLM.from_pretrained(
    quantized_model_dir,
    device_map="auto",
    torch_dtype=torch.float16 # 以 float16 格式加载权重，内核处理 int4
)

print("量化模型加载成功！")
print(f"量化模型内存占用: {quantized_model.get_memory_footprint() / 1e9:.2f} GB")

# 示例：使用量化模型运行推理
prompt = "The future of AI is "
inputs = tokenizer(prompt, return_tensors="pt").to(quantized_model.device)

# 生成文本
output_sequences = quantized_model.generate(**inputs, max_new_tokens=50)
generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

print("\n生成的文本:")
print(generated_text)
```

您应该会观察到量化后模型内存占用显著减少。尽管详细的性能评估（速度、准确性、困惑度）将在第 6 章中说明，但运行这样一个简单的生成任务可以快速进行定性检查，确认模型功能正常。



![近似模型内存占用 (OPT-125m)](plots/4635-0.json)



> OPT-125m 模型在 4 位 GPTQ 量化前后（FP16 和 INT4）的近似内存使用情况比较。实际数字取决于实现细节和测量方法。

本次实践练习体现了应用 GPTQ 的核心工作流程。您加载了模型，准备了校准数据，使用 `optimum` 和 `auto-gptq` 配置并执行了 GPTQ 算法，最后保存并加载了量化模型进行推理。此过程实现了显著的模型压缩，使得在内存资源有限的硬件上运行大型模型成为可能，而像 GPTQ 这样的高级技术则有助于相比更简单的 PTQ 方法保持高水平的准确性。尝试不同的校准数据集、`group_size` 和模型将有助于您对其中的权衡取舍形成直观认识。

## 参考资料

- [Accelerate inference with GPTQ](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEhWS5Mlf7er68O9YMbKDcTTfUW78rFA1axauXIJa9oThZItoblGnBhNdHU46HlN2JX07Re6YiMSEEjI95a2lYyyVd25Ul19-BngC9XnX4Pe8e27mAajM9Q2nXSwLy96M7DXK8xtdVTX29zPq7-Gt0D-bqX2eJTkQ==) — Hugging Face (2024)
  Publisher: Hugging Face
  关于如何通过Hugging Face Optimum库高效使用GPTQ的官方指南，与实际操作直接相关。
- [🤗 Transformers Documentation](https://huggingface.co/docs/transformers/index) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face Transformers库的官方文档，对加载和管理模型及分词器至关重要。
