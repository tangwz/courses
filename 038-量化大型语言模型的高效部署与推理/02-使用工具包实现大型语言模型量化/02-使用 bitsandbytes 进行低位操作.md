# 使用 bitsandbytes 进行低位操作

来源：[原文](https://apxml.com/zh/courses/quantized-llm-deployment/chapter-2-implementing-llm-quantization-toolkits/using-bitsandbytes)

[返回章节目录](README.md) · [返回课程目录](../README.md)

低位量化 (quantization)的实际实现需要专门的库。`bitsandbytes` 是这方面的一个主要工具，以其与 Hugging Face 生态系统的高效结合而闻名。这个库提供优化的 CUDA 内核，通过启用4位和8位计算，使得在内存有限的硬件上运行大型语言模型成为可能。

`bitsandbytes` 主要通过处理变压器 (Transformer)模型的两个主要组成部分来加速推理 (inference)：线性层（矩阵乘法）和归一化 (normalization)层。它通过将权重 (weight)存储在4位（INT4、NF4、FP4）或8位（INT8）等较低精度格式中，实现显著的内存节省，同时通常使用混合精度执行实际计算以保持模型准确性。

### `bitsandbytes` 中的思想

理解 `bitsandbytes` 的工作方式需要掌握一些核心思想：

1. **混合精度矩阵乘法：** 该库实现了矩阵乘法（通常表示为 `GEMM`，即通用矩阵乘法），其中输入激活可能采用较高精度格式（如16位 BrainFloat16 `BF16` 或 Float16 `FP16`），而权重 (weight)则以高度压缩的低位格式（例如4位 NormalFloat `NF4`）存储。乘法运算可能会在内部实时解量化 (quantization)权重，或者在可能的情况下使用直接对量化数据进行操作的专用内核。非常重要的一点是，乘法过程中的中间累加通常以较高精度格式（如 `FP32`）进行，以防止在最终结果可能转换回 `BF16` 或 `FP16` 之前出现数值溢出或精度丢失。这种平衡对于保持模型的预测性能至关重要。
2. **分块量化：** `bitsandbytes` 通常采用分块量化，而不是对整个权重张量（逐张量量化）或逐行/逐列（逐通道量化）使用一组单一的量化参数 (parameter)（如比例因子和零点）。权重矩阵被划分为更小的块（例如，每块64个值），每个块独立地使用自己的比例因子（以及可能的零点）进行量化。这使得量化过程能够更有效地适应矩阵不同部分权重大小的变化，与更简单的方案相比，通常能带来更好的准确性，尤其是在4位等非常低的比特率下。

> 分块量化的示意图。原始权重矩阵被划分成块（用不同颜色表示）。每个块使用共享数据类型（例如 NF4）进行量化，得到量化值 (Q)。每个块单独存储一个比例因子 (Scale)，通常采用 FP16 等更高精度格式。

3. **支持的数据类型 (NF4, FP4)：** 如第1章所述，`bitsandbytes` 支持多种低位格式。`NF4`（4位 NormalFloat）尤其值得注意。它基于模型权重在归一化 (normalization)后通常遵循零均值正态分布的假设而设计。`NF4` 数据点通过分位数量化选择，以匹配理论正态分布的分位数，这使其对于此类数据在信息论上具有最佳性。`FP4`（4位 Float）提供了另一种浮点表示形式，同时标准 `INT4`（4位 Integer）也可用。
4. **双重量化 (DQ)：** 为了进一步压缩内存，`bitsandbytes` 引入了双重量化。在分块量化中，每个块存储一个比例因子。对于大型模型，这些比例因子本身可能占用大量内存（例如，对于64个值的块大小，开销是原始参数数量的 $1/64$ 倍，再乘以比例因子类型的大小，通常是 FP16 或 BF16）。双重量化使用二级分块量化方案对这些比例因子进行量化，从而进一步减少内存占用，同时通常对模型准确性影响很小。

### 将 `bitsandbytes` 与 Hugging Face Transformers 配合使用

`bitsandbytes` 的一个主要优点是它能够直接集成到 Hugging Face `Transformers` 库中。加载4位或8位量化 (quantization)模型通常只需向 `from_pretrained` 方法添加特定参数 (parameter)即可。

首先，确保您已安装必要的库：

```bash
pip install torch transformers bitsandbytes accelerate
```

请注意，`bitsandbytes` 通常需要特定的 CUDA 版本和 GPU 计算能力（通常8位优化需要 Maxwell 架构或更新，4位优化需要 Turing 或更新）。有关精确的硬件要求，请查阅 `bitsandbytes` 文档。

**加载8位模型：**

这是 `bitsandbytes` 提供的最简单的量化形式。它对线性层使用分块 INT8 量化。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "NousResearch/Llama-2-7b-chat-hf" # 示例模型

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 加载8位量化模型
model_8bit = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto", # 自动将层分布到 GPU/CPU
    load_in_8bit=True
)

print(f"Model loaded on: {model_8bit.device}")
# 观察与不使用 load_in_8bit=True 加载时的内存使用情况
# print(model_8bit) # 查看模型层——您会看到 Linear8bitLt
```

设置 `load_in_8bit=True` 会指示 `Transformers` 对线性层使用 `bitsandbytes` 的8位内核。此处 `device_map="auto"` 很有用，因为它使用 `accelerate` 库自动将模型权重 (weight)放置到可用设备（GPU、CPU RAM）上，这种放置方式会考虑量化带来的内存节省。如果没有量化，一个7B参数模型在 FP16 精度下可能需要约14GB的显存 (VRAM)，但在 INT8 精度下，这将降至接近7GB，使其可以在更多消费级 GPU 上运行。

**加载4位模型：**

4位量化提供更大的内存节省，与8位相比，需求大致减半。然而，它涉及通过 `BitsAndBytesConfig` 类暴露的更多配置选项。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "NousResearch/Llama-2-7b-chat-hf" # 示例模型

# 配置4位量化设置
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # 使用 NF4 数据类型
    bnb_4bit_use_double_quant=True, # 启用双重量化
    bnb_4bit_compute_dtype=torch.bfloat16 # 矩阵乘法期间的计算类型
)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 加载4位量化配置模型
model_4bit = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=quantization_config
)

print(f"Model loaded on: {model_4bit.device}")
# 观察内存使用情况（7B 模型约为 3.5GB-4GB）
# print(model_4bit) # 查看层——您会看到 Linear4bit
```

在此示例中：

- `load_in_4bit=True` 启用4位模式。
- `bnb_4bit_quant_type` 指定4位格式。常见选项是 `"nf4"`（推荐用于提高准确性）和 `"fp4"`。
- `bnb_4bit_use_double_quant` 为分块比例因子启用内存节省的双重量化技术。
- `bnb_4bit_compute_dtype` 设置矩阵乘法期间用于中间计算的数据类型（例如 `torch.bfloat16` 或 `torch.float16`）。在兼容硬件（Ampere GPU 及更新型号）上，通常更推荐使用 `bfloat16`，因为它在范围和精度之间取得了平衡，可能比 `float16` 更好地保持准确性。

使用4位量化将7B参数模型的内存占用减少到大约3.5-4GB，使更大的模型也能够在消费级硬件上运行。

### 性能影响

虽然 `bitsandbytes` 大幅减少了加载和运行 LLM 所需的内存，但其对推理 (inference)速度的影响更为复杂。

- **内存节省：** 这是主要优点。与16位精度相比，4位和8位量化 (quantization)大约将模型权重 (weight)所需的内存减半或减至四分之一。
- **推理速度：** 您可能预期由于使用较少位数会带来显著加速，但这并非总是如此。与实时解量化权重（特别是分块）或执行混合精度操作相关的开销可能会抵消部分来自更快速低位算术的收益（如果存在专用内核并被充分利用）。通常，推理速度可能与在高端 GPU 上使用 FlashAttention 等框架进行优化的 FP16 推理相似，有时甚至略慢。然而，在内存受限的系统中，如果 FP16 推理不可能实现，`bitsandbytes` 则能够使模型运行。与仅仅让模型能够运行相比，速度变成了次要的好处。
- **准确性权衡：** 量化，特别是到4位，本质上是有损的。虽然 NF4、分块量化和混合精度计算等技术旨在最大程度地减少这种损失，但与原始 FP16 或 BF16 模型相比，模型性能（通过困惑度或下游任务准确性衡量）预计会有一定程度的下降。实际影响因模型、具体任务和所使用的量化参数 (parameter)而异。第3章提供了评估这些权衡的详细方法。

`bitsandbytes` 提供了一种强大且易于使用的方式来应用低位量化，以运行大型语言模型。它与 Hugging Face 的紧密集成使其成为处理内存限制的研究人员和实践者的热门选择。虽然它提供了显著的内存减少，但了解相关的性能特点和潜在的准确性权衡很重要，我们将在后续章节中进一步查看。在接下来的部分中，我们将查看其他工具包，如 AutoGPTQ 和 AutoAWQ，它们实现了针对 LLM 的不同训练后量化算法。

## 参考资料

- [bitsandbytes GitHub Repository](https://github.com/bitsandbytes-foundation/bitsandbytes) — Tim Dettmers and the bitsandbytes contributors (2024)
  Publisher: GitHub
  提供库的源代码、安装说明以及关于低位量化、混合精度操作和CUDA内核功能的详细信息。
- [QLoRA: Efficient Finetuning of Quantized LLMs on Consumer Hardware](https://arxiv.org/abs/2305.14314) — Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023)
  Journal: NeurIPS 2023; DOI: [10.48550/arXiv.2305.14314](https://doi.org/10.48550/arXiv.2305.14314)
  介绍了NF4（4位NormalFloat）量化、双重量化和块级k位量化，这些是bitsandbytes在4位LLM操作中实现的核心概念。
- [Quantization - Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/main/en/main_classes/quantization) — Hugging Face (2024)
  Publisher: Hugging Face
  解释了如何使用BitsAndBytesConfig类和相关参数，通过Hugging Face Transformers库加载使用bitsandbytes量化的模型。

---

[上一节](01-LLM%E9%87%8F%E5%8C%96%E5%BA%93%E6%A6%82%E8%A7%88.md) · [下一节](03-%E4%BD%BF%E7%94%A8%20Hugging%20Face%20Transformers%20%E5%92%8C%20Accelerate%20%E5%AE%9E%E7%8E%B0%E9%87%8F%E5%8C%96.md)
