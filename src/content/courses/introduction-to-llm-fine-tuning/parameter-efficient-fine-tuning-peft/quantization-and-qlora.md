---
course: "introduction-to-llm-fine-tuning"
chapter: "parameter-efficient-fine-tuning-peft"
lesson: "quantization-and-qlora"
sourceId: 7284
sourceUrl: "https://apxml.com/zh/courses/introduction-to-llm-fine-tuning/chapter-4-parameter-efficient-fine-tuning-peft/quantization-and-qlora"
title: "量化及其对微调的影响 (QLoRA)"
description: "了解模型量化如何减少内存使用，并支持 QLoRA 等技术，从而实现更高效的微调。"
order: 4
plots: []
sourceHash: "4eb029c7a8e2d7f6877f10b2aa00eabe5048f484e6bb5765548f9587a1e353fa"
sourceCorrections: []
---

大型语言模型 (LLM) 对 GPU 内存有很高的要求，通常仅加载完整且冻结的基础模型就需要大量显存 (VRAM)。例如，一个使用 16 位精度 (`bfloat16`) 的 70 亿参数 (parameter)模型，在训练开始前就需要大约 14 GB 的显存。尽管 LoRA 等技术可以减少*可训练*参数的数量，但基础模型本身的内存占用仍然是一个显著的瓶颈。量化 (quantization)提供了一种缩小基础模型本身的方案，直接解决这一内存问题。

### 量化 (quantization)原理

深度学习 (deep learning)中的量化，是降低模型权重 (weight)数值精度的过程。大多数模型使用 32 位浮点数 (`float32`) 或 16 位脑浮点数 (`bfloat16`) 进行训练。这些格式为表示权重和梯度提供了宽广的数值范围和高准确度。

量化将这些高精度值映射到较低精度的数据类型，例如 8 位整数 (`int8`) 甚至 4 位数字。可以将其想象成减少高分辨率照片中的颜色数量。整体图像仍可辨认，但文件大小会大幅缩小。通过用更少的比特表示每个权重，模型的总内存需求会急剧降低。

- 一个 **32 位 (`float32`)** 权重使用 4 字节。
- 一个 **16 位 (`bfloat16`)** 权重使用 2 字节。
- 一个 **8 位 (`int8`)** 权重使用 1 字节。
- 一个 **4 位** 权重仅使用 0.5 字节。

以 4 位量化权重而不是 16 位权重加载模型，可以将其内存占用减少四倍。然而，这种效率也带来了一个问题。降低的精度可能导致信息丢失。如果处理不当，这种信息丢失会降低模型性能，而且更进一步来说，还会破坏训练过程的稳定性，因为低精度格式无法精确表示微调 (fine-tuning)所需的微小梯度更新。

### QLoRA 介绍：在量化 (quantization)模型上进行微调 (fine-tuning)

QLoRA (Quantized Low-Rank Adaptation) 是一种突破性的技术，它成功地将量化带来的内存节省与 LoRA 的训练稳定性结合起来。它使得你可以对基础权重 (weight)已量化到极低精度（例如 4 位）的模型进行微调。

QLoRA 方法通过巧妙的技术组合实现其功能：

1. **4 位 NormalFloat (NF4) 量化：** QLoRA 的核心是将冻结的、预训练 (pre-training)的模型权重量化为一种特殊的 4 位数据类型，称为 NormalFloat4 (NF4)。这种数据类型在理论上对于正态分布的权重是最佳的，而正态分布是神经网络 (neural network)权重的一个常见特点。与更简单的 4 位量化方案相比，这种专用格式最大限度地减少了信息损失。
2. **双重量化：** 为了节省更多内存，QLoRA 使用了一种称为双重量化的技术。这包括对量化常数本身进行量化，从而额外节省了约每参数 (parameter) 0.5 比特的内存。
3. **分页优化器：** 为了处理训练期间可能出现的内存峰值，尤其是在长序列长度下，QLoRA 采用 NVIDIA 的统一内存功能在 CPU 和 GPU 之间“分页”优化器状态。这可以防止在梯度检查点期间出现内存不足错误。
4. **高精度计算：** 这是该过程中的主要一环。基础模型权重虽然存储为 4 位，但在需要进行前向或反向传播 (backpropagation)时，它们会按需**反量化**为更高精度的格式（例如 `bfloat16`）。只有 LoRA 适配器权重会被训练，它们始终保持较高的 `bfloat16` 精度。

这意味着实际的矩阵乘法在 16 位精度下进行，从而保持了性能和训练的稳定性。梯度只为 16 位 LoRA 权重计算，这样可以正确地累积微小更新。然而，内存密集型基础模型在整个过程中都以 4 位精度存储在显存 (VRAM)中。

> QLoRA 过程。大型基础模型以内存高效的 4 位格式存储。在计算时，其权重会暂时反量化为 `bfloat16`，并与可训练的、更高精度的 LoRA 适配器结合。梯度仅针对 LoRA 权重计算。

### Hugging Face 实践

Hugging Face 生态系统，特别是 `transformers`、`peft` 和 `bitsandbytes` 库，使得 QLoRA 的实现变得简单明了。你可以通过创建一个 `BitsAndBytesConfig` 对象来加载一个 4 位量化 (quantization)模型。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "mistralai/Mistral-7B-Instruct-v0.2"

# 配置 4 位量化
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 使用指定的量化配置加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto" # 自动将模型映射到可用设备
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
```

在此代码片段中：

- `load_in_4bit=True` 是启用 4 位加载的主要开关。
- `bnb_4bit_quant_type="nf4"` 指定使用 NormalFloat4 数据类型，这对于获得最佳性能是推荐的。
- `bnb_4bit_compute_dtype=torch.bfloat16` 告诉库在按需反量化和计算时使用 `bfloat16`，这对于现代 GPU 来说是理想的。

只需添加此配置，你就可以在 5 GB 以下的显存 (VRAM)中加载像 Mistral-7B 这样的模型，而该模型通常在 `bfloat16` 精度下需要超过 14 GB 的显存。这使得在单个消费级 GPU 上（例如具有 24 GB 显存的 NVIDIA RTX 3090 或 4090）对 70 亿参数 (parameter)模型进行微调 (fine-tuning)成为可能。QLoRA 有效地使大型模型的微调变得易于实现，将其从大型工业实验室的范畴带给个人开发者和小型研究团队。

## 参考资料

- [QLoRA: Efficient Finetuning of Quantized LLMs on Consumer GPUs](https://arxiv.org/abs/2305.14314) — Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023)
  Journal: arXiv preprint arXiv:2305.14314; DOI: [10.48550/arXiv.2305.14314](https://doi.org/10.48550/arXiv.2305.14314)
  介绍了QLoRA方法，该方法通过4位量化实现大型语言模型的内存高效微调。
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: arXiv preprint arXiv:2106.09685; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍了用于预训练模型参数高效适应的LoRA基础技术。
- [bitsandbytes GitHub Repository](https://github.com/bitsandbytes-foundation/bitsandbytes) — Tim Dettmers and collaborators (2024)
  bitsandbytes库的官方存储库，它提供了QLoRA中使用的4位量化实现。
- [Parameter-Efficient Fine-Tuning (PEFT) library documentation](https://huggingface.co/docs/peft/index) — Hugging Face (2024)
  Hugging Face PEFT库的官方文档，提供了使用包括QLoRA在内的参数高效微调方法的实用指南。
