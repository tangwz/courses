---
course: "introduction-to-llm-fine-tuning"
chapter: "full-parameter-fine-tuning"
lesson: "saving-loading-fine-tuned-models"
sourceId: 7279
sourceUrl: "https://apxml.com/zh/courses/introduction-to-llm-fine-tuning/chapter-3-full-parameter-fine-tuning/saving-loading-fine-tuned-models"
title: "保存和加载精调模型"
description: "保存模型检查点和加载完全精调模型进行推理的最佳实践。"
order: 6
plots: []
sourceHash: "1b079b13912c239a1c0dd79e7fa686e921ff6c3f44b409a3b62f9c23f546b17b"
sourceCorrections: []
---

精调过程完成后，且训练损失已收敛，模型的调整后权重 (weight)仅存在于内存中。为了使您的工作持久并可用于推理 (inference)，您必须将此状态保存到磁盘。这不仅涉及保存原始参数 (parameter)；可复现的模型成果也包括模型配置和训练时使用的特定分词 (tokenization)器 (tokenizer)。

### `save_pretrained` 方法

Hugging Face Transformers 库提供了一个简单的方法来保存模型：`save_pretrained()`。此函数将模型的权重 (weight)及其配置文件（`config.json`）序列化到指定目录中。为了得到一个完整且自包含的成果，您还应该将分词 (tokenization)器 (tokenizer)保存到同一目录。这种做法确保任何使用您模型的人都将使用其训练时所用的确切词汇表 (vocabulary)和分词规则加载它。

考虑一个名为 `model` 的精调模型对象及其对应的 `tokenizer`。您可以按如下方式保存它们：

```python
# 假设 'model' 和 'tokenizer' 是您的精调对象
output_dir = "./my_finetuned_model"

# 保存模型权重和配置文件
model.save_pretrained(output_dir)

# 保存分词器的词汇表和配置
tokenizer.save_pretrained(output_dir)

print(f"模型和分词器已保存到 {output_dir}")
```

执行此代码将创建一个包含多个文件的目录。主要组成部分包括模型权重（例如 `pytorch_model.bin` 或 `model.safetensors`）、模型配置（`config.json`）以及分词器文件（例如 `tokenizer.json`、`vocab.json`）。

> 由 `save_pretrained` 创建的目录结构。将模型、配置和分词器一同保存可确保复现性。

### 训练期间管理检查点

完整精调可能是一个耗时且计算成本高的过程。硬件故障或中断可能导致您丢失数小时的进展。为降低此风险，在整个训练过程中保存模型的中间版本（称为检查点）是一种标准做法。

Hugging Face `Trainer` API 通过其 `TrainingArguments` 简化了这一点。您可以将其配置为根据训练步数或每个 epoch 结束时自动保存检查点。

例如，在每个 epoch 结束后保存检查点：

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./training_checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    # 策略可以是 "steps"（步）或 "epoch"（轮）
    save_strategy="epoch", 
    save_total_limit=2, # 可选：只保留最后 2 个检查点
    logging_dir='./logs',
)
```

`Trainer` 将在指定间隔内，在 `./training_checkpoints` 内部创建子目录（例如 `checkpoint-500`、`checkpoint-1000`）。每个子目录都是一个完整、可加载的模型版本，使您能够从中间点恢复训练或评估性能。

### 加载模型进行推理 (inference)

模型保存后，您可以使用对应的 `from_pretrained()` 类方法将其加载回内存进行评估或部署。`AutoModel` 类在此处特别有用，因为它们能自动从指定目录中的 `config.json` 文件推断出正确的模型架构。

加载过程与保存过程对称：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./my_finetuned_model"

# 从目录加载精调模型和分词器
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 如果可用，将模型移动到 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
```

通过将 `from_pretrained()` 指向目录路径，库会自动找到并加载权重 (weight)、配置和分词 (tokenization)器 (tokenizer)文件，重建您保存的确切状态。

### 使用 SafeTensors 进行序列化

您可能在图表中注意到了 `model.safetensors` 文件。这是一种现代序列化格式，旨在作为 Python 默认 `pickle` 格式（`pytorch_model.bin` 中使用）的一种更安全、性能更好的替代方案。Pickled 文件可能被利用来执行任意代码，构成安全风险。

相比之下，`SafeTensors` 仅存储张量数据及其元数据，从而避免了此类漏洞。它还支持更快的模型加载，特别是对于非常大的模型，因为张量可以直接加载而无需中间内存分配。如果已安装 `safetensors` 库（`pip install safetensors`），`transformers` 库最新版本中的 `save_pretrained()` 方法通常默认使用 `safetensors`。这是共享和部署模型的一种推荐做法。

## 参考资料

- [Hugging Face Transformers Documentation: Saving and Loading Models](https://huggingface.co/docs/transformers/main/en/model_sharing#saving-and-loading) — Hugging Face (2024)
  Publisher: Hugging Face
  模型和分词器保存与加载的官方文档，包含`save_pretrained`和`from_pretrained`方法，以及检查点策略概述。
- [SafeTensors GitHub Repository](https://github.com/huggingface/safetensors) — Hugging Face and the safetensors contributors (2022)
  `safetensors`格式的GitHub仓库和文档，说明其在大模型存储中相对于`pickle`格式的安全性和加载速度优势。
- [The Hugging Face Course: Training a Model - Saving your model](https://huggingface.co/docs/transformers/training#saving-your-model) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face官方课程中关于模型保存、加载以及训练期间检查点管理的专门章节。
