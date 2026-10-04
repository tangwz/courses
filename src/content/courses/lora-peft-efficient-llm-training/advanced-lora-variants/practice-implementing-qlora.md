---
course: "lora-peft-efficient-llm-training"
chapter: "advanced-lora-variants"
lesson: "practice-implementing-qlora"
sourceId: 4154
sourceUrl: "https://apxml.com/zh/courses/lora-peft-efficient-llm-training/chapter-4-advanced-lora-variants/practice-implementing-qlora"
title: "动手实践：实现 QLoRA"
description: "实践环节：运用 QLoRA 高效微调大型语言模型。"
order: 7
plots: []
sourceHash: "3dc1a0989088521cad29502681d7f659b17bd3ce7379c1bc5abe04f05df3d066"
sourceCorrections: []
---

一个动手指南介绍了如何实现 QLoRA，这是一种整合了 4 比特 NormalFloat ($NF4$) 量化 (quantization)、双重量化和分页优化器等原理的技术。这项实践应用利用流行的 Hugging Face 体系，特别是 `transformers`、`peft` 和 `bitsandbytes` 库。我们假设您已有一个安装了 PyTorch 和这些库的 Python 工作环境。

### 环境准备

首先，请确认您已安装所需的库。您通常需要 `transformers`、`peft`、`accelerate`、`datasets` 和 `bitsandbytes`。

```bash
pip install -q transformers peft accelerate datasets bitsandbytes
```

*注意：`bitsandbytes` 通常需要特定 CUDA 版本。请确保您的安装与您的 GPU 环境兼容。*

### 加载量化 (quantization)后的基础模型

QLoRA 的核心思想是以量化格式加载基础大型语言模型 (LLM)，从而大幅减少其内存占用。这通过在加载模型时使用 `transformers` 库的 `BitsAndBytesConfig` 来实现。

我们将其配置为 4 比特量化 ($NF4$)，启用双重量化，并指定计算数据类型（在兼容硬件上，`bfloat16` 通常可带来更优的性能）。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 定义基础模型 ID（例如，Llama 或 Mistral 变体）
model_id = "meta-llama/Llama-2-7b-hf" # 替换为您希望使用的模型

# 配置 BitsAndBytes 量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                     # 启用 4 比特加载
    bnb_4bit_quant_type="nf4",             # 使用 NF4 量化
    bnb_4bit_compute_dtype=torch.bfloat16, # 设置计算数据类型以提高效率
    bnb_4bit_use_double_quant=True,        # 启用双重量化
)

# 使用指定的量化配置加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto", # 自动将模型分发到可用的 GPU/CPU
    # trust_remote_code=True # 某些模型需要
)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_id)
# 确保为批量处理设置填充（padding）标记
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 可选：训练时禁用缓存使用
model.config.use_cache = False
```

此配置指示 `transformers` 使用 $NF4$ 格式加载 `meta-llama/Llama-2-7b-hf` 的权重 (weight)。前向传播过程中的实际矩阵乘法将使用 `bfloat16` 以提高速度，而权重本身仍以 4 比特存储，从而节省大量 GPU 内存。双重量化进一步优化了量化元数据的内存使用。`device_map="auto"` 负责将模型层高效地放置到可用设备上。

### 配置 LoRA 适配器

在加载量化 (quantization)后的基础模型后，我们现在使用 `peft` 库的 `LoraConfig` 来定义 LoRA 配置。这指定要适配的层、分解的秩 ($r$)、缩放因子 ($\alpha$)、dropout 以及其他参数 (parameter)。

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 为 k 比特训练准备模型（对 QLoRA 非常重要）
model = prepare_model_for_kbit_training(model)

# 定义 LoRA 配置
lora_config = LoraConfig(
    r=16,                            # 更新矩阵的秩
    lora_alpha=32,                   # LoRA 缩放因子
    target_modules=["q_proj", "v_proj"], # 将 LoRA 应用于查询和值投影
    lora_dropout=0.05,               # LoRA 层的 Dropout 概率
    bias="none",                     # 不训练偏置项
    task_type="CAUSAL_LM",           # 指定任务类型
)

# 使用 LoRA 配置将基础模型与 PEFT 模型包装
peft_model = get_peft_model(model, lora_config)

# 打印可训练参数以验证
peft_model.print_trainable_parameters()
# 示例输出：trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622
```

这里的重要步骤包括：

1. `prepare_model_for_kbit_training`：此实用函数为使用 PEFT 适配器进行训练准备量化模型。它处理确保某些层保持更高精度以提高稳定性等任务。
2. `LoraConfig`：我们设置秩 `r`、`lora_alpha`、目标模块（通常是注意力投影，如 `q_proj`、`k_proj`、`v_proj`、`o_proj`，有时是前馈层，如 `gate_proj`、`up_proj`、`down_proj` —— 请查阅模型架构）、dropout 和 `task_type`。
3. `get_peft_model`：此函数将 LoRA 层（由 `lora_config` 定义）注入到基础 `model` 中。
4. `print_trainable_parameters`：这确认总参数中只有一小部分（即 LoRA 适配器）被标记 (token)为可训练，这显示了参数效率。

### 数据集准备

为作演示，我们假设您有一个适合因果语言建模（例如，指令微调 (fine-tuning)）的数据集。我们将使用一个使用 `datasets` 库的占位示例。您需要将其替换为针对您任务的实际数据加载和预处理。

```python
from datasets import load_dataset

# 加载示例数据集（替换为您实际的数据集）
data = load_dataset("Abirate/english_quotes") # 示例数据集
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

# 确保数据集已准备好进行训练（已分词，已格式化）
# ... 在此处添加您的特定数据处理步骤 ...
```

### 配置训练器

我们使用 `transformers.Trainer` 来管理训练循环。我们需要定义 `TrainingArguments`，注意与 QLoRA 相关且可能在内存受限环境中重要的设置。

```python
from transformers import TrainingArguments, Trainer

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./qlora-finetune-results", # 结果保存目录
    per_device_train_batch_size=4,       # 每个 GPU 的批处理大小
    gradient_accumulation_steps=4,       # 在 4 个步骤中累积梯度
    learning_rate=2e-4,                  # 学习率
    logging_steps=10,                    # 每 10 步记录一次日志
    num_train_epochs=1,                  # 训练轮数
    max_steps=-1,                        # 使用 num_train_epochs 而非 max_steps
    save_steps=100,                      # 每 100 步保存检查点
    fp16=False,                          # 禁用 fp16/混合精度（计算数据类型通过 bnb_config 为 bf16）
    bf16=True,                           # 启用 bf16 精度（与 bnb_config 计算数据类型匹配）
    optim="paged_adamw_8bit",            # 使用分页 AdamW 优化器以提高内存效率
    # 其他参数，如评估策略、热身步数等
    # report_to="wandb" # 可选：启用 Weights & Biases 日志记录
)

# 初始化训练器
trainer = Trainer(
    model=peft_model,                      # PEFT 模型（量化基础模型 + LoRA）
    args=training_args,
    train_dataset=data["train"],         # 您预处理过的训练数据
    # eval_dataset=data["validation"],   # 您预处理过的验证数据（可选）
    tokenizer=tokenizer,
    # data_collator=... # 如有需要，指定数据整理器
)
```

QLoRA 中 `TrainingArguments` 的重要配置：

- `bf16=True`：这通常应与 `bnb_4bit_compute_dtype` 匹配，以获得最佳性能和兼容性。如果您的硬件不支持 `bfloat16`，您可以尝试使用 `fp16=True` 并将 `bnb_4bit_compute_dtype` 调整为 `torch.float16`，但如果可用，`bfloat16` 通常是首选。
- `optim="paged_adamw_8bit"`：这激活了 `bitsandbytes` 提供分页版本的 AdamW 优化器，它通过在 GPU 内存不足时将优化器状态卸载到 CPU 内存来进一步减少内存使用。替代选项包括 `paged_adamw_32bit`。
- `per_device_train_batch_size` 和 `gradient_accumulation_steps`：根据您的 GPU 内存调整这些参数 (parameter)。QLoRA 允许使用比在相同硬件上进行全量微调 (fine-tuning)更大的有效批处理大小。

### 运行微调 (fine-tuning)任务

一切准备就绪后，开始训练过程：

```python
# 开始微调
print("开始 QLoRA 微调...")
trainer.train()

# 保存训练好的 LoRA 适配器权重
peft_model.save_pretrained("./qlora-adapter-checkpoint")
print("QLoRA 适配器已保存。")
```

`trainer.train()` 调用执行微调循环。只有 LoRA 适配器权重 (weight)（`A` 和 `B` 矩阵）会得到更新。基础模型权重在其 4 比特量化 (quantization)状态下保持冻结。训练完成后，`save_pretrained` 只保存训练好的适配器权重，这些权重通常非常小（兆字节级别）。

### 可视化：QLoRA 架构概述

> 此图演示了 QLoRA 在前向传播过程中的运作方式。输入 `x` 通过冻结且量化 (quantization)的基础模型权重 (weight) `W0` 以及可训练的低秩适配器 `ΔW = B * A`。这些输出被求和以产生最终输出 `h_final`。训练期间，只有矩阵 `A` 和 `B` 会更新。

这项实践练习演示了如何配置和执行 QLoRA 微调 (fine-tuning)任务。通过量化大型基础模型且只训练小型适配器层，QLoRA 大幅降低了在常用硬件上微调强大 LLM 的障碍。请记住根据您的特定模型和任务要求调整 `model_id`、`LoraConfig` 目标模块、数据集加载和训练参数 (parameter)。

## 参考资料

- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023)
  Journal: arXiv preprint arXiv:2305.14314; DOI: [10.48550/arXiv.2305.14314](https://doi.org/10.48550/arXiv.2305.14314)
  介绍了 QLoRA 方法，包括 4 位 NormalFloat (NF4) 量化、双重量化和分页优化器，这些都是本节的核心内容。
- [PEFT (Parameter-Efficient Fine-Tuning) documentation](https://huggingface.co/docs/peft/en/index) — Hugging Face (2024)
  提供了 Hugging Face PEFT 库的官方文档，包含 `LoraConfig`、`get_peft_model` 以及如何准备模型进行低位训练的详细指南。
