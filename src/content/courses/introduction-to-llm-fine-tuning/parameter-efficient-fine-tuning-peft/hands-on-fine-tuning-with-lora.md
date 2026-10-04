---
course: "introduction-to-llm-fine-tuning"
chapter: "parameter-efficient-fine-tuning-peft"
lesson: "hands-on-fine-tuning-with-lora"
sourceId: 7287
sourceUrl: "https://apxml.com/zh/courses/introduction-to-llm-fine-tuning/chapter-4-parameter-efficient-fine-tuning-peft/hands-on-fine-tuning-with-lora"
title: "动手实践：使用 LoRA 进行微调"
description: "一个循序渐进的编码会话，您将使用 LoRA 和 PEFT 库微调大型语言模型。"
order: 7
plots: ["plots/7287-0.json"]
sourceHash: "86e99081163a1c1b4a10caa0e920fe34c26eaa49a75d0a4d4f96daee83d0b2f5"
sourceCorrections: []
---

使用低秩适应（LoRA）微调 (fine-tuning)大型语言模型，涉及将参数 (parameter)高效微调（PEFT）技术应用于实际操作。Hugging Face 生态系统，包括 `transformers`、`peft` 和 `accelerate` 库，用于高效地执行此任务。其目标是让预训练 (pre-training)模型更好地遵循指令，同时在普通消费级 GPU 的内存限制下运行。

### 1. 环境搭建

首先，请确保已安装所需的库。我们将需要 `transformers` 来处理模型，`peft` 用于 LoRA 的实现，`accelerate` 用于简化在任何基础设施上运行 PyTorch，`datasets` 用于数据加载，以及 `bitsandbytes` 来启用模型量化 (quantization)。

```bash
pip install transformers peft accelerate datasets bitsandbytes
```

这些库构成了我们微调 (fine-tuning)流程的基本构成，它们提供了加载模型、应用适配器和管理训练过程的工具。

### 2. 加载模型和分词 (tokenization)器 (tokenizer)

PEFT 的起点是一个能用的基础模型。在本次练习中，我们将使用一个中等大小的模型，并使用 `bitsandbytes` 库以 4 位精度加载它。这一量化 (quantization)步骤显著减少了模型所需的 GPU 内存，使得在 VRAM 有限的硬件上微调 (fine-tuning)数十亿参数 (parameter)的模型成为可能。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "mistralai/Mistral-7B-Instruct-v0.1"

# Configure quantization to load the model in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load the model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True
)
model.config.use_cache = False # Disable cache for training

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token # Set padding token
```

通过设置 `load_in_4bit=True`，我们指示 `transformers` 在加载模型时应用量化。基础模型的权重 (weight)现在以内存高效的 4 位格式冻结，已准备好附加 LoRA 适配器。

### 3. 准备数据集

对于指令微调 (fine-tuning)，模型学习根据指令生成响应。我们将使用 `databricks/dolly-v2-12k` 数据集的一个子集，该数据集包含指令-响应对。我们需要将每个数据点格式化为单个文本字符串，通常使用提示模板。此模板构建输入，将指令与模型响应的空间明确分开。

让我们定义一个简单的提示模板和格式化函数。

```python
from datasets import load_dataset

# Load a sample of the dataset
dataset = load_dataset("databricks/dolly-v2-12k", split="train[:1000]")

# Function to format the prompts
def format_prompt(sample):
    instruction = sample["instruction"]
    context = sample["context"]
    response = sample["response"]
    if context:
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{context}

### Response:
{response}"""
    else:
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}"""
    return {"text": prompt}

# Apply the formatting
formatted_dataset = dataset.map(format_prompt)
```

这些格式化的文本将在训练期间被分词 (tokenization)并输入到模型中。模型的任务是学习根据前面的指令和上下文 (context)生成 `### Response:` 部分的文本。

### 4. 配置 LoRA

在这里，我们使用 `peft` 库中的 `LoraConfig` 类定义 LoRA 特定的配置。在此，我们指定要调整模型的哪些部分以及如何调整。

- `r`：低秩矩阵的秩。较小的 `r` 意味着更少的参数 (parameter)可训练，训练速度更快，但表达能力可能较弱。常见范围是 8 到 64。
- `lora_alpha`：LoRA 更新的缩放因子，计算方式为 `alpha / r`。它充当适配器的学习率。常见做法是将 `lora_alpha` 设置为 `r` 值的两倍。
- `target_modules`：这是一个我们希望应用 LoRA 的模块名称列表。对于 Transformer 模型，这些通常是注意力块的投影（`q_proj`、`k_proj`、`v_proj`、`o_proj`）。
- `lora_dropout`：LoRA 层的一个 dropout 概率，用于防止过拟合 (overfitting)。
- `task_type`：指定任务类型，对于我们的生成文本模型，其为 `CAUSAL_LM`。

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"], # Target attention blocks
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Wrap the base model with LoRA config
peft_model = get_peft_model(model, lora_config)
```

创建配置后，我们使用 `get_peft_model` 来包装我们的量化 (quantization)基础模型。此函数会查找 `target_modules` 中指定的模块并注入 LoRA 适配器。

让我们验证一下可训练参数的显著减少。

```python
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"Trainable params: {trainable_params} || All params: {all_param} || "
        f"Trainable %: {100 * trainable_params / all_param:.2f}"
    )

print("Trainable parameters with LoRA:")
print_trainable_parameters(peft_model)
```

您应该会看到一个输出，表明可训练参数仅占模型总参数的一小部分，通常小于 1%。这就是 LoRA 的核心效率所在。



![可训练参数：完全微调与 LoRA](plots/7287-0.json)



> LoRA 微调 (fine-tuning)期间更新的参数数量比完全微调小几个数量级，而模型的总参数数量保持不变。

### 5. 训练模型

我们的 PEFT 模型准备就绪后，可以使用 `transformers.Trainer` 进行训练。我们首先定义 `TrainingArguments` 来指定学习率、训练轮数和批次大小等超参数 (parameter) (hyperparameter)。

```python
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

# Define training arguments
training_args = TrainingArguments(
    output_dir="./mistral-7b-lora-dolly",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    fp16=True, # Use mixed precision
    save_total_limit=2,
    overwrite_output_dir=True,
)

# Create the Trainer
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=formatted_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Start training
trainer.train()
```

`Trainer` 处理整个训练循环，包括梯度更新、日志记录和保存检查点。由于我们只更新小的 LoRA 适配器而不是庞大的基础模型，因此优化器状态和梯度所需的内存极少，这使得该过程可以在单个 GPU 上运行。

### 6. 执行推理 (inference)

训练完成后，`peft_model` 对象现在包含基础模型和训练好的 LoRA 适配器。要生成文本，您可以使用标准的 `generate` 方法。PEFT 库确保适配器权重 (weight)在前向传递期间自动应用。

让我们用一个新的指令来测试我们微调 (fine-tuning)后的模型。

```python
# Create a prompt for inference
prompt_text = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Explain the difference between supervised and unsupervised machine learning in simple terms.

### Response:
"""

# Tokenize the input
inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")

# Generate a response
output = peft_model.generate(
    **inputs, 
    max_new_tokens=150, 
    eos_token_id=tokenizer.eos_token_id
)

# Decode and print the response
response_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(response_text)
```

模型的输出现在应该遵循它在微调期间学到的指令-响应格式。将此与原始基础模型的输出进行比较；您应该会看到它在遵循指令风格方面的能力有显著改善。

本次动手实践环节展示了使用 LoRA 微调大型模型的端到端流程。您已成功加载了量化 (quantization)模型，准备了数据集，配置并应用了 LoRA 适配器，并执行了训练循环。下一章将介绍如何严格评估新微调模型的性能并为部署做好准备。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: arXiv preprint arXiv:2106.09685; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍了低秩适应（LoRA）方法，一种参数高效的大型语言模型微调技术。
- [Parameter-Efficient Fine-Tuning (PEFT)](https://huggingface.co/docs/peft/en/index) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face PEFT库的官方文档，详细说明了如何实现和使用LoRA进行微调。
- [Quantize a model for inference and training with bitsandbytes](https://huggingface.co/docs/transformers/main/en/quantization) — Hugging Face (2024)
  Publisher: Hugging Face
  关于在Hugging Face Transformers中使用`bitsandbytes`库进行4位模型量化的指南。
- [Dolly: Introducing the World's First Truly Open Instruction-Following LLM](https://www.databricks.com/blog/2023/04/12/dolly-v2-databricks-community-edition.html) — Mike Conover, et al. (2023)
  Journal: Databricks Blog; Publisher: Databricks
  介绍了Dolly-v2数据集和模型，提供了所用指令遵循数据的背景和创建细节。
