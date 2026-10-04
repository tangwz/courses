# 实操：使用QLoRA进行微调

来源：[原文](https://apxml.com/zh/courses/fine-tuning-adapting-large-language-models/chapter-4-parameter-efficient-fine-tuning/practice-qlora-fine-tuning)

[返回章节目录](README.md) · [返回课程目录](../README.md)

QLoRA（量化 (quantization)低秩适应）结合了低秩适应策略与模型量化，进一步提升了参数 (parameter)效率。具体来说，它在一个权重 (weight)已量化（通常为4位精度）的基础模型上微调 (fine-tuning)LoRA适配器。这大幅减少了微调所需的内存占用，使得在消费级硬件上微调大得多模型成为可能。

本实践练习将引导您完成使用QLoRA微调大型语言模型。我们将加载一个4位精度的预训练 (pre-training)模型，配置LoRA适配器，准备数据集，并使用Hugging Face的`transformers`和`peft`库执行微调过程。目标是在有效管理内存限制的同时，实现特定任务的适应。

您应熟悉Python、PyTorch以及Hugging Face生态系统的基本组件（`transformers`、`datasets`）。请确保您已安装必要的库，特别是处理量化的`bitsandbytes`。

### 1. 环境设置

首先，让我们安装所需的库。QLoRA高度依赖`bitsandbytes`进行量化 (quantization)，`peft`用于LoRA的实现，`accelerate`用于方便的设备部署和分布式训练工具，以及`transformers`和`datasets`用于模型/数据处理。

```bash
pip install -q transformers datasets accelerate peft bitsandbytes trl
```

- `transformers`：提供对预训练 (pre-training)模型和`Trainer` API的访问。
- `datasets`：便于数据集的加载和处理。
- `accelerate`：简化PyTorch代码在各种硬件配置（CPU、GPU、多GPU）上的运行。
- `peft`：参数 (parameter)高效微调 (fine-tuning)库，包含LoRA、QLoRA等的实现。
- `bitsandbytes`：对QLoRA很重要，支持4位量化和高效矩阵运算。
- `trl`：提供如`SFTTrainer`等工具，简化监督微调过程。

### 2. 加载量化 (quantization)模型和分词 (tokenization)器 (tokenizer)

QLoRA的主要思想是以量化格式（通常是4位）加载基础模型。我们使用`transformers`库中的`BitsAndBytesConfig`来指定加载模型时的量化参数 (parameter)。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 指定预训练模型名称
model_name = "NousResearch/Llama-2-7b-chat-hf" # 示例：Llama-2 7B 对话模型

# 使用BitsAndBytesConfig配置量化
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,                # 启用4位量化
    bnb_4bit_quant_type="nf4",        # 使用NF4（正态浮点4）量化类型
    bnb_4bit_compute_dtype=torch.bfloat16, # 将计算数据类型设置为bfloat16以提高速度
    bnb_4bit_use_double_quant=True,  # 启用嵌套量化以节省更多内存
)

# 使用量化配置加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto", # 自动将模型层分布到可用的GPU/CPU上
    trust_remote_code=True # 信任来自模型中心的模型代码执行（请谨慎使用）
)

# 加载与模型关联的分词器
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# 设置填充标记（如果尚未设置，这是训练的常见要求）
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # 通常对因果语言模型进行右侧填充

print(f"Model loaded: {model_name}")
print(f"Model memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")
```

`BitsAndBytesConfig`中的重要参数：

- `load_in_4bit=True`：此标志激活4位加载。
- `bnb_4bit_quant_type="nf4"`：指定量化数据类型。“nf4”（正态浮点4）通常推荐用于良好性能。另一个选项是“fp4”。
- `bnb_4bit_compute_dtype=torch.bfloat16`：虽然权重 (weight)以4位存储，但计算（例如前向传播期间的矩阵乘法）通常以更高精度格式进行，例如`bfloat16`（或`float16`），以提高稳定性和速度。如果您的硬件支持`bfloat16`（如Ampere GPU及更新版本），通常优先选用它。
- `bnb_4bit_use_double_quant=True`：启用一种嵌套量化技术，其中量化常数也被量化，从而节省更多内存。

`device_map="auto"`参数能智能地将模型层分布到可用的GPU和CPU内存中，使得加载可能无法完全放入单个GPU的模型成为可能。

### 3. 准备数据集

对于本示例，我们使用`databricks-dolly-15k`数据集的一个子集，该数据集包含指令遵循示例。我们会将其格式化为适合微调 (fine-tuning)聊天或指令遵循模型的提示模板。

```python
from datasets import load_dataset

# 加载数据集的一个子集
dataset_name = "databricks/databricks-dolly-15k"
dataset = load_dataset(dataset_name, split="train[:500]") # 为演示目的使用一小部分

# 定义一个函数来格式化提示
def format_prompt(example):
    # 简单的指令遵循格式
    instruction = example.get("instruction", "")
    context = example.get("context", "")
    response = example.get("response", "")

    if context:
        prompt = f"""以下是描述任务的指令，以及提供更多上下文的输入。请编写一个适当完成请求的响应。

### 指令：
{instruction}

### 输入：
{context}

### 响应：
{response}"""
    else:
        prompt = f"""以下是描述任务的指令。请编写一个适当完成请求的响应。

### 指令：
{instruction}

### 响应：
{response}"""
    # 我们需要将此作为名为“text”的列返回，供SFTTrainer使用
    return {"text": prompt}

# 应用格式化函数
formatted_dataset = dataset.map(format_prompt)

print("示例格式化提示：")
print(formatted_dataset[0]['text'])
```

这种格式化会创建一个单一的文本字段，其中包含指令、上下文 (context)（如果可用）以及期望的响应，并用清晰的标记 (token)分隔。在监督微调期间，模型将基于这段组合文本进行训练。

### 4. 配置LoRA

现在，我们使用`peft`库中的`LoraConfig`来配置LoRA参数 (parameter)。我们指定要适应的层、低秩矩阵的秩`r`、缩放因子`alpha`以及其他超参数 (hyperparameter)。

```python
# 为k位训练准备模型（梯度检查点，层归一化缩放）
model = prepare_model_for_kbit_training(model)

# 配置LoRA
lora_config = LoraConfig(
    r=16,                         # 更新矩阵的秩（值越高=参数越多）
    lora_alpha=32,                # LoRA缩放因子（alpha/r控制幅度）
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # 应用LoRA的模块（特定于Llama-2架构）
    lora_dropout=0.05,            # LoRA层的Dropout概率
    bias="none",                  # 不训练偏置参数
    task_type="CAUSAL_LM"         # 指定任务类型
)

# 将LoRA配置应用到量化模型
model = get_peft_model(model, lora_config)

# 打印可训练参数的百分比
model.print_trainable_parameters()
```

- `prepare_model_for_kbit_training(model)`：此辅助函数对模型进行必要修改，以实现稳定的k位训练，例如启用梯度检查点（通过在反向传播 (backpropagation)期间重新计算激活而不是存储它们来节省内存）并确保层归一化 (normalization)层兼容。
- `r=16`：秩的常见起始点。值越高会增加可训练参数的数量和潜在的表达能力，但也会增加计算成本。
- `lora_alpha=32`：通常设置为秩`r`的两倍，但可以调整。它用于缩放学习到的低秩更新。
- `target_modules`：这很重要。它指定了Transformer架构中注入LoRA矩阵的线性层的名称。这些名称取决于特定的模型架构（例如，Llama类模型的`q_proj`、`v_proj`）。您可能需要查看模型结构（`print(model)`）以确定不同模型的正确模块名称。通常会目标设定注意力投影层（`q_proj`、`k_proj`、`v_proj`、`o_proj`）和前馈层（`gate_proj`、`up_proj`、`down_proj`）。
- `lora_dropout`：应用于LoRA权重 (weight)的正则化 (regularization)。
- `bias="none"`：通常，在LoRA设置中不训练偏置 (bias)项。
- `task_type="CAUSAL_LM"`：指定任务类型，确保模型架构得到正确处理（例如，用于顺序生成文本）。

`print_trainable_parameters()`方法突出了PEFT/QLoRA的核心优势。它会显示总参数中只有极小一部分（通常不到1%）实际在训练。



[交互图表：参数对比：完整模型与QLoRA（示例7B模型）](https://apxml.com/zh/courses/fine-tuning-adapting-large-language-models/chapter-4-parameter-efficient-fine-tuning/practice-qlora-fine-tuning#plot-1gwlufb)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "type": "bar",
      "x": [
        "总参数",
        "可训练参数 (QLoRA)"
      ],
      "y": [
        7000000000,
        41943040
      ],
      "marker": {
        "color": [
          "#4c6ef5",
          "#f76707"
        ]
      }
    }
  ],
  "layout": {
    "title": {
      "text": "参数对比：完整模型与QLoRA（示例7B模型）"
    },
    "yaxis": {
      "title": "参数数量（十亿/百万）"
    },
    "xaxis": {
      "title": "参数类型"
    },
    "width": 600,
    "height": 400,
    "showlegend": false
  }
}
```

</details>



> 示例对比显示了在7B参数模型上使用QLoRA时，可训练参数的显著减少。具体数字取决于基础模型和LoRA配置（`r`、`target_modules`）。

### 5. 设置训练参数 (parameter)和训练器

我们使用`transformers`库中的`TrainingArguments`类来定义训练超参数 (hyperparameter)，以及`trl`库中的`SFTTrainer`，它专门为指令遵循等监督微调 (fine-tuning)任务设计。`SFTTrainer`通过内部处理数据格式化和打包来简化过程。

```python
import transformers
from trl import SFTTrainer

# 配置训练参数
training_args = transformers.TrainingArguments(
    output_dir="./qlora_finetuned_model",      # 保存检查点和日志的目录
    per_device_train_batch_size=4,          # 每个GPU的批处理大小
    gradient_accumulation_steps=4,          # 积累4步梯度（有效批处理大小 = 4 * 4 = 16）
    learning_rate=2e-4,                     # 学习率
    num_train_epochs=1,                     # 训练轮次（根据数据集大小调整）
    logging_steps=20,                       # 每20步记录训练指标
    save_steps=50,                          # 每50步保存检查点
    fp16=True,                              # 启用混合精度训练（如果支持，可使用bf16=True）
    optim="paged_adamw_8bit",               # 使用分页AdamW优化器以提高内存效率
    lr_scheduler_type="cosine",             # 学习率调度器类型
    warmup_ratio=0.03,                      # 学习率调度器的预热比例
    report_to="none",                       # 在此示例中禁用向Weights & Biases等服务报告
    # SFTTrainer特定参数
    max_seq_length=1024,                    # 打包的最大序列长度
    dataset_text_field="text",              # 数据集中包含格式化文本的列名
)

# 初始化SFTTrainer
trainer = SFTTrainer(
    model=model,                         # 经过PEFT封装的量化模型
    train_dataset=formatted_dataset,     # 格式化后的训练数据集
    args=training_args,                  # 训练参数
    peft_config=lora_config,             # LoRA配置
    tokenizer=tokenizer,                 # 分词器
    # 可选：您可以添加packing=True以提高效率，但这需要仔细处理序列长度
)
```

重要参数：

- `per_device_train_batch_size` 和 `gradient_accumulation_steps`：这些控制有效批处理大小。由于大型模型存在内存限制，会使用较小的每设备批处理大小，并通过多步梯度累积来模拟更大的批处理。
- `learning_rate`：与完整微调相比，PEFT通常使用相对较高的学习率（例如1e-4到3e-4）。
- `fp16=True`（或`bf16=True`）：启用混合精度训练，减少内存使用并加快计算速度。如果您的硬件（例如Ampere）支持`bf16`，请使用它，因为它通常对LLM训练更稳定。
- `optim="paged_adamw_8bit"`：QLoRA从`bitsandbytes`提供的分页优化器中受益匪浅。这些优化器将优化器状态卸载到CPU内存，进一步减少GPU内存使用。
- `max_seq_length`：`SFTTrainer`特有，定义了分词 (tokenization)后序列的最大长度。更长的序列需要更多内存。
- `dataset_text_field="text"`：告知`SFTTrainer`数据集中哪一列包含用于训练的文本。

### 6. 开始微调 (fine-tuning)

现在，我们可以开始训练过程。

```python
print("开始QLoRA微调...")
trainer.train()
print("训练完成。")
```

训练期间，请监控您的GPU内存使用情况。QLoRA应该使其显著低于完整微调。`transformers`的`Trainer`将输出日志，显示训练损失、学习率和轮次进度。持续时间将取决于数据集大小、硬件和训练配置。

### 7. 保存适配器

训练完成后，我们保存训练好的适配器权重 (weight)。请注意，我们只保存一小部分LoRA参数 (parameter)，而不是整个基础模型。

```python
# 定义保存适配器权重的路径
adapter_output_dir = "./qlora_adapter_weights"

# 保存LoRA适配器权重
trainer.save_model(adapter_output_dir)
# 或者：model.save_pretrained(adapter_output_dir)

print(f"QLoRA适配器权重已保存到：{adapter_output_dir}")
```

这会创建一个包含`adapter_model.bin`文件和`adapter_config.json`的目录。这通常只有几兆字节或几十兆字节大小，这表明了PEFT方法的存储效率。

### 8. 使用微调 (fine-tuning)后的适配器进行推断

为了使用微调后的模型进行推断，我们首先再次加载原始的*量化 (quantization)*基础模型，然后使用`PeftModel`在其上应用已保存的适配器权重 (weight)。

```python
from peft import PeftModel
import time

# 重新加载基础量化模型（如果尚未在内存中）
# 确保使用与训练时相同的量化配置
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)

# 通过将适配器权重合并到基础模型来加载PEFT模型
model_tuned = PeftModel.from_pretrained(base_model, adapter_output_dir)
model_tuned = model_tuned.eval() # 将模型设置为评估模式

# --- 推断示例 ---

# 准备一个示例提示（使用与训练相同的格式，但不包含响应）
instruction = "What is the difference between LoRA and QLoRA?"
prompt_template = f"""以下是描述任务的指令。请编写一个适当完成请求的响应。

### 指令：
{instruction}

### 响应：
"""

# 对输入提示进行分词
inputs = tokenizer(prompt_template, return_tensors="pt").to(model_tuned.device)

print("\n--- 正在生成响应 ---")
start_time = time.time()

# 使用微调模型生成文本
with torch.no_grad(): # 推断时禁用梯度计算
    outputs = model_tuned.generate(
        **inputs,
        max_new_tokens=200,       # 生成的新标记的最大数量
        do_sample=True,           # 启用采样
        temperature=0.7,          # 控制随机性（值越低 = 越确定）
        top_k=50,                 # 采样时考虑前k个标记
        top_p=0.95,               # 使用核采样（累积概率截止）
        eos_token_id=tokenizer.eos_token_id # 遇到EOS标记时停止生成
    )

end_time = time.time()

# 解码生成的标记
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
print(f"\n生成时间：{end_time - start_time:.2f} 秒")
```

这演示了如何将轻量级适配器加载到量化基础模型上进行推断。生成过程使用标准的`transformers`生成方法。将输出质量与基础模型对相同提示的输出进行比较，以评估微调的效果。

### 总结

本实践演练演示了使用QLoRA微调 (fine-tuning)LLM的核心步骤：

1. 使用4位量化 (quantization)加载基础模型（`BitsAndBytesConfig`）。
2. 准备指令遵循数据集。
3. 配置LoRA参数 (parameter)（`LoraConfig`）并将其应用于量化模型（`get_peft_model`）。
4. 使用`SFTTrainer`以及适当的`TrainingArguments`（包括分页优化器和混合精度）进行内存高效训练。
5. 训练后仅保存小型适配器权重 (weight)。
6. 将适配器权重加载到量化基础模型上进行推断。

QLoRA大幅降低了微调大型模型的硬件门槛，通过大幅减少训练期间模型权重和优化器状态的内存需求，使在更易获得的GPU设置上适应成为可能。这使其成为一种为特定下游任务定制大型语言模型的强大且实用的技术。

## 参考资料

- [QLoRA: Efficient Finetuning of Quantized LLMs on Consumer Hardware](https://arxiv.org/abs/2305.14314) — Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2305.14314](https://doi.org/10.48550/arXiv.2305.14314)
  介绍 QLoRA 的原始研究论文，详细说明了其方法，如 4 位 NormalFloat (NF4) 量化和用于大语言模型内存高效微调的分页优化器。
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍低秩适应（LoRA）的基础论文，LoRA 是 QLoRA 构建的参数高效微调核心技术。
- [Parameter-Efficient Fine-tuning (PEFT)](https://huggingface.co/docs/peft/en/index) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face PEFT 库的官方文档，提供了实现 LoRA、QLoRA 和其他 PEFT 方法的实用指南和 API 参考。
- [Supervised Fine-tuning (SFT) Trainer](https://huggingface.co/docs/trl/main/en/sft_trainer) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face TRL 库中 `SFTTrainer` 的官方文档，该工具简化了语言模型的监督微调过程，并在此实践练习中使用。

---

[上一节](09-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BD%BF%E7%94%A8%20LoRA%20%E8%BF%9B%E8%A1%8C%E5%BE%AE%E8%B0%83.md) · [下一节](../05-%E9%AB%98%E9%98%B6%E5%BE%AE%E8%B0%83%E7%AD%96%E7%95%A5/01-%E5%A4%9A%E4%BB%BB%E5%8A%A1%E5%BE%AE%E8%B0%83.md)
