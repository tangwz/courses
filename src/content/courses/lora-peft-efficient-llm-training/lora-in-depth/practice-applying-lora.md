---
course: "lora-peft-efficient-llm-training"
chapter: "lora-in-depth"
lesson: "practice-applying-lora"
sourceId: 4129
sourceUrl: "https://apxml.com/zh/courses/lora-peft-efficient-llm-training/chapter-2-lora-in-depth/practice-applying-lora"
title: "实际操作：应用基础LoRA"
description: "练习会话，使用基础LoRA配置实现和训练语言模型。"
order: 8
plots: []
sourceHash: "43f8e640e3129580d2b0fa074ee00955bd6b280393c6b20d12b60cef2c269e73"
sourceCorrections: []
---

一个分步操作指南，将LoRA应用于预训练 (pre-training)模型进行微调 (fine-tuning)。过程中利用Hugging Face的`transformers`和`peft`库，这些库提供了便捷的抽象功能，用于应用LoRA。该过程包含加载模型、配置LoRA、进行应用，并准备训练等步骤。

本次实践假定您拥有已安装`torch`、`transformers`、`peft`和`datasets`的可用Python环境。

```bash
pip install -q torch transformers datasets peft accelerate bitsandbytes
```

### 1. 设置：加载模型和数据

首先，我们导入必要的库并加载预训练 (pre-training)模型和一个数据集。为进行演示，我们将使用GPT-2（一种常见的因果语言模型）和ELI5数据集的一个小部分，该数据集包含问答对。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_dataset

# 加载模型和分词器
model_name = "gpt2" # 示例：使用'gpt2'或其他合适的因果语言模型
base_model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token # 设置用于批处理的填充标记

# 加载和准备数据集（使用小部分进行演示）
dataset = load_dataset("eli5", split="train_asks[:1000]") # 使用小切片

# 基本预处理：对文本进行分词
def preprocess_function(examples):
    # 为因果语言模型训练连接问题和答案
    texts = [q + " " + a[0] for q, a in zip(examples['title'], examples['answers.text'])]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

print("基础模型已加载:", model_name)
print("数据集已加载并分词。")
```

### 2. 配置LoRA

通过`peft`库应用LoRA的要点在于`LoraConfig`。此对象指定LoRA应如何结合到基础模型中。我们需要定义秩 $r$、缩放因子 $\alpha$ 以及LoRA应作用于哪些层。

如前所述：

- `r`：分解 $BA$ 的秩。较小的 $r$ 意味着更少的可训练参数 (parameter)，但可能限制适应能力。典型值范围为 4 到 64。
- `lora_alpha`：应用于LoRA输出的缩放因子（$\frac{\alpha}{r} BAx$）。它控制适应相对于原始权重 (weight)的幅度。常见做法是将`lora_alpha`设置为等于或两倍于`r`的值。
- `target_modules`：基础模型中模块名称（或正则表达式模式）的列表，LoRA矩阵应在此处替换或增强线性层。对于GPT-2，常见的目标是注意力投影层（例如，`c_attn`）或前馈网络层。您可以检查`base_model.named_modules()`以找到合适的名称。
- `task_type`：指定任务目标，影响PEFT结构可能与模型头部交互的方式。对于GPT-2微调 (fine-tuning)，`TaskType.CAUSAL_LM`是合适的。

```python
# 定义LoRA配置
lora_config = LoraConfig(
    r=16,                             # 秩 r
    lora_alpha=32,                    # 缩放因子 alpha
    target_modules=["c_attn"],        # 将LoRA应用于注意力中的查询、键、值投影
    lora_dropout=0.05,                # LoRA层的Dropout概率
    bias="none",                      # 不训练偏置参数
    task_type=TaskType.CAUSAL_LM      # 因果语言模型的任务类型
)

print("LoRA配置:")
print(lora_config)
```

### 3. 将LoRA应用于模型

定义`LoraConfig`后，使用`get_peft_model`将其应用于基础模型很简单。此函数根据配置修改模型架构，冻结原始权重 (weight)并加入可训练的LoRA适配器。

让我们比较应用LoRA前后可训练参数 (parameter)的数量。

```python
# 计算原始可训练参数
original_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
print(f"原始可训练参数: {original_params:,}")

# 将LoRA配置应用于基础模型
lora_model = get_peft_model(base_model, lora_config)

# 计算LoRA可训练参数
lora_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
print(f"LoRA可训练参数:     {lora_params:,}")

# 计算减少量
reduction = (original_params - lora_params) / original_params * 100
print(f"参数减少:           {reduction:.2f}%")

# 打印可训练模块以供验证
lora_model.print_trainable_parameters()
```

您应该观察到可训练参数数量大幅减少（通常大于99%）。`print_trainable_parameters`的输出确认只有新添加的LoRA组件（`lora_A`和`lora_B`）需要梯度。

> 完全微调 (fine-tuning)与LoRA微调中可训练参数的比较。LoRA通过只训练小型适配器矩阵，同时保持基础模型冻结，大幅减少了参数数量。

### 4. 设置训练循环

现在，我们可以使用`transformers.Trainer`设置标准训练过程。主要区别在于我们传递的是`lora_model`（PEFT修改后的模型），而不是`base_model`。`Trainer`将自动处理优化，只关注可训练的LoRA参数 (parameter)。

```python
# 定义训练参数
output_dir = "./lora_gpt2_eli5_results"
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,             # 为演示目的缩短周期
    per_device_train_batch_size=4,  # 根据您的GPU内存调整
    logging_steps=50,
    save_steps=200,
    learning_rate=2e-4,             # LoRA的典型学习率
    fp16=torch.cuda.is_available(), # 如果可用，使用混合精度
    # 根据需要添加其他参数：weight_decay, warmup_steps等。
)

# 因果语言模型的数据整理器
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 初始化训练器
trainer = Trainer(
    model=lora_model,               # 使用PEFT模型
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("训练器已初始化。开始训练...")
```

### 5. 训练模型

启动训练过程。`Trainer`将处理前向和反向传播 (backpropagation)，只更新LoRA权重 (weight)（$A$和$B$）。基础模型权重保持不变。

```python
# 开始训练
trainer.train()

print("训练完成。")
```

在训练期间，请观察日志。损失应该下降，表明LoRA适配器正在学习调整模型的行为以适应目标任务（在本例中，模仿ELI5的问答风格）。

### 6. 保存LoRA适配器

训练后，您不需要保存整个模型。相反，您只保存训练好的LoRA适配器权重 (weight)。这是LoRA等PEFT方法的主要优点之一——生成的产物非常小。

```python
# 定义保存适配器的路径
adapter_path = f"{output_dir}/final_adapter"

# 保存LoRA适配器权重
lora_model.save_pretrained(adapter_path)

print(f"LoRA适配器已保存到: {adapter_path}")

# 您可以验证保存的适配器目录的小尺寸
# !ls -lh {adapter_path}
```

保存的目录（`adapter_path`）将包含`adapter_model.bin`（LoRA权重）和`adapter_config.json`（使用的LoRA配置）等文件。其大小将以兆字节计，明显小于完整基础模型所需的千兆字节。

### 7. 加载并使用训练好的适配器

要使用微调 (fine-tuning)后的模型进行推理 (inference)，您需要先加载原始基础模型，然后应用已保存的LoRA适配器权重 (weight)。

```python
# 再次加载基础模型（或使用内存中已有的模型）
base_model_reloaded = AutoModelForCausalLM.from_pretrained(model_name)

# 通过合并适配器权重加载PEFT模型
inference_model = PeftModel.from_pretrained(base_model_reloaded, adapter_path)

# 确保模型处于评估模式并在正确的设备上
inference_model.eval()
if torch.cuda.is_available():
    inference_model.to("cuda")

print("基础模型已加载并应用LoRA适配器进行推理。")

# 示例推理（可选）
prompt = "What is the main cause of climate change?"
inputs = tokenizer(prompt, return_tensors="pt")

if torch.cuda.is_available():
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

# 生成文本
with torch.no_grad():
    outputs = inference_model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n--- 示例生成 ---")
print("提示:", prompt)
print("生成内容:", generated_text)
print("------------------------")
```

这就完成了一个应用LoRA的基本周期：配置、封装模型、训练、保存适配器以及加载使用。您通过只训练大型语言模型极小一部分参数 (parameter)，就有效地对其进行了微调，展示了LoRA的效率。接下来的部分和章节将在此根基上进行构建，研习更高级的配置、QLoRA等变体以及评估技术。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍了低秩适配（LoRA）的研究论文，描述了其理论基础、架构以及用于高效微调大型预训练模型的实证结果。
- [Parameter-Efficient Fine-Tuning (PEFT) Documentation](https://huggingface.co/docs/peft/en/index) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face PEFT 库的官方文档，提供了应用参数高效微调方法（包括 LoRA）到各种模型的指南和 API 参考。
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/en/index) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face Transformers 库的官方文档，提供了加载预训练模型、分词器和使用 `Trainer` 类进行微调任务的资源。
