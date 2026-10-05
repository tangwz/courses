# 实践：使用LoRA和QLoRA进行微调

来源：[原文](https://apxml.com/zh/courses/llm-compression-acceleration/chapter-5-parameter-efficient-fine-tuning-peft/practice-lora-qlora-finetuning)

[返回章节目录](README.md) · [返回课程目录](../README.md)

本次动手练习将提供LoRA和QLoRA的实际操作指导。它将指导您使用LoRA和QLoRA技术对大型语言模型进行微调 (fine-tuning)。您将直接体验这些方法的配置、训练过程的执行，并观察与完全微调相比的效率提升。我们假设您在具有合适GPU资源的环境中操作，并已安装了所需库，例如`transformers`、`peft`、`accelerate`、`datasets`和`bitsandbytes`。

### 准备工作：环境和基础模型

首先，确保您的环境配置正确。`peft`、`accelerate`和`bitsandbytes`库对于实现LoRA，尤其是QLoRA非常重要。

我们将从加载预训练 (pre-training)的基础模型开始。本次练习，我们选用`meta-llama/Llama-2-7b-hf`或类似大小的Transformer模型。访问某些模型可能需要通过Hugging Face Hub等平台进行身份验证。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import transformers

# 定义基础模型ID
model_id = "meta-llama/Llama-2-7b-hf" # 或另一个合适的模型
# 如果模型需要，请使用身份验证令牌
# from huggingface_hub import login
# login()

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_id)
# 如果缺少填充令牌，则设置填充令牌
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载示例数据集（例如，指令微调）
# 将 'databricks/databricks-dolly-15k' 替换为您目标数据集
data = load_dataset("databricks/databricks-dolly-15k", split="train[:1000]") # 为提高速度使用子集

# 预处理数据
def format_instruction(sample):
    # 根据所选数据集结构调整格式
    return f"""### 指令:
{sample['instruction']}

### 上下文:
{sample['context']}

### 回复:
{sample['response']}
"""

data = data.map(lambda sample: tokenizer(format_instruction(sample), truncation=True, max_length=512, padding="max_length"))

print("环境设置和数据准备完成。")
```

此初始设置加载分词 (tokenization)器 (tokenizer)和示例数据集，为微调 (fine-tuning)过程做准备。特定的数据集和预处理函数（`format_instruction`）应根据您的目标任务（例如，摘要、问答、指令遵循）进行调整。

### 使用LoRA进行微调 (fine-tuning)

LoRA通过在特定层（通常是注意力机制 (attention mechanism)的线性投影）中引入低秩矩阵来调整模型。我们使用`LoraConfig`进行配置。

#### LoRA配置

`LoraConfig`对象定义了LoRA的应用方式：

- `r`: 更新矩阵的秩。较小的`r`意味着更少的训练参数 (parameter)。常见值范围为8到64。
- `lora_alpha`: LoRA更新的缩放因子，通常设置为`2 * r`。
- `target_modules`: 基础模型中将注入LoRA矩阵的模块名称列表（例如，注意力机制中查询和值投影的`['q_proj', 'v_proj']`）。
- `lora_dropout`: 应用于LoRA层的Dropout概率。
- `bias`: 指定偏差的处理方式（'none'、'all'或'lora\_only'）。通常设置为'none'。
- `task_type`: 任务类型（例如，"CAUSAL\_LM"）。

```python
# 加载基础模型（确保VRAM充足）
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto", # 自动分布到可用GPU上
    torch_dtype=torch.float16 # 使用float16以减少内存
)

# 定义LoRA配置
lora_config = LoraConfig(
    r=16,                             # 更新矩阵的秩
    lora_alpha=32,                    # 缩放因子 alpha
    target_modules=["q_proj", "v_proj"], # 将LoRA应用于查询和值投影
    lora_dropout=0.05,                # Dropout概率
    bias="none",                      # 不训练偏差
    task_type="CAUSAL_LM"             # 任务类型
)

# 使用PeftModel封装基础模型
model = get_peft_model(model, lora_config)

# 打印可训练参数的百分比
model.print_trainable_parameters()
# 示例输出：可训练参数：4,194,304 || 所有参数：6,742,609,920 || 可训练%：0.0622
```

`print_trainable_parameters`方法提供关于参数效率的即时反馈。请注意可训练参数占总参数的百分比很小。

#### 训练LoRA Adapter

我们现在可以使用`transformers.Trainer`进行训练。它自动处理PEFT模型，确保只更新LoRA参数。

```python
# 定义训练参数
training_args = transformers.TrainingArguments(
    output_dir="./lora_finetuned_model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1, # 根据数据集大小和收敛情况调整训练轮数
    logging_steps=10,
    save_steps=50,
    fp16=True, # 使用混合精度训练
    # 添加其他相关参数，如评估策略、权重衰减等
)

# 初始化Trainer
trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=data,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# 开始训练
print("开始LoRA微调...")
trainer.train()
print("LoRA微调完成。")

# 保存训练好的LoRA adapter
lora_adapter_path = "./lora_adapter"
model.save_pretrained(lora_adapter_path)
print(f"LoRA adapter 已保存到 {lora_adapter_path}")
```

#### LoRA推理 (inference)

要进行推理，请加载原始基础模型，然后应用保存的LoRA adapter权重 (weight)。

```python
from peft import PeftModel

# 再次加载基础模型（如果尚未在内存中）
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 加载LoRA adapter
model_with_adapter = PeftModel.from_pretrained(base_model, lora_adapter_path)
model_with_adapter.eval() # 将模型设置为评估模式

# 推理示例
prompt = "### 指令:\nLoRA的主要优点是什么？\n\n### 回复:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model_with_adapter.device)

with torch.no_grad():
    outputs = model_with_adapter.generate(**inputs, max_new_tokens=100)

print("生成回复 (LoRA):")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 使用QLoRA进行微调 (fine-tuning)

QLoRA在LoRA的基础上，通过使用`bitsandbytes`将基础模型量化 (quantization)到4位精度。这大幅减少了加载和微调期间的内存占用，使得在消费级硬件上微调更大的模型成为可能。

#### QLoRA配置和模型加载

主要区别在于基础模型的加载方式。我们使用`BitsAndBytesConfig`来指定4位量化参数 (parameter)。

```python
# 定义量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                # 启用4位量化
    bnb_4bit_quant_type="nf4",        # 使用NF4（Normal Float 4）数据类型
    bnb_4bit_compute_dtype=torch.bfloat16, # 计算数据类型以加快训练
    bnb_4bit_use_double_quant=True,   # 使用双重量化以额外节省内存
)

# 使用量化配置加载基础模型
qlora_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto", # 对于量化模型的分布很重要
)

# 准备量化模型进行k位训练
qlora_model = prepare_model_for_kbit_training(qlora_model)

# 定义LoRA配置（可以与之前相同）
qlora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"], # 可能需要根据模型架构和观察到的稳定性进行调整
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 使用PeftModel封装量化模型
qlora_model = get_peft_model(qlora_model, qlora_config)

# 打印可训练参数
qlora_model.print_trainable_parameters()
```

`prepare_model_for_kbit_training`函数执行必要的调整，例如将层归一化 (normalization)和语言模型头部转换为float32以提高稳定性。`LoraConfig`保持相似，但现在应用于4位量化的基础模型。

#### QLoRA训练和推理 (inference)

训练和推理过程与LoRA所用的相同。您可以重用相同的`transformers.Trainer`设置和推理代码。主要优点是在`trainer.train()`调用期间内存消耗大幅降低。

```python
# 重用或重新定义训练参数
qlora_training_args = transformers.TrainingArguments(
    output_dir="./qlora_finetuned_model",
    per_device_train_batch_size=4, # 由于内存使用量较低，可能会增加批量大小
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=50,
    fp16=False, # QLoRA 使用在BitsAndBytesConfig中指定的bf16计算类型
    bf16=True,  # 启用bf16训练
    # 添加其他相关参数
)

# 为QLoRA初始化Trainer
qlora_trainer = transformers.Trainer(
    model=qlora_model,
    args=qlora_training_args,
    train_dataset=data,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# 开始QLoRA训练
print("开始QLoRA微调...")
qlora_trainer.train()
print("QLoRA微调完成。")

# 保存QLoRA adapter
qlora_adapter_path = "./qlora_adapter"
qlora_model.save_pretrained(qlora_adapter_path)
print(f"QLoRA adapter 已保存到 {qlora_adapter_path}")

# 推理遵循与LoRA相同的模式，将adapter加载到量化的基础模型上
# 确保基础模型使用与训练时相同的BitsAndBytesConfig加载
```

### 评估与分析

使用两种方法训练adapter后，进行细致的评估很有必要。

1. **任务表现**：使用任务特定指标（例如，语言模型的困惑度、摘要的ROUGE、分类的准确率）在保留的测试集上评估LoRA和QLoRA微调 (fine-tuning)后的模型。将这些结果与基础模型的表现进行比较，如果可能，再与完全微调的模型进行比较。
2. **资源消耗**：记录LoRA和QLoRA训练期间的峰值GPU内存使用量。注意训练时间差异。QLoRA应展现明显更低的内存需求。
3. **参数 (parameter)数量**：比较两种配置中由`print_trainable_parameters()`报告的可训练参数数量。如果`LoraConfig`相同，它们应该相同，这表明QLoRA的内存节省来自基础模型量化 (quantization)，而非更少的adapter参数。

考虑可视化这些权衡：



[交互图表：PEFT方法对比（示例）](https://apxml.com/zh/courses/llm-compression-acceleration/chapter-5-parameter-efficient-fine-tuning-peft/practice-lora-qlora-finetuning#plot-1a49sz8)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "PEFT方法对比（示例）",
    "xaxis": {
      "title": "方法"
    },
    "yaxis": {
      "title": "峰值GPU显存 (GB)",
      "range": [
        0,
        25
      ]
    },
    "yaxis2": {
      "title": "可训练参数 (%)",
      "overlaying": "y",
      "side": "right",
      "range": [
        0,
        0.1
      ],
      "showgrid": false
    },
    "legend": {
      "x": 0.1,
      "y": 0.9
    }
  },
  "data": [
    {
      "type": "bar",
      "name": "峰值显存 (GB)",
      "x": [
        "完全微调",
        "LoRA",
        "QLoRA"
      ],
      "y": [
        40,
        18,
        10
      ],
      "marker": {
        "color": "#228be6"
      }
    },
    {
      "type": "scatter",
      "name": "可训练参数 (%)",
      "x": [
        "完全微调",
        "LoRA",
        "QLoRA"
      ],
      "y": [
        100,
        0.062,
        0.062
      ],
      "yaxis": "y2",
      "mode": "lines+markers",
      "line": {
        "color": "#f76707"
      },
      "marker": {
        "symbol": "diamond",
        "size": 8
      }
    }
  ]
}
```

</details>



> 示例性对比展示了完全微调、LoRA和QLoRA的潜在显存 (VRAM)使用量和可训练参数百分比。实际值很大程度上取决于模型大小、硬件和批量大小。请注意，为了突出显示，此处完全微调的显存使用量超出了图表范围。

### 讨论与考虑

本次实践练习说明了LoRA和QLoRA在高效LLM微调 (fine-tuning)中的实现。

- **LoRA**：与完全微调相比，提供了显著的参数 (parameter)效率，减少了计算和存储需求，同时通常保持高任务性能。
- **QLoRA**：通过量化 (quantization)基础模型，进一步大幅节省内存，使得在可用硬件上微调非常大的模型成为可能。然而，4位量化在某些复杂任务上可能引入轻微的性能下降。需要细致评估。

在它们之间进行选择时，请考虑可用的硬件资源和所需的性能保真度。如果内存是主要限制，QLoRA是一个很好的选择。如果需要最大性能且内存允许，标准的LoRA（如果资源允许，甚至完全微调）可能是更优选择。超参数 (hyperparameter)调整，特别是`r`、`lora_alpha`和学习率，对于两种技术优化结果仍然很重要。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: arXiv preprint arXiv:2106.09685; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍了LoRA方法，通过注入低秩分解矩阵，实现大型预训练模型的效率微调。
- [Parameter-Efficient Fine-tuning (PEFT)](https://huggingface.co/docs/peft/en/index) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face PEFT库的官方文档，为实现LoRA、QLoRA和其他PEFT方法提供实用指导和示例。
- [🤗 Transformers](https://huggingface.co/docs/transformers/en/index) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face Transformers库的官方文档，是加载模型、分词器和设置LLM训练流程的基础。

---

[上一节](07-PEFT%E6%8A%80%E6%9C%AF%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90.md) · [下一节](../06-%E7%A1%AC%E4%BB%B6%E5%8A%A0%E9%80%9F%E4%B8%8E%E7%B3%BB%E7%BB%9F%E4%BC%98%E5%8C%96/01-%E5%B0%86LLM%E6%93%8D%E4%BD%9C%E6%98%A0%E5%B0%84%E5%88%B0%E7%A1%AC%E4%BB%B6%E6%9E%B6%E6%9E%84.md)
