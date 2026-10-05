# 动手实践：使用 LoRA 进行微调

来源：[原文](https://apxml.com/zh/courses/fine-tuning-adapting-large-language-models/chapter-4-parameter-efficient-fine-tuning/practice-lora-fine-tuning)

[返回章节目录](README.md) · [返回课程目录](../README.md)

低秩适配 (LoRA) 提供了一种对预训练 (pre-training) Transformer 模型进行下游任务微调 (fine-tuning)的实用方法。使用 Hugging Face 的 `peft` 库，该方法能显著减少可训练参数 (parameter)的数量，与完全微调相比，使训练过程更快、所需的内存更少，同时在许多任务中性能损失不大。

我们将逐步介绍这些重要步骤：环境配置、数据准备、LoRA 配置、适配器训练，以及使用微调后的模型进行推理 (inference)。

### 1. 设置与环境

首先，请确保已安装必要的库。我们将主要使用 `transformers` 处理基础模型和训练工具，`peft` 用于实现 LoRA，`datasets` 用于数据处理，以及 `accelerate` 来简化 PyTorch 代码在任何基础架构上的运行。

```bash
pip install transformers datasets peft accelerate torch
```

现在，让我们导入所需的模块并定义基础模型检查点。在本示例中，我们将使用一个相对较小的序列到序列模型，例如 `google/flan-t5-small`，并对其进行文本摘要任务的微调 (fine-tuning)。使用较小的模型可以使过程更快，即使没有高端 GPU 也能进行操作。

```python
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# 定义基础模型检查点
model_checkpoint = "google/flan-t5-small"

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# 加载基础模型。我们使用 device_map="auto" 来利用 accelerate 在不同设备上放置层。
# 我们还以 8 位模式加载，以进一步节省内存，并与 LoRA 兼容。
# 注意：8 位加载是可选的，但对于大型模型很有用。
# 如果不使用 8 位模式，请移除 load_in_8bit 和 prepare_model_for_kbit_training
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, load_in_8bit=True, device_map="auto")

# 准备模型进行 k 位训练（如果使用量化）
# 当以 8 位或 4 位模式加载模型时，此步骤是必需的
model = prepare_model_for_kbit_training(model)
```

### 2. 数据准备

我们需要一个适合我们所选任务（文本摘要）的数据集。包含对话及其摘要的 `samsum` 数据集是一个不错的选择。我们将使用 `datasets` 库加载并预处理它。为了效率，本次演示我们只使用数据集的一小部分。

```python
# 加载数据集
dataset_name = "samsum"
dataset = load_dataset(dataset_name, split="train[:1%]") # 演示仅使用 1%
dataset = dataset.train_test_split(test_size=0.1) # 创建训练/测试集

print(f"训练集大小: {len(dataset['train'])}")
print(f"测试集大小: {len(dataset['test'])}")
# 示例：训练集大小: 132
# 示例：测试集大小: 15

# 预处理函数
max_input_length = 512
max_target_length = 128

def preprocess_function(examples):
    # 为 T5 模型添加前缀
    inputs = ["summarize: " + doc for doc in examples["dialogue"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")

    # 为目标设置分词器
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    # 将标签中的 tokenizer.pad_token_id 替换为 -100，以便在损失计算中忽略填充
    model_inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in model_inputs["labels"]
    ]
    return model_inputs

# 应用预处理
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 移除训练不需要的列
tokenized_datasets = tokenized_datasets.remove_columns(["id", "dialogue", "summary"])

print(f"分词后数据集中包含的列: {tokenized_datasets['train'].column_names}")
# 示例：分词后数据集中包含的列: ['input_ids', 'attention_mask', 'labels']
```

我们还需要一个数据收集器来处理批次创建时的动态填充。`DataCollatorForSeq2Seq` 适用于序列到序列任务。

```python
# 创建数据收集器
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100, # 重要：确保标签正确填充
    pad_to_multiple_of=8 # 可选：优化硬件使用
)
```

### 3. LoRA 配置

在这里，我们定义 LoRA 将如何修改基础模型。我们使用 `peft` 库中的 `LoraConfig` 类。

- `r`: 低秩矩阵 ($A$ 和 $B$) 的秩。较小的 `r` 意味着更少的可训练参数 (parameter)，但可能捕获的任务特定信息也较少。常见取值范围为 4 到 32。
- `lora_alpha`: LoRA 更新的缩放因子。它通常设置为 `r` 或 `2*r`。更新按 $\frac{\alpha}{r}$ 比例缩放。
- `target_modules`: 基础模型中 LoRA 矩阵将被注入的模块名称列表。对于 T5 模型，目标设定为自注意力 (self-attention)机制 (attention mechanism)中的查询 (`q`) 和值 (`v`) 投影是常规做法。你可以通过检查 `model.named_modules()` 找到这些名称。
- `lora_dropout`: 应用于 LoRA 层的 Dropout 比率。
- `bias`: 指定要训练的偏置 (bias)。`"none"` 很常见，表示冻结所有原始偏置且不添加新的偏置。
- `task_type`: 定义模型类型和任务。对于 `flan-t5`，它是 `TaskType.SEQ_2_SEQ_LM`。

```python
# 定义 LoRA 配置
lora_config = LoraConfig(
    r=16, # 更新矩阵的秩
    lora_alpha=32, # 缩放因子
    target_modules=["q", "v"], # 将 LoRA 应用于查询和值投影
    lora_dropout=0.05, # Dropout 概率
    bias="none", # 不训练偏置
    task_type=TaskType.SEQ_2_SEQ_LM # 序列到序列模型的任务类型
)
```

### 4. 使用 PEFT 包装模型

现在，我们使用 `get_peft_model` 将 LoRA 配置应用于我们的基础模型。

```python
# 获取 PEFT 模型
peft_model = get_peft_model(model, lora_config)

# 打印可训练参数的数量
peft_model.print_trainable_parameters()
# 示例输出：可训练参数: 884,736 || 所有参数: 77,822,464 || 可训练百分比: 1.13685...
```

注意到显著的减少了吗！我们只训练了总参数 (parameter)的约 1%。与更新 `flan-t5-small` 的全部 7700 万参数相比，这极大地降低了内存需求并加快了训练速度。

> 图示 LoRA 如何在已冻结的预训练 (pre-training)权重 (weight)矩阵 (W) 旁边注入可训练的低秩矩阵 (A 和 B)。训练期间仅更新 A 和 B。

### 5. 训练 LoRA 适配器

我们使用 `transformers` 库中标准的 `Trainer`。配置与完全微调 (fine-tuning)几乎相同，但 `Trainer` 会自动处理 PEFT 模型，仅更新 LoRA 参数 (parameter)。

```python
# 定义训练参数
output_dir = "flan-t5-small-samsum-lora"
training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True, # 自动寻找合适的批次大小
    learning_rate=1e-3, # LoRA 通常使用更高的学习率
    num_train_epochs=3, # 训练轮数
    logging_strategy="epoch", # 每轮记录指标
    save_strategy="epoch", # 每轮保存检查点
    # evaluation_strategy="epoch", # 如果评估数据可用，每轮评估一次
    report_to="none", # 在此示例中禁用向 wandb/tensorboard 报告
    # 如果支持，使用 fp16 以加快训练
    # fp16=torch.cuda.is_available(),
)

# 创建 Trainer 实例
trainer = Trainer(
    model=peft_model, # 传入 PEFT 模型
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"], # 可选：传入评估数据集
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# 显式设置 LoRA 层为可训练（有时需要）
peft_model.config.use_cache = False # 训练时禁用缓存

print("开始 LoRA 训练...")
trainer.train()
print("训练完成。")
```

与完全微调 `flan-t5-small` 模型相比，训练应该会显著加快，并且需要更少的 GPU 内存。

### 6. 保存适配器

训练完成后，我们保存训练好的 LoRA 适配器权重 (weight)。重要的一点是，这只保存适配器参数 (parameter)（每个目标模块的矩阵 A 和 B），而不是整个基础模型。这使得保存的文件非常小。

```python
# 定义适配器保存路径
adapter_path = f"{output_dir}/final_adapter"

# 保存适配器权重
peft_model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path) # 同时保存分词器和适配器

print(f"LoRA 适配器已保存至: {adapter_path}")

# 你可以检查保存的适配器大小——它应该相对较小（MB 级别）。
# 例如，使用: !ls -lh {adapter_path}
```

### 7. 使用 LoRA 适配器进行推理 (inference)

要使用微调 (fine-tuning)后的模型进行推理，我们首先加载原始的基础模型，然后在其之上加载 LoRA 适配器权重 (weight)。

```python
from peft import PeftModel, PeftConfig

# 再次加载基础模型（如果尚未在内存中）
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# 加载带有已保存适配器的 PEFT 模型
lora_model = PeftModel.from_pretrained(base_model, adapter_path)
lora_model = lora_model.to("cuda" if torch.cuda.is_available() else "cpu") # 确保模型在正确的设备上
lora_model.eval() # 将模型设置为评估模式

# 从测试集（或任何新对话）准备一个示例输入
sample_idx = 5
dialogue = dataset['test'][sample_idx]['dialogue']
reference_summary = dataset['test'][sample_idx]['summary']

input_text = "summarize: " + dialogue
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(lora_model.device)

print("对话:")
print(dialogue)
print("\n参考摘要:")
print(reference_summary)

# 使用 LoRA 模型生成摘要
with torch.no_grad():
    outputs = lora_model.generate(input_ids=input_ids, max_new_tokens=100, do_sample=True, top_p=0.9)
generated_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n生成摘要 (LoRA):")
print(generated_summary)

# 可选：与基础模型生成结果进行比较
# base_model.to(lora_model.device)
# base_model.eval()
# with torch.no_grad():
#    base_outputs = base_model.generate(input_ids=input_ids, max_new_tokens=100)
# base_summary = tokenizer.decode(base_outputs[0], skip_special_tokens=True)
# print("\n生成摘要 (基础模型):")
# print(base_summary)
```

你应该会观察到，LoRA 适配模型生成的摘要与摘要任务的契合度更高，比原始基础模型（未经微调时可能会重复部分输入或给出不相关的回答）的输出更佳。

### 8. 合并 LoRA 权重 (weight) (可选)

对于无需频繁切换不同适配器的部署情况，你可以将 LoRA 权重直接合并到基础模型的权重中。这会创建一个标准 `transformers` 模型，其中包含了微调 (fine-tuning)的调整。合并后，进行推理 (inference)时不再需要 `peft` 库。

```python
# 将适配器权重合并到基础模型中
# merged_model = lora_model.merge_and_unload()

# 现在 'merged_model' 是一个应用了 LoRA 更新的标准 transformers 模型。
# 它可以像任何常规 Hugging Face 模型一样保存和加载。
# merged_model.save_pretrained(f"{output_dir}/final_merged_model")
# tokenizer.save_pretrained(f"{output_dir}/final_merged_model")

# 注意：合并后，模型大小会恢复到原始基础模型的大小，
# 因为低秩更新现在已成为主权重矩阵的一部分。
```

本次实践阐述了如何使用 LoRA 进行高效微调。你成功地使用显著更少的可训练参数 (parameter)适配了一个预训练 (pre-training)模型，配置了 LoRA 参数，训练了适配器，并进行了推理。尝试不同的秩 (`r`)、`lora_alpha` 值和 `target_modules` 有助于优化特定任务和数据集的性能。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍低秩适配（LoRA）的原始研究论文，提供了其在高效微调大型语言模型方面的理论基础和实证证据。
- [PEFT Documentation](https://huggingface.co/docs/peft/en/index) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face PEFT（参数高效微调）库的官方文档，本实践部分中用于实现LoRA。
- [Transformers Documentation](https://huggingface.co/docs/transformers/en/index) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face Transformers库的官方文档，对于理解微调过程中使用的基础模型、分词器和`Trainer`类至关重要。

---

[上一节](08-%E4%BD%BF%E7%94%A8%20Hugging%20Face%20PEFT%20%E5%BA%93%E8%BF%9B%E8%A1%8C%E5%AE%9E%E7%8E%B0.md) · [下一节](10-%E5%AE%9E%E6%93%8D%EF%BC%9A%E4%BD%BF%E7%94%A8QLoRA%E8%BF%9B%E8%A1%8C%E5%BE%AE%E8%B0%83.md)
