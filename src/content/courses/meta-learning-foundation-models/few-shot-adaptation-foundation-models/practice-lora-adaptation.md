---
course: "meta-learning-foundation-models"
chapter: "few-shot-adaptation-foundation-models"
lesson: "practice-lora-adaptation"
sourceId: 4369
sourceUrl: "https://apxml.com/zh/courses/meta-learning-foundation-models/chapter-5-few-shot-adaptation-foundation-models/practice-lora-adaptation"
title: "动手实践：使用LoRA适配基础模型"
description: "实施 LoRA，将预训练的基础模型适配到小样本任务。"
order: 7
plots: []
sourceHash: "e4f69badbf9eb9a9d5c601a208b62d508c3fb7969d41ab5d6d3a5df069a0ef8f"
sourceCorrections: []
---

低秩适配 (LoRA) 的实践应用包括使用 `peft` 库为小样本任务适配预训练 (pre-training)的基础模型，该库简化了各种参数 (parameter)高效微调 (fine-tuning)技术的应用。我们假设你已安装好包含 PyTorch 和 Hugging Face 生态系统（`transformers`、`datasets`、`peft`）的 Python 环境。强烈建议使用 GPU 进行高效训练，即使是参数高效的方法也一样。

我们的目标是取一个大型的预训练模型（已冻结），并仅在一个表示新任务的小数据集上训练轻量级的 LoRA 适配器。

### 1. 设置与准备工作

首先，请确保已安装必要的库：

```python
# pip install transformers datasets peft torch accelerate
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import os

# 配置（请替换为你的具体设置）
BASE_MODEL_NAME = "bert-base-uncased" # 示例模型
DATASET_NAME = "imdb" # 用于分类的示例数据集
NUM_CLASSES = 2 # 示例：正面/负面情感
FEW_SHOT_SAMPLES = 16 # K 样本学习的 K 值（每个类别）
OUTPUT_DIR = "./lora-bert-few-shot-adapter"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
LORA_R = 8 # LoRA 秩
LORA_ALPHA = 16 # LoRA 缩放因子
LORA_DROPOUT = 0.1
# 根据模型架构指定目标模块（例如，对于 BERT）
LORA_TARGET_MODULES = ["query", "value"]

# 确保设备设置正确（如果 GPU 可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用的设备：{device}")

# 如果输出目录不存在则创建
os.makedirs(OUTPUT_DIR, exist_ok=True)
```

我们为基础模型、数据集、LoRA 参数 (parameter)和训练超参数 (hyperparameter)定义常量。选择合适的 `LORA_TARGET_MODULES` 很要紧；对于许多 Transformer 模型，将 LoRA 应用于自注意力 (self-attention)机制 (attention mechanism)中的查询和值投影矩阵是有效的。你可能需要检查模型架构（`print(model)`）来确定正确的模块名称。

### 2. 加载基础模型和分词 (tokenization)器 (tokenizer)

我们加载预训练 (pre-training)模型及其对应的分词器。该模型将作为基础，其原始权重 (weight)在适配期间被冻结。

```python
# 加载分词器和基础模型
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_NAME,
    num_labels=NUM_CLASSES
)

# 冻结基础模型的所有参数
for param in model.parameters():
  param.requires_grad = False

print(f"已加载基础模型：{BASE_MODEL_NAME}")
```

### 3. 准备小样本数据集

“对于小样本学习，我们需要一个小的支持集来训练适配器。我们将通过从标准数据集中抽样少量示例来模拟这一点。在实际情况中，这将是你实际有限的任务特定数据。”

```python
# 加载数据集
dataset = load_dataset(DATASET_NAME)

# 创建一个小型、均衡的小样本训练子集
train_dataset_full = dataset['train'].shuffle(seed=42)
sampled_train_indices = []
for label in range(NUM_CLASSES):
    label_indices = [
        i for i, ex in enumerate(train_dataset_full)
        if ex['label'] == label
    ][:FEW_SHOT_SAMPLES]
    sampled_train_indices.extend(label_indices)

few_shot_train_dataset = train_dataset_full.select(sampled_train_indices).shuffle(seed=42)

# 使用子集以便更快评估
eval_dataset = dataset['test'].shuffle(seed=42).select(range(1000))

# 预处理函数
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

# 应用预处理
encoded_train_dataset = few_shot_train_dataset.map(preprocess_function, batched=True)
encoded_eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# 为 PyTorch 格式化数据集
encoded_train_dataset.set_format("torch", columns=['input_ids', 'attention_mask', 'label'])
encoded_eval_dataset.set_format("torch", columns=['input_ids', 'attention_mask', 'label'])

print(f"已准备好包含 {len(encoded_train_dataset)} 个训练样本的小样本数据集。")
print(f"正在使用 {len(encoded_eval_dataset)} 个样本进行评估。")
```

这段代码从训练集中按类别抽取 `FEW_SHOT_SAMPLES` 个示例，并通过对文本输入进行分词 (tokenization)来准备训练和评估数据集。

### 4. 配置和应用 LoRA

接下来，我们使用 `LoraConfig` 定义 LoRA 配置，并使用 `get_peft_model` 将其应用于已冻结的基础模型。此函数会修改模型架构，使其包含指定目标模块中的低秩适配器。

```python
# 定义 LoRA 配置
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none", # 通常为 'none'、'all' 或 'lora_only'
    task_type=TaskType.SEQ_CLS # 特定任务类型
)

# 将 LoRA 应用于模型
lora_model = get_peft_model(model, lora_config)

# 打印可训练参数
lora_model.print_trainable_parameters()

# 将模型移动到相应的设备
lora_model.to(device)
```

`print_trainable_parameters()` 方法展现了 LoRA 的高效性。你会看到可训练参数 (parameter)的数量只占原始基础模型总参数的很小一部分。

> LoRA 适配的简化视图。原始权重 (weight)矩阵 W 被冻结。可训练的低秩矩阵 B 和 A（秩 $r \ll d, k$）并行添加。最终输出结合了两个分支的输出。

### 5. 训练 LoRA 适配器

我们设置一个标准的 PyTorch 训练循环。与完整微调 (fine-tuning)的主要不同是，优化器只需要管理 LoRA 适配器的参数 (parameter)，`get_peft_model` 会方便地将其标记 (token)为可训练。

```python
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
from tqdm.notebook import tqdm # Use tqdm for progress bars

# 创建数据加载器
train_dataloader = DataLoader(encoded_train_dataset, batch_size=8, shuffle=True)
eval_dataloader = DataLoader(encoded_eval_dataset, batch_size=16)

# 优化器 - 仅优化 LoRA 参数
optimizer = AdamW(lora_model.parameters(), lr=LEARNING_RATE)

# 学习率调度器
num_training_steps = NUM_EPOCHS * len(train_dataloader)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

print("开始训练 LoRA 适配器...")

for epoch in range(NUM_EPOCHS):
    lora_model.train() # 将模型设置为训练模式
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)

    for batch in progress_bar:
        # 将批次数据移动到设备
        batch = {k: v.to(device) for k, v in batch.items()}

        # 前向传播
        outputs = lora_model(input_ids=batch['input_ids'],
                           attention_mask=batch['attention_mask'],
                           labels=batch['label'])

        # 计算损失
        loss = outputs.loss
        total_loss += loss.item()

        # 反向传播
        loss.backward()

        # 优化器步进
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        progress_bar.set_postfix({'loss': loss.item()})

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} Average Training Loss: {avg_train_loss:.4f}")

    # 可选：循环内的评估步骤（参见下一节）
    # evaluate(lora_model, eval_dataloader, device)

print("训练完成。")

# 保存训练好的 LoRA 适配器
lora_model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR) # 也保存分词器以便于加载
print(f"LoRA 适配器已保存到 {OUTPUT_DIR}")
```

这个循环会遍历小型小样本数据集，进行指定数量的训练周期，计算损失并仅更新 LoRA 权重 (weight)（矩阵 A 和 B）。

### 6. 评估已适配模型

训练完成后，我们评估带有训练好的 LoRA 适配器的模型在保留评估集上的性能。

```python
from sklearn.metrics import accuracy_score

def evaluate(model, dataloader, device):
    model.eval() # 将模型设置为评估模式
    all_preds = []
    all_labels = []
    total_eval_loss = 0
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

    with torch.no_grad(): # 禁用梯度计算
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['label'])

            loss = outputs.loss
            total_eval_loss += loss.item()

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch['label'].cpu().numpy())

    avg_eval_loss = total_eval_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Evaluation Loss: {avg_eval_loss:.4f}")
    print(f"Evaluation Accuracy: {accuracy:.4f}")
    return accuracy, avg_eval_loss

# 执行最终评估
print("\n正在执行最终评估...")
evaluate(lora_model, eval_dataloader, device)
```

此评估函数计算评估集上的损失和准确率，提供了适配器从小样本训练示例中泛化能力好坏的衡量。

### 7. 加载和使用适配器

你可以轻松加载带有训练好的 LoRA 适配器的基础模型，以便后续进行推理 (inference)：

```python
from peft import PeftModel, PeftConfig

# 加载配置和基础模型
config = PeftConfig.from_pretrained(OUTPUT_DIR)
base_model = AutoModelForSequenceClassification.from_pretrained(
    config.base_model_name_or_path, # 加载原始基础模型名称
    num_labels=NUM_CLASSES
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# 加载 LoRA 模型（将适配器合并到基础模型中）
loaded_lora_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
loaded_lora_model.to(device)
loaded_lora_model.eval()

print("成功加载已适配模型。")

# 推理示例
text = "This movie was fantastic, great acting and plot!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

with torch.no_grad():
    outputs = loaded_lora_model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()

print(f"输入文本：'{text}'")
print(f"预测类别 ID：{predicted_class_id}") # 如有需要，将 ID 映射到标签名称
```

### 分析与考量

本次实践练习展示了使用 LoRA 适配基础模型的核心工作流程：加载、冻结、配置 PEFT、在小样本数据上训练适配器，以及评估。

- **效率：** 你观察到可训练参数 (parameter)相较于完整微调 (fine-tuning)有明显减少。这意味着更低的内存需求和更快的训练时间，对于超大型模型尤为重要。
- **性能：** 尽管只训练了极少部分参数，LoRA 在许多任务上通常能达到与完整微调相当的性能。性能取决于任务、数据集大小、基础模型以及超参数 (hyperparameter)调整（`r`、`alpha`、`target_modules`）。
- **超参数调整：** 秩 `r` 是一个要紧参数。更高的 `r` 可以捕获更复杂的适配，但会增加可训练参数。`lora_alpha` 作为 LoRA 更新的缩放因子；通常设置为 `r` 或 `2*r`。需要通过实验寻找最优值。
- **与元学习的比较：** 这种 LoRA 适配过程比 MAML 等元学习方法更简单。它不需要涉及多个任务的复杂元训练阶段。相反，它直接将预训练 (pre-training)模型适配到目标小样本任务。元学习旨在为快速适配学习一个好的*初始化*或*学习程序*，而 LoRA 为适配本身提供了一种*参数高效*的机制。它们之间的选择取决于具体问题、可用数据（多个元训练任务 vs. 单个小样本任务）以及计算预算。结合两者特点的混合方法也是一个活跃的研究方向。

这个动手实践示例提供了一个起点。你可以在此基础上，试验不同的基础模型（视觉 Transformer、其他 LLM），尝试 `peft` 库中可用的不同 PEFT 技术（如 Prefix Tuning 或 Adapters），并将其应用于更复杂的小样本数据集和任务。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: arXiv preprint arXiv:2106.09685; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍了低秩适应（LoRA）方法，通过训练少量参数来实现大型模型的高效微调。
- [PEFT (Parameter-Efficient Fine-Tuning)](https://huggingface.co/docs/peft/en/index) — Hugging Face (2024)
  Publisher: Hugging Face
  `peft`库的官方文档，提供了参数高效微调技术的指南和API参考。
- [Fine-tuning a pretrained model](https://huggingface.co/docs/transformers/en/training#fine-tuning-a-pretrained-model) — Hugging Face (2024)
  Publisher: Hugging Face
  `transformers`库的官方模型微调指南，涵盖了分词和数据集准备等方面。
