---
course: "practical-llm-quantization"
chapter: "quantization-aware-training-qat"
lesson: "practice-simple-qat"
sourceId: 4650
sourceUrl: "https://apxml.com/zh/courses/practical-llm-quantization/chapter-4-quantization-aware-training-qat/practice-simple-qat"
title: "动手实践：搭建简单量化感知训练运行环境"
description: "关于为小型模型或组件设置基本 QAT 实验的指导实践会话。"
order: 8
plots: []
sourceHash: "edd4765736c75aaca0e0bc45d46627035acc2f4e3935a85d7886e382be12e87f"
sourceCorrections: []
---

量化 (quantization)感知训练 (QAT) 包含量化模拟和直通估计器 (STE)。设置一个基本的QAT实验是一个实践任务。这涉及微调 (fine-tuning)一个预训练 (pre-training)的Transformer模型，同时加入量化模拟。目标是获得一个量化模型，其准确性可能比单独使用训练后量化 (PTQ) 更好。

我们将借助 Hugging Face 生态系统，特别是使用 `transformers` 库进行模型处理，以及 `optimum` 库，它为量化技术（包括 QAT）提供了便捷的抽象接口。

### 前提条件

在开始之前，请确保你已安装所需的库。通常可以使用 pip 进行安装：

```bash
pip install torch torchvision torchaudio
pip install transformers datasets evaluate accelerate optimum[neural-compressor]
```

我们将使用 PyTorch 作为后端，并在本示例中通过 `optimum` 使用 Intel 的 Neural Compressor 来实现 QAT。请确保你有一个可用的 PyTorch 环境。

### 1. 加载模型和数据集

首先，我们需要一个预训练 (pre-training)模型和一个用于微调 (fine-tuning)的数据集。为了演示，我们使用一个较小的Transformer模型，例如 `distilbert-base-uncased` 以及 SST-2 (斯坦福情感树库) 数据集，这是一个常见的文本分类任务。

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset, load_metric
import torch

# 定义模型检查点和任务
model_checkpoint = "distilbert-base-uncased"
task = "sst2" # 用于情感分析的GLUE任务

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

# 加载数据集
dataset = load_dataset("glue", task)

# 预处理数据
def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)

encoded_dataset = dataset.map(preprocess_function, batched=True)

# 使用较小的子集以加快演示速度
train_dataset = encoded_dataset["train"].shuffle(seed=42).select(range(1000)) # 使用1000个样本进行训练
eval_dataset = encoded_dataset["validation"].shuffle(seed=42).select(range(500)) # 使用500个样本进行评估

# 加载评估指标
metric = load_metric("glue", task)
```

这样的设置为我们提供了标准的序列分类模型、分词 (tokenization)器 (tokenizer)和已预处理的数据集，它们已准备好进行训练。

### 2. 准备进行量化 (quantization)感知训练

`optimum` 库通过提供 `QATrainer` 类来简化 QAT，该类封装了标准的 `transformers.Trainer`。我们需要定义量化配置，然后使用 `QATrainer`。

```python
from optimum.intel.neural_compressor import QATrainer, IncQuantizationMode
from transformers import TrainingArguments
import numpy as np

# 定义用于评估的 compute_metrics 函数
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

# 定义 TrainingArguments
# 使用少量轮次进行演示
training_args = TrainingArguments(
    output_dir="./qat_output",
    num_train_epochs=1, # 演示用，保持较短；实际QAT需要更多调优
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir='./qat_logs',
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none" # 为简化起见，禁用外部报告
)

# 初始化 QATrainer
# IncQuantizationMode.DYNAMIC 在训练模拟期间对激活使用动态量化，对权重使用静态量化
# IncQuantizationMode.STATIC 则会对两者都使用静态量化
trainer = QATrainer(
    model=model,
    quantization_mode=IncQuantizationMode.DYNAMIC, # 或者 IncQuantizationMode.STATIC
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# QATrainer 自动修改模型
# 以根据 quantization_mode 插入伪量化节点。
print("模型已准备好进行QAT。")
print("原始模型类型：", type(model))
# 注意：底层模型在 QATrainer 初始化时会被原地修改
```

在这里，`QATrainer` 会接收原始模型，并根据指定的 `quantization_mode` 自动插入必要的“伪量化”操作。这些操作在训练的前向和后向传播过程中模拟低精度算术的效果。我们在这里使用 `IncQuantizationMode.DYNAMIC`，这通常是一个很好的起点，它模拟激活的动态量化和权重 (weight)的静态量化。你也可以尝试 `IncQuantizationMode.STATIC`。

### 3. 运行 QAT 微调 (fine-tuning)

现在，我们可以像使用标准 `transformers.Trainer` 一样启动微调过程。`QATrainer` 在内部处理了带模拟量化 (quantization)训练的复杂部分。

```python
print("开始QAT微调...")
train_result = trainer.train()

# 评估经过QAT训练的模型（仍在模拟量化）
print("正在评估QAT模型...")
eval_metrics = trainer.evaluate()
print(f"QAT后的评估指标： {eval_metrics}")

# 保存经过QAT训练的模型（包含量化模拟节点）
# 此模型尚未完全量化，但已学习到对量化有利的权重。
trainer.save_model("./qat_trained_model")
tokenizer.save_pretrained("./qat_trained_model")
print("QAT训练的模型已保存（带模拟节点）。")
```

在 `trainer.train()` 这一步中，模型学习到的权重 (weight)能够有效地应对量化带来的影响，因为模拟在整个训练过程中都处于活跃状态。梯度是使用直通估计器 (STE) 等方法计算的，以使其能够流经模拟的量化步骤。

### 4. 转换为最终量化 (quantization)模型

`trainer.train()` 之后保存的模型仍处于一种 *模拟* 量化的格式。为了获得最终的、真正量化好的模型以进行高效部署（使用整数算术），我们需要一个明确的转换步骤。`optimum` 提供了相应的工具。

```python
from optimum.intel import INCQuantizer

# 加载经过QAT训练的模型状态
quantizer = INCQuantizer.from_pretrained("./qat_trained_model")

# 定义最终转换的量化配置
# 通常与QAT模式或期望的最终状态匹配
from neural_compressor.config import PostTrainingQuantConfig, AccuracyCriterion

# 示例配置：INT8 静态量化
quantization_config = PostTrainingQuantConfig(
    approach="static", # 对最终模型使用静态量化
    backend="pytorch_fx", # 确保后端与QAT期间指定的匹配
    accuracy_criterion=AccuracyCriterion(tolerable_loss=0.01) # 可选：如果准确性下降太多则停止
)

# 定义校准函数（可重用评估数据集）
def calibration_func(model):
    # 使用训练或验证数据的一个小部分进行校准
    num_samples = 100
    calib_dataloader = trainer.get_eval_dataloader(eval_dataset.select(range(num_samples)))
    for batch in calib_dataloader:
        # 确保批次在正确的设备上
        inputs = {k: v.to(trainer.args.device) for k, v in batch.items() if k != "labels"}
        _ = model(**inputs)

# 量化模型（从QAT状态转换为完全量化的INT8）
quantizer.quantize(
    quantization_config=quantization_config,
    calibration_dataset=eval_dataset.select(range(100)), # 如果配置需要，提供用于校准的数据集
    # calib_func=calibration_func # 备选：提供校准函数
    save_directory="./final_quantized_model",
)

print("最终量化模型已保存到 ./final_quantized_model")

# 你现在可以加载这个最终量化模型进行推理
# from optimum.intel import INCModelForSequenceClassification
# quantized_model = INCModelForSequenceClassification.from_pretrained("./final_quantized_model")
# tokenizer = AutoTokenizer.from_pretrained("./final_quantized_model")
# 现在使用 quantized_model 和 tokenizer 进行推理
```

最后这一步通过 `optimum` 使用 Intel Neural Compressor 后端，根据 QAT 阶段学习到的权重 (weight)来执行实际的量化。如果使用静态量化，校准数据有助于确定激活的合适缩放因子，从而利用 QAT 期间学习到的稳定性。存储在 `./final_quantized_model` 中的模型包含整数权重和可能优化的操作，以实现更快的推理 (inference)。

### 预期结果和注意事项

通过运行此过程，你应能在 `./final_quantized_model` 中获得一个量化 (quantization)模型。与对原始 `distilbert-base-uncased` 模型 *不* 进行微调 (fine-tuning)而直接应用基本 PTQ（如朴素静态或动态量化）相比，这种 QAT 方法通常在评估集上产生更好的准确性，尤其是在目标位深较低时（尽管本例侧重于 INT8）。

请记住这些实用要点：

- **计算成本：** QAT 需要完整的微调周期，这比 PTQ 的计算成本高出很多。
- **超参数 (parameter) (hyperparameter)：** QAT 会带来一些需要考虑的方面。学习率、训练轮次和量化配置可能需要仔细调整才能获得最佳结果。训练稳定性有时也可能是一个难题。
- **复杂性：** QAT 的设置比 PTQ 更复杂，需要仔细整合到训练流程中。像 `optimum` 这样的库能大大减轻这方面的负担。
- **何时使用：** 当 PTQ 导致不可接受的准确性下降，并且计算预算允许进行微调时，选择 QAT。

本实用示例提供了一个起始模板。你可以将其应用于不同的模型、任务、数据集，并试用 `optimum` 及其底层后端提供的各种 QAT 配置，以根据你的具体需求在模型压缩和准确性之间取得最佳平衡。

## 参考资料

- [Hugging Face Optimum Documentation](https://huggingface.co/docs/optimum/index) — Hugging Face (2024)
  Hugging Face Optimum 的官方文档，提供了加速模型的关键指南和 API 参考，包括与 Intel Neural Compressor 等各种后端进行量化感知训练的实际实现，正符合实践章节中的用法。
