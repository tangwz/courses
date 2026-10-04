---
course: "introduction-to-llm-fine-tuning"
chapter: "full-parameter-fine-tuning"
lesson: "practice-full-fine-tuning"
sourceId: 7280
sourceUrl: "https://apxml.com/zh/courses/introduction-to-llm-fine-tuning/chapter-3-full-parameter-fine-tuning/practice-full-fine-tuning"
title: "实践：小型模型上的全参数微调"
description: "一个在更小、更易于管理的语言模型上执行端到端全参数微调的动手环节。"
order: 7
plots: []
sourceHash: "8cc6fe69b2fd4bd2afb7677688565c7ad530acd193c93a98a82bb40121f014d0"
sourceCorrections: []
---

在本次动手环节中，你将把全参数 (parameter)微调 (fine-tuning)的原则应用到一个小型生成模型上。我们的目标是选用一个预训练 (pre-training)模型，并使其适应特定的问答风格。本次练习将引导你完成整个流程，从加载数据到使用你新训练的模型生成文本。

我们将使用 **`Qwen/Qwen2.5-0.5B`** 模型，它是 Qwen 的一个更小、更易于管理的版本，这使得它非常适合在单个 GPU 上运行此示例，例如 Google Colab 或 Kaggle Kernels 中提供的那些。数据集将是 **`squad`** 的一个小型子集，这是一个包含问题和**基于阅读段落**的回答的数据集。

### 微调 (fine-tuning)流程

在编写代码之前，我们先概述一下将要采取的步骤。整个过程遵循一个结构化流程，这在机器学习 (machine learning)项目中是一种常见模式。

> 整个过程始于数据和模型的准备，然后进入训练环节，在此环节中模型的权重 (weight)会得到更新，最后以保存模型以便将来使用而结束。

### 第一步：环境配置

首先，请确保你已安装所需的库。`transformers` 库提供模型和 `Trainer` API，`datasets` 帮助我们管理数据，而 `accelerate` 则优化 PyTorch 的训练代码。

```bash
pip install datasets tokenizers huggingface-hub transformers accelerate evaluate torch vllm
```

如果我们想在线保存模型检查点，还需要登录 Hugging Face Hub。此步骤是可选的，但这是一个好的习惯。

```python
from huggingface_hub import notebook_login

notebook_login()
```

### 第二步：加载和准备数据集

我们将使用 **`squad`** 数据集的一小部分以缩短训练时间。我们的目标是将问答对格式化成模型将学习生成的一个单一字符串。

让我们加载数据集并创建一个简单的格式化函数。我们将把每个示例格式化为 `question: [QUESTION] answer: [ANSWER]`。这种结构会教会模型在看到“question:”前缀时生成一个回答。

```python
from datasets import load_dataset

# 加载训练集的一小部分
train_dataset = load_dataset("squad", split="train[:5000]")

# 将5000个示例分割为训练集和验证集
train_test_split = train_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# 定义格式化函数
def format_qa(example):
    # SQuAD 数据集有答案列表，取第一个
    question = example["question"]
    answer = example["answers"]["text"][0]
    return f"question: {question} answer: {answer}"
```

> **数据格式化即任务定义**
> 你组织数据的方式是基础性的。通过将数据格式化为 `question: ... answer: ...`，我们就在无形中教会模型一个特定任务：即给定一个以 `question:` 开头的字符串，用一个相关的 `answer:` 来补全它。

现在，我们需要加载分词 (tokenization)器 (tokenizer)并将其应用到我们格式化的数据集上。分词器将我们的文本字符串转换为模型能理解的整数 ID。我们还将设置 `pad_token` 来处理不同长度的输入。

```python
from transformers import AutoTokenizer

# 为 Qwen/Qwen2.5-0.5B 加载分词器
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token # 设置填充标记

def tokenize_function(examples):
    formatted_examples = [
        format_qa({"question": q, "answers": a})
        for q, a in zip(examples["question"], examples["answers"])
    ]
    tokenized = tokenizer(formatted_examples, truncation=True, padding="max_length", max_length=128)
    tokenized["labels"] = tokenized["input_ids"].copy()  # Add labels for training
    return tokenized

# 对数据集应用分词
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)
```

### 第三步：配置和初始化模型

数据准备就绪后，我们就可以加载 **`Qwen/Qwen2.5-0.5B`** 模型了。我们使用 `AutoModelForCausalLM` 是因为我们的任务是文本生成，也称因果语言建模。

```python
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

# 加载预训练模型
model = AutoModelForCausalLM.from_pretrained(model_name)
```

接下来，我们定义 `TrainingArguments`。此对象包含训练运行所需的所有超参数 (parameter) (hyperparameter)，例如学习率、训练轮数和批量大小。这些设置直接控制本章开头讨论的梯度下降 (gradient descent)更新过程。

```python
training_args = TrainingArguments(
    output_dir="qwen2.5-0.5b-squad-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    load_best_model_at_end=True,
    push_to_hub=False, # 如果已登录并想推送，请设置为 True
)
```

### 第四步：启动微调 (fine-tuning)过程

现在，我们已准备好将所有部分整合起来，放入 `Trainer` 对象中。它需要模型、训练参数 (parameter)、数据集和分词 (tokenization)器 (tokenizer)。

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
)

# 开始微调
trainer.train()
```

当你运行 `trainer.train()` 时，你将看到一个进度条，显示训练损失和其他指标。此输出是你监控过程的主要工具，正如“监控训练”部分所述。持续下降的损失表明模型正在从你的数据中学习。

```text
/opt/conda/lib/python3.10/site-packages/transformers/trainer.py:...
    ***** Running training *****
    Num examples = 4500
    Num Epochs = 3
    Instantaneous batch size per device = 8
    ...
    Step | Training Loss | Validation Loss
    100  | 1.503200      | N/A
    200  | 1.354100      | N/A
    ...
    563  | 1.298700      | 1.251142
    ...
```

### 第五步：评估和保存模型

训练完成后，`Trainer` 将在每个训练轮次结束时自动评估模型在验证集上的表现。你也可以手动触发一次最终评估。

```python
import math
eval_results = trainer.evaluate()
print(f"困惑度: {math.exp(eval_results['eval_loss']):.2f}")
```

困惑度是语言模型的一种常用评估指标；它衡量模型预测一段文本样本的准确程度。困惑度得分越低，性能越好。

`Trainer` 会将最佳模型检查点保存到 `TrainingArguments` 中指定的 `output_dir` 目录下。你也可以手动将其保存到其他位置。

```python
# 保存模型和分词器
trainer.save_model("my_finetuned_qwen2.5-0.5b")
tokenizer.save_pretrained("my_finetuned_qwen2.5-0.5b")
```

### 第六步：测试你的微调 (fine-tuning)模型

最后一步是使用你的模型进行推理 (inference)。让我们看看它是否已学会以期望的格式回答问题。我们可以使用 `pipeline` 工具进行一个简单的测试。

```python
from transformers import pipeline

# 加载微调后的模型进行推理
finetuned_model = AutoModelForCausalLM.from_pretrained("my_finetuned_qwen2.5-0.5b")
finetuned_tokenizer = AutoTokenizer.from_pretrained("my_finetuned_qwen2.5-0.5b")

# 创建文本生成流程
generator = pipeline("text-generation", model=finetuned_model, tokenizer=finetuned_tokenizer)

# 一个符合我们模型期望格式的新问题
prompt = "question: What is the main purpose of the immune system?"

# 生成一个回答
result = generator(prompt, max_length=100, num_return_sequences=1)
print(result[0]['generated_text'])
```

你应该会看到一个以你的提示开始，并接着生成回答的输出，遵循 **`squad`** 数据集的风格。模型已成功调整其行为，从一个通用文本生成器变成一个更专业的问答器。

本次动手练习展现了全参数 (parameter)微调的端到端过程。虽然有效，但更新每一个权重 (weight)计算成本较高。下一章中，我们将介绍更高效的技术，这些技术能以一小部分计算成本达到类似的效果。

## 参考资料

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/en/index) — Hugging Face (2024)
  提供 transformers 库的全面指南和 API 参考，包括 Trainer、TrainingArguments 和 AutoModelForCausalLM，这些在微调工作流中是核心。
- [Qwen: A Super Large-Scale Pretrained Language Model Family with Strong Chinese and English Abilities](https://arxiv.org/abs/2309.16609) — Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, Tianhang Zhu (2023)
  Journal: arXiv preprint arXiv:2309.16609; DOI: [10.48550/arXiv.2309.16609](https://doi.org/10.48550/arXiv.2309.16609)
  描述 Qwen 模型系列，包括架构和训练方法，这是本节中用于微调的基础模型。
