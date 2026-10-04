---
course: "fine-tuning-small-language-model"
chapter: "environment-and-library-setup"
lesson: "managing-datasets-hugging-face"
sourceId: 8509
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-small-language-model/chapter-3-environment-and-library-setup/managing-datasets-hugging-face"
title: "使用 Hugging Face Datasets 管理数据集"
description: "利用 datasets 库高效地加载、缓存和处理训练数据。"
order: 3
plots: []
sourceHash: "769789566d1d448f4e57ca002fe28bc33612dc7bb6386fc28b8620ba778a0361"
sourceCorrections: []
---

管理训练数据是准备微调 (fine-tuning)流水线的主要组成部分。模型的表现完全依赖于其学习数据的质量和结构。如何高效地将海量文本输入神经网络 (neural network)是机器学习 (machine learning)中的一个主要挑战。Hugging Face 的 datasets 库专门用于处理大规模数据集，能够快速处理这些数据并将其格式化以用于模型训练，且不会耗尽系统内存。

### Arrow 后端与内存映射

在处理小型语言模型时，你经常需要处理包含成千上万条指令对的数据集。将所有这些文本加载到标准的 Python 字典或 pandas DataFrame 中会迅速消耗可用内存 (RAM)。datasets 库通过使用 Apache Arrow 作为其后端解决了这个问题。

Apache Arrow 提供了一种零拷贝、内存映射的格式。数据不会被全部加载到内存中，而是安全地保留在存储盘上。该库利用内存映射技术，仅在需要时读取特定的数据部分。这使你能够在远超物理内存限制的数据集上训练模型。

### 加载数据

加载数据集只需调用一个函数。你可以直接从 Hugging Face Hub 加载标准化数据集，也可以读取本地的 JSON 和 CSV 格式文件。对于监督式微调 (fine-tuning)，你通常会处理包含自定义指令的本地 JSON Lines 文件。

```python
from datasets import load_dataset

# 加载本地 JSON 数据集
dataset = load_dataset("json", data_files="custom_instructions.jsonl")
```

生成的对象行为类似于标准的 Python 字典，但包含 `train`（训练）和 `test`（测试）等数据集切片。你可以使用标准索引访问单行数据，该操作会透明地从磁盘获取所需的字节。

### 使用 Map 进行批处理

原始文本不能直接输入神经网络 (neural network)。你必须使用从 Transformers 库初始化的分词 (tokenization)器 (tokenizer) 将文本转换为数值形式的标记 (token)。逐一针对数千个样本应用这种转换在计算上非常缓慢。

datasets 库提供了一个 `map` 函数，用于在整个数据集上应用转换。通过设置 `batched=True` 参数 (parameter)，该函数可以同时处理多行数据。这种方法允许分词器优化其内部循环，从而大幅加快文本处理速度。

```python
def tokenize_function(examples):
    # 对输入文本进行分词，并应用填充和截断
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

# 在数据集上分批应用分词处理
tokenized_datasets = dataset.map(tokenize_function, batched=True)
```

> 使用 datasets 库准备训练文本的数据处理流水线。

### 智能缓存

机器学习 (machine learning)需要不断的迭代。你会经常调整分词 (tokenization)策略、更改最大序列长度或过滤掉有问题的样本。处理大型数据集可能需要几分钟时间。

为了节省时间，`map` 函数会自动将结果缓存到磁盘。当你再次运行完全相同的 `map` 操作时，该库会根据处理函数和数据集状态的哈希值检测现有缓存，并立即加载缓存版本，而不是重复计算。如果你修改了 `tokenize_function` 中的任何一个参数 (parameter)，该库会计算一个新的哈希值并重新计算数据集。

### 为 PyTorch 格式化

管理数据的最后一步是为 PyTorch 做准备。默认情况下，datasets 库返回标准的 Python 列表。而 PyTorch 在名为张量 (tensor) 的多维矩阵上运行。

你必须明确指示数据集为模型所需的特定列返回 PyTorch 张量。这些列通常包括 `input_ids`、`attention_mask` 和 `labels`。

```python
# 定义模型训练所需的列
columns_to_keep = ["input_ids", "attention_mask", "labels"]

# 将数据集格式设置为 PyTorch 张量
tokenized_datasets.set_format(type="torch", columns=columns_to_keep)
```

格式设置完成后，从数据集中获取批次数据将返回可直接使用的张量。这个格式化后的数据集现在可以直接传递给 PyTorch 的 `DataLoader` 或 Hugging Face 的 `Trainer` API。数据将直接从硬盘出发，通过内存映射的 Arrow 后端，进入 GPU 内存进行损失计算。

## 参考资料

- [Hugging Face Datasets Documentation](https://huggingface.co/docs/datasets/index) — Hugging Face Inc. (2024)
  官方技术文档，涵盖了数据集加载、批处理映射以及与 PyTorch 的集成。
- [Natural Language Processing with Transformers: Building Language Applications with Hugging Face](https://www.oreilly.com/library/view/natural-language-processing-with-transformers/9781098136789/) — Lewis Tunstall, Leandro von Werra, Thomas Wolf (2022)
  Publisher: O'Reilly Media; Pages: 51-78
  关于在 NLP 流水线中使用 datasets 库进行数据准备和分词的指南，修订版。
- [Apache Arrow Documentation](https://arrow.apache.org/docs/index.html) — Apache Software Foundation (2024)
  Publisher: Apache Software Foundation
  Apache Arrow 官方文档，详细介绍了 Hugging Face Datasets 用于处理大规模数据的底层内存映射技术。
