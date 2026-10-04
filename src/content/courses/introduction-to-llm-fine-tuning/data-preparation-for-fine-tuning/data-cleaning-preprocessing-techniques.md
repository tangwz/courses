---
course: "introduction-to-llm-fine-tuning"
chapter: "data-preparation-for-fine-tuning"
lesson: "data-cleaning-preprocessing-techniques"
sourceId: 7270
sourceUrl: "https://apxml.com/zh/courses/introduction-to-llm-fine-tuning/chapter-2-data-preparation-for-fine-tuning/data-cleaning-preprocessing-techniques"
title: "数据清洗和预处理技术"
description: "清洗文本数据的实用技术，以提高模型在训练期间的性能和稳定性。"
order: 3
plots: []
sourceHash: "0c39f6d0729564025066080461bdb0c39e104a90c96438cc976c294d781b79c7"
sourceCorrections: []
---

高质量数据集是成功微调 (fine-tuning)过程的基础。原始数据，无论是从网络抓取还是来自内部日志，通常都不够干净，无法直接使用。它常含有不相关的残余、不一致性以及结构性噪声，这些会在训练期间使模型困惑，导致性能不佳、不稳定或出现意外行为。本节介绍系统清洗和预处理文本数据的实用技术，以创建高质量、模型可用的数据集。

数据清洗的目标不仅是去除错误，还在于标准化输入。通过减少噪声，您可以帮助模型专注于您希望它学习的潜在模式，而不是源数据的特性。

### 识别和移除数字残余

来自数字平台的文本数据常包含与语义内容不相关的残余。这些可能包括HTML标签、URL、电子邮件地址及其他元数据。将这些留在数据集中可能引入噪声，并导致模型学习到错误的关联。

正则表达式是完成这项任务的有效工具。让我们看一个清理可能从网页抓取的一段文本的常见示例。

```python
import re

def clean_text(text):
    """
    移除常见的数字残余，例如URL、HTML标签和电子邮件地址。
    """
    # 移除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 移除URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # 移除电子邮件地址
    text = re.sub(r'\S+@\S+', '', text)
    # 移除多余的空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 混乱数据样本示例
messy_sample = """
    <p>Check out our new product at https://example.com! For questions, email us at support@example.com.</p>
    <br>It's      amazing.
"""

cleaned_sample = clean_text(messy_sample)
print(f"原始: {messy_sample}")
print(f"清洗后:  {cleaned_sample}")
# 预期输出:
# Original: ... (原始的混乱字符串)
# Cleaned:  Check out our new product at! For questions, email us at. It's amazing.
```

该函数为移除常见类型的噪声提供了一个可靠的基础。您可以使用额外的正则表达式扩展它，以处理数据集中特定的其他模式，例如移除用户句柄（如`@username`）或话题标签。

### 规范化文本以保持一致性

文本规范化旨在通过将其转换为更标准的形式来减少文本的可变性。尽管过度规范化对依赖标点和大小写细微线索的生成式LLM有害，但适度的方法可以提高训练稳定性。

#### 大小写转换

将所有文本转换为单一大小写，通常是小写，称为大小写转换。

- **优点：** 它减少了模型需要考虑的词汇量。例如，“Model”、“model”和“MODEL”被视为相同的标记 (token)。
- **缺点：** 它会丢失信息。大小写可以表示专有名词（例如“US”与“us”）、首字母缩略词或强调。

对于大多数指令遵循或聊天微调 (fine-tuning)任务，通常最好**保留原始大小写**，除非您的数据集噪声很大或您有特定理由忽略大小写。如果您选择转换为小写，请在整个数据集中一致应用。

```python
text = "Fine-Tuning is an IMPORTANT technique."
lowercase_text = text.lower()
# Output: "fine-tuning is an important technique."
```

#### 处理标点符号

是否移除标点符号的决定完全取决于您的任务。为了生成格式良好、人类可读的文本，标点符号是必不可少的。但是，您可能需要标准化或移除那些没有价值的非标准标点符号或特殊字符。例如，您可能希望将“智能”引号（`“`和`”`）规范化为标准引号（`"`），或者移除`~`或`*`等装饰性字符。

```python
import string

def handle_punctuation(text):
    # 示例：移除特定标点符号，但保留重要的。
    # 这高度取决于您的任务。
    # 对于本示例，我们移除一组特定符号。
    unwanted_punctuation = "#~*&"
    translator = str.maketrans('', '', unwanted_punctuation)
    return text.translate(translator)

text = "This is a *great* idea, right? #LLM"
processed_text = handle_punctuation(text)
print(processed_text)
# Output: "This is a great idea, right? LLM"
```

在这里，我们选择性地移除了字符，同时保留了逗号和问号，这些对句子结构有帮助。

### 处理数据冗余和不一致性

高质量数据集不仅干净，而且不冗余。重复或近似重复的示例可能导致模型过拟合 (overfitting)，使其响应偏向重复样本。

#### 去重

始终检查并移除数据集中相同的条目。如果您处理的是指令-响应对，重复项可以定义为相同的指令或相同的（指令，响应）对。

使用`pandas`库是管理和去重结构化数据的有效方法。

```python
import pandas as pd

# 假设数据是字典列表，一种常见格式
data = [
    {"instruction": "Summarize the following text.", "input": "Text A...", "output": "Summary A..."},
    {"instruction": "What is the capital of France?", "input": "", "output": "Paris"},
    {"instruction": "Summarize the following text.", "input": "Text A...", "output": "Summary A..."}, # 重复项
    {"instruction": "What is the capital of Japan?", "input": "", "output": "Tokyo"}
]

df = pd.DataFrame(data)
deduplicated_df = df.drop_duplicates()

print(f"原始计数: {len(df)}")
print(f"去重后计数: {len(deduplicated_df)}")
# 预期输出:
# Original count: 4
# Deduplicated count: 3
```

对于非常大的数据集，您也可以考虑使用MinHash等技术进行近似重复检测，以识别和移除语义相似但不完全相同的条目，但这属于更进阶的步骤。

### 清洗流程

将这些步骤视为一个顺序流程会很有帮助，其中原始数据在每个阶段都会被转换。以一致的顺序应用这些步骤可确保您的整个数据集得到统一处理。

> 一个典型的微调 (fine-tuning)数据清洗流程。规范化步骤通常是可选的，并取决于任务。

### 平衡之道：激进与谨慎的清洗

数据清洗需要权衡。过度激进的清洗，例如移除所有标点符号和数字，可能会剥离高质量生成所需的上下文 (context)和细节。清洗不足会留下噪声，从而降低模型性能。

恰当的平衡取决于两个因素：

1. **源数据质量：** 您的初始数据噪声越多，可能需要的清洗就越多。
2. **目标任务：** 对于需要创意或对话式文本的任务，保留原始风格，包括俚语和某些标点符号，可能是有益的。对于更正式的任务，如代码生成或技术摘要，更严格、更标准化的格式更受欢迎。

在每个清洗步骤之后，务必检查数据样本，以确保您没有无意中移除有价值的信息。这种清洗和验证的迭代过程是准备数据集的基础，它将使您的模型表现最佳。

## 参考资料

- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](https://books.google.com/books/about/Python_for_Data_Analysis.html?id=zI5YDwAAQBAJ) — Wes McKinney (2017)
  Publisher: O'Reilly Media
  提供了使用Python的pandas库进行数据处理和数据清洗的实用指南，包括高效的去重技术。第4版。
- [A Comprehensive Survey on Data Preprocessing for Deep Learning](https://link.springer.com/article/10.1007/s42979-023-01777-6) — Md. Tahmidul Islam, Md. Rakib Hasan, Md. Hasanuzzaman (2023)
  Journal: SN Computer Science; Publisher: Springer Nature; Volume: 4; Pages: Article number 170; DOI: [10.1007/s42979-023-01777-6](https://doi.org/10.1007/s42979-023-01777-6)
  综述了与深度学习相关的多种数据预处理方法，包括文本数据技术，强调了它们对模型性能的重要性。
- [The Hugging Face Course: Fine-tune a pretrained model](https://huggingface.co/docs/course/en/chapter3/3) — Hugging Face (2024)
  Publisher: Hugging Face
  为大型语言模型微调的数据准备提供了实用指导，涵盖了分词、数据集格式化和特定任务的最佳实践。
