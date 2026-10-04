---
course: "how-to-build-a-large-language-model"
chapter: "data-cleaning-preprocessing-pipelines"
lesson: "strategies-quality-filtering"
sourceId: 5979
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-7-data-cleaning-preprocessing-pipelines/strategies-quality-filtering"
title: "质量筛选策略"
description: "使用启发式方法和模型去除低质量或含噪的文本内容。"
order: 1
plots: []
sourceHash: "1b968721976aa095151d986b83df7c75fcb1f65fdf48303842db0ace738574e2"
sourceCorrections: []
---

从网络等来源收集的原始文本数据，除了有用信息外，还包含大量噪声。文档可能被截断，主要包含导航链接，由机器生成日志组成，包含大量样板文本，或者只是与目标语言或领域不相关。将此类低质量数据直接送入LLM预训练 (pre-training)，可能会降低性能，浪费计算资源，并可能引入不良偏见或安全问题。因此，实施有效的质量筛选策略是预处理流程中的主要一步。

这些策略从简单的启发式方法到更复杂的基于模型的方法不等。通常，多种方法结合能提供最佳效果，形成一个多阶段的筛选过程。

### 基于启发式方法的筛选

启发式方法是根据对典型低质量文本的观察得出的经验法则。它们通常计算成本较低，并且能有效去除明显的噪声样本。

- **文档长度：** 极短的文档通常缺少有意义的内容（例如，仅仅是标题、页眉/页脚片段或错误消息）。反之，极长的文档可能表明解析错误、文件串联或非散文内容。常见做法是设置最小和最大字符数或词数阈值。

  ```python
  # 示例：简单的长度筛选
  MIN_CHARS = 100
  MAX_CHARS = 100000

  def filter_by_length(text):
      char_count = len(text)
      # 基本词数估算
      word_count = len(text.split())
      if char_count < MIN_CHARS or char_count > MAX_CHARS:
          return False
      # 如果需要，添加词数检查
      # if word_count < MIN_WORDS or word_count > MAX_WORDS:
      #     return False
      return True

  document = "This is a sample document..."
  if filter_by_length(document):
      # 处理文档
      pass
  ```

  具体阈值（如 `MIN_CHARS`、`MAX_CHARS`）是超参数 (parameter) (hyperparameter)，通常需要根据数据集特性和后续目标进行调整。
- **符号和标点符号比例：** 主要由非字母数字字符组成的文本通常表示源代码、配置文件、表格或损坏的文本。计算字母字符与总字符的比例，或特定符号（如 `#`、`{`、`}`）的比例，有助于识别此类内容。

  ```python
  # 示例：基于字母字符比例的筛选
  MIN_ALPHA_RATIO = 0.7

  def filter_by_alpha_ratio(text):
      if not text: # 处理空字符串
          return False
      alpha_chars = sum(c.isalpha() for c in text)
      total_chars = len(text)
      ratio = alpha_chars / total_chars
      return ratio >= MIN_ALPHA_RATIO

  code_snippet = "if (x > 0) { y = x * 2; } // 示例代码"
  prose = "This sentence contains mostly alphabetic characters."

  print(f"Code snippet passes filter: {filter_by_alpha_ratio(code_snippet)}")
  print(f"Prose passes filter: {filter_by_alpha_ratio(prose)}")
  # 预期输出：
  # Code snippet passes filter: False
  # Prose passes filter: True
  ```
- **“不良词汇”或样板短语的存在：** 创建常见于低质量网络内容的特定词汇或短语列表（例如，“javascript is required”、“terms of use”、错误消息、像“lorem ipsum”这样的占位符文本），允许直接进行筛选。正则表达式可用于匹配与样板相关的模式，如长串导航链接或版权声明。必须注意不要让这些列表过于严格，因为它们可能会无意中过滤掉有效内容。
- **重复内容：** 包含过度重复（例如，同一行或段落重复多次）的文档通常是低质量的。这可以通过查找重复的行、段落，甚至长n-gram来检测。计算文档的压缩比（使用gzip等标准算法）也可以作为一种近似方法；高度重复的文档压缩效果极好。

### 统计和基于模型的筛选

这些方法通常运用文本的统计属性或预训练 (pre-training)模型来评估质量。

- **语言识别：** 对于单语或特定多语模型，识别每个文档的语言非常必要。`fastText` 或 `pycld3` 等库提供预训练的语言识别模型。不匹配目标语言的文档或模型分配较低置信度分数的文档可以被筛选掉。

# 使用语言识别库的示例

````
# import language_identifier_lib

# model = language_identifier_lib.load_model()
TARGET_LANG = 'en'
MIN_CONFIDENCE = 0.90

def filter_by_language(text):
    # detected_lang, confidence = model.predict(text)
    # return detected_lang == TARGET_LANG and confidence >= MIN_CONFIDENCE
    # 仅为说明而进行的模拟实现：
    if "esto es espa\u00f1ol" in text:
         return False # 模拟非目标语言
    if "confidence low" in text:
         return False # 模拟低置信度
    return True # 否则假定为英语

spanish_text = "esto es espa\u00f1ol"
low_confidence_text = "mixed signals confidence low maybe"
english_text = "This is primarily English text."

print(f"Spanish text passes: {filter_by_language(spanish_text)}")
print(f"Low confidence text passes: {filter_by_language(low_confidence_text)}")
print(f"English text passes: {filter_by_language(english_text)}")
# 预期输出：
# Spanish text passes: False
# Low confidence text passes: False
# English text passes: True
```
````

- **困惑度筛选：** 在高质量参考语料库（例如维基百科）上训练的语言模型，会为其训练数据相似的文本分配较低的困惑度（或较高的概率）。我们可以使用这样的模型对原始数据集中的文档进行评分。超过特定困惑度阈值的文档可能与参考语料库不相似，并且潜在地质量较低。这需要访问合适的质量参考语言模型，并且可能计算密集。

# 使用预训练模型进行困惑度评分的示例

```
import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer

# 假设模型和分词器已加载
# model = AutoModelForCausalLM.from_pretrained("gpt2") # 示例
# tokenizer = AutoTokenizer.from_pretrained("gpt2")
PERPLEXITY_THRESHOLD = 50 # 示例阈值

def calculate_perplexity(text, model, tokenizer):
    # 占位符：实际实现包括编码、通过模型、计算损失并转换为困惑度。
    # inputs = tokenizer(text, return_tensors="pt")
    # with torch.no_grad():
    #     outputs = model(**inputs, labels=inputs["input_ids"])
    # loss = outputs.loss
    # perplexity = torch.exp(loss)
    # return perplexity.item()
    # 模拟实现：
    if "gibberish" in text:
        return 100.0
    else:
        return 20.0
```

# 示例用法

````
# high_quality_text = "This is well-formed text."
# low_quality_text = "asdf qwer gibberish zxcv."
# if calculate_perplexity(document, model, tokenizer) < PERPLEXITY_THRESHOLD:
#     # 保留文档
#     pass
```
````

- **基于分类器的筛选：** 如果有区分高质量和低质量文档的小型标注数据集可用或可以创建，则可以训练一个分类模型。特征可以从简单的TF-IDF向量 (vector)到预训练模型的嵌入 (embedding)向量不等。训练好的分类器随后会预测大型数据集中每个文档的质量分数，从而允许根据此分数进行筛选。这需要标注工作，但可以捕获比单独使用启发式方法更不明显的质量指标。

### 在流程中结合策略

没有单一的筛选策略是完美的。有效的数据清洗通常涉及在流程中将多个筛选器串联起来。一种常见的方法是首先使用计算成本较低的启发式方法（长度、语言识别）来快速去除明显噪声，随后对剩余数据采用困惑度或基于分类器筛选等更耗资源的方法。

> 文本数据的典型多阶段筛选流程。

### 注意事项

- **偏见放大：** 筛选规则，特别是那些基于启发式方法或在可能存在偏见的参考语料库上训练的模型，可能会无意中删除来自特定人群、领域或方言的有效文本，从而放大最终数据集中的偏见。分析被丢弃的数据以了解可能引入的偏见非常重要。
- **阈值调整：** 设置合适的长度、比例、置信度分数或困惑度阈值通常取决于数据集，需要仔细实验。评估不同阈值对后续模型性能的影响，并分析每个阶段被筛选掉的文档类型。
- **计算成本：** 虽然启发式方法成本低，但将大型语言模型或复杂分类器应用于数TB或数PB数据进行筛选，需要大量计算资源。权衡筛选的严格性与处理时间/成本。

成功运用这些质量筛选策略，是准备构建高性能大型语言模型所需的大量（但通常杂乱）数据集的不可或缺的环节。它为后续的规范化和去重等预处理步骤做好了铺垫。

## 参考资料

- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://jmlr.org/papers/v21/20-074.html) — Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu (2020)
  Journal: Journal of Machine Learning Research; Publisher: JMLR; Volume: 21; Pages: 1-67; DOI: [10.48550/arXiv.1910.10683](https://doi.org/10.48550/arXiv.1910.10683)
  详细介绍了C4数据集的构建，这是LLM预训练的重要语料库，阐述了用于质量控制的各种过滤技术，包括启发式和统计方法。
- [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759) — Armand Joulin, Edouard Grave, Piotr Bojanowski, Tomas Mikolov (2016)
  Journal: arXiv; DOI: [10.48550/arXiv.1607.01759](https://doi.org/10.48550/arXiv.1607.01759)
  介绍了fastText，这是一个用于高效文本分类和语言识别的重要库，常用于数据过滤流程中。
