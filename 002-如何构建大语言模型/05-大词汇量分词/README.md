# 第 5 章：大词汇量分词

来源：[原章节](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-5-tokenization-large-vocabularies)

[返回课程目录](../README.md)

在像 Transformer 这样的模型处理文本之前，原始字符序列必须被转换为一系列数字 ID。这个转换过程被称为分词。虽然像按空格分割文本这样的简单方法对小任务有效，但它们难以应对用于大型语言模型（LLM）的海量数据集中的庞大词汇量和形态变化。处理未知词（词汇表外词或 OOV）以及管理可能数百万个独特的词需要更复杂的方法。

本章侧重于旨在解决这些挑战的子词分词算法。您将了解字节对编码（BPE）和 WordPiece，这些技术基于频繁的子词单元而非整个词来构建词汇表。我们还将讨论 SentencePiece 框架、特殊标记（如 `[CLS]`、`[SEP]`）的作用和管理，以及选择词汇表大小 ($|V|$)、平衡模型表达能力和计算效率的实际考量。学习结束时，您将明白如何有效地为大型模型准备文本数据。

## 小节

- 1. [子词分词的必要性](01-%E5%AD%90%E8%AF%8D%E5%88%86%E8%AF%8D%E7%9A%84%E5%BF%85%E8%A6%81%E6%80%A7.md)
- 2. [字节对编码 (BPE) 算法](02-%E5%AD%97%E8%8A%82%E5%AF%B9%E7%BC%96%E7%A0%81%20%28BPE%29%20%E7%AE%97%E6%B3%95.md)
- 3. [WordPiece 分词](03-WordPiece%20%E5%88%86%E8%AF%8D.md)
- 4. [SentencePiece 实现](04-SentencePiece%20%E5%AE%9E%E7%8E%B0.md)
- 5. [处理特殊分词](05-%E5%A4%84%E7%90%86%E7%89%B9%E6%AE%8A%E5%88%86%E8%AF%8D.md)
- 6. [词汇量大小选择的权衡](06-%E8%AF%8D%E6%B1%87%E9%87%8F%E5%A4%A7%E5%B0%8F%E9%80%89%E6%8B%A9%E7%9A%84%E6%9D%83%E8%A1%A1.md)
