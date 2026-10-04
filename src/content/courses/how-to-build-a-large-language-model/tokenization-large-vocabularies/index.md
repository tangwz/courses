---
course: "how-to-build-a-large-language-model"
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-5-tokenization-large-vocabularies"
sourceId: 1083
chapter: "tokenization-large-vocabularies"
title: "大词汇量分词"
order: 5
description: "了解用于处理庞大和开放词汇量的子词分词算法，如 BPE、WordPiece 和 SentencePiece。"
hasQuiz: false
---

在像 Transformer 这样的模型处理文本之前，原始字符序列必须被转换为一系列数字 ID。这个转换过程被称为分词。虽然像按空格分割文本这样的简单方法对小任务有效，但它们难以应对用于大型语言模型（LLM）的海量数据集中的庞大词汇量和形态变化。处理未知词（词汇表外词或 OOV）以及管理可能数百万个独特的词需要更复杂的方法。

本章侧重于旨在解决这些挑战的子词分词算法。您将了解字节对编码（BPE）和 WordPiece，这些技术基于频繁的子词单元而非整个词来构建词汇表。我们还将讨论 SentencePiece 框架、特殊标记（如 `[CLS]`、`[SEP]`）的作用和管理，以及选择词汇表大小 ($|V|$)、平衡模型表达能力和计算效率的实际考量。学习结束时，您将明白如何有效地为大型模型准备文本数据。
