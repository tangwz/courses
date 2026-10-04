---
course: "introduction-to-transformer-models"
chapter: "training-implementing-transformers"
lesson: "data-preparation-tokenization"
sourceId: 3020
sourceUrl: "https://apxml.com/zh/courses/introduction-to-transformer-models/chapter-4-training-implementing-transformers/data-preparation-tokenization"
title: "数据准备：分词"
description: "讨论Transformer模型中常见的分词策略，例如WordPiece、BPE和SentencePiece。"
order: 1
plots: []
sourceHash: "cc78fb9be323eb9505938c36f20eaee21e46fda456f1e172320484b99a3d5363"
sourceCorrections: []
---

在 Transformer 模型处理文本之前，原始字符串必须转换为它能理解的数值形式。这种转换过程称为**分词 (tokenization)**。尽管存在更简单的方法，例如通过空格拆分文本以获取单词，但它们很快就会遇到大词汇表 (vocabulary)和训练期间未见的词（词汇表外词或 OOV 词）问题。Transformer 模型通常使用更高级的技术，称为**子词 (subword)分词**算法。

子词分词的核心思想是将单词分解为更小、频繁出现的单元。这种方法提供了一种平衡：它能维持词汇表大小易于管理，同时大幅降低遇到未知令牌的可能性。子词分词器 (tokenizer)可能不会将“transformer”和“transformers”表示为两个完全独立的令牌，而是将它们表示为 `["transform", "er"]` 和 `["transform", "ers"]`。根词“transform”被学习，而“er”和“ers”等常见词缀则单独学习。这使得模型能够根据其学习到的子词单元，潜在地理解词语的新颖组合或变体。

下面介绍 Transformer 模型中常用的子词分词算法：

### 字节对编码 (BPE)

字节对编码 (BPE) 最初是一种数据压缩算法，后被修改用于文本分词 (tokenization)。它以迭代方式进行：

1. **初始化：** 从训练语料库中所有单个字符组成的词汇表 (vocabulary)开始。
2. **迭代：** 重复计算语料库中所有相邻的符号对（最初是字符，后来是合并后的子词 (subword)）。
3. **合并：** 找出最常见的符号对（例如，'t' 后跟 'h' 形成 'th'），并将它们合并成一个新符号 'th'。将此新符号添加到词汇表。
4. **重复：** 继续此合并过程，直至达到预设的步数或期望的词汇表大小。

以一个简单的例子来说明，包含少量语料和几次合并：

- **语料：** `{"low low low", "lowest lowest", "newer newer", "wider wider"}`
- **初始词汇表（字符）：** `{'l', 'o', 'w', ' ', 's', 't', 'n', 'e', 'r', 'i', 'd'}`
- **步骤1：** `('l', 'o')`、`('o', 'w')`、`('w', ' ')`、`('e', 'r')` 等对频繁出现。假设 `('e', 'r')` 是最频繁的对。
- **合并1：** 将 'e' 和 'r' 合并为 'er'。**新词汇表：** `{'l', 'o', 'w', ' ', 's', 't', 'n', 'er', 'i', 'd'}`。语料变为 `{"low low low", "lowest lowest", "new'er' new'er'", "wid'er' wid'er'"}`。
- **步骤2：** 现在计算涉及 'er' 的对，如 `('w', 'er')` 或 `(' ', 'er')`。也许 `('l', 'o')` 现在是最频繁的。
- **合并2：** 将 'l' 和 'o' 合并为 'lo'。**新词汇表：** `{'lo', 'w', ' ', 's', 't', 'n', 'er', 'i', 'd'}`。语料变为 `{"'lo'w 'lo'w 'lo'w", "'lo'west 'lo'west", "new'er' new'er'", "wid'er' wid'er'"}`。
- **步骤3：** 也许 `('lo', 'w')` 变得频繁。
- **合并3：** 将 'lo' 和 'w' 合并为 'low'。**新词汇表：** `{'low', ' ', 's', 't', 'n', 'er', 'i', 'd', 'e', 'w'}`（注意：'e' 和 'w' 对于“lowest”仍然是必需的）。语料：`{"'low' 'low' 'low'", "'low'est 'low'est", "new'er' new'er'", "wid'er' wid'er'"}`。

此过程会继续进行，逐渐构建出“lowest”、“newer”、“wider”等常见子词，也可能根据期望的词汇表大小提前停止。

> “newer”一词的BPE合并步骤简化图。初始字符根据频率组合形成“er”等子词，可能还有“new”（假设'n','e'已合并），最终形成“newer”。

### WordPiece

WordPiece 与 BPE 相似，但采用不同的合并标准。WordPiece 不合并最常见的对，而是合并在给定词汇表 (vocabulary)的情况下使训练数据似然度最大化的对。它基本上是询问：“在由词汇表定义的简单语言模型下，哪种合并能使训练数据最可能？”。WordPiece 显著用于 BERT 及相关模型。它通常会产生与语言词素良好对齐 (alignment)的子词 (subword)，尽管这不是其明确目标。WordPiece 的一个常见约定是，用 `##` 作为继续一个单词的子词前缀（例如，“transformer”可能变为 `["transform", "##er", "##s"]`）。

### SentencePiece

SentencePiece 将输入文本视为原始序列，包括空格。与 BPE 和 WordPiece 通常需要预分词 (tokenization)（如按空格拆分）不同，SentencePiece 直接操作原始字节流或 Unicode 字符。它明确编码空格，通常使用  (U+2581) 等特殊字符表示令牌内的空格。这使得它对于词边界未由空格明确定义的语言尤其有效，并允许在不同语言之间采用单一、一致的分词/反分词过程，而无需特定语言的逻辑。

### 特殊令牌

除了从数据中提取的子词 (subword)外，Transformer 分词 (tokenization)器 (tokenizer)还包含几个对模型运行不可或缺的特殊令牌：

- **`[PAD]` (填充令牌)：** 用于使批次中的序列长度一致。模型通过注意力掩码学习忽略这些令牌。
- **`[UNK]` (未知令牌)：** 表示分词器词汇表 (vocabulary)中不存在的任何子词。理想情况下，子词分词应最大程度地减少 `[UNK]` 的出现。
- **`[CLS]` (分类令牌)：** 通常添加到输入序列的开头。此令牌对应的最终隐藏状态常用于分类任务的聚合序列表示。
- **`[SEP]` (分隔令牌)：** 用于在单个输入序列中分隔不同的文本段（例如，问答中问题与上下文 (context)的分隔，或用于下一句预测的两个句子）。
- **`[MASK]` (掩码令牌)：** 专门用于掩码语言模型预训练 (pre-training)（如BERT中），其中输入令牌被随机替换为 `[MASK]`，模型学习预测原始令牌。

### 使用分词 (tokenization)器 (tokenizer)库

从零开始实现这些分词算法可能很复杂。幸运的是，有出色的库可以高效地处理此问题。Hugging Face 的 `tokenizers` 库提供了高度优化的 BPE、WordPiece 及其他算法实现，让您能够轻松加载预训练 (pre-training)的分词器或在特定语料库上训练自己的分词器。

```python
# 使用 Hugging Face 分词器的示例
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# 1. 使用 BPE 模型初始化分词器
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# 2. 自定义预分词（例如，首先按空格拆分）
tokenizer.pre_tokenizer = Whitespace()

# 3. 定义训练器（词汇表大小、特殊令牌）
trainer = BpeTrainer(vocab_size=10000, special_tokens=[
    "[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"
])

# 4. 在文本文件上训练
files = ["path/to/your/corpus.txt", ...] # 文本文件列表
tokenizer.train(files, trainer)

# 5. 保存分词器
tokenizer.save("my-bpe-tokenizer.json")

# 6. 加载并使用
tokenizer = Tokenizer.from_file("my-bpe-tokenizer.json")
output = tokenizer.encode("This is example text.")

print(f"Tokens: {output.tokens}")
# 示例输出：令牌：['This', 'Ġis', 'Ġexample', 'Ġtext', '.']
# （注意：'\u0120' 在 SentencePiece/BPE 变体中常表示空格）

print(f"IDs: {output.ids}")
# 示例输出：ID：[713, 164, 1794, 1036, 5]
```

理解分词是为 Transformer 准备数据的第一个实际步骤。算法和词汇表 (vocabulary)大小的选择会影响模型如何“看待”文本，从而影响其性能和泛化能力。一旦文本被转换为这些整数 ID 序列（如上文的 `output.ids`），您就可以继续创建适合训练的批次，这将在下一步讨论。

## 参考资料

- [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/) — Rico Sennrich, Barry Haddow, Alexandra Birch (2016)
  Journal: Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers); Volume: 1; Pages: 1715-1725; DOI: [10.18653/v1/P16-1162](https://doi.org/10.18653/v1/P16-1162)
  这篇基础性论文将字节对编码（BPE）引入自然语言处理领域，用于处理神经机器翻译中的稀有词和未登录词，使其成为一种常见的子词分词方法。
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/) — Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova (2019)
  Journal: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers); Volume: 1; Pages: 4171-4186; DOI: [10.18653/v1/N19-1423](https://doi.org/10.18653/v1/N19-1423)
  这篇开创性论文介绍了BERT模型，该模型显著使用了WordPiece分词。它展示了子词分词在大型Transformer模型中进行语言理解的实际应用和重要性。
- [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://aclanthology.org/D18-2012/) — Taku Kudo (2018)
  Journal: Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP); Pages: 166-173; DOI: [10.18653/v1/D18-2012](https://doi.org/10.18653/v1/D18-2012)
  这篇论文介绍了SentencePiece，一个与语言无关的子词分词器，它直接处理原始文本，包括空格，使其特别适用于多种语言和一致的分词/去分词过程。
- [tokenizers: Fast State-of-the-Art Tokenizers](https://huggingface.co/docs/tokenizers/index) — Hugging Face (2023)
  Publisher: Hugging Face
  Hugging Face `tokenizers`库的官方文档，该库提供了BPE、WordPiece和其他用于Transformer模型的子词分词算法的优化实现。
- [Natural Language Processing with Transformers](https://www.oreilly.com/library/view/natural-language-processing/9781098103231/) — Lewis Tunstall, Leandro von Werra, Thomas Wolf (2022)
  Publisher: O'Reilly Media
  一本实用指南，包含各种分词方法、特殊标记及其在Hugging Face Transformers生态系统中的应用的详细解释和代码示例。
