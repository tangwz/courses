---
course: "how-to-build-a-large-language-model"
chapter: "tokenization-large-vocabularies"
lesson: "sentencepiece-implementation"
sourceId: 5971
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-5-tokenization-large-vocabularies/sentencepiece-implementation"
title: "SentencePiece 实现"
description: "介绍 SentencePiece 这一包含 BPE 和 Unigram 模型的框架。"
order: 4
plots: []
sourceHash: "daac0d4e1bfac2209b76abffd49e6309936f802fba245837481bf02d0341ef99"
sourceCorrections: []
---

尽管像字节对编码（BPE）和WordPiece这样的算法比简单的词语切分有许多优点，但它们通常对预分词 (tokenization)文本进行操作，这些文本一般通过空格切分。这一预处理步骤可能存在问题。它带来了对词语边界的假定，而这些假定并非适用于所有语言（例如中文、日文、泰语），并且可能永久丢失有关空格变化的有用信息。此外，管理单独的预分词脚本会增加数据处理流程的复杂度。

由Google开发的SentencePiece提供了一个统一的框架，解决了这些局限。它直接处理原始文本序列，将输入视为Unicode字符流。这避免了对特定语言预分词器 (tokenizer)的需求，使其成为多语言模型的一种适应性强的工具。

### 直接处理原始文本

SentencePiece的主要区别在于它不假定空格表示词语边界。相反，它将空格视为任何其他字符。在构建词表时，SentencePiece通常会明确地编码空格，通常在应用子词 (subword)算法（如BPE或Unigram）之前，将其替换为像 `▁`（U+2581，一个下八分之一块）这样的元符号。

考虑文本 "Hello world."。

- 依赖空格的BPE可能会首先将其切分为 `["Hello", "world."]`，然后对每个部分应用BPE。
- SentencePiece将其视为 `"Hello world."`。它可能会学习像 `He`、`llo`、`_world`、`.` 这样的分词 (tokenization)单元（其中 `_` 表示编码的空格）。这使得它能够从分词序列精确地重构原始字符串，包括空格。

这种方法使得SentencePiece与语言无关。它不需要知道词语的开始或结束位置；它直接从数据中学习常用的字符序列。

### 支持BPE和Unigram模型

SentencePiece并非单一算法，而是一个实现多种子词 (subword)分词 (tokenization)策略的框架。主要有两种：

1. **BPE：** 它包含字节对编码算法的一个实现，与我们之前讨论的相似，但经过调整以直接处理Unicode字符流，并且通常包含明确的空格处理。
2. **Unigram语言模型：** 这是一种不同的方法。它从一个包含大量可能子词（例如，训练数据中的所有子字符串）的初始词表开始，然后根据Unigram语言模型，迭代地移除对训练语料库总体似然贡献最小的分词单元，直到达到预设的词表大小。在分词过程中，它会根据学习到的词表分词单元的Unigram概率，找到使概率最大的切分。虽然它功能强大，但在许多大型Transformer模型中，BPE仍然更常见。

在训练SentencePiece模型时，您通常选择所需的算法（`--model_type=bpe` 或 `--model_type=unigram`）。

### 规范化与可逆性

SentencePiece将其文本规范化功能直接整合到其流程中。它可以应用标准的Unicode规范化形式（如NFKC），并支持通过正则表达式定义的自定义规范化规则。这确保了在分词 (tokenization)开始前文本表示的一致性。

重要的一点是，SentencePiece分词被设计为可逆的。因为它直接处理原始Unicode字符流并明确处理空格，您几乎总是可以将ID序列解码回精确的原始（规范化）文本字符串。这比那些在预分词过程中丢弃空格信息的现有分词器 (tokenizer)有很大的优势。

### 训练和使用SentencePiece模型

`sentencepiece` 库提供了命令行工具和Python绑定，用于训练和使用模型。

**1. 训练：**
您通常从原始文本文件训练模型。假设您有一个文件 `corpus.txt`。

```bash
# 使用BPE的命令行训练示例
spm_train --input=corpus.txt --model_prefix=my_sp_model --vocab_size=16000 --model_type=bpe --character_coverage=1.0 --normalization_rule_name=nmt_nfkc_cf
```

- `--input`：您的原始训练文本数据路径。
- `--model_prefix`：输出文件（`my_sp_model.model` 和 `my_sp_model.vocab`）的基本名称。
- `--vocab_size`：目标词表大小 $|V|$。
- `--model_type`：要使用的算法（`bpe` 或 `unigram`）。
- `--character_coverage`：目标是用基本单字符分词 (tokenization)单元覆盖至少这个比例的输入字符。对于处理多样的文字系统很重要。
- `--normalization_rule_name`：预定义的规范化规则（例如，`nmt_nfkc_cf` 应用NFKC规范化和大小写折叠）。

这会生成 `my_sp_model.model`（包含词表和合并规则/概率）和 `my_sp_model.vocab`（一个人类可读的词表列表）。

**2. 在Python中使用（PyTorch环境下）：**
训练完成后，您加载 `.model` 文件，并将其用于编码和解码。

```python
import sentencepiece as spm
import torch

# 加载训练好的SentencePiece模型
sp = spm.SentencePieceProcessor()
sp.load('my_sp_model.model') # 加载 my_sp_model.model

# 示例文本
text = "SentencePiece is useful for LLMs."

# 将文本编码为ID
ids = sp.encode_as_ids(text)
print(f"原始文本: {text}")
print(f"编码ID: {ids}")

# 将ID转换为PyTorch张量
tensor_ids = torch.tensor(ids, dtype=torch.long)
print(f"PyTorch张量: {tensor_ids}")

# 将ID解码回文本
decoded_text = sp.decode_ids(ids)
print(f"解码文本: {decoded_text}")

# 将文本编码为子词片段
pieces = sp.encode_as_pieces(text)
print(f"编码片段: {pieces}")
# 示例输出（将根据训练数据和词表大小而异）：
# [' S', 'entence', 'P', 'iece', ' is', ' useful', ' for', ' L', 'L', 'Ms', '.']
# 注意 ' ' 表示空格。

# 获取词表大小和特殊分词单元ID
vocab_size = sp.get_piece_size()
bos_id = sp.bos_id() # 句子开始
eos_id = sp.eos_id() # 句子结束
pad_id = sp.pad_id() # 填充
unk_id = sp.unk_id() # 未知分词单元

print(f"词表大小: {vocab_size}")
print(f"BOS ID: {bos_id}, EOS ID: {eos_id}, PAD ID: {pad_id}, UNK ID: {unk_id}")
```

在此示例中，`sp.encode_as_ids` 直接将原始字符串转换为整数列表。`sp.decode_ids` 执行逆向操作。通过 `encode_as_pieces` 获取片段的能力有助于理解分词过程。SentencePiece还定义了特殊分词单元的标准ID，如序列开始符（`BOS`）、序列结束符（`EOS`）、填充符（`PAD`）和未知符（`UNK`），这些对于为Transformer等模型准备输入批次很重要。

### 大型模型的优点

SentencePiece的设计为构建大型语言模型带来了几个好处：

- **无需预分词 (tokenization)：** 简化了数据处理流程，避免了特定语言的脚本。
- **语言无关：** 能够有效地处理具有不同书写系统和词语切分规则的多种语言。
- **无损重构：** 通常可以完美地重构原始规范化文本，保留空格。
- **高效率：** 用C++实现以提高速度，并提供Python绑定以便于使用。
- **集成规范化：** 一致地处理文本规范化。

通过将文本视为原始序列并采用BPE或Unigram等数据驱动的方法，SentencePiece提供了一种灵活的分词方法，非常适合现代大型语言模型开发中使用的多样和海量数据集。它避开了依赖空格的分词器 (tokenizer)的限制，并为准备文本数据提供了一个统一的解决方案。

## 参考资料

- [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://aclanthology.org/D18-2012/) — Taku Kudo, John Richardson (2018)
  Journal: Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations; Publisher: Association for Computational Linguistics; Pages: 66-71; DOI: [10.18653/v1/D18-2012](https://doi.org/10.18653/v1/D18-2012)
  介绍SentencePiece，详细说明其语言无关的子词分词设计、显式空格处理以及BPE和Unigram模型的集成。
- [SentencePiece GitHub Repository and Documentation](https://github.com/google/sentencepiece) — Google (2024)
  提供官方源代码、安装说明、命令行工具使用、Python API示例，以及训练参数和模型类型的详细解释。
- [Tokenizers - Hugging Face Documentation](https://huggingface.co/docs/tokenizers/index) — Hugging Face (2024)
  解释了包括SentencePiece在内的各种子词分词算法在现代自然语言处理模型中的原理，并提供实用的实现见解。
- [Natural Language Processing with Transformers](https://learning.oreilly.com/library/view/natural-language-processing/9781098136789/) — Lewis Tunstall, Leandro von Werra, Mario Šaško (2022)
  Publisher: O'Reilly Media; Pages: Chapter 2: Text Classification, From Text to Tokens
  概述分词技术，包括SentencePiece，在构建和应用Transformer模型中的作用，涵盖理论和实践两方面。
