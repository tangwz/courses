# WordPiece 分词

来源：[原文](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-5-tokenization-large-vocabularies/wordpiece-tokenization)

[返回章节目录](README.md) · [返回课程目录](../README.md)

WordPiece 词元 (token)化是一种构建大型词汇表 (vocabulary)的方法，由 Google 开发，并特别用于 BERT（来自 Transformer 的双向编码器表示）模型及其变体。此方法通过迭代合并单元来构建词汇表。与字节对编码（BPE）合并最常出现的词元对不同，WordPiece 的合并准则基于最大化训练数据的似然性，而非原始频率计数。

该过程与 BPE 类似：用训练数据中所有单个字符初始化词汇表。然后，迭代地考虑合并相邻词元。主要区别在于选择合并哪个对。WordPiece 选择一对（比如 $A$ 和 $B$），使得将它们合并成单个词元 $AB$ 会导致训练语料库的似然性增加幅度最大，这假设词元上使用单一词语言模型。

### 似然性最大化准则

设想训练语料库是从我们当前词汇表 (vocabulary) $V$ 生成的一系列词元 (token)。语料库的似然性 $L$ 是序列中每个词元出现概率的乘积：


$$
L = \prod_{token \in Corpus} P(token)
$$


词元 $P(token)$ 的概率通常估算为其在语料库中的频率除以词元总数：


$$
P(token) = \frac{count(token)}{\sum_{t \in V} count(t)}
$$


当我们考虑将两个词元 $A$ 和 $B$ 合并成一个新词元 $AB$ 时，我们实际上会修改所有 $A$ 和 $B$ 相邻出现位置的词元序列。这会改变 $A$、$B$ 和 $AB$ 的计数，以及语料库中词元的总数（因为每次合并都会使词元计数减少一）。WordPiece 会评估语料库当前分词 (tokenization)中所有可能的相邻对的这种变化。选择合并后对修改后的语料库产生最高似然性 $L$ 的对 $(A, B)$，并将其添加到词汇表以进行下一次迭代。

实际操作中，计算完整的似然性变化可能很复杂。通常，会使用近似值或与似然性相关的分数。一种常见的考虑方式是选择能使如下分数最大化的合并 $(A, B) \to AB$：


$$
score(A, B) = \frac{count(AB)}{count(A) \times count(B)}
$$


虽然这并非精确的似然性变化，但它反映了相似的直觉：相对于它们各自的频率，将经常一起出现的对进行合并是有益的。确切的评分函数在不同实现之间可能有所不同。

### WordPiece 的实践：前缀和实现

与许多 BPE 实现一样，WordPiece 通常通过给表示单词延续部分的词元 (token)添加一个特殊前缀（通常是 `##`）来处理单词内部的子词 (subword)。最初的单词划分可能发生在空白处，然后 WordPiece 分词 (tokenization)会在这些初始单词单元内部进行。

例如，单词“hugging”最初可能会被拆分为字符：`h`、`u`、`g`、`g`、`i`、`n`、`g`。通过基于似然性最大化的迭代合并，词汇表 (vocabulary)最终可能包含 `hug` 和 `##ging`。单词“hugging”随后将被分词为 `['hug', '##ging']`。`##` 表示 `ging` 无空格地连接到前一个词元。这使得模型能够区分一个词元是单词的开头部分还是中间部分，并实现明确的逆分词。

我们来看看如何在 PyTorch 中通过 Hugging Face `transformers` 库使用预训练 (pre-training)的 WordPiece 分词器 (tokenizer)，特别是 BERT 所用的分词器。

```python
# 确保你已安装 transformers 和 torch
# pip install transformers torch

from transformers import BertTokenizer

# 加载预训练的 BERT 分词器（它使用 WordPiece）
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

sentence = "WordPiece maximizes likelihood."

# 对句子进行分词
tokens = tokenizer.tokenize(sentence)

# 将词元转换为对应的 ID
input_ids = tokenizer.convert_tokens_to_ids(tokens)

# 获取包含特殊词元 ([CLS], [SEP]) 的完整输出
encoded_input = tokenizer(sentence)

print(f"原始句子: {sentence}")
print(f"词元: {tokens}")
print(f"词元 ID: {input_ids}")
print(f"完整编码输入（含特殊词元）:")
print(f"{encoded_input['input_ids']}")
print(f"解码后的完整输入: {tokenizer.decode(encoded_input['input_ids'])}")
```

运行此代码会产生如下所示的输出：

```
原始句子: WordPiece maximizes likelihood.
词元: ['word', '##piece', 'max', '##imi', '##zes', 'like', '##li', '##hood', '.']
词元 ID: [2773, 19352, 4011, 24027, 16464, 2066, 2135, 12731, 1012]
完整编码输入（含特殊词元）:
[101, 2773, 19352, 4011, 24027, 16464, 2066, 2135, 12731, 1012, 102]
解码后的完整输入: [CLS] wordpiece maximizes likelihood. [SEP]
```

请注意“WordPiece”如何被拆分为 `word` 和 `##piece`，“maximizes”被拆分为 `max`、`##imi` 和 `##zes`，“likelihood”被拆分为 `like`、`##li` 和 `##hood`。`##` 前缀清楚地标记 (token)了不作为单词开头的子词单元。最终的 `encoded_input['input_ids']` 在开头包含特殊词元 ID `[CLS]` (101)，在结尾包含 `[SEP]` (102)，这些是 BERT 风格模型的标准要求。

### WordPiece 与 BPE 的比较

WordPiece 和 BPE 之间的主要区别在于词元 (token)合并的准则：BPE 使用频率，而 WordPiece 使用语料库似然性最大化（或相关分数）。这种差异可能导致最终词汇表 (vocabulary)以及单词分段方式的不同。WordPiece 可能偏向于创建统计上更可能的单元的合并，即使它们在计数上并非严格地最频繁的对。

> BPE 和 WordPiece 之间合并准则的不同。

实际应用中，这两种方法都能有效地创建子词 (subword)词汇表，从而很好地处理大型文本语料库，缓解 OOV 问题并保持词汇量在可管理范围内。它们之间的选择通常取决于它们最初开发的特定模型架构（例如，BERT 使用 WordPiece，GPT-2 使用 BPE）或在特定任务和数据集上的实际表现。实现方式随处可见，尤其是在 Hugging Face 的 `tokenizers` 库等框架中，这些框架通常会为最终用户抽象掉细微差别。然而，了解其基本机制有助于理解原始文本如何转换为大型语言模型处理的数字序列。

## 参考资料

- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — Rico Sennrich, Barry Haddow, Alexandra Birch (2016)
  Journal: Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers); Publisher: Association for Computational Linguistics; Pages: 1715-1725; DOI: [10.48550/arXiv.1508.07909](https://doi.org/10.48550/arXiv.1508.07909)
  引入了字节对编码 (BPE) 用于神经网络机器翻译中的子词分词，该方法在概念上与 WordPiece 类似。
- [Tokenizers in 🤗 Transformers](https://huggingface.co/docs/transformers/main/en/tokenizer_summary) — Hugging Face team (2024)
  提供了包括 BertTokenizer 及其 WordPiece 实现的各种分词器的实用指南和 API 参考。

---

[上一节](02-%E5%AD%97%E8%8A%82%E5%AF%B9%E7%BC%96%E7%A0%81%20%28BPE%29%20%E7%AE%97%E6%B3%95.md) · [下一节](04-SentencePiece%20%E5%AE%9E%E7%8E%B0.md)
