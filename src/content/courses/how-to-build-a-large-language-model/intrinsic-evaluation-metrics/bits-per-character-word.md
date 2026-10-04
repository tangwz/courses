---
course: "how-to-build-a-large-language-model"
chapter: "intrinsic-evaluation-metrics"
lesson: "bits-per-character-word"
sourceId: 6058
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-21-intrinsic-evaluation-metrics/bits-per-character-word"
title: "每字符/词比特数"
description: "阐释与信息论相关的其他内在评估指标。"
order: 4
plots: []
sourceHash: "59830c7149db2fc439ceac098296c8728341763c803cc3801c31e6cc3571c96f"
sourceCorrections: []
---

困惑度是语言模型最常用的一种内在评估指标，如前所述。与信息论紧密关联的指标则提供不同视角，尤其在比较采用不同分词 (tokenization)方案的模型时。这些指标计算根据模型的概率分布，编码每个文本单位（字符或词/词元 (token)）平均所需的比特数。

训练期间使用的交叉熵损失本质上是真实下一个词元的平均负对数似然，通常使用自然对数（以$e$为底）计算：
$H(p, q) = -\frac{1}{N}\sum_{i=1}^N \log_e q(w_i | w_{<i}; \theta)$
$q$ 是模型的分布，$N$ 是词元数量。困惑度是此损失的指数化：$PPL = \exp(H(p, q))$。

然而，信息论通常以*比特*（以2为底的对数）来衡量信息内容。一个为观测序列分配更高概率的模型，其“确定性”更强，根据其概率分配，平均编码该序列所需的比特数更少。这使得我们关注每字符比特数和每词/词元比特数等指标。

### 每字符比特数 (BPC)

每字符比特数 (BPC) 衡量给定模型预测下，编码文本序列中每个*字符*所需的平均比特数。它的计算方法是，取使用以2为底的对数计算的交叉熵损失，然后除以序列中的总字符数 ($C$)，而不是词元 (token)数量：

$BPC = -\frac{1}{C}\sum_{j=1}^C \log_2 p(char_j | context; \theta)$

然而，大多数大型语言模型是基于词元（词或子词 (subword)）工作的，而非单个字符。计算每个字符的精确概率需要一个字符级别模型，或对词元概率进行复杂的边缘化处理。对于基于词元的模型，一个更实用的方法是计算每个词元的标准交叉熵损失（以$e$为底），然后将其转换为比特（以2为底），并除以原始文本中的字符数量。

每词元计算的交叉熵损失（以$e$为底，表示为$H_e$）与 BPC 的关系如下：

$BPC = \frac{H_e \times N}{C \times \ln(2)}$

此处，$N$ 是词元数量，$C$ 是评估文本中的字符数量。$\ln(2)$ 因子将自然对数转换为以2为底的对数（$\log_2(x) = \frac{\ln(x)}{\ln(2)}$）。

BPC 的主要优点是它对所选分词 (tokenization)方法相对不敏感。由于它通过字符数量进行归一化 (normalization)，因此可以在相同的原始文本数据上，更公正地比较使用不同分词器 (tokenizer)（例如，BPE、WordPiece、字符级别）的模型。一个模型可能仅因为其分词器将词分解成许多小片段而获得不错的困惑度分数，但其BPC可以表明它是否在建模底层字符序列方面确实表现更佳。

我们来看一个 PyTorch 示例。假设您已从评估循环中获得每词元的平均交叉熵损失以及字符/词元计数：

```python
import torch
import math

# 假设这些值来自评估结果
avg_cross_entropy_loss_per_token = 1.85 # 示例损失 (自然对数底)
total_tokens_in_eval_set = 50000      # 示例词元数量
total_chars_in_eval_set = 200000     # 示例字符数量

# 计算以2为底的交叉熵，按词元归一化
avg_bits_per_token = avg_cross_entropy_loss_per_token / math.log(2)

# 根据模型计算数据集的总比特数
total_bits = avg_bits_per_token * total_tokens_in_eval_set

# 计算每字符比特数 (BPC)
bpc = total_bits / total_chars_in_eval_set

print(
    f"每词元平均交叉熵损失（以e为底）: "
    f"{avg_cross_entropy_loss_per_token:.4f}"
)
print(f"每词元平均比特数 (BPT): {avg_bits_per_token:.4f}")
print(f"每字符比特数 (BPC): {bpc:.4f}")

# 示例输出：
# 每词元平均交叉熵损失（以e为底）: 1.8500
# 每词元平均比特数 (BPT): 2.6691
# 每字符比特数 (BPC): 0.6673
```

### 每词/词元 (token)比特数 (BPW/BPT)

每词比特数 (BPW) 或每词元比特数 (BPT) 对于基于词元的模型来说更简单。它表示根据模型的概率分配，编码每个*词元*（如果使用词级别分词 (tokenization)，则为词）所需的平均比特数。

它与在词元上计算的交叉熵损失（以$e$为底，$H_e$）和困惑度（PPL）直接关联：

$BPT = -\frac{1}{N}\sum_{i=1}^N \log_2 p(词元_i | 词元_{<i}; \theta) = \frac{H_e}{\ln(2)}$

此外，BPW/BPT 与困惑度有直接关联：

$PPL = 2^{BPT}$
$BPT = \log_2(PPL)$

这意味着 BPW/BPT 只是困惑度的以2为底的对数。如果一个模型在数据集上的困惑度为 16，这意味着平均而言，预测下一个词元的难度，相当于在 $16 = 2^4$ 个选项中均匀选择。这对应于每词元 4 比特的 BPT。

```python
import torch
import math

# 示例值
perplexity = 16.0
avg_cross_entropy_loss_per_token = math.log(perplexity) # H_e = ln(PPL)

# 从困惑度计算 BPT
bpt_from_ppl = math.log2(perplexity)

# 从交叉熵损失（以e为底）计算 BPT
bpt_from_loss = avg_cross_entropy_loss_per_token / math.log(2)

print(f"困惑度 (PPL): {perplexity:.4f}")
print(
    f"每词元平均交叉熵损失（以e为底）: "
    f"{avg_cross_entropy_loss_per_token:.4f}"
)
print(f"从 PPL 计算的每词元比特数 (BPT): {bpt_from_ppl:.4f}")
print(f"从损失计算的每词元比特数 (BPT): {bpt_from_loss:.4f}")

# 示例输出：
# 困惑度 (PPL): 16.0000
# 每词元平均交叉熵损失（以e为底）: 2.7726
# 从 PPL 计算的每词元比特数 (BPT): 4.0000
# 从损失计算的每词元比特数 (BPT): 4.0000
```

尽管 BPW/BPT 是困惑度的直接转换，但以“比特”来思考，能与信息论及压缩限制建立更强的关联。更低的 BPT 表明模型根据其学习到的概率分布，在压缩文本序列中的信息方面更高效。然而，与 BPC 不同，BPW/BPT 高度依赖于所使用的分词方法。使用子词 (subword)单元的模型在相同文本上通常会比字符级别模型具有更低的 BPT，仅仅因为预测一个更长的词元单元一次能覆盖更多字符。

总之，BPC 和 BPW/BPT 都提供了关于模型内在性能的信息论视角。BPC 提供了一种与分词器 (tokenizer)无关的衡量方法，在比较根本不同的模型时很有用；而 BPW/BPT 是困惑度的直接对数转换，常用于评估基于特定分词方案的词元模型。这两种指标都量化 (quantization)了模型的概率分布与数据的匹配程度，数值越低表示匹配度越好。

## 参考资料

- [Elements of Information Theory](https://books.google.com/books/about/Elements_of_Information_Theory.html?id=lq-WAAAAQBAJ) — Thomas M. Cover and Joy A. Thomas (2006)
  Publisher: Wiley-Interscience
  一本经典教科书，提供了信息论的数学基础，包括熵、交叉熵和比特作为信息内容度量的概念。
- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  一本权威的自然语言处理教科书，详细涵盖了语言建模、困惑度和各种评估指标。
- [Exploring the Limits of Language Modeling](http://proceedings.mlr.press/v48/jozefowicz16.pdf) — Rafał Jozefowicz, Oriol Vinyals, Samy Bengio, Mohammad Norouzi (2016)
  Journal: Proceedings of the 33rd International Conference on Machine Learning; Pages: 1721-1729; DOI: [10.5555/3045390.3045437](https://doi.org/10.5555/3045390.3045437)
  本文研究了大规模语言模型的性能，并使用每字符比特（BPC）作为评估和比较模型的关键指标。
- [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/) — Rico Sennrich, Barry Haddow, Alexandra Birch (2016)
  Journal: Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers); Publisher: Association for Computational Linguistics; Pages: 1715-1725; DOI: [10.18653/v1/P16-1162](https://doi.org/10.18653/v1/P16-1162)
  引入了字节对编码（BPE），一种广泛使用的子词分词方法，阐明了需要BPC等指标来公平比较使用不同分词方案的模型。
