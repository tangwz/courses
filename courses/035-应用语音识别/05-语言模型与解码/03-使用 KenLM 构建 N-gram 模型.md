# 使用 KenLM 构建 N-gram 模型

来源：[原文](https://apxml.com/zh/courses/applied-speech-recognition/chapter-5-language-modeling-decoding/building-n-gram-model-kenlm)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管声学模型擅长识别声音，但它们缺乏对语言的理解。为了避免出现“wreck a nice beach”这样无意义的转录，我们引入了一个语言模型，它根据词语序列的统计可能性进行评分。一种对此有效且广泛使用的方法是 N-gram 模型，我们可以使用 KenLM 工具包高效构建它。

### 基于 N-gram 的统计语言建模

N-gram 模型基于一个简单而有力的前提，即马尔可夫假设。我们不计算一个词语基于其之前所有词语历史的概率（这在计算上不可行），而是通过仅查看最后 $n-1$ 个词语来近似此概率。

对于词语序列 $W = w_1, w_2, ..., w_m$，其概率为：


$$
P(W) = P(w_1) P(w_2|w_1) P(w_3|w_1, w_2) ... P(w_m|w_1, ..., w_{m-1})
$$


N-gram 模型简化了此计算。例如：

- **一元**模型（$n=1$）假定每个词语是独立的：$P(W) \approx \prod_i P(w_i)$。
- **二元**模型（$n=2$）基于前一个词语来确定每个词语的条件概率：$P(W) \approx \prod_i P(w_i|w_{i-1})$。
- **三元**模型（$n=3$）基于前两个词语来确定每个词语的条件概率：$P(W) \approx \prod_i P(w_i|w_{i-2}, w_{i-1})$。

这些概率通过计算大量文本（称为文本语料库）中出现的次数来学习。对于二元模型，概率 $P(\text{speech} | \text{recognize})$ 通过计算语料库中短语“recognize speech”出现的次数，并除以“recognize”出现的总次数来估计。

### KenLM 工具包介绍

KenLM 是一个高度优化的库，专门用于创建、存储和查询 N-gram 语言模型。它因速度快和内存高效这两个主要原因，受到许多语音识别系统的青睐。它可以处理千兆字节的文本，生成紧凑的二进制格式语言模型，并以非常低的延迟进行查询，这对于构建响应式 ASR 系统极为重要。

### 语言模型构建流程

使用 KenLM 创建语言模型涉及一个清晰的三步流程：获取和清理文本语料库、构建模型以及将其转换为高效的二进制格式。

> 创建 KenLM 语言模型的过程，从原始文本到最终优化过的二进制文件。

#### 1. 准备训练语料库

你的语言模型质量完全取决于文本语料库的质量和相关性。对于通用 ASR，来自维基百科或公共领域书籍的大型语料库是一个不错的起点。对于特定应用场景，如转录医疗记录，你会使用医疗文献语料库。

在将文本输入 KenLM 之前，必须对其进行清理和规范化。这通常包括：

- 将所有文本转换为小写。
- 去除标点符号。
- 扩展缩写（例如：“don't”->“do not”）。
- 规范化数字（例如：“10”->“ten”）。

下面是一个简单的 Python 脚本，用于对文件 `raw_text.txt` 执行基本清理，并将结果保存到 `corpus.txt`。

```python
import re

# 用于创建词汇表的简单词集
# 在实际系统中，你会从训练数据的转录文本中生成它
vocab = {"the", "a", "an", "is", "of", "and", "recognize", "speech", "wreck", "nice", "beach", "it", "was"}

def normalize_text(text):
    text = text.lower()
    # 只保留字母字符和空格
    text = re.sub(r'[^a-z\s]', '', text)
    # 根据词汇表进行分词和过滤
    words = [word for word in text.split() if word in vocab]
    return " ".join(words)

with open('raw_text.txt', 'r') as infile, open('corpus.txt', 'w') as outfile:
    for line in infile:
        cleaned_line = normalize_text(line)
        if cleaned_line: # 避免写入空行
            outfile.write(cleaned_line + '\n')

print("语料库预处理完成。输出已保存到 corpus.txt")
```

此脚本确保输入 KenLM 的文本一致且不含可能被误解为词语的字符。词汇过滤步骤对于确保语言模型只包含声学模型能生成的词语非常重要。

#### 2. 构建 ARPA 模型

语料库准备就绪后，你可以使用 KenLM 的 `lmplz` 命令来构建 N-gram 模型。该工具会读取你处理过的文本，并以名为 ARPA 的标准文本格式输出模型。

让我们构建一个三元（$n=3$）模型。`-o` 标志指定了 N-gram 的阶数。

```bash
# 假设 KenLM 已安装且在你的 PATH 环境变量中
lmplz -o 3 < corpus.txt > model.arpa
```

此命令会将 `corpus.txt` 文件流式传输到 `lmplz`，`lmplz` 将计算 N-gram 的数量并计算它们的概率。输出将保存到 `model.arpa`。ARPA 文件是可读的，并列出了所有已学习的一元、二元和三元模型的对数概率和回退权重 (weight)。

#### 3. 创建高效二进制模型

尽管 ARPA 格式便于检查，但在解码期间进行快速查找的效率不高。KenLM 提供了另一个工具 `build_binary`，用于将 `.arpa` 文件转换为压缩的二进制格式，这显著减少了内存使用并提高了查询速度。

```bash
build_binary model.arpa model.binary
```

生成的 `model.binary` 文件是你将在 ASR 解码器中使用的文件。

### 查询语言模型

为了查看模型的实际效果，你可以使用 KenLM 的 `query` 工具。这允许你对句子进行评分并查看模型为其分配的 log10 概率。更高的分数（即负值更小）表明根据语料库该句子更可能出现。

让我们测试一下我们的运行示例。

```bash
$ echo "recognize speech" | query model.binary
-1.875333  recognize speech   p: -0.983 -0.892 </s> -0.521

$ echo "wreck a nice beach" | query model.binary
-4.129871  wreck a nice beach   p: -1.341 -1.112 -0.998 -0.678 </s> -0.521
```

输出显示了每个句子的总 log10 概率。如预期，“recognize speech”获得了明显更好的分数（$-1.87$），而“wreck a nice beach”则为（$-4.12$）。在解码过程中，这种差异正是当声学证据不明确时，系统能够选择正确转录的原因。有了这个 `model.binary` 文件，我们现在可以将其集成到解码算法中，以提高我们 ASR 系统的准确性。

## 参考资料

- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Online manuscript
  一本权威教材，全面涵盖了n-gram语言模型、其数学基础以及在语音识别中的实际应用。
- [KenLM: Faster and Smaller Language Model Queries](https://aclanthology.org/W11-2114) — Kenneth Heafield (2011)
  Journal: Proceedings of the Sixth Workshop on Statistical Machine Translation; Publisher: Association for Computational Linguistics; Pages: 187-197; DOI: [10.3115/v1/W11-2114](https://doi.org/10.3115/v1/W11-2114)
  介绍KenLM的原创研究论文，详细阐述了其用于n-gram语言模型构建和查询的高效算法，对大规模语音识别系统至关重要。

---

[上一节](02-N-gram%20%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.md) · [下一节](04-%E6%A8%A1%E5%9E%8B%E6%95%B4%E5%90%88%E7%9A%84%E8%A7%A3%E7%A0%81%E5%9B%BE.md)
