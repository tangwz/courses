# 使用N-gram来获取语境信息

来源：[原文](https://apxml.com/zh/courses/nlp-fundamentals/chapter-2-nlp-feature-engineering/n-grams-for-context)

[返回章节目录](README.md) · [返回课程目录](../README.md)

文本表示方法，例如词袋模型和TF-IDF，通常侧重于单个词项的存在和频率。尽管TF-IDF通过根据词语的重要性赋予权重 (weight)，比简单的词袋模型有所改进，但这两种方法都存在一个主要限制：它们忽略了词序。考虑短语“good service”和“service not good”。在标准的词袋模型或TF-IDF表示中，它们可能看起来非常相似，因为它们包含相同的词（“good”、“service”、“not”），只是重新排列了。然而，它们的含义是相反的。这种序列信息的缺失对许多自然语言处理（NLP）任务（如情感分析或机器翻译）可能不利。

为解决这个问题，我们可以扩展特征表示以包含相邻词序列。这就是N-gram发挥作用的地方。

### 什么是N-gram？

N-gram是从给定文本或语音样本中提取的包含$N$个元素的连续序列。“元素”通常是词语，但也可以是字符。

- **Unigram（1-gram）：** 这些是我们到目前为止在词袋模型和TF-IDF模型中处理的单个词语。对于句子“The quick brown fox”，其unigram是“The”、“quick”、“brown”、“fox”。
- **Bigram（2-gram）：** 这些是两个相邻词语的序列。对于同一个句子，其bigram是（“The”、“quick”）、（“quick”、“brown”）、（“brown”、“fox”）。通常表示为“The quick”、“quick brown”、“brown fox”。
- **Trigram（3-gram）：** 三个相邻词语的序列。例子：（“The”、“quick”、“brown”）、（“quick”、“brown”、“fox”），或者“The quick brown”、“quick brown fox”。
- **N-gram（通用）：** $N$个相邻词语的序列。

通过不仅基于单个词语（unigram）还基于词对（bigram）、三元组（trigram）或更长序列来生成特征，我们将局部词序纳入到文本表示中。

### 使用N-gram获取语境信息

这有何帮助？让我们再次审视我们的例子：“good service”对比“service not good”。

- **Unigram特征：** {good, service}, {service, not, good}
- **Bigram特征：** {“good service”}, {“service not”, “not good”}
- **组合Unigram+Bigram特征：** {good, service, “good service”}, {service, not, good, “service not”, “not good”}

注意添加bigram如何创建独特特征。Bigram“not good”清晰地捕获了负面情感，如果只单独查看unigram“not”和“good”，这部分信息就会遗漏。同样，像“New York”或“San Francisco”这样的bigram表示与其构成词语不同的含义。

使用N-gram使得模型能够根据这些短语学习模式，从而对文本有更丰富的理解。

> 示意图说明了从示例句子中生成unigram、bigram和trigram的过程。

### 权衡：特征空间爆炸

尽管N-gram增强了语境表示，但它们也带来了代价：潜在特征数量显著增加。

设想一个包含10,000个独立词语（unigram）的词汇表 (vocabulary)。

- 潜在bigram的数量是$10,000^2 = 100,000,000$。
- 潜在trigram的数量是$10,000^3 = 1,000,000,000,000$。

尽管这些潜在N-gram中有许多永远不会实际出现在语料库中，但*观察到的*N-gram数量仍然比unigram数量大得多。这导致：

1. **高维度：** 特征向量 (vector)变得非常长。
2. **稀疏性：** 任何给定文档的特征向量中，大多数条目将为零，因为只会出现所有可能N-gram中的一小部分。
3. **计算成本增加：** 处理和存储这些更大的特征集需要更多内存和计算能力。



[交互图表：示例：语料库中独立N-gram的增长](https://apxml.com/zh/courses/nlp-fundamentals/chapter-2-nlp-feature-engineering/n-grams-for-context#plot-zjdrme)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "x": [
        "Unigram (N=1)",
        "Bigram (N=2)",
        "Trigram (N=3)"
      ],
      "y": [
        10000,
        250000,
        1500000
      ],
      "type": "bar",
      "marker": {
        "color": [
          "#228be6",
          "#12b886",
          "#fab005"
        ]
      }
    }
  ],
  "layout": {
    "title": {
      "text": "示例：语料库中独立N-gram的增长",
      "x": 0.5,
      "xanchor": "center"
    },
    "xaxis": {
      "title": "N-gram类型"
    },
    "yaxis": {
      "title": "近似独立特征数量",
      "type": "log"
    },
    "margin": {
      "l": 60,
      "r": 20,
      "t": 40,
      "b": 40
    },
    "height": 350
  }
}
```

</details>



> 在一个包含1万个unigram词汇的语料库上，为不同N-gram大小生成的近似独立特征数量。请注意Y轴上的对数刻度，这突显了快速增长。实际数量很大程度上取决于语料库大小和语言特点。

### 实际应用

在实际中，我们很少只使用高阶N-gram（例如只使用trigram）。一种常见策略是将unigram与bigram结合，或者unigram、bigram和trigram都结合使用。这在获取语境信息的同时，也保留了unigram的信息。

$N$的选择（最大序列长度）以及是否包含低阶gram取决于具体任务和可用的计算资源。Bigram通常在获取语境信息和管理特征空间大小之间提供良好平衡。

您可以使用scikit-learn等库生成N-gram特征。例如，`CountVectorizer`或`TfidfVectorizer`类接受`ngram_range`参数 (parameter)。将`ngram_range`设置为`(1, 2)`会指示向量 (vector)化器生成unigram和bigram。

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    'this is the first document',
    'this document is the second document',
    'and this is the third one',
    'is this the first document',
]

# 生成unigram和bigram
vectorizer_1_2 = CountVectorizer(ngram_range=(1, 2))
X_1_2 = vectorizer_1_2.fit_transform(corpus)

print("词汇表 (Unigram + Bigram):")
# 展示组合词汇表的一部分
print(sorted(vectorizer_1_2.vocabulary_.keys())[:15]) 

print("\n特征矩阵形状 (文档数, 特征数):")
print(X_1_2.shape)

# 只生成bigram
vectorizer_2_2 = CountVectorizer(ngram_range=(2, 2))
X_2_2 = vectorizer_2_2.fit_transform(corpus)

print("\n词汇表 (只包含Bigram):")
print(sorted(vectorizer_2_2.vocabulary_.keys()))

print("\n特征矩阵形状 (文档数, 特征数):")
print(X_2_2.shape)
```

这段代码展示了您可以多么轻松地纳入N-gram。请注意，`ngram_range=(1, 2)`的设置比`ngram_range=(2, 2)`产生了更多特征，因为它既包含单个词语也包含词对。输出的形状`(documents, features)`清晰地显示了维度的增加。

### 字符N-gram

值得一提的是**字符N-gram**。我们不将文本分割成词语，而是可以创建$N$个字符的序列。例如，词语“context”的字符trigram将包括“con”、“ont”、“nte”、“tex”、“ext”。

字符N-gram特别适用于：

- 处理拼写错误和打字错误（例如，“service”和“servce”将共享许多字符N-gram）。
- 处理没有明确词界限的语言。
- 获取形态学信息（前缀、后缀）。

然而，它们也导致比词N-gram更大的特征空间，并且与词N-gram相比，可能获取较少语义信息。

N-gram提供了一种直接而有效的方法，可以将局部词序信息注入到文本特征中。尽管它们增加了维度，但所增加的语境信息通常会使基于这些特征构建的模型获得更好的性能，特别是与TF-IDF等技术结合时。接下来，我们将看到词嵌入 (embedding)等其他方法如何提供不同的方式来获取语义关系，但N-gram在文本特征工程过程中仍是一种有价值的工具。

## 参考资料

- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Pearson
  涵盖N-gram在语言建模和文本表示背景下的综合性教材。这是第3版。
- [sklearn.feature_extraction.text.CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) — scikit-learn developers (2024)
  Publisher: scikit-learn project
  scikit-learn CountVectorizer的官方文档，详细说明其用于生成N-gram特征的用法。
- [Foundations of Statistical Natural Language Processing](https://nlp.stanford.edu/fsnlp/) — Christopher D. Manning and Hinrich Schütze (1999)
  Publisher: MIT Press
  一本开创性的教材，为NLP提供了严谨的统计和数学基础，包括对N-gram模型的广泛介绍。

---

[上一节](02-%E8%AE%A1%E7%AE%97%20TF-IDF%20%E5%88%86%E6%95%B0.md) · [下一节](04-%E7%89%B9%E5%BE%81%E5%93%88%E5%B8%8C%E7%AE%80%E4%BB%8B.md)
