# 从词袋模型到TF-IDF

来源：[原文](https://apxml.com/zh/courses/nlp-fundamentals/chapter-2-nlp-feature-engineering/bag-of-words-tf-idf)

[返回章节目录](README.md) · [返回课程目录](../README.md)

将文本转换为数值格式对机器学习 (machine learning)模型而言十分必要。最简单的方法通常是词袋（Bag-of-Words, BoW）模型。词袋（Bag-of-Words, BoW）模型将每份文档表示为一个向量 (vector)，其中每个元素对应总词汇表 (vocabulary)中某个特定词的出现次数。

设想两个简单句子：

1. “The cat sat on the mat.”
2. “The dog chased the cat.”

词汇表为 {"The", "cat", "sat", "on", "mat", "dog", "chased"}。为简化起见，忽略大小写和标点，BoW 向量将是：

- 句子 1: `[2, 1, 1, 1, 1, 0, 0]` (对应词汇表中每个词的计数)
- 句子 2: `[2, 1, 0, 0, 0, 1, 1]`

这种表示方式直观且易于计算。然而，它有一个重要局限：它平等对待每个词。可以发现，“The”这个词在两个句子中都出现次数最多 (2)。“The”能告诉我们多少关于这些句子的具体内容或差异吗？可能不会。像“the”、“is”、“in”、“a”这样的常见词通常在计数中占主导，但与“sat”、“mat”、“dog”或“chased”等词相比，它们携带的区分性信息较少。BoW 向量可能被这些频繁但通常无信息量的词严重倾斜。

此外，BoW 只考虑单个文档*内*的词计数。它没有衡量一个词在整个文档集合（即语料库）中是常见还是罕见。一个词可能在某个特定文档中出现多次，仅凭BoW会使其看起来很重要。但如果同一个词在语料库中几乎*所有*其他文档中也频繁出现，那么它区分该特定文档的能力就会减弱。反之，一个词在某文档中出现频率适中，但在其他地方非常罕见，这可能是该文档独特主题的有力标志。

### 超越简单计数：权重 (weight)化的必要性

为解决这些局限，我们需要一种方法来调整数值表示，以反映词的*重要性*，而不仅仅是其原始频率。我们希望对在*特定*文档中频繁出现但在*整个*语料库中相对罕见的词赋予更高权重。这正是词频-逆文档频率 (TF-IDF) 加权方案的缘由所在。

TF-IDF 旨在量化 (quantization)一个词在文档集合中与某文档的相关程度。它通过结合两个不同指标来实现此目的：

1. **词频 (TF):** 衡量一个词在特定文档中出现的频率。
2. **逆文档频率 (IDF):** 通过考虑一个词在语料库中所有文档中的常见或稀有程度，衡量其信息量。

让我们逐一看每个组成部分。

### 词频 (TF)

词频量化 (quantization)了一个词在单个文档中的显著性。其基本思想简单明了：一个词 $t$ 在文档 $d$ 中出现的次数越多，它就越可能与 $d$ 的主题相关。

有几种计算 TF 的方法：

- **原始计数：** 简单来说就是词项 $t$ 在文档 $d$ 中出现的次数。这就是 BoW 计数。
  
  $$
  TF_{raw}(t, d) = \text{t在d中的计数}
  $$
  
- **频率：** 原始计数除以文档中的词项总数。这会根据文档长度进行归一化 (normalization)，避免较长文档仅因包含更多词而固有地拥有更高的 TF 值。
  
  $$
  TF_{freq}(t, d) = \frac{\text{t在d中的计数}}{\text{d中的总词数}}
  $$
  
- **对数缩放：** 减小非常高词频的影响。一个词项出现 100 次不一定比出现 10 次重要 10 倍。对数缩放能减弱这种效应。
  
  $$
  TF_{log}(t, d) = \log(1 + \text{t在d中的计数})
  $$
  

TF 计算方法的选择取决于具体的应用和数据集特点。频率归一化很常用。

### 逆文档频率 (IDF)

TF 告诉我们一个词在单个文档中的频率，而逆文档频率则告诉我们它在整个语料库 $D$ 中的稀有性或常见性。直观上看，在许多文档中出现的词（如“the”、“a”、“is”）在区分文档方面不太可能具有区分性或信息量。只在少数文档中出现的词可能更具体，因此信息量更大。

IDF 赋予更罕见的词更高的值，对常见词赋予更低的值。它通常按以下方式计算：


$$
IDF(t, D) = \log\left(\frac{N}{df_t}\right)
$$


其中：

- $N$ 是语料库 $D$ 中的文档总数。
- $df_t$ (文档频率) 是语料库 $D$ 中包含词项 $t$ 的文档数量。

对数（通常是自然对数或以 10 为底的对数）用于减弱 IDF 的影响。如果没有它，非常罕见的词可能会完全主导加权。

> **注意：** 如果一个词项出现在每个文档中 ($df_t = N$)，则 IDF 变为 $\log(1) = 0$，有效地消除了该词项的贡献。如果一个词项没有出现在任何文档中（例如，在测试期间遇到但不在训练语料库中的新词），$df_t$ 将为 0，导致除以零。为处理这种情况，通常会在分母中加 1 或使用“平滑”处理：
>
> 
> $$
> IDF_{smooth}(t, D) = \log\left(\frac{N}{1 + df_t}\right) + 1
> $$
> 
>
> 或
>
> 
> $$
> IDF_{smooth}(t, D) = \log\left(\frac{1 + N}{1 + df_t}\right) + 1
> $$
> 
>
> 末尾添加的“+ 1”确保了在许多文档中出现的词项仍具有小的正权重 (weight)，从而避免了下游计算中可能出现的问题。

### 结合 TF 和 IDF

词项 $t$ 在语料库 $D$ 中文档 $d$ 的 TF-IDF 分数，简单来说就是其 TF 分数与 IDF 分数的乘积：


$$
TFIDF(t, d, D) = TF(t, d) \times IDF(t, D)
$$


这个组合分数达到了预期效果：

- 对在特定文档中频繁出现（高 TF）但在整个语料库中罕见（高 IDF）的词项最高。这些可能是该文档最具特征的词项。
- 对在文档中不常出现（低 TF）或在许多文档中出现（低 IDF）的词项较低。
- 对几乎所有文档中都出现的词项最低（通常接近零）。

再次考虑我们简单的例子：“The cat sat on the mat.” 和 “The dog chased the cat.”。

- “The”：在两者中 TF 高，但出现在两个（所有）文档中，因此 IDF 非常低。TF-IDF 分数会很低。
- “cat”：在两者中 TF 高，出现在两个（所有）文档中，因此 IDF 非常低。TF-IDF 分数会很低。
- “sat”、“on”、“mat”：在文档 1 中 TF 中等 (1)，仅出现在文档 1 中（更罕见）。IDF 将高于“the”或“cat”。TF-IDF 分数将是中等。
- “dog”、“chased”：在文档 2 中 TF 中等 (1)，仅出现在文档 2 中（更罕见）。IDF 将高于“the”或“cat”。TF-IDF 分数将是中等。

这个简单例子说明了 TF-IDF 如何自动降低“the”和“cat”等常见词（在这个小型语料库中）的权重 (weight)，并赋予区分文档的词（“sat”、“mat”、“dog”、“chased”）更高的显著性。

在实际应用中，我们不会手动计算这些，而是使用像 Scikit-learn 这样的库，它们提供了高效实现 (`TfidfVectorizer`)。通过从简单的 BoW 计数转向 TF-IDF 分数，我们创建了更详细且通常更有效的文本数值表示，不仅捕获了词的存在，还衡量了词相对于文档和语料库的重要性。这种改进的表示方式为后续步骤（例如训练机器学习 (machine learning)模型，我们将在稍后介绍）提供了更好的基础。我们现在将了解如何实际计算这些分数，并讨论 N-gram 等其他技术以进一步改进我们的文本特征。

## 参考资料

- [A statistical interpretation of term specificity and its application in automatic indexing](https://doi.org/10.1108/eb026526) — Karen Spärck Jones (1972)
  Journal: Journal of Documentation; Publisher: Emerald; Volume: 28; Pages: 11-21; DOI: [10.1108/eb026526](https://doi.org/10.1108/eb026526)
  这篇基础论文介绍了逆文档频率（IDF）的概念，它是TF-IDF加权方案的核心。
- [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/information-retrieval-book.html) — Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze (2008)
  Publisher: Cambridge University Press; Pages: Chapter 6
  一本关于信息检索的综合教科书，详细解释了词项加权方案，包括词袋模型和TF-IDF。
- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Stanford University; Pages: Chapter 6 (3rd ed. draft)
  一本广泛使用的自然语言处理教科书，在更广泛的自然语言处理背景下，清晰地解释了文本表示方法，包括词袋模型和TF-IDF。
- [TfidfVectorizer - scikit-learn 1.3.2 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) — scikit-learn developers (2023)
  Scikit-learn `TfidfVectorizer` 的官方文档，它是将原始文档集合转换为TF-IDF特征矩阵的有效实现。对实际应用很有帮助。

---

[上一节](../01-%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%E4%B8%8E%E9%AB%98%E7%BA%A7%E6%96%87%E6%9C%AC%E5%A4%84%E7%90%86/07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E9%A2%84%E5%A4%84%E7%90%86%E6%B5%81%E7%A8%8B.md) · [下一节](02-%E8%AE%A1%E7%AE%97%20TF-IDF%20%E5%88%86%E6%95%B0.md)
