---
course: "building-ml-recommendation-system"
chapter: "content-based-filtering"
lesson: "vectorizing-text-data-tfidf"
sourceId: 7457
sourceUrl: "https://apxml.com/zh/courses/building-ml-recommendation-system/chapter-2-content-based-filtering/vectorizing-text-data-tfidf"
title: "使用 TF-IDF 进行文本数据向量化"
description: "关于如何使用词频-逆文档频率 (TF-IDF) 将文本物品描述转换为数值向量的指南。"
order: 3
plots: []
sourceHash: "ff4dc242f1e7fdce21bda4826921a14d82f7248e6c0fffd2498c0c4af61dd5a0"
sourceCorrections: []
---

虽然像类型或演员这样的结构化数据可以直接使用，但有关物品的大量丰富信息（如电影剧情简介或产品描述）通常是非结构化文本。为了构建基于内容的过滤系统，我们必须先将这些文本转换为数值格式，这个过程称为向量 (vector)化。机器学习 (machine learning)模型无法直接理解“冒险”或“星系”之类的词汇，但它可以处理数值向量。

一种简单的方法是统计每个词出现的次数，通常称为词袋（Bag-of-Words，简称 BoW）模型。在这个模型中，每篇文档都变成一个向量，其中的每个元素对应一个词频计数。然而，这种方法有一个明显的缺点：像“the”、“a”和“in”这样非常常见的词会主导向量，尽管它们几乎没有描述性含义。因此，具有相似常用词的文档看起来可能比实际情况更相似。我们需要一种能够识别出不仅频率高，而且能*描述*文档内容的词汇的方法。这正是词频-逆文档频率（TF-IDF）所实现的功能。

### TF-IDF 的组成部分

TF-IDF 是一种数值统计指标，反映了一个词在集合或语料库中对某篇文档的权重 (weight)。它由两部分组成：词频（Term Frequency）和逆文档频率（Inverse Document Frequency）。通过将这两个值相乘，我们可以得到一个分值。对于在特定文档中经常出现但在整个语料库中很少见的词，这个分值会很高。

> TF-IDF 过程将一系列文本档转换为数值矩阵，其中每个文档都由一个词分值向量 (vector)表示。

#### 词频 (TF)

词频衡量一个词在文档中出现的相对频率。它的计算方法是用一个词在文档中出现的次数除以该文档中的总词数。这种归一化 (normalization)处理可以防止较长的文档比较短的文档占据不公平的优势。

词频的公式为：


$$
\text{TF}(t, d) = \frac{\text{词 } t \text{ 在文档 } d \text{ 中出现的次数}}{\text{文档 } d \text{ 中的总词数}}
$$


例如，如果“knight”一词在一段包含 100 个词的电影剧情简介中出现了 3 次，那么它的 TF 就是 $3/100 = 0.03$。

#### 逆文档频率 (IDF)

逆文档频率衡量一个词提供的信息量。它会降低常用词的权重，并增加不常用词的权重。一个词的 IDF 计算方法是语料库中总文档数的对数除以包含该词的文档数。

逆文档频率的公式为：


$$
\text{IDF}(t, D) = \log\left(\frac{\text{文档总数 } D}{\text{包含词 } t \text{ 的文档数}}\right)
$$


像“and”或“the”这样的词几乎会出现在所有文档中，因此它们的 IDF 将接近 $\log(1)$，即 0。而像“interstellar”这样可能只出现在少数科幻电影描述中的稀有词，将具有更高的 IDF 分值，表明它更具代表性。对数有助于缓冲这种效果，使极稀有的词不会完全主导评分。

#### 最终的 TF-IDF 分值

文档中某个词的 TF-IDF 分值就是其 TF 和 IDF 分值的乘积。


$$
\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)
$$


这个最终得分会给那些在某篇文档中频繁出现（高 TF）但在整个文档集中很少见（高 IDF）的词赋予更高的权重。这种组合有效地突出了最能体现特定物品特征的词汇。

### 使用 Scikit-learn 实现

虽然理解数学原理很有帮助，但你不需要从零开始实现 TF-IDF。`scikit-learn` 库提供了一个高度优化且灵活的实现，称为 `TfidfVectorizer`。这个类处理了分词 (tokenization)、学习词汇表 (vocabulary)、计算 IDF 权重 (weight)以及生成最终 TF-IDF 矩阵的整个流程。

以下是如何使用它将一组电影剧情描述转换为 TF-IDF 矩阵：

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# 示例电影剧情描述
documents = [
    'A brave knight battles a dragon to save a princess.',
    'A space explorer discovers a new galaxy of brave pioneers.',
    'A knight starts a search for a magical sword.'
]

# 初始化向量化器，移除常用的英文停用词
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# 将向量化器拟合到数据并进行转换
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# 生成的矩阵包含每个文档的 TF-IDF 向量
print(f"TF-IDF 矩阵的形状: {tfidf_matrix.shape}")

# 查看特征名称（学习到的词汇表）
print(f"词汇表: {tfidf_vectorizer.get_feature_names_out()}")

# 矩阵是稀疏的，打印其稠密表示以进行检查
print("\nTF-IDF 矩阵 (稠密):")
print(tfidf_matrix.toarray())
```

`tfidf_vectorizer.fit_transform` 的输出是一个稀疏矩阵。这是一种内存效率高的表示方式，因为 TF-IDF 矩阵中的大多数单元格都是零，毕竟任何给定的文档只包含总词汇表中的一小部分。

`TfidfVectorizer` 有几个用于调整其行为的有用参数 (parameter)：

- `stop_words`: 可以设置为 `'english'` 以自动移除常见的英文停用词。
- `min_df`: 忽略文档频率严格低于给定阈值的词。这对于移除稀有词或拼写错误很有用。
- `max_df`: 忽略文档频率严格高于给定阈值的词。这可以用来过滤掉那些因过于常见而失去描述性的词。
- `ngram_range`: 允许你考虑词的 n-gram 而不仅仅是单个单词。例如，`ngram_range=(1, 2)` 将同时包含单个词（unigrams）和相邻的词对（bigrams），从而捕捉更多上下文 (context)信息。

通过应用 TF-IDF，我们成功地将非结构化文本转换为了有意义的数值向量 (vector)。`tfidf_matrix` 中的每一行都代表一个物品，为下一步衡量相似度做好了准备。

## 参考资料

- [A statistical interpretation of term specificity and its application in automatic indexing and information retrieval](https://doi.org/10.1108/eb026526) — KAREN SPARCK JONES (1972)
  Journal: Journal of Documentation; Publisher: Emerald Publishing; Volume: 28; Pages: 11-21; DOI: [10.1108/eb026526](https://doi.org/10.1108/eb026526)
  提出逆文档频率（IDF）概念的原始论文，是TF-IDF的基础。
- [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/html/htmledition/irbook.html) — Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze (2008)
  Publisher: Cambridge University Press
  信息检索标准教材，涵盖TF-IDF理论及文本处理中的实际应用。
- [sklearn.feature_extraction.text.TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) — Scikit-learn Developers (2024)
  TfidfVectorizer类的官方文档，提供参数和Python用法详情。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  使用scikit-learn进行文本向量化（包括TF-IDF）的实践指南，适用于机器学习应用。
