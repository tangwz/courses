---
course: "getting-started-with-llm-toolkit"
chapter: "embeddings-and-semantic-search"
lesson: "vector-similarity-fundamentals"
sourceId: 7770
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-5-embeddings-and-semantic-search/vector-similarity-fundamentals"
title: "向量相似度的基本原理"
description: "了解用于比较嵌入向量的相似度指标，例如余弦相似度。"
order: 3
plots: []
sourceHash: "c616c95af632d9998f4f11eac09c9cd9b42d8c47c3b2b2d531db97c7739b71b2"
sourceCorrections: []
---

嵌入 (embedding)将文本意义有效地以数值形式表示。为了找到相关内容，你需要判断这些数值向量 (vector)之间的‘接近程度’或‘相似度’。此项判定是通过向量相似度指标来完成的。

可以将每个嵌入看作高维空间 (high-dimensional space)中的一个点。在这个空间中，意义相似的文本彼此靠近，而意义不同的文本则相距较远。相似度指标是计算这些点之间距离或角度关系的数学函数。

> 相关术语在向量空间中紧密聚集，根据其语义形成不同的群组。

存在多种指标，但对于文本嵌入，有一个最为常用和有效：那就是余弦相似度。

### 余弦相似度：按角度衡量

余弦相似度计算两个向量 (vector)夹角的余弦值。此指标特别适合文本嵌入 (embedding)，因为它注重向量的方向而非其大小。方向捕获语义内容，而大小有时并不那么关键，特别是考虑到大多数现代嵌入模型生成的是*归一化 (normalization)*向量（长度为1的向量）。

基本思路很简单：

- 如果两个向量指向相同方向，它们之间的夹角为 0°，余弦相似度为 1。这表示它们的语义完全相同。
- 如果向量正交（夹角为 90°），余弦相似度为 0。这表明它们不相关。
- 如果它们指向相反方向（夹角为 180°），相似度为 -1，表示意义相反。

两个向量 $A$ 和 $B$ 之间余弦相似度的公式是：


$$
\text{相似度} = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|}
$$


当向量归一化为长度 1 时，分母 $\|A\| \|B\|$ 变为 1，余弦相似度就简化为两个向量的点积 $A \cdot B$。

你可以使用 `cosine_similarity` 函数轻松计算此值。我们来比较相关词和不相关词之间的相似度。

```python
from kerb.embedding import embed, cosine_similarity

# 生成两个相似词的嵌入
vec1 = embed("Hello")
vec2 = embed("Hi")
sim_related = cosine_similarity(vec1, vec2)
print(f"“Hello”和“Hi”之间的相似度：{sim_related:.4f}")

# 生成两个不相关词的嵌入
vec3 = embed("Car")
sim_unrelated = cosine_similarity(vec1, vec3)
print(f"“Hello”和“Car”之间的相似度：{sim_unrelated:.4f}")
```

如预期的那样，“Hello”和“Hi”的相似度得分较高，而“Hello”和“Car”的得分非常低，这反映了它们之间语义关联 (semantic relationship)的缺乏。

### 其他距离指标

虽然余弦相似度最为常用，但该工具包也提供在不同情况下有用的其他指标。与余弦相似度不同，这些是*距离*指标，其中**较低**的值表示更高的相似度。

#### 欧几里得距离（L2 距离）

欧几里得距离是向量 (vector)空间中两点之间的直线距离。它兼顾向量的方向和大小。距离为 0 表示向量完全相同。

#### 曼哈顿距离（L1 距离）

曼哈顿距离，也称为“城市街区”距离，是向量各分量绝对差值的总和。它通常比欧几里得距离对异常值不那么敏感。

#### 点积

点积衡量一个向量在另一个向量方向上的分量。对于归一化 (normalization)向量，它与余弦相似度相同。但是，对于非归一化向量，它对向量的大小敏感，这意味着较长的向量可能产生不成比例的较大影响。

我们来看看这些指标在同一组文本上的比较情况。

```python
from kerb.embedding import embed, euclidean_distance, manhattan_distance, dot_product

vec_a = embed("Python programming language")
vec_b = embed("Python coding and development")
vec_c = embed("JavaScript web framework")

# 欧几里得距离（值越小越相似）
dist_ab = euclidean_distance(vec_a, vec_b)
dist_ac = euclidean_distance(vec_a, vec_c)
print(f"欧几里得距离（相似文本）：{dist_ab:.4f}")
print(f"欧几里得距离（不同文本）：{dist_ac:.4f}")

# 点积（值越大越相似）
dot_ab = dot_product(vec_a, vec_b)
dot_ac = dot_product(vec_a, vec_c)
print(f"\n点积（相似文本）：{dot_ab:.4f}")
print(f"点积（不同文本）：{dot_ac:.4f}")
```

你可以看到，对于相似的文本对，欧几里得距离更小，而点积更大，两者都正确指明了更紧密的关联。

### 选择合适的指标

对于大多数基于文本的语义搜索应用，**余弦相似度是推荐的指标**。由于现代嵌入 (embedding)模型生成归一化 (normalization)向量 (vector)，它们的长度不带有太多信息，而关注角度（方向）能提供最可靠的语义关联 (semantic relationship)判定。

`batch_similarity` 函数提供一种高效的方法，可以使用任何可用指标将单个查询向量与文档向量集合进行比较。这是语义搜索系统的基础。

```python
from kerb.embedding import embed, embed_batch, batch_similarity

query_text = "cloud computing infrastructure"
documents = [
    "Cloud services and platforms",
    "Infrastructure as a service",
    "Traditional on-premise servers",
    "Mobile app development",
]

query_emb = embed(query_text)
doc_embeddings = embed_batch(documents)

# 比较不同指标的得分
cosine_scores = batch_similarity(query_emb, doc_embeddings, metric="cosine")
print("余弦相似度（值越大越好）：")
for doc, score in zip(documents, cosine_scores):
    print(f"  [{score:.4f}] {doc}")

euclidean_scores = batch_similarity(query_emb, doc_embeddings, metric="euclidean")
print("\n欧几里得距离（值越小越好）：")
for doc, score in zip(documents, euclidean_scores):
    print(f"  [{score:.4f}] {doc}")
```

请注意，两个指标都正确地将“Cloud services and platforms”和“Infrastructure as a service”识别为最相关的文档。然而，它们的评分范围和解读有所不同。余弦相似度提供了一个介于 -1 和 1 之间的归一化、易于理解的分数，从而更容易设定一致的相关性阈值。

在对嵌入及其比较方式有了扎实掌握后，你现在可以开始构建你的第一个语义搜索功能了。

## 参考资料

- [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/information-retrieval-book.html) — Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze (2008)
  Publisher: Cambridge University Press
  介绍向量空间模型，这是将文档表示为向量并计算其相似度的基本框架，是信息检索的核心。
- [Speech and Language Processing (3rd edition draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Stanford University; Pages: Chapter 6
  介绍词嵌入、其语义属性，以及如何应用余弦相似度比较文本含义。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer; Pages: Chapter 2
  在统计学习背景下，阐释各种距离度量的数学基础，包括欧几里得距离和曼哈顿距离。
- [OpenAI Embeddings API](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFkKd3RevG8wimD9OzbZeNpdKwmwHqz_u0Gx8yl9S4lGjmM3LSlBRThnRf_g3L5xKaggmcakEXif8qNAFktGvl8yEt0yX07InPp4Ldd265kr_M_f7qxSWhoMd9OWL5efTDbz1OIDIYuP1WFICWh_npX8ls2wQ==) — OpenAI (2024)
  Publisher: OpenAI
  OpenAI 嵌入模型的官方指南，详细说明其特性、归一化以及建议使用余弦相似度进行语义比较。
