---
course: "vector-databases-semantic-search"
chapter: "embeddings-vector-spaces-foundation"
lesson: "vector-similarity-metrics"
sourceId: 3515
sourceUrl: "https://apxml.com/zh/courses/vector-databases-semantic-search/chapter-1-embeddings-vector-spaces-foundation/vector-similarity-metrics"
title: "测量向量空间中的相似度"
description: "比较常见距离指标：余弦相似度、欧几里得距离和点积。"
order: 5
plots: ["plots/3515-0.json"]
sourceHash: "6d8ea9b7ca520146f6392853c6f0a72e9b73e2ee5fd56128fd08e7021d9b8c54"
sourceCorrections: []
---

现在我们已理解文本或图片等数据如何转换为高维空间 (high-dimensional space)中的数值向量 (vector)，下一步便是弄清如何量化 (quantization)这些向量之间的*关联*。若我们的嵌入 (embedding)模型已准确捕获语义信息，代表相似事物的向量在该空间中应“靠近”，而不相似的事物则应“远离”。衡量这种接近程度对于查找相关文档、推荐相似商品或执行语义搜索等任务是极为重要的。

在向量空间中，“接近程度”通常通过距离或相似度指标来衡量。这些数学函数以两个向量为输入，输出一个标量值，指示它们的相似或不相似程度。接下来，我们来看看向量嵌入中常用的指标。

### 余弦相似度

余弦相似度可以说是比较高维嵌入 (embedding)最常用的指标，尤其是在自然语言处理中。它不是测量两个向量 (vector)端点之间的绝对距离，而是测量它们之间夹角的余弦值。

向量 $\vec{a}$ 和 $\vec{b}$ 之间的余弦相似度公式为：


$$
\text{余弦相似度}(\vec{a}, \vec{b}) = \frac{\vec{a} \cdot \vec{b}}{||\vec{a}|| ||\vec{b}||} = \frac{\sum_{i=1}^{n} a_i b_i}{\sqrt{\sum_{i=1}^{n} a_i^2} \sqrt{\sum_{i=1}^{n} b_i^2}}
$$


这里：

- $\vec{a} \cdot \vec{b}$ 是向量 $\vec{a}$ 和 $\vec{b}$ 的点积。
- $||\vec{a}||$ 和 $||\vec{b}||$ 分别是向量 $\vec{a}$ 和 $\vec{b}$ 的欧几里得范数（或大小）。
- $n$ 是向量的维度。

**直观理解：** 余弦相似度纯粹关注向量的*指向*或*方向*，而不考虑它们的大小。设想两支从同一点发出的箭。如果它们指向完全相同的方向，夹角为 0°，余弦值为 1（最大相似度）。如果它们指向相反的方向，夹角为 180°，余弦值为 -1（最大不相似度）。如果它们是正交的（垂直的），夹角为 90°，余弦值为 0（方向上无关联或相似度）。

结果总是在 [-1, 1] 范围内。对于许多嵌入模型，向量会被归一化 (normalization)为单位长度（$||\vec{v}|| = 1$）。在这种情况下，分母变为 1，余弦相似度就简化为点积（$\vec{a} \cdot \vec{b}$）。处理仅相对方向重要的嵌入时（文本语义中常见），余弦相似度通常是优选。值越接近 1 表示语义相似度越高。

### 欧几里得距离

欧几里得距离是我们在几何学中了解的标准“直尺”距离。它测量向量 (vector)空间中两个向量端点之间的直线距离。

向量 $\vec{a}$ 和 $\vec{b}$ 之间的欧几里得距离公式为：


$$
\text{欧几里得距离}(\vec{a}, \vec{b}) = ||\vec{a} - \vec{b}||_2 = \sqrt{\sum_{i=1}^{n}(a_i - b_i)^2}
$$


这里：

- $a_i$ 和 $b_i$ 是向量 $\vec{a}$ 和 $\vec{b}$ 在维度 $i$ 上的分量。
- $n$ 是向量的维度。

**直观理解：** 欧几里得距离同时考虑向量的方向和大小。如果两个向量的端点在空间中彼此接近，则它们被认为是接近的。结果总是非负的（$\ge 0$），其中 0 表示向量完全相同。值越大表示差异越大。

与余弦相似度不同，欧几里得距离对向量的长度（大小）敏感。如果一个向量比另一个长得多，但指向相似的方向，欧几里得距离可能很大，而余弦相似度仍然很高。对于大小有意义的嵌入 (embedding)（尽管在纯语义搜索中不常见），或对于特定类型的聚类，欧几里得距离可能有用。请记住，在高维空间 (high-dimensional space)中，距离的含义会变得不那么直观（维度灾难），这是余弦相似度常受青睐的一个原因。



![比较余弦相似度与欧几里得距离](plots/3515-0.json)



> 2D 空间中的向量 A 和 B。余弦相似度取决于角度 $\theta$，而欧几里得距离是连接向量端点的红色虚线的长度。

### 点积

点积（或内积）是比较向量 (vector)的另一种方式。它与余弦相似度密切相关。

向量 $\vec{a}$ 和 $\vec{b}$ 之间的点积公式为：


$$
\vec{a} \cdot \vec{b} = \sum_{i=1}^{n} a_i b_i
$$


它也可以用几何方式定义为：


$$
\vec{a} \cdot \vec{b} = ||\vec{a}|| ||\vec{b}|| \cos(\theta)
$$


这里 $\theta$ 是向量之间的夹角。

**直观理解：** 点积同时考虑角度（如余弦相似度）和向量的大小。较大的点积可能源于向量具有较大的大小、方向上紧密对齐 (alignment)，或两者兼有。点积的范围是 $(-\infty, \infty)$。

如果向量被归一化 (normalization)为单位长度（$||\vec{a}|| = ||\vec{b}|| = 1$），则点积就与余弦相似度相同（$\vec{a} \cdot \vec{b} = \cos(\theta)$）。许多向量数据库和库会优化对归一化向量的计算，使得点积成为在实际中计算余弦相似度的一种计算效率高的方式。然而，如果向量未归一化，其相似度的解释就不那么直接，因为大小会严重影响得分。

### 选择合适的指标

- **余弦相似度：** 通常适用于语义搜索和比较文本嵌入 (embedding)，此时*方向*（语义）比大小更重要。它在高维空间 (high-dimensional space)中表现良好，并且在嵌入归一化 (normalization)时是标准选择。
- **欧几里得距离：** 当嵌入向量 (vector)的大小包含重要信息时，或对于某些聚类算法，可使用此方法。请留意它在极高维空间中的表现。通常，使用平方欧几里得距离（$||\vec{a} - \vec{b}||_2^2$）来避免计算成本更高的平方根运算，因为它能保持距离的顺序。
- **点积：** 当向量归一化时，常作为余弦相似度的有效计算方式。如果向量未归一化，其值会混合大小和角度信息，这在特定情境中可能合用，但对于一般的语义相似度来说则不那么直观。

在大多数以语义搜索为主的向量数据库应用中，您会主要遇到并使用余弦相似度，它通常通过对归一化向量进行优化的点积计算来实现。理解这些指标对于解释搜索结果和正确配置您的向量数据库索引非常重要，这一点我们将在后续章节中看到。

## 参考资料

- [Introduction to Linear Algebra](http://math.mit.edu/linearalgebra) — Gilbert Strang (2016)
  Publisher: Wellesley-Cambridge Press
  提供向量、点积、范数和向量空间中几何距离的基础处理。
- [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/information-retrieval-book.html) — Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze (2008)
  Publisher: Cambridge University Press
  阐述信息检索中的向量空间模型以及余弦相似度在衡量文档相似性方面的应用。
- [Neural Network Methods for Natural Language Processing](https://doi.org/10.2200/S00762ED1V01Y201703HLT037) — Yoav Goldberg (2017)
  Publisher: Morgan & Claypool Publishers; Volume: 37; DOI: [10.2200/S00762ED1V01Y201703HLT037](https://doi.org/10.2200/S00762ED1V01Y201703HLT037)
  讨论密集向量嵌入及其在现代自然语言处理中利用相似度指标进行比较。
