---
course: "advanced-vector-search-llms"
chapter: "hybrid-search-approaches"
lesson: "integrating-keyword-search"
sourceId: 3873
sourceUrl: "https://apxml.com/zh/courses/advanced-vector-search-llms/chapter-3-hybrid-search-approaches/integrating-keyword-search"
title: "结合关键词搜索（BM25, TF-IDF）"
description: "关于稠密向量检索与稀疏关键词方法（如BM25）结合的方式。"
order: 2
plots: []
sourceHash: "56868b0c6f88641881e5db5313c1c56f7ff2d06f0cd2a7675ad492ce7f7fb81c"
sourceCorrections: []
---

向量 (vector)搜索擅长捕获语义并找到相似项，但它有时可能遗漏那些特定关键词、标识符或精确短语很重要的文档。例如，搜索像 `XG-500` 这样的产品代码或 `calculate_fft` 这样的特定函数名，如果语义表示未能充分突出该精确术语而非其更广泛的上下文 (context)，可能无法可靠地找出最相关的文档。在此，传统关键词搜索技术仍具有显著价值。

结合关键词搜索，尤其使用Okapi BM25等有效算法，与向量搜索相结合，使您能够构建一个混合系统，该系统结合了两类方法的优势：向量提供的语义理解和关键词提供的精确术语匹配。

### 从TF-IDF到BM25

您可能了解TF-IDF（词频-逆文档频率），这是一种根据关键词相关性对文档评分的经典方法。它根据词语在文档中出现的频率（TF）以及该词语在整个文档集中的稀有程度（IDF），为文档中的每个词语计算权重 (weight)。基本思想是，在特定文档中频繁出现但整体稀有的词语，能很好地指示该文档的内容。

尽管TF-IDF奠定了基础，但它仍有局限。它没有考虑词频饱和效应（即一个词第10次出现对相关性贡献小于第1次出现），且缺少一种精细的方式来处理文档长度的差异。

BM25（最佳匹配25）是一种更高级的概率检索函数，它直接解决了这些局限，通常为关键词搜索带来更好效果。它已成为许多现代搜索引擎的标准。BM25改进了TF和IDF的思路，并引入了文档长度归一化 (normalization)。

在BM25中，给定查询 $Q$（包含词语 $q_1, ..., q_n$）时，文档 $D$ 的评分公式通常表示为：


$$
\text{评分}(D, Q) = \sum_{i=1}^{n} \text{逆文档频率}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgdl}})}
$$


我们来逐一分析各个组成部分：

1. **逆文档频率（IDF）：** 类似于TF-IDF，该部分衡量查询词 $q_i$ 的稀有性或信息量。在较少文档中出现的词语会得到更高的IDF分数。计算IDF有多种方式；一种常用的方式是：

   
   $$
   \text{逆文档频率}(q_i) = \log \left( \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1 \right)
   $$
   

   $N$ 指集合中文档的总数，$n(q_i)$ 是包含词语 $q_i$ 的文档数量。添加 $+0.5$ 项是为了平滑处理，避免语料库中不存在的词语引起除零错误，并避免词语在超过半数文档中出现时引发的问题。
2. **词频饱和：** 分数 $\frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot (\dots)}$ 部分调整了原始词频 $f(q_i, D)$。参数 (parameter) $k_1$（通常在1.2到2.0之间）控制分数饱和的速度。随着一个词语在文档中出现的频率增加，它对分数的贡献也增加，但增加的速率会减小。这避免了仅仅因为一个词语重复出现多次的文档不公平地主导搜索结果。
3. **文档长度归一化：** 术语 $(1 - b + b \cdot \frac{|D|}{\text{avgdl}})$ 根据文档长度 $|D|$ 相对于集合中平均文档长度 `avgdl` 来归一化分数。参数 $b$（通常在0.75左右）控制归一化的程度。当 $b=1$ 时，归一化完全生效；当 $b=0$ 时，没有长度归一化。这有助于避免很长的文档仅仅因为有更多机会包含查询词而获得不公平的优势。

通过调整 $k_1$ 和 $b$，您可以将BM25的行为调整以适应您的特定数据集和查询模式。

### 实际集成策略

将BM25（或其他关键词算法）与向量 (vector)搜索结合通常涉及并行运行两种搜索类型，然后组合结果。

1. **并行查询执行：** 当用户查询到来时，它会同时发送到：

   - 向量搜索系统（使用ANN索引）根据查询嵌入 (embedding)查找语义相似的文档。
   - 关键词搜索系统（如Elasticsearch、OpenSearch，或向量数据库中支持稀疏向量/BM25的组件）根据BM25分数查找匹配查询词的文档。
2. **索引要求：** 这种方法需要维护两种不同类型的索引：

   - **向量索引：** 存储用于语义搜索的稠密向量嵌入（例如，使用HNSW、IVF）。
   - **关键词索引：** 存储原始文本内容，经过处理（分词 (tokenization)、词干提取、停用词移除）以适应关键词查找和BM25评分。这通常是由Apache Lucene（Elasticsearch和OpenSearch背后的引擎）等系统管理的倒排索引。
3. **系统架构：** 您可以使用在应用层连接的独立专业系统，或采用为混合搜索设计的平台，这些平台内部管理稠密和稀疏（关键词）表示。选择取决于您的规模、性能需求和现有基础架构。

> 混合搜索系统在组合结果之前并行执行向量和关键词搜索的流程。

独立执行这些搜索会产生两组不同的候选文档，每组都根据各自的评分机制（向量的相似度分数，关键词的BM25分数）进行排序。接下来的重要一步，我们将在下一节讨论，是如何有效地合并或整合这些独立的结果列表，形成一个单一、一致的排序，既反映语义相关性，也反映关键词的重要性。

## 参考资料

- [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/) — Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze (2008)
  Publisher: Cambridge University Press
  一本内容全面的教科书，阐述了TF-IDF、BM25以及其他基础信息检索算法。
- [Okapi at TREC-3](http://trec.nist.gov/pubs/trec3/papers/city.ps.gz) — Stephen E. Robertson, Steve Walker, Susan Jones, Micheline Hancock-Beaulieu, Mike Gatford (1994)
  Journal: Proceedings of The Third Text REtrieval Conference, TREC 1994; Publisher: National Institute of Standards and Technology (NIST); Volume: NIST Special Publication 500-226; Pages: 109-126
  这份原始文献展示了Okapi BM25排序函数在信息检索中的有效性。
- [Fusing IR Systems using Rank Aggregation](https://dl.acm.org/doi/10.1145/1277741.1277855) — William W. Cohen, Einat Minkov, and Alon Kornblith (2007)
  Journal: Proceedings of the 30th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '07); Publisher: ACM; Pages: 639-646; DOI: [10.1145/1277741.1277855](https://doi.org/10.1145/1277741.1277855)
  探讨了使用排序聚合技术合并来自多个搜索系统结果的策略，与混合搜索结果融合相关。
- [Lucene's Practical Scoring Function](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHxvalxBw2w-N1PsSv9VjXqhWqv9IR6Qeq99aVJzLvRGRHGvR4ks7gz5AWDZv2SmbqQFIHytz5rwxV4P8ReLiQ6pXdhu9UrcWmxozDBUtwuPUlGGHyMOxYxMDQrkXAsOEhzv1Q4p6vUBVh1zVI=) — The Apache Lucene Project (2023)
  解释了Apache Lucene（版本9.9.2）中BM25及其他评分函数的实际实现，包括参数细节。
