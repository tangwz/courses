---
course: "getting-started-with-llm-toolkit"
chapter: "building-rag-systems"
lesson: "improving-relevance-with-reranking"
sourceId: 7776
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-6-building-rag-systems/improving-relevance-with-reranking"
title: "通过重排提升相关性"
description: "在将检索到的文档发送到大型语言模型之前，使用重排模型来提升其相关性。"
order: 4
plots: []
sourceHash: "2f0a014f821d19841e06bed40e2983a4bcc75ec541c26c6b27cfa053fb39c1e6"
sourceCorrections: []
---

用于增强语言模型的检索方法能够有效识别大量潜在相关文档。然而，这些方法通常优先考虑找到所有可能的匹配项（召回率），而不是确保排名靠前的结果最有用（精确率）。这种做法可能导致上下文 (context)窗口充满冗余或略微相关的信息，这可能会让大型语言模型困惑或浪费宝贵的 token 空间。重排作为第二阶段过程，对最初检索到的文档集进行优化。它应用更精密的评分方法，以识别对生成模型来说最相关、最多样化和最有用的文档。

### 应用不同的重排策略

提高相关性最简单的方法是根据额外信号对初始搜索结果进行重新排序。这些信号通常来自文档元数据，例如发布日期或流行度指标。`rerank_results` 函数提供了一种直接的方式来应用这些策略。

例如，在执行初始搜索后，您可以重排结果，以优先显示更新或更受欢迎的文档。这在新闻摘要或社区论坛问答等应用中特别有帮助。

我们从一组初始搜索结果开始。

```python
from kerb.retrieval import keyword_search, rerank_results

# 假设 'documents' 是一个包含元数据的 Document 对象列表
query = "python async programming"
initial_results = keyword_search(query, documents, top_k=10)

print("初始关键词搜索结果（前4项）：")
for r in initial_results[:4]:
    print(f"  {r.rank}. {r.document.id} (score: {r.score:.3f})")
```

现在，我们可以应用不同的重排方法。为了偏向较新的文档，我们使用 `recency` 方法，该方法检查元数据中的日期信息。

```python
# 按新近度重排
recency_ranked = rerank_results(query, initial_results, method="recency", top_k=4)

print("\n按新近度（日期）重排：")
for r in recency_ranked:
    date = r.document.metadata.get('date', 'N/A')
    print(f"    {r.rank}. {r.document.id} (score: {r.score:.3f}, date: {date})")
```

同样，如果您的文档具有像浏览量这样的流行度元数据，您可以使用 `popularity` 方法。

```python
# 按流行度重排
popularity_ranked = rerank_results(query, initial_results, method="popularity", top_k=4)

print("\n按流行度（浏览量）重排：")
for r in popularity_ranked:
    views = r.document.metadata.get('views', 0)
    print(f"    {r.rank}. {r.document.id} (score: {r.score:.3f}, views: {views})")
```

### 实现自定义评分逻辑

内置方法很有用，但您经常需要实现特定于您应用的业务逻辑。`rerank_results` 函数通过自定义的 `scorer` 来支持这一点。您可以传入一个函数，为每个文档计算新的分数，从而组合多个信号。

例如，您可能希望提升来自特定作者的文档，或属于某个重要类别的文档。您的自定义评分函数接收查询和 `Document` 对象，并应返回一个浮点分数，该分数随后会乘以原始相关性分数。

```python
from kerb.retrieval import Document

def category_booster(query: str, doc: Document) -> float:
    """提升“编程”类别和作者为“Alice”的文档分数。"""
    score_multiplier = 1.0
    if doc.metadata.get('category') == 'programming':
        score_multiplier *= 1.5  # 将编程文档提升50%

    if doc.metadata.get('author') == 'Alice':
        score_multiplier *= 1.2  # 将 Alice 的文档提升20%

    return score_multiplier

# 应用自定义评分器
custom_ranked = rerank_results(
    query, 
    initial_results, 
    method="custom", 
    scorer=category_booster,
    top_k=4
)

print("\n使用自定义评分器重排：")
for r in custom_ranked:
    category = r.document.metadata.get('category')
    author = r.document.metadata.get('author')
    print(f"  {r.rank}. {r.document.id} (new score: {r.score:.3f})")
    print(f"       类别: {category}, 作者: {author}")
```

这种方法提供了一种有效途径，可以将领域特定知识和业务规则直接注入您的检索管道。

### 使用最大边际相关性（MMR）提高多样性

检索中一个常见问题是排名靠前的结果可能高度冗余。例如，搜索“Python async”可能会返回多个文档，它们都以略有不同的方式解释 `async/await` 语法。这不是对大型语言模型上下文 (context)窗口的有效利用。

最大边际相关性（MMR）是一种技术，用于选择一组既与查询相关又具有多样性的结果。它通过优化一个平衡这两个方面的公式，迭代地选择文档。`diversify_results` 函数实现了 MMR。

`diversity_factor` 参数 (parameter)控制这种平衡：

- 值为 `0` 时，优先考虑纯粹的相关性，选择得分最高的文档。
- 值为 `1` 时，优先考虑纯粹的多样性，选择彼此差异最大的文档。
- 介于 `0` 和 `1` 之间的值平衡两者。一个常见的起始点是 `0.5`。

```python
from kerb.retrieval import diversify_results

# 检索一组更大的初始候选文档
initial_results_for_mmr = keyword_search(query, documents, top_k=8)

print("多样化之前（前5项）：")
for r in initial_results_for_mmr[:5]:
    print(f"  {r.rank}. {r.document.id} - {r.document.content[:50]}...")

# 应用多样化
diverse_results = diversify_results(
    initial_results_for_mmr,
    max_results=5,
    diversity_factor=0.5
)

print(f"\n多样化之后（diversity_factor=0.5）：")
for r in diverse_results:
    print(f"  {r.rank}. {r.document.id} - {r.document.content[:50]}...")
```

使用 MMR 确保提供给大型语言模型的上下文覆盖更广泛的信息，减少冗余并提高最终生成答案的质量。

### 合并多查询结果

对于复杂问题，单个查询可能不足以检索所有必要信息。一种常见模式是将复杂查询分解为几个子查询，对每个子查询运行搜索，然后组合结果。倒数排名融合（RRF）是一种简单有效的算法，用于合并多个排名列表。

RRF 根据每个文档在各个结果列表中的排名计算新的分数。该公式对在不同搜索中始终排名靠前的文档给予更高的权重 (weight)。


$$
Score_{RRF}(d) = \sum_{i=1}^{N} \frac{1}{k + rank_i(d)}
$$


这里，$rank_i(d)$ 是文档 $d$ 在结果列表 $i$ 中的排名，$k$ 是一个常数（通常设置为 60），它减弱了低排名的影响。

`reciprocal_rank_fusion` 函数实现了这一点。您可以使用它来组合来自关键词搜索、语义搜索或针对不同查询变体的混合方法的结果。

```python
from kerb.retrieval import reciprocal_rank_fusion

# 从不同查询创建多个结果集
results1 = keyword_search("python async", documents, top_k=5)
results2 = keyword_search("concurrent programming", documents, top_k=5)
results3 = keyword_search("asyncio library", documents, top_k=5)

# 将结果融合成一个排名列表
fused_results = reciprocal_rank_fusion([results1, results2, results3], k=60)

print("来自三个不同查询的融合排名靠前结果：")
for r in fused_results[:5]:
    print(f"  {r.rank}. {r.document.id} (RRF score: {r.score:.3f})")
```

RRF 是一种提高召回率的有效技术，可确保您的 RAG 系统在最终上下文 (context)选择之前考虑更广泛的潜在相关文档。

### 构建多阶段管道

在生产环境中，这些重排技术通常组合成多阶段管道，以逐步提炼检索到的上下文 (context)。

> 多阶段重排管道将初始检索结果提炼为最终的、更优的上下文。

这个管道确保您从一组广泛的文档开始，并系统地将其缩小为对大型语言模型来说最相关、最多样化和最有用的集合。

```python
# 阶段1：检索大量候选文档
stage1 = keyword_search("python async web development", documents, top_k=10)
print(f"阶段1 - 初始检索：{len(stage1)} 个结果")

# 阶段2：按相关性重排，得到更小、更相关的集合
stage2 = rerank_results(query, stage1, method="relevance", top_k=6)
print(f"阶段2 - 相关性重排：{len(stage2)} 个结果")

# 阶段3：应用多样性以避免冗余
stage3 = diversify_results(stage2, max_results=4, diversity_factor=0.4)
print(f"阶段3 - 多样化：{len(stage3)} 个结果")

# 阶段4：基于流行度或新近度的最终提升
final_context_docs = rerank_results(query, stage3, method="popularity", top_k=3)
print(f"阶段4 - 最终上下文：{len(final_context_docs)} 个结果")

print("\n用于大型语言模型上下文的最终排名文档：")
for r in final_context_docs:
    print(f"  {r.rank}. {r.document.id} (score: {r.score:.3f})")
```

通过组合这些技术，您可以显著提高提供给 RAG 系统的上下文质量，从而得到更准确、相关和全面的答案。

## 参考资料

- [The use of MMR for reordering documents and producing summaries](http://doi.acm.org/10.1145/290941.291025) — Jaime G. Carbonell and Jade Goldstein (1998)
  Journal: Proceedings of the 21st annual international ACM SIGIR conference on Research and development in information retrieval; Publisher: ACM; Pages: 335-336; DOI: [10.1145/290941.291025](https://doi.org/10.1145/290941.291025)
  介绍了最大边际相关性 (MMR)，这是一种用于选择相关且多样化文档的方法，旨在减少结果中的冗余。
- [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/) — Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze (2008)
  Publisher: Cambridge University Press
  涵盖信息检索基础知识，包括排序算法、精确度和召回率，这些是理解重排的核心。
- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, Haofen Wang (2023)
  Journal: arXiv preprint arXiv:2312.10997; DOI: [10.48550/arXiv.2312.10997](https://doi.org/10.48550/arXiv.2312.10997)
  概述了检索增强生成 (RAG) 系统，讨论了检索和重排等不同阶段以提高性能。
