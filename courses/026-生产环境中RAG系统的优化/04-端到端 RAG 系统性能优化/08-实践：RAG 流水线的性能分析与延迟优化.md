# 实践：RAG 流水线的性能分析与延迟优化

来源：[原文](https://apxml.com/zh/courses/optimizing-rag-for-production/chapter-4-end-to-end-rag-performance/hands-on-rag-latency-optimization)

[返回章节目录](README.md) · [返回课程目录](../README.md)

动手实践 RAG 系统优化，对示例 RAG 流水线进行性能分析。识别延迟瓶颈并应用有针对性的优化，并衡量这些改动的影响。系统化的方法可以带来显著的性能提升，因为低延迟通常是面向用户应用程序的重要的要求。

我们这里的目标不仅仅是向你展示 *优化什么*，更是教你 *如何* 在自己的 RAG 系统中进行性能分析。

### 搭建示例 RAG 流水线

我们来定义一个简单的 RAG 流水线，它包含：

1. **查询嵌入 (embedding)**：将输入查询转换为向量 (vector)。
2. **检索**：从向量存储中获取相关文档。
3. **重排序**：使用更强大的模型重新排序检索到的文档，以提高相关性。
4. **生成**：根据查询和重排序后的文档生成答案。

本练习将使用 Python。首先，请确保你已安装必要的库：
`pip install sentence-transformers faiss-cpu numpy`

这是我们流水线的基本结构。为简单起见，我们将使用内存中的 FAISS 索引并模拟生成器的处理时间。

```python
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

# 1. 初始化模型
print("正在加载模型...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("模型加载完成。")

# 2. 准备示例数据和向量存储
documents = [
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
    "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy.",
    "The Amazon rainforest is the largest tropical rainforest, famed for its biodiversity.",
    "Quantum computing studies theoretical computation systems that make direct use of quantum-mechanical phenomena.",
    "The Colosseum is an oval amphitheatre in the centre of the city of Rome, Italy, built of travertine limestone, tuff, and brick-faced concrete."
]

print("正在嵌入文档...")
doc_embeddings = embedding_model.encode(documents)
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)
print("FAISS 索引已创建。")

# 3. 定义流水线阶段
def embed_query(query_text):
    start_time = time.perf_counter()
    query_vector = embedding_model.encode([query_text])
    end_time = time.perf_counter()
    print(f"查询嵌入延迟: {end_time - start_time:.4f}s")
    return query_vector

def retrieve_documents(query_vector, top_k=3):
    start_time = time.perf_counter()
    distances, indices = index.search(query_vector, top_k)
    retrieved_docs = [documents[i] for i in indices[0]]
    end_time = time.perf_counter()
    print(f"检索延迟: {end_time - start_time:.4f}s")
    return retrieved_docs, indices[0]

def rerank_documents(query_text, retrieved_docs):
    if not retrieved_docs:
        return []
    start_time = time.perf_counter()
    pairs = [[query_text, doc] for doc in retrieved_docs]
    scores = reranker_model.predict(pairs)
    # 根据重排序器得分对文档进行排序
    reranked_docs_with_scores = sorted(zip(scores, retrieved_docs), key=lambda x: x[0], reverse=True)
    reranked_docs = [doc for score, doc in reranked_docs_with_scores]
    end_time = time.perf_counter()
    print(f"重排序延迟: {end_time - start_time:.4f}s")
    return reranked_docs

def generate_answer(query_text, context_docs):
    start_time = time.perf_counter()
    # 模拟 LLM 生成延迟
    # 在真实系统中，这涉及格式化上下文并调用 LLM
    prompt = f"查询: {query_text}\n\n上下文:\n" + "\n".join(context_docs)
    # print(f"LLM 的提示长度: {len(prompt)} 个字符")
    time.sleep(0.5) # 模拟 LLM 处理时间
    generated_text = f"根据上下文，与 '{query_text}' 相关联的答案在此合成。"
    end_time = time.perf_counter()
    print(f"生成延迟: {end_time - start_time:.4f}s (模拟)")
    return generated_text

# 4. 端到端 RAG 函数
def full_rag_pipeline(query_text):
    print(f"\n正在处理查询: '{query_text}'")
    total_start_time = time.perf_counter()

    query_vector = embed_query(query_text)
    retrieved_docs, _ = retrieve_documents(query_vector, top_k=3)
    print(f"已检索到: {retrieved_docs}")

    reranked_docs = rerank_documents(query_text, retrieved_docs)
    print(f"已重排序: {reranked_docs}")

    # 使用重排序后的前 N 个文档作为生成上下文
    context_for_generation = reranked_docs[:2] if reranked_docs else []

    answer = generate_answer(query_text, context_for_generation)

    total_end_time = time.perf_counter()
    print(f"生成的答案: {answer}")
    print(f"流水线总延迟: {total_end_time - total_start_time:.4f}s")
    return answer

# 运行流水线
sample_query = "Tell me about ancient Rome"
_ = full_rag_pipeline(sample_query)
```

这个脚本提供了一个基本的 RAG 流程，带有打印语句，用于记录每个主要步骤的时间。运行它将使你初步了解时间消耗在哪里。

### 使用 `cProfile` 和 `SnakeViz` 进行性能分析

虽然 `time.perf_counter()` 对于粗粒度计时很有用，但 Python 内置的 `cProfile` 模块提供了更详细的函数调用时间明细。`SnakeViz` 随后可以可视化这些性能分析数据，使发现瓶颈更容易。

安装 SnakeViz:
`pip install snakeviz`

要对我们的 `full_rag_pipeline` 函数进行性能分析，你可以使用 `cProfile` 运行你的脚本：

```bash
python -m cProfile -o rag_profile.prof your_script_name.py
```

将 `your_script_name.py` 替换为你的 Python 文件名。此命令将执行你的脚本并将性能分析数据保存到 `rag_profile.prof`。

然后，使用 SnakeViz 可视化它：

```bash
snakeviz rag_profile.prof
```

这将打开一个网页浏览器界面。查找具有高“TotalTime”或“CumTime”（累计时间，包括子函数调用）的函数。你可能会看到模型推理 (inference)（sentence-transformers 的 `encode`，cross-encoders 的 `predict`）以及我们模拟的用于生成的 `time.sleep` 消耗了大量时间。

### 识别初步瓶颈

从初步运行和 `cProfile` 输出中，你可能会发现：

1. **查询嵌入 (embedding)**：消耗一定时间，但对于单个查询而言通常少于其他步骤。
2. **检索**：对于小型、本地的 FAISS 索引，这非常快。对于大型、基于磁盘或网络化的向量 (vector)数据库，这可能是一个重要因素。
3. **重排序**：交叉编码器计算密集。即使是少数文档的重排序也可能消耗可观的时间。这通常是优化的首要考虑对象。
4. **生成**：LLM 调用（此处通过 `time.sleep(0.5)` 模拟）通常是 RAG 流水线中最耗时的部分。

我们假设性能分析突出显示重排序步骤和 LLM 生成是主要的时间消耗者。

### 实践优化 1：选择性重排序

重排序器处理每个查询-文档对。如果初始检索返回 `k` 个文档，我们将执行 `k` 次交叉编码器预测。我们可以通过仅对最初检索到的顶部文档中较小的子集进行重排序来减少这种负载，例如 `top_n_rerank`，其中 `top_n_rerank < k`。

我们稍微修改一下 `full_rag_pipeline` 和 `retrieve_documents` 函数。我们将最初检索更多文档（例如，`top_k_retrieve = 10`），但只对更少的数量进行重排序（例如，`top_n_rerank = 3`）。

更新 `retrieve_documents` 以接受 `top_k_retrieve` 参数 (parameter)：

```python
# ... (之前的代码) ...

def retrieve_documents(query_vector, top_k_retrieve=3): # 将 top_k 重命名为 top_k_retrieve
    start_time = time.perf_counter()
    distances, indices = index.search(query_vector, top_k_retrieve)
    retrieved_docs = [documents[i] for i in indices[0]]
    end_time = time.perf_counter()
    print(f"检索延迟 ({top_k_retrieve} 个文档): {end_time - start_time:.4f}s")
    return retrieved_docs, indices[0]

# ... (流水线其他阶段) ...
```

现在，修改 `full_rag_pipeline` 以实现选择性重排序：

```python
# ... (之前的函数：embed_query, retrieve_documents, rerank_documents, generate_answer)

def full_rag_pipeline_optimized_reranking(query_text):
    print(f"\n正在处理具有优化重排序的查询: '{query_text}'")
    total_start_time = time.perf_counter()

    query_vector = embed_query(query_text)

    # 最初检索更多，例如，前 5 个
    initial_retrieval_count = 5 
    documents_to_consider, _ = retrieve_documents(query_vector, top_k_retrieve=initial_retrieval_count)
    print(f"最初检索到 ({initial_retrieval_count} 个): {documents_to_consider[:3]}...") # 显示前几个

    # 只对这些文档中的顶部，例如，3 个，进行重排序
    docs_for_reranking = documents_to_consider[:3] 

    reranked_docs = rerank_documents(query_text, docs_for_reranking)
    print(f"已重排序 (从 {len(docs_for_reranking)} 个文档中): {reranked_docs}")

    # 使用重排序后的前 N 个文档作为生成上下文
    context_for_generation = reranked_docs[:2] if reranked_docs else []

    answer = generate_answer(query_text, context_for_generation)

    total_end_time = time.perf_counter()
    print(f"生成的答案: {answer}")
    print(f"流水线总延迟 (优化重排序): {total_end_time - total_start_time:.4f}s")
    return answer

# 运行原始和优化后的流水线进行比较
sample_query = "Tell me about ancient Rome"
print("\n--- 正在运行基线流水线 ---")
_ = full_rag_pipeline(sample_query)

print("\n--- 正在运行具有优化重排序的流水线 ---")
_ = full_rag_pipeline_optimized_reranking(sample_query)
```

运行此代码后，比较“重排序延迟”和“流水线总延迟”的输出。如果重排序是与文档数量成比例的瓶颈，你应该会看到重排序时间的减少。权衡是，初始检索器排名低于 `top_n_rerank` 的潜在相关文档将没有机会被重排序器提升。性能和准确性之间的这种平衡在 RAG 优化中很常见。

### 实践优化 2：缓存 LLM 响应

如果某些类型的查询经常生成相似的上下文 (context)，缓存 LLM 的响应可以节省大量时间和成本。这里，我们将为 `generate_answer` 函数实现一个简单的内存缓存。对于生产环境，你会使用更完善的方案，例如 Redis。

```python
# ... (之前的代码，包括模型初始化和其他流水线阶段) ...

llm_response_cache = {}

def generate_answer_with_cache(query_text, context_docs):
    # 从查询和上下文创建缓存键
    # 更完善的方法可能涉及哈希或文本归一化
    cache_key_list = [query_text] + sorted(context_docs) # 对文档进行排序以保持键的一致性
    cache_key = "##".join(cache_key_list)

    if cache_key in llm_response_cache:
        start_time = time.perf_counter()
        cached_answer = llm_response_cache[cache_key]
        end_time = time.perf_counter()
        print(f"生成延迟 (缓存命中): {end_time - start_time:.4f}s (可忽略)")
        return cached_answer

    # 如果不在缓存中，则继续生成
    start_time = time.perf_counter()
    prompt = f"查询: {query_text}\n\n上下文:\n" + "\n".join(context_docs)
    time.sleep(0.5) # 模拟 LLM 处理时间
    generated_text = f"根据上下文，与 '{query_text}' 相关联的答案在此合成 (全新生成)。"
    end_time = time.perf_counter()

    llm_response_cache[cache_key] = generated_text # 存储到缓存中
    print(f"生成延迟 (缓存未命中 - 模拟): {end_time - start_time:.4f}s")
    return generated_text

# 更新优化后的流水线以使用带缓存的生成器
def full_rag_pipeline_optimized_reranking_and_cache(query_text):
    print(f"\n正在处理具有优化重排序和缓存的查询: '{query_text}'")
    total_start_time = time.perf_counter()

    query_vector = embed_query(query_text)
    initial_retrieval_count = 5
    documents_to_consider, _ = retrieve_documents(query_vector, top_k_retrieve=initial_retrieval_count)
    docs_for_reranking = documents_to_consider[:3]
    reranked_docs = rerank_documents(query_text, docs_for_reranking)
    context_for_generation = reranked_docs[:2] if reranked_docs else []

    # 使用带缓存的生成器
    answer = generate_answer_with_cache(query_text, context_for_generation) 

    total_end_time = time.perf_counter()
    print(f"生成的答案: {answer}")
    print(f"流水线总延迟 (优化重排序和缓存): {total_end_time - total_start_time:.4f}s")
    return answer

# 测试缓存
sample_query = "Eiffel Tower information"
print("\n--- 正在运行具有优化重排序和缓存的流水线 (第一次) ---")
_ = full_rag_pipeline_optimized_reranking_and_cache(sample_query)

print("\n--- 正在运行具有优化重排序和缓存的流水线 (第二次 - 应命中缓存) ---")
_ = full_rag_pipeline_optimized_reranking_and_cache(sample_query)
```

当你运行此代码时，第一次调用 `full_rag_pipeline_optimized_reranking_and_cache` 对于给定查询，生成步骤将是缓存未命中。第二次使用 *完全相同的查询和结果上下文* 的调用应该会命中缓存，你将看到“生成延迟”和“流水线总延迟”显著减少。

### 可视化性能提升

应用优化后，可视化其影响很有用。假设我们的初始计时如下：

- 查询嵌入 (embedding)：0.05秒
- 检索 (3 个文档)：0.01秒
- 重排序 (3 个文档)：0.30秒
- LLM 生成：0.50秒 (模拟)
- 总计：约 0.86秒

选择性重排序后（例如，检索 5 个，重排序 2 个）：

- 查询嵌入：0.05秒
- 检索 (5 个文档)：0.015秒
- 重排序 (2 个文档)：0.20秒 (从 0.30秒 减少)
- LLM 生成：0.50秒
- 总计：约 0.765秒

加上缓存后（第二次相同请求时）：

- 查询嵌入：0.05秒
- 检索 (5 个文档)：0.015秒
- 重排序 (2 个文档)：0.20秒
- LLM 生成 (缓存命中)：0.0001秒 (可忽略)
- 总计：约 0.2651秒

我们可以用图表表示：



[交互图表：RAG 流水线延迟优化影响](https://apxml.com/zh/courses/optimizing-rag-for-production/chapter-4-end-to-end-rag-performance/hands-on-rag-latency-optimization#plot-1p1mi2w)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "type": "bar",
      "name": "基线",
      "x": [
        "查询嵌入",
        "检索",
        "重排序",
        "LLM 生成"
      ],
      "y": [
        0.05,
        0.01,
        0.3,
        0.5
      ],
      "marker": {
        "color": "#ff6b6b"
      },
      "text": [
        0.05,
        0.01,
        0.3,
        0.5
      ],
      "textposition": "auto"
    },
    {
      "type": "bar",
      "name": "选择性重排序",
      "x": [
        "查询嵌入",
        "检索",
        "重排序",
        "LLM 生成"
      ],
      "y": [
        0.05,
        0.015,
        0.2,
        0.5
      ],
      "marker": {
        "color": "#fcc419"
      },
      "text": [
        0.05,
        0.015,
        0.2,
        0.5
      ],
      "textposition": "auto"
    },
    {
      "type": "bar",
      "name": "优化 + 缓存 (命中)",
      "x": [
        "查询嵌入",
        "检索",
        "重排序",
        "LLM 生成"
      ],
      "y": [
        0.05,
        0.015,
        0.2,
        0.0001
      ],
      "marker": {
        "color": "#40c057"
      },
      "text": [
        0.05,
        0.015,
        0.2,
        0.0001
      ],
      "textposition": "auto"
    }
  ],
  "layout": {
    "title": "RAG 流水线延迟优化影响",
    "barmode": "group",
    "yaxis": {
      "title": "延迟 (秒)"
    },
    "xaxis": {
      "title": "流水线阶段"
    },
    "legend": {
      "orientation": "h",
      "yanchor": "bottom",
      "y": 1.02,
      "xanchor": "right",
      "x": 1
    }
  }
}
```

</details>



> 不同 RAG 流水线配置的延迟细分。基线显示初始计时。“选择性重排序”减少重排序延迟。“优化 + 缓存（命中）”说明由于缓存，生成延迟显著减少。

### 进一步考虑和后续步骤

本次实践练习涉及了几个重要方面。在实际应用中，你还会考虑：

- **向量 (vector)数据库优化**：对于大型数据集，优化你的向量数据库（索引、分片、硬件）很重要。直接对数据库的搜索查询进行性能分析。
- **硬件加速**：如果使用 GPU 进行嵌入 (embedding)、重排序或生成，确保它们得到充分利用。像 `nvidia-smi` 这样的工具可以监控 GPU 使用情况。PyTorch Profiler 或 TensorFlow Profiler 可以提供关于 GPU 内核执行时间的见解。
- **模型量化 (quantization)/蒸馏**：如第 3 章所述，使用更小或量化模型进行嵌入、重排序或生成可以显著减少延迟。在应用此类方法前后进行性能分析。
- **异步处理**：对于可以并行运行或不需要阻塞主请求线程的组件（例如日志记录、一些后处理），考虑使用 `asyncio` 进行异步执行。
- **批处理**：虽然本次练习侧重于单个查询延迟，但如果你的系统同时处理多个请求，对模型（嵌入、重排序、生成）输入进行批处理可以提高吞吐量 (throughput)。

性能分析是一个迭代过程。优化一个瓶颈后，重新进行性能分析以找出下一个。始终衡量你的更改对延迟和整体系统质量（例如，检索和生成准确性）的影响。这种识别和解决性能问题的实践方法对于构建快速、响应迅速且可扩展的生产 RAG 系统是基础性的。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  介绍了结合检索与生成的RAG框架，这是提升LLM响应的核心方法。
- [Faiss: A Library for Efficient Similarity Search](https://arxiv.org/abs/1702.08734) — Jeff Johnson, Matthijs Douze, Hervé Jégou (2017)
  Journal: Proceedings of the European Conference on Computer Vision (ECCV) Workshops; DOI: [10.48550/arXiv.1702.08734](https://doi.org/10.48550/arXiv.1702.08734)
  阐述了FAISS，一个用于密集向量高效相似性搜索的库，这在RAG检索中很关键。
- [The Python Profilers](https://docs.python.org/3/library/profile.html) — Python Software Foundation (2024)
  Python官方`cProfile`模块文档，用于性能分析。
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) — Nils Reimers, Iryna Gurevych (2019)
  Journal: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP); DOI: [10.48550/arXiv.1908.10084](https://doi.org/10.48550/arXiv.1908.10084)
  介绍了Sentence-BERT，一个生成语义丰富句子嵌入的模型，与查询和文档嵌入有关。

---

[上一节](07-%E7%94%9F%E4%BA%A7RAG%E7%B3%BB%E7%BB%9F%E7%9A%84%E8%B4%9F%E8%BD%BD%E5%9D%87%E8%A1%A1%E4%B8%8E%E8%87%AA%E5%8A%A8%E4%BC%B8%E7%BC%A9.md) · [下一节](../05-%E7%94%9F%E4%BA%A7RAG%E7%9A%84%E6%88%90%E6%9C%AC%E4%BC%98%E5%8C%96/01-%E8%AF%86%E5%88%AB%E7%94%9F%E4%BA%A7RAG%E4%B8%AD%E7%9A%84%E6%88%90%E6%9C%AC%E8%A6%81%E7%B4%A0.md)
