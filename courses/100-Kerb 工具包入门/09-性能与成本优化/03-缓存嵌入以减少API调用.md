# 缓存嵌入以减少API调用

来源：[原文](https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-9-performance-and-cost-optimization/caching-embeddings)

[返回章节目录](README.md) · [返回课程目录](../README.md)

缓存文本嵌入 (embedding)带来了很多好处，显著提升了许多应用的性能。比如，在检索增强生成（RAG）中，嵌入过程常是主要的瓶颈。每个文档块都需转换为向量 (vector)，而每次转换都需要一次API调用，这会耗费时间和金钱。

在开发期间，您可能运行索引流程数十次。在生产环境中，文档可能被不同服务重复处理。若无缓存，您将为完全相同的文本重复计算嵌入，白白消耗资源。

### 嵌入 (embedding)缓存的原理

嵌入缓存通过存储一段文本的向量 (vector)表示来实现作用。当您的应用需要嵌入文本时，它首先检查缓存中是否已有该精确文本的向量。

该流程遵循以下步骤：

1. **生成缓存键**：通过哈希文本内容和嵌入模型的名称来创建一个唯一的键。这保证了用不同模型（例如 `text-embedding-3-small` 与本地模型）嵌入的相同文本会有不同的缓存条目。
2. **检查缓存**：系统在缓存中查找此项。
3. **缓存命中**：如果找到该标识符，立即返回存储的嵌入向量，避免了一次API调用。
4. **缓存未命中**：如果未找到键，应用会调用嵌入API，获取向量，并使用生成的键将其存储在缓存中，然后返回。

此机制特别有用，因为嵌入是确定性的；相同的文本和模型总是会产生相同的向量。`cache` 模块提供了专门的工具来处理此流程。

### 实现嵌入 (embedding)缓存

`LLMCache` 类提供了专门用于处理嵌入的高级方法：`cache_embedding` 和 `get_cached_embedding`。这些方法与 `generate_embedding_key` 协同作用，以创建一致且唯一的键。

我们先创建一个简单的内存缓存，并手动检查命中和未命中的情况。我们将模拟一次嵌入API调用，以明确何时使用了缓存。

```python
import time
from typing import List
from kerb.cache import create_llm_cache, generate_embedding_key

# 模拟一个耗时的嵌入API调用
def get_embedding_from_api(text: str, model: str) -> List[float]:
    print(f"  -> 正在为 '{text[:20]}...' 调用嵌入API，模型：{model}")
    time.sleep(0.5)  # 模拟网络延迟
    # 根据文本长度返回一个模拟嵌入
    return [len(text) / 100.0] * 5

# 1. 创建一个LLM专用缓存
llm_cache = create_llm_cache()

# 2. 定义文本和模型
text_to_embed = "This is the document chunk we need to embed."
model_name = "text-embedding-3-small"

# --- 第一次请求（缓存未命中）---
print("首次请求文本：")
embedding = llm_cache.get_cached_embedding(text=text_to_embed, model=model_name)

if embedding is None:
    print("  ✗ 缓存未命中。生成并缓存嵌入。")
    # 调用API
    new_embedding = get_embedding_from_api(text=text_to_embed, model=model_name)
    # 存储到缓存
    llm_cache.cache_embedding(text=text_to_embed, embedding=new_embedding, model=model_name)
    embedding = new_embedding
else:
    print("  ✓ 缓存命中！")

print(f"  嵌入向量（前5个值）：{embedding[:5]}")

# --- 第二次请求（缓存命中）---
print("\n第二次请求相同文本：")
embedding = llm_cache.get_cached_embedding(text=text_to_embed, model=model_name)

if embedding is None:
    print("  ✗ 缓存未命中。这不应该发生！")
else:
    print("  ✓ 缓存命中！立即返回存储的嵌入。")

print(f"  嵌入向量（前5个值）：{embedding[:5]}")
```

在第一次请求中，缓存为空，因此我们看到“缓存未命中”消息，并调用了我们模拟的API函数。生成的向量 (vector)随后被存储。在第二次请求完全相同的文本和模型时，缓存会立即返回存储的向量，完全跳过了耗时的API调用。

### 将缓存整合到RAG流程中

在实际应用中，您会将此逻辑封装在一个辅助函数中，以保持主代码整洁。让我们看看这在典型的RAG文档处理流程中会是怎样的，我们会处理一个文档块列表。这也是缓存价值最大的地方，因为相同文档在开发和测试期间常被重复处理。

在这里，我们将追踪节省的成本。`LLMCache` 旨在通过追踪命中和未命中情况，并估算节省的时间和金钱来提供帮助。

```python
from kerb.cache import create_llm_cache, create_tiered_cache
from kerb.cache.backends import LLMCache

# 模拟一个带成本的嵌入API调用
def embed_with_cost(text: str, model: str) -> dict:
    print(f"  -> 正在为 '{text[:20]}...' 调用嵌入API")
    time.sleep(0.5)
    cost_per_token = 0.00002  # ada-002的示例成本
    num_tokens = len(text.split())
    return {
        "embedding": [num_tokens / 100.0] * 5,
        "cost": num_tokens * cost_per_token
    }

def get_embedding_with_cache(text: str, model: str, cache: LLMCache) -> List[float]:
    """获取嵌入，如果可能则使用缓存。"""
    cached_embedding = cache.get_cached_embedding(text=text, model=model)
    if cached_embedding:
        print(f"  ✓ 缓存命中 '{text[:20]}...' ")
        return cached_embedding

    print(f"  ✗ 缓存未命中 '{text[:20]}...' ")
    result = embed_with_cost(text, model)

    cache.cache_embedding(
        text=text,
        embedding=result["embedding"],
        model=model,
        cost=result["cost"]
    )
    return result["embedding"]

# 在此示例中使用持久化分层缓存
# 这会将嵌入保存到磁盘，以便跨脚本运行保持有效
persistent_cache_backend = create_tiered_cache(disk_cache_dir=".embedding_cache")
llm_cache = create_llm_cache(backend=persistent_cache_backend)

document_chunks = [
    "RAG combines retrieval with generation.",
    "Embeddings convert text into vectors.",
    "Caching reduces latency and cost.",
    "RAG combines retrieval with generation.", # 重复
    "Embeddings convert text into vectors.", # 重复
]

print("处理RAG流程的文档块（首次运行）：")
for chunk in document_chunks:
    get_embedding_with_cache(text=chunk, model="text-embedding-3-small", cache=llm_cache)

stats_run1 = llm_cache.get_stats()
print(f"\n--- 首次运行统计 ---")
print(f"缓存命中: {stats_run1.hits}")
print(f"缓存未命中: {stats_run1.misses}")
print(f"预计节省成本: ${stats_run1.estimated_cost_saved:.6f}")

print("\n再次处理相同的文档块（第二次运行）：")
for chunk in document_chunks:
    get_embedding_with_cache(text=chunk, model="text-embedding-3-small", cache=llm_cache)

stats_run2 = llm_cache.get_stats()
print(f"\n--- 第二次运行后的累计统计 ---")
print(f"缓存命中: {stats_run2.hits}")
print(f"缓存未命中: {stats_run2.misses}")
print(f"预计节省成本: ${stats_run2.estimated_cost_saved:.6f}")
```

请留意，第一次运行对唯一的文档块产生了三次缓存未命中。然而，第二次运行产生了五次缓存命中，因为所有文档块（包括重复的）都已处理并存储。每次后续运行都会累积成本节省。

### 选择缓存后端

缓存后端的选择决定了嵌入 (embedding)的存储方式和位置。`cache` 模块提供了几种适用于不同情形的选项。

- `create_memory_cache()`: 创建一个内存缓存。它非常快但易失；当应用停止时，缓存会被清除。这非常适合开发、测试或短期进程，您希望在单次运行中避免冗余调用。
- `create_disk_cache(cache_dir="...")`：在磁盘上创建一个持久化缓存。嵌入会保存到指定目录的文件中，因此它们在应用重启后依然存在。这是您在开发过程中多次运行RAG索引流程的最佳选择。
- `create_tiered_cache()`：这会创建一个两级缓存，结合了快速内存缓存和持久化磁盘缓存。它提供了两方面的优势：内存中常用项的即时访问，以及磁盘上更大的持久存储。这是大多数生产应用推荐的后端。

创建高级 `LLMCache` 时，您可以轻松切换后端。例如，为您的应用设置一个持久化缓存：

```python
from kerb.cache import create_llm_cache, create_tiered_cache

# 创建一个分层缓存，内存大小为200项
# 并在'.cache/embeddings'目录中设置持久化磁盘缓存
tiered_backend = create_tiered_cache(
    memory_max_size=200,
    disk_cache_dir=".cache/embeddings"
)

# 用LLMCache封装，以获取嵌入专用方法
llm_cache = create_llm_cache(backend=tiered_backend)

# 现在像以前一样使用缓存
# embedding = get_embedding_with_cache(text="...", model="...", cache=llm_cache)
```

通过实现嵌入缓存，您可以构建一个更高效、更经济、更快速的应用。这是一种基本的优化技术，能带来可观的回报，特别是在处理大量文本的系统中。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  本文介绍了检索增强生成（RAG）框架，该框架依赖于高效的文本嵌入和检索，使嵌入缓存成为一项重要优化。
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings) — OpenAI (2024)
  Publisher: OpenAI
  官方文档，提供了使用OpenAI嵌入模型的实用指南，详细说明了其功能、可用模型和API集成，与管理嵌入API调用相关。
- [Designing Data-Intensive Applications: The Art of Robust, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  本书深入讨论了分布式系统中的缓存机制和策略，解释了适用于优化性能和减少资源消耗的基本原理。
- [Building Machine Learning Powered Applications: Going from Idea to Product](https://www.barnesandnoble.com/w/building-machine-learning-powered-applications-emmanuel-ameisen/1133345863) — Emmanuel Ameisen (2020)
  Publisher: O'Reilly Media; Pages: 257
  本资源涵盖了构建和部署机器学习应用程序的实用考虑因素，包括生产环境中性能改进和成本管理的策略。

---

[上一节](02-%E5%AE%9E%E7%8E%B0LLM%E5%93%8D%E5%BA%94%E7%BC%93%E5%AD%98.md) · [下一节](04-%E7%BC%93%E5%AD%98%E5%A4%B1%E6%95%88%E7%AD%96%E7%95%A5.md)
