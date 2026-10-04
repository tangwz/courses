---
course: "getting-started-with-llm-toolkit"
chapter: "performance-and-cost-optimization"
lesson: "cache-invalidation-strategies"
sourceId: 7791
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-9-performance-and-cost-optimization/cache-invalidation-strategies"
title: "缓存失效策略"
description: "了解不同的缓存失效策略，以确保你的应用程序使用最新数据。"
order: 4
plots: []
sourceHash: "8f67b65e829053950af3daa374e791bf51aa3d5be4737005a9a8289aabdfb173"
sourceCorrections: []
---

虽然缓存是提升性能和降低成本的有效手段，但也带来了一个新问题：数据陈旧。提供过时信息的缓存可能导致应用程序行为不正确，这往往比完全没有缓存更糟糕。缓存失效是指当底层数据发生变化时，从缓存中移除或更新条目的过程，以确保你的应用程序数据准确且及时。

选择合适的失效策略取决于你的数据如何随时间变化。我们将介绍几种常见且有效的策略供你使用。

### 存活时间 (TTL) 失效

最简单的失效策略是存活时间 (TTL)。使用 TTL，你可以为每个缓存条目设置一个过期时间。一旦时间过期，该条目就会被视为无效，并在下次访问时自动或在周期性清理过程中从缓存中移除。

此策略非常适合在可预测时间段后会变旧的数据。例如，天气预报、股票价格或新闻头条都是基于 TTL 缓存的不错选择。如果你正在构建一个报告当前天气的应用程序，你可能会将结果缓存 15 分钟。

在向缓存添加条目时，你可以使用 `set` 方法中的 `ttl` 参数 (parameter)来设置存活时间。时间以秒为单位。

```python
from kerb.cache import create_memory_cache, generate_prompt_key

# 为时间敏感数据创建缓存
weather_cache = create_memory_cache()

prompt = "What is the current stock price for AAPL?"
key = generate_prompt_key(prompt, model="gpt-4")

# 模拟 API 调用以获取股票价格
response = {"stock": "AAPL", "price": 180.25, "timestamp": "2024-10-26T10:00:00Z"}

# 将响应缓存，设置 5 分钟 (300 秒) 的 TTL
weather_cache.set(key, response, ttl=300)

print(f"已缓存 '{prompt}' 的股票价格，并设置 5 分钟 TTL。")
```

300 秒后，对此键的 `get` 调用将返回 `None`，从而触发一次新的 API 调用以获取最新的股票价格。

### 手动失效

对于变化时间无法预测的数据，手动失效是必需的。这种方法让你能够明确控制何时移除缓存条目。它适用于由用户操作或外部事件更新的内容，例如编辑知识库中的文档、更改产品描述或更新用户档案。

当底层数据变化时，你的应用程序逻辑应显式删除缓存中对应的条目。你可以使用 `delete` 方法来完成此操作。

```python
from kerb.cache import create_memory_cache, generate_prompt_key

# 产品描述缓存
product_cache = create_memory_cache()

# 总结产品的提示
product_id = "prod_123"
product_description = "A durable, high-performance laptop for professionals."
prompt = f"Summarize this product: {product_description}"
key = generate_prompt_key(prompt, model="gpt-4", product_id=product_id)

# 缓存初始摘要
summary = "A powerful and reliable laptop for professional use."
product_cache.set(key, summary)

print(f"已缓存 {product_id} 的初始摘要。")

# 之后，产品描述被更新...
updated_description = "An ultra-light, high-performance laptop for creative professionals."

# 你的应用程序现在应该使旧的缓存条目失效。
# 我们生成之前使用的相同键来删除它。
was_deleted = product_cache.delete(key)

if was_deleted:
    print(f"{product_id} 的缓存因更新而失效。")
else:
    print(f"未找到 {product_id} 的缓存条目。")

# 下次请求摘要时，将出现缓存未命中，
# 强制根据更新后的描述生成新摘要。
```

这种事件驱动的方法确保你的缓存始终与你的数据源（例如数据库或内容管理系统）保持同步。

### 基于版本的键

处理失效的一种整洁方法是将版本标识符整合到你的缓存键中。当你的提示、模型或底层逻辑发生变化时，你只需递增版本号。这会自动创建新的缓存键，从而有效地使所有旧条目失效，而无需删除它们。旧的、未被引用的条目最终将被缓存的替换策略（如 LRU）清除。

此策略对于管理提示更新或应用程序配置变更非常有效。它避免了复杂的删除逻辑，并提供清晰的变更历史记录。

`generate_prompt_key` 函数旨在处理任意关键字参数 (parameter)，从而轻松添加版本。

```python
from kerb.cache import generate_prompt_key

prompt = "Analyze customer sentiment"
model = "gpt-4o-mini"

# 生成我们提示逻辑的 1.0 版本
key_v1 = generate_prompt_key(prompt, model=model, version="1.0")
print(f"对于 v1.0: {key_v1[:24]}...")

# 改进提示后，我们将版本更新到 1.1
key_v2 = generate_prompt_key(prompt, model=model, version="1.1")
print(f"对于 v1.1: {key_v2[:24]}...")

print(f"\n键不同: {key_v1 != key_v2}")
```

因为 `key_v1` 和 `key_v2` 不同，版本更新后的请求将导致缓存未命中，从而强制使用更新后的逻辑进行新的 LLM 调用。旧的 `v1.0` 缓存响应会保留，但将不再被访问。

### 使用前缀进行批量失效

有时你需要一次性使一整组相关条目失效。例如，如果你更新 RAG 系统中的整个文档，你会想要清除所有与该文档相关的缓存块和嵌入 (embedding)。手动删除每个条目将效率低下。

`LLMCache` 类提供了 `invalidate_by_prefix` 方法，它会移除所有键以特定字符串开头的条目。这要求在命名时采用规范的方法，即为相关项包含一个共同的前缀。

```python
from kerb.cache import create_llm_cache, create_memory_cache

# 我们使用 LLMCache，因为它具有高级失效功能
llm_cache = create_llm_cache(backend=create_memory_cache())

# 为文档的不同块缓存嵌入
document_id = "doc_abc"
chunks = ["First part of the text.", "Second part of the text."]
models = ["text-embedding-3-small", "text-embedding-3-large"]

# 创建带有共同前缀的键：doc_abc:embedding:
for i, chunk in enumerate(chunks):
    for model in models:
        # 一个简单的键策略，用于演示
        key = f"{document_id}:embedding:{model}:{i}"
        llm_cache.cache_embedding(text=chunk, embedding=[0.1, 0.2], model=model) # 模拟嵌入
        print(f"已缓存键为 '{key}' 的条目")

# 之后，文档 'doc_abc' 被更新。我们需要使其所有嵌入失效。
prefix_to_invalidate = f"{document_id}:embedding:"
invalidated_count = llm_cache.invalidate_by_prefix(prefix_to_invalidate)

print(f"\n文档 '{document_id}' 已更新。正在使所有相关嵌入失效...")
print(f"已使 {invalidated_count} 个带有前缀 '{prefix_to_invalidate}' 的条目失效。")
```

这种模式在处理结构化、互联数据的系统中，对于管理缓存一致性非常高效。

### 选择失效策略

选择正确的策略在于理解你的数据生命周期。以下是一些指南：

- **使用 TTL** 针对按可预测时间表过期的数据，如新闻源或临时会话数据。
- **使用手动失效** 当数据因外部事件而发生不可预测的变化时，例如用户更新其个人资料。
- **使用基于版本的键** 当你更新应用程序的逻辑、提示或模型时。这是最整洁、最安全的方法之一。
- **使用批量失效** 用于管理相关数据组，例如 RAG 系统中属于单个文档的所有块。

实际上，应用程序通常会结合使用这些策略，以有效地处理不同类型的缓存数据。

## 参考资料

- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  一本关于理解复杂数据系统的基础资源，讨论了分布式缓存、一致性模型和有效管理数据陈旧性的策略。
- [Caching best practices](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG2w3AlyWvEDmg9IU3sHd69Y7gcwDd_zHgjy_2PQ8ZbV3FkprQDeUnsAIEjQkxqS5HpEcAF6bxOfiTYNuwjE-LKtS0b_r3TFRLDWIBaJkb_DCfVCPhsL3zHALlCmdv_27D5k9eIy19sV1QBuyqKnb_yF_sYRKcvCkhn5z6vBRkYSlWbkuNVNhje1jPrAaO0eg06SWNSwDHxkcXLEIVvZRJhlI5wbuE=) — Google Cloud (2025)
  Publisher: Google Cloud
  领先云服务商的官方架构指南，为构建可伸缩应用程序提供了关于缓存失效技术和一致性考量的实用建议。
