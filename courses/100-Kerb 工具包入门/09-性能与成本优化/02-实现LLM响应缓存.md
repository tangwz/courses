# 实现LLM响应缓存

来源：[原文](https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-9-performance-and-cost-optimization/llm-response-caching)

[返回章节目录](README.md) · [返回课程目录](../README.md)

反复调用LLM API会带来延迟并增加成本。一个直接且有效的缓解办法是使用缓存。LLM响应缓存是指保存生成调用结果，并在再次发出完全相同的请求时重复使用它。这避免了不必要的API调用，从而实现更快的响应速度和显著的成本节省，特别是在具有重复查询的应用中。

任何缓存系统的**核心**是**缓存键**。此键是一个根据函数输入生成的独特标识符。对于LLM调用，输出不仅取决于提示文本，还会受到模型名称、温度及其他生成参数 (parameter)的影响。一个有效的缓存键必须包含所有这些因素，以避免结果不准确。使用`gpt-4o-mini`请求摘要与使用`claude-3-5-haiku`请求摘要不同，两者都应有各自独立的缓存条目。

此工具包提供了一个实用函数`generate_prompt_key`，专为此目的设计。它根据提示和任何指定的模型参数创建一个确定性哈希值。

```python
from kerb.cache import generate_prompt_key

same_prompt = "Explain Python programming"

# 相同的提示，但参数不同会生成不同的键
key_gpt4 = generate_prompt_key(same_prompt, model="gpt-4", temperature=0.7)
key_gpt35 = generate_prompt_key(same_prompt, model="gpt-3.5-turbo", temperature=0.7)
key_temp1 = generate_prompt_key(same_prompt, model="gpt-4", temperature=1.0)

print(f"model=gpt-4, temp=0.7:         {key_gpt4[:16]}...")
print(f"model=gpt-3.5-turbo, temp=0.7: {key_gpt35[:16]}...")
print(f"model=gpt-4, temp=1.0:         {key_temp1[:16]}...")
```

可以看到，提示和参数的每个独特组合都会生成一个独特键，这保证了只有当请求完全相同时，我们才会重复使用缓存的响应。

### 缓存的运作方式

实现响应缓存遵循一个直接的模式：“先检查，再计算”。在进行API调用之前，您会检查缓存中是否存在具有对应标识符的条目。

1. **生成缓存键**：使用`generate_prompt_key`及提示和模型参数 (parameter)。
2. **检查缓存**：尝试使用该标识符从缓存中获取条目。
3. **缓存命中**：如果找到数据，立即返回。这会完全避免API调用。
4. **缓存未命中**：如果未找到数据，则执行LLM API调用。
5. **存储结果**：使用相同的标识符将新响应存储到缓存中，以便将来的请求可以使用。

下图显示了这个流程，突出了缓存如何绕过昂贵的API调用。

> 缓存逻辑拦截请求。缓存命中直接返回已存储的响应，而未命中则会继续调用API并存储新结果。

我们来看看实际操作。首先，我们创建一个内存缓存，它对于单会话应用来说既快速又简单。

```python
from kerb.cache import create_memory_cache

# 创建一个简单的内存缓存
cache = create_memory_cache(max_size=100)
```

现在，我们可以为给定提示实现该流程。

```python
# 假设 mock_llm_api_call 存在并返回一个包含响应和成本的字典
# from some_module import mock_llm_api_call 

prompt = "What is the weather like today?"
model_params = {"model": "gpt-4", "temperature": 0.7}

# 1. 生成
key = generate_prompt_key(prompt, **model_params)

# 2. 检查缓存
cached_response = cache.get(key)

if cached_response:
    # 3. 缓存命中
    print("✓ Cache hit - no API call needed!")
    response = cached_response
else:
    # 4. 缓存未命中
    print("✗ Cache miss - calling API")
    response = mock_llm_api_call(prompt, **model_params)

    # 5. 存储新响应
    cache.set(key, response)

print(f"Response: {response['response']}")
```

这种模式有效，但可能给每个调用LLM的函数增加重复代码。一个更好的做法是将此逻辑封装在一个专门的客户端类中。

### 构建一个缓存客户端

为了更简洁的应用代码，您可以围绕LLM客户端构建一个包装类，它将自动处理缓存。此类将管理缓存实例，并在其生成方法中实现“先检查，再计算”的逻辑。

这是一个`CachedLLMClient`的示例，它提供了一个带有内置缓存的`generate`方法。

```python
class CachedLLMClient:
    """带有自动缓存功能的LLM客户端。"""

    def __init__(self):
        self.cache = create_memory_cache()
        self.api_calls = 0
        self.cache_hits = 0

    def generate(self, prompt, model="gpt-4", temperature=0.7, **kwargs):
        """生成带有自动缓存的响应。"""
        # 根据所有相关参数生成缓存标识符
        key = generate_prompt_key(
            prompt=prompt,
            model=model,
            temperature=temperature,
            **kwargs
        )

        # 检查缓存
        cached = self.cache.get(key)
        if cached:
            self.cache_hits += 1
            return cached["response"]

        # 如果未命中，调用实际API
        self.api_calls += 1
        response = mock_llm_api_call(prompt, model, temperature, **kwargs)

        # 将新响应存储到缓存中
        self.cache.set(key, response)

        return response["response"]

    def stats(self):
        """获取使用统计。"""
        total = self.api_calls + self.cache_hits
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        return {
            "total_requests": total,
            "api_calls": self.api_calls,
            "cache_hits": self.cache_hits,
            "hit_rate": f"{hit_rate:.1f}%"
        }

# 使用客户端
client = CachedLLMClient()

print("Using CachedLLMClient:")
client.generate("What is machine learning?")
client.generate("What is machine learning?") # 这将是缓存命中
client.generate("Explain neural networks")
client.generate("What is machine learning?") # 这将是另一次缓存命中

# 显示统计信息
stats = client.stats()
print(f"\nClient Statistics:")
print(f"  Total requests: {stats['total_requests']}")
print(f"  API calls:      {stats['api_calls']}")
print(f"  Cache hits:     {stats['cache_hits']}")
print(f"  Hit rate:       {stats['hit_rate']}")
```

该客户端发出了四个请求，但只进行了两次实际API调用，实现了50%的命中率。在有大量重复查询的应用中，这个命中率可能会高得多，从而带来性能和成本的显著提升。

### 跟踪成本节省

缓存的一个主要好处是降低成本。您可以通过存储每次API调用及其响应的成本来量化 (quantization)这些节省。当缓存命中时，您可以记录节省的金额。

缓存实例上的`set`方法接受可选的`metadata`参数 (parameter)，用于存储不属于缓存值本身的额外信息。这是存储原始API调用成本的理想位置。

```python
# 在我们的CachedLLMClient或手动流程中...
# 当缓存未命中时:
response = mock_llm_api_call(prompt, model)
# 将响应及其相关成本存储在元数据中
cache.set(key, response, metadata={"cost": response["cost"]})

# 当缓存命中时:
cached_entry = cache.get_entry(key) # get_entry 获取值和元数据
if cached_entry:
    response = cached_entry.value
    saved_cost = cached_entry.metadata.get("cost", 0.0)
    total_saved += saved_cost
```

通过跟踪`total_saved`，您可以直接衡量缓存实现的财务影响。在生产系统中，这些数据对于监控运营开支和展示性能优化的投资回报具有极高价值。

### 缓存后端和持久性

`kerb.cache`模块提供多种存储后端，适用于不同的使用场景。

- **MemoryCache**：一个内存缓存，通过`create_memory_cache()`创建。它速度极快但易失，即应用程序重启时缓存会被清除。它非常适合在单个运行进程中服务状态，例如网络服务器。
- **DiskCache**：一个基于文件系统的缓存，通过`create_disk_cache()`创建。它通过写入磁盘来持久保存数据，即使应用程序重启也不会丢失。虽然比`MemoryCache`慢，但它对于需要重复使用先前运行结果的命令行工具或批处理任务很有用。

  ```python
  # 创建一个将数据存储在本地.cache/llm目录中的缓存
  disk_cache = create_disk_cache(cache_dir=".cache/llm", serializer="json")
  ```
- **TieredCache**：此后端通过`create_tiered_cache()`创建，结合了`MemoryCache`和`DiskCache`。它为常用项目提供内存缓存的速度，同时使用磁盘缓存进行持久性保存，并作为更大、更慢的备份。这通常是需要高性能和数据持久性的生产应用的优选。

## 参考资料

- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  涵盖数据存储、分布式系统和缓存机制的基本概念，对设计高效的LLM缓存架构有价值。
- [Caching LLM calls](https://github.com/openai/openai-cookbook/blob/main/examples/Caching_LLM_calls.ipynb) — OpenAI (2023)
  Journal: OpenAI Cookbook; Publisher: OpenAI
  提供专门针对大型语言模型API调用实现缓存的实际示例和指导，涵盖键生成和工作流程。
- [High Performance Python: Practical Performant Programming for Humans](https://www.oreilly.com/library/view/high-performance-python-2nd/9781492055020/) — Micha Gorelick, Ian Ozsvald (2020)
  Publisher: O'Reilly Media; Pages: 468
  讨论各种优化技术，包括记忆化和缓存，有助于提高与外部服务交互的Python应用程序的效率和速度。

---

[上一节](01-%E5%AE%9A%E4%BD%8D%E6%80%A7%E8%83%BD%E7%93%B6%E9%A2%88.md) · [下一节](03-%E7%BC%93%E5%AD%98%E5%B5%8C%E5%85%A5%E4%BB%A5%E5%87%8F%E5%B0%91API%E8%B0%83%E7%94%A8.md)
