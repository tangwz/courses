---
course: "langchain-production-llm"
chapter: "optimizing-scaling-langchain"
lesson: "batch-processing-offline"
sourceId: 3922
sourceUrl: "https://apxml.com/zh/courses/langchain-production-llm/chapter-6-optimizing-scaling-langchain/batch-processing-offline"
title: "离线任务的批量处理"
description: "高效地将LangChain用于大规模离线数据处理或分析任务。"
order: 6
plots: []
sourceHash: "43f7f0653fe874b175c66a93eedef6f2e71cb29300bb202f8da5ac2f88ad77f7"
sourceCorrections: []
---

许多LangChain应用侧重于交互式、实时响应，但并非所有任务都需要即时结果。处理大量数据、生成报告或进行广泛分析通常能从离线、面向批处理的方法中受益。这种策略对于优化和扩展LangChain应用十分有益，能够为非时间敏感型任务实现吞吐量 (throughput)最大化、有效管理成本和高效利用资源。

批量处理涉及对输入集合（例如链或智能体调用）顺序或并行地执行LangChain管道，通常在运行时不直接与用户交互。这与请求-响应模式不同，在请求-响应模式中，每个输入都会触发即时计算和响应。采用批量处理在以下情况中特别有利：

- **需要成本效益：** 在非高峰时段运行计算或使用可能较慢、更经济的LLM端点可以大幅降低运营开支。
- **吞吐量比延迟更重要：** 对于丰富大型数据集等任务，处理所有项的总时间比处理任何单个项所需的时间更重要。
- **处理大型数据集：** 处理数千或数百万条记录需要一种能高效管理规模的方法，这与单个实时请求不同。
- **资源调度：** 批处理作业可以在计算资源可用时安排运行，从而平衡基础设施的负载。

### 批量处理的常见用例

LangChain组件可以轻松应用于批处理工作流，用于各种离线任务：

- **批量数据丰富：** 设想有一个包含大量客户评论的数据库。批处理作业可以使用LangChain链来分类每条评论的情绪，提取重要主题，或将其翻译成不同语言，从而将有价值的结构化信息添加回您的数据库。
- **大规模文档分析：** 处理整个研究论文库或法律文件，以生成摘要、提取实体、回答预设问题或发现语料库中的趋势。
- **合成数据生成：** 通过LangChain使用LLM生成现有数据点的变体，或创建全新的示例来训练机器学习 (machine learning)模型或增强测试套件。
- **定期报告：** 通过查询多个数据源、汇总信息并使用LLM链对其进行结构化，自动生成每日、每周或每月报告。例如，将销售数据、支持工单趋势和市场新闻汇总成一份整合的执行摘要。
- **离线评估运行：** 如第5章所述，评估模型或链的性能通常涉及针对大型静态数据集运行。批量处理非常适合定期执行这些评估管道。

### 实现批量处理工作流

基本思想很简单：遍历输入数据并应用LangChain逻辑。然而，简单的实现可能缓慢且效率低下。

#### 基本顺序处理

最直接的方法是使用一个简单的循环：

```python
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 假设 'inputs' 是一个字典列表，例如 [{'topic': 'AI ethics'}, {'topic': 'quantum computing'}]
# 假设 'llm' 和 'prompt' 已配置
# 示例链：为主题生成一个简短解释
prompt = ChatPromptTemplate.from_template("Explain the basics of {topic} in one sentence.")
llm = ChatOpenAI(model="gpt-4o-mini") # 用您的API密钥配置
chain = prompt | llm

inputs = [
    {"topic": "large language models"},
    {"topic": "vector databases"},
    {"topic": "prompt engineering"},
    # ... 可能还有数千个
]

results = []
start_time = time.time()

for item in inputs:
    try:
        result = chain.invoke(item)
        results.append(result.content)
        # 基本进度指示
        if len(results) % 10 == 0:
             print(f"Processed {len(results)} items...")
    except Exception as e:
        print(f"Error processing item {item}: {e}")
        results.append(None) # 失败项的占位符

end_time = time.time()
print(f"Sequential processing took: {end_time - start_time:.2f} seconds")

# 'results' 现在包含生成的解释或错误情况下的None
```

这可行，但会逐个处理项。如果每次LLM调用需要一秒，处理1000个项将耗时超过16分钟，外加任何开销。对于大型作业，这通常太慢了。

#### 并行处理以提高速度

由于大多数涉及LLM调用的LangChain操作是I/O密集型（等待网络响应），并行执行可以显著提速。Python的`concurrent.futures`模块是管理线程池或进程池的便捷方法。对于API调用等I/O密集型任务，`ThreadPoolExecutor`通常适用。

LangChain的表达语言（LCEL）提供了用于并行执行的内置方法，例如`.batch()`和`.map()`，这些方法简化了部分样板代码。

```python
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
# 假设 'llm' 和 'prompt' 已像之前一样配置
# chain = prompt | llm （之前已定义）

inputs = [
    {"topic": "large language models"},
    {"topic": "vector databases"},
    {"topic": "prompt engineering"},
    {"topic": "retrieval-augmented generation"},
    {"topic": "agentic systems"},
    # ... 可能还有数千个
]

start_time = time.time()

# 使用LCEL的.batch()方法进行并行调用
# max_concurrency 控制并行线程的数量
try:
    # 使用 return_exceptions=True 将错误捕获到结果列表中，而不是中止执行
    results_batch = chain.batch(inputs, config=RunnableConfig(max_concurrency=10), return_exceptions=True)
    
    # 过滤结果，将成功输出与异常分开
    successful_results = [res.content for res in results_batch if not isinstance(res, Exception)]
    errors = [err for err in results_batch if isinstance(err, Exception)]
    
    print(f"Processed {len(inputs)} items. Success: {len(successful_results)}, Errors: {len(errors)}")
    if errors:
        print("Sample Error:", errors[0])

except Exception as e:
    # 捕获批处理设置过程中可能出现的错误
    print(f"An error occurred during batch processing setup: {e}")
    successful_results = [] # 确保即使批处理提前失败，结果列表也存在

end_time = time.time()
print(f"Batch processing took: {end_time - start_time:.2f} seconds")

# 'successful_results' 包含成功调用的内容
```

使用`.batch()`（或`.map()`）通过并发运行多个调用，可以显著减少执行时间。`max_concurrency`参数 (parameter)对于根据API速率限制和资源约束调整性能非常重要。

#### 处理规模：分布式框架

对于超出单个机器内存或计算能力的超大型数据集，请考虑分布式计算框架：

- **Ray：** 一个用于构建分布式应用的开源框架。您可以将LangChain组件封装在Ray任务或Actor中。
- **Dask：** 提供模拟NumPy数组和Pandas DataFrame的并行和分布式集合，允许并行处理大型数据集。
- **Apache Spark：** 一个广泛用于大规模数据处理的引擎。LangChain组件可以在Spark UDF（用户定义函数）中应用，或使用连接Spark和Python执行的库，例如`pyspark.sql.functions.pandas_udf`。

与这些系统的集成需要更多设置，但可以在更大规模上进行处理。

#### 关键说明：速率限制和错误处理

批量处理通常涉及快速连续地进行多次API调用。

- **速率限制：** LLM提供商强制执行速率限制（每分钟请求数、每分钟令牌数）。您的批处理代码*必须*遵守这些限制。实现指数退避和重试逻辑（使用`tenacity`等库）以优雅地处理瞬时速率限制错误。根据您的API计划仔细配置并发（`.batch()`中的`max_concurrency`，或`ThreadPoolExecutor`中的工作线程数）。
- **错误处理：** 批处理中的单个项可能因各种原因（无效输入、API错误、超时）而失败。您的批处理作业应具有容错性。清楚地记录错误，查明是哪个输入导致了失败，并决定是存储失败项以便后续重新处理还是跳过它们。在`.batch()`中使用`return_exceptions=True`是管理此问题的一种方法。

### 优化批处理作业

请考虑以下优化点：

- **成本跟踪：** 在批处理运行时密切监控令牌使用情况。记录每个项的令牌计数，或使用LangSmith（第5章）进行详细追踪。这有助于估算成本并发现意外昂贵的项。
- **模型选择：** 选择经济高效且满足您的批处理任务质量要求的LLM。与复杂的推理 (inference)任务相比，一个更简单、更便宜的模型（如`gpt-4o-mini`）可能足以进行批量分类。
- **缓存：** 如果相同的输入可能在不同的批处理运行中（甚至在同一次运行中）被多次处理，请为LLM响应实现缓存。LangChain提供缓存集成（例如`InMemoryCache`、`SQLiteCache`、`RedisCache`）。
- **输入/输出效率：** 优化加载输入数据和存储结果的方式。高效读取大文件或使用优化的数据库查询可以影响整体作业时长。
- **检查点：** 对于运行时间很长的作业（数小时或数天），请实现检查点。定期保存已处理项和结果的状态。如果作业失败，可以从最后一个检查点恢复，而不是从头开始，从而节省大量时间和成本。

批量处理是扩展LangChain应用的一种有效技术。通过仔细选择您的实现策略（顺序、并行、分布式）、管理API限制和错误，以及优化成本和效率，您可以使用LangChain有效处理大规模离线任务。这补充了您应用的实时能力，为各种生产场景提供了全面的工具集。

## 参考资料

- [How to use .batch and .map in LCEL](https://python.langchain.com/docs/expression_language/) — LangChain Team (2024)
  Publisher: LangChain
  官方LangChain文档，解释如何利用LangChain表达式语言（LCEL）的.batch()和.map()方法高效并行执行链，这对于批处理至关重要。
- [Learning Spark: Lightning-Fast Data Analytics](https://www.oreilly.com/library/view/learning-spark-2nd/9781492050032/) — Jules S. Damji, Brooke Wenig, Tathagata Das, Denny Lee (2020)
  Publisher: O'Reilly Media; Pages: 397
  Apache Spark的综合指南，涵盖分布式数据处理，为使用Spark、Dask和Ray等框架扩展大规模数据集的批处理提供了基础知识。
- [Tenacity Documentation](https://tenacity.readthedocs.io/en/latest/) — sethmlarson and Tenacity Contributors (2024)
  Tenacity库的官方文档，展示如何实现带有指数退避的健壮重试机制，这对于批处理工作流中处理API速率限制和瞬时错误至关重要。
- [Cloud FinOps: Collaborative, Real-Time, Cost Management](https://www.oreilly.com/library/view/cloud-finops/9781492098355/) — J. R. Storment, Mike Fuller (2023)
  Publisher: O'Reilly Media; Pages: 456
  这本书提供了管理和优化云支出的见解，包括成本效率、资源调度和适当服务选择的策略，直接适用于优化LangChain批处理作业。
