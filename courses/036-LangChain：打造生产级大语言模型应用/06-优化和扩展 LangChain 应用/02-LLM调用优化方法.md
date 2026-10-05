# LLM调用优化方法

来源：[原文](https://apxml.com/zh/courses/langchain-production-llm/chapter-6-optimizing-scaling-langchain/llm-call-optimization)

[返回章节目录](README.md) · [返回课程目录](../README.md)

与大型语言模型（LLM）的交互常常是LangChain应用中延迟和运营成本的主要因素。因此，在准备生产部署时，优化这些交互是首要任务。这里将介绍几种提高LLM调用效率的实用方法。

### 缓存LLM响应

许多LLM应用会遇到重复的请求或子请求。对于相同输入多次执行相同的LLM调用会浪费时间和金钱（API成本）。缓存提供了一种简单方法来存储和重用之前的LLM响应。

LangChain提供了内置的缓存机制。最简单的形式是内存缓存，适用于开发或单进程应用：

```python
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

set_llm_cache(InMemoryCache())

# 现在，此进程中任何后续相同的LLM调用（相同模型、参数、提示）
# 都将命中缓存。
# llm.invoke("天空为什么是蓝色的？") # 第一次调用 - 访问LLM
# llm.invoke("天空为什么是蓝色的？") # 第二次调用 - 命中缓存
```

对于生产环境，特别是涉及多个服务器实例或需要持久化的场景，外部缓存是必需的。选项包括数据库支持的缓存（如用于SQL数据库的`SQLAlchemyCache`或`RedisCache`）或专门的向量 (vector)存储缓存（`GPTCache`，但需要单独配置）。

```python
# 使用Redis的例子（需要Redis服务器和redis-py包）
# from langchain.globals import set_llm_cache
# from langchain_community.cache import RedisCache
# import redis

# client = redis.Redis(decode_responses=True)
# set_llm_cache(RedisCache(client))
```

**缓存失效：** 缓存的一个主要问题是确定缓存数据何时失效。如果LLM可能访问的底层知识发生变化，或者特定提示的预期行为发生改变，则需要更新缓存。方法有：

- **生存时间 (TTL)：** 在设定的持续时间后使缓存条目过期。方法简单，但在过期前可能提供过期数据。
- **事件驱动失效：** 当相关数据源更新时清除特定缓存条目（实现更复杂）。
- **语义缓存：** 不使用精确提示匹配，而是基于提示的语义相似性进行缓存。这可以提高缓存命中率，但会增加复杂性，如果调整不当，可能会出现不正确的匹配。需要嵌入 (embedding)提示并比较向量。

实施适当的缓存策略很大程度上依据应用对潜在过期数据的容忍度，以及其对低延迟和成本降低的需求。

### 减少令牌使用

LLM API成本通常根据输入和输出令牌的数量计算。延迟也通常与处理的令牌数量相关。因此，最小化令牌数量是一种直接的优化手段。

**1. 提示工程 (prompt engineering)：**

- **简洁性：** 重新措辞提示，使其更短，同时保留必要的上下文 (context)和指令。删除冗余短语或例子。
- **指令调整：** 对于支持的模型，微调 (fine-tuning)特定的指令格式有时可以用更少的主提示中的明确指令达到预期输出。
- **少样本学习 (few-shot learning)优化：** 分析是否可以使用更少的例子（样本）来达到所需性能，从而减少提示令牌计数。

**2. 上下文管理：**

- **概括：** 不传递整个长对话历史或文档，而是使用另一个（可能更便宜/更快）LLM调用在主要的生成调用之前概括相关部分（如第三章“内存”所述）。
- **选择性上下文：** 实现逻辑，只在提示中包含历史记录或检索文档中最相关部分（例如，基于时新性或与当前查询的语义相似性）。第四章讨论的RAG方法在此也有效。

**3. 输出限制：**

- 明确指示模型简洁或限制其响应长度（例如，“回复不超过100字”）。虽然不总是完美遵循，但这可以显著减少输出令牌。
- 在适用情况下，使用结构化输出解析（第一章）。请求JSON或特定字段通常比自由形式文本生成产生更简洁、可预测的输出。

### 模型选择

并非所有任务都需要使用最强大（且通常最昂贵、最慢）的LLM。请权衡模型能力、速度和成本。

- **分层方法：** 对复杂推理 (inference)或生成任务使用高能力模型（如GPT-4），但对更简单的任务，如概括、分类、数据提取或基本问答，则使用更小、更快、更便宜的模型（如GPT-3.5-Turbo、Claude Haiku，或Llama 3 8B或Mistral 7B等开源替代品）。
- **路由：** 在您的链或代理中实现逻辑，根据复杂程度、预估成本或用户要求将请求路由到不同的模型。
- **微调 (fine-tuning)：** 在特定数据集上微调小型模型，有时可以在狭窄任务上达到与大型通用模型相当的性能，通常成本和延迟仅为其一小部分。

下面的图表展示了比较结果：



[交互图表：模型成本与延迟的权衡](https://apxml.com/zh/courses/langchain-production-llm/chapter-6-optimizing-scaling-langchain/llm-call-optimization#plot-1wkudz7)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "模型成本与延迟的权衡",
    "xaxis": {
      "title": "相对延迟（越低越快）"
    },
    "yaxis": {
      "title": "每百万令牌相对成本（越低越便宜）"
    },
    "showlegend": true
  },
  "data": [
    {
      "type": "scatter",
      "mode": "markers+text",
      "x": [
        1.0,
        0.3,
        0.2,
        0.15
      ],
      "y": [
        1.0,
        0.1,
        0.05,
        0.02
      ],
      "marker": {
        "size": [
          20,
          14,
          12,
          10
        ],
        "color": [
          "#fa5252",
          "#ff922b",
          "#40c057",
          "#228be6"
        ]
      },
      "text": [
        "大型模型 (GPT-4 类)",
        "中型模型 (GPT-3.5 类)",
        "小型微调模型",
        "微型模型 (开源)"
      ],
      "textposition": "top right",
      "name": "模型"
    }
  ]
}
```

</details>



> 不同类型语言模型的相对成本和延迟。实际值因供应商和具体模型版本而异。

### 并行化和异步执行

如果您的LangChain应用涉及多个独立的LLM调用（例如，处理列表中的项目，查询一个主题的多个方面），并发执行这些调用可以显著缩短总耗时。

LangChain的表达式语言（LCEL）内置支持异步操作和并行执行，如第一章介绍的。使用 `ainvoke`、`abatch` 和 `astream` 等方法，以及Python的 `asyncio` 库，可以使LLM API调用等I/O密集型操作并发运行。

```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel

# 假设llm是一个已初始化的ChatOpenAI实例
prompt = ChatPromptTemplate.from_template("讲一个关于{topic}的笑话")
chain = prompt | llm

# 并发运行多个独立调用的例子
topics = ["bears", "programmers", "cheese"]
coroutines = [chain.ainvoke({"topic": t}) for t in topics]

# 并发运行它们
results = await asyncio.gather(*coroutines)
# results将包含每个主题的响应
```

对于更复杂的工作流，LCEL的 `RunnableParallel` 允许在链中定义并行步骤：

```python
# 在链中使用RunnableParallel进行并行LLM调用的例子
joke_chain = ChatPromptTemplate.from_template("讲一个关于{topic}的笑话") | llm
poem_chain = ChatPromptTemplate.from_template("写一首关于{topic}的短诗") | llm

map_chain = RunnableParallel(joke=joke_chain, poem=poem_chain)

# 这将并发执行joke_chain和poem_chain
result = await map_chain.ainvoke({"topic": "robots"})
# result将是一个字典：{'joke': ..., 'poem': ...}
```

利用异步执行对于构建在每个用户请求中执行多次LLM交互的响应式应用程序非常重要。

### 流式响应

对于涉及长文本 (long context)生成（如聊天机器人或内容创建工具）的应用，在显示任何内容之前等待整个LLM响应可能会导致糟糕的用户体验。流式传输允许应用在LLM输出生成时，逐个令牌接收和显示。

大多数LangChain LLM集成通过 `stream` 或 `astream` 方法支持流式传输。

```python
# 流式输出的例子
async for chunk in llm.astream("写一个关于勇敢骑士的短故事。"):
    # 处理每个到达的块（例如，打印到控制台，发送到UI）
    print(chunk.content, end="", flush=True)
```

尽管流式传输不会减少*总*处理时间或成本，但它显著改善了最终用户的*感知*延迟，使应用感觉更具响应性。实现流式传输通常需要改变应用前端处理传入数据的方式，但它是生产级对话式AI的标准做法。

通过系统地应用这些方法：缓存、减少令牌、选择合适的模型、并行化和流式传输，您可以显著提高LangChain应用的性能和成本效益，使其适用于要求高的生产工作负载。后面的章节将讨论扩展其他组件，如数据检索系统。

## 参考资料

- [LangChain Documentation: Production Readiness](https://python.langchain.com/docs/concepts/#production-readiness) — LangChain Team (2024)
  提供LangChain内置优化功能的实现指导，包括缓存、使用LCEL的并行执行以及面向生产环境LLM应用的流式传输。
- [OpenAI Best Practices for Prompt Engineering](https://platform.openai.com/docs/guides/prompt-engineering) — OpenAI (2023)
  Publisher: OpenAI
  提供通过有效的提示工程减少令牌使用和优化模型选择的策略，直接解决成本和延迟问题。
- [asyncio - Asynchronous I/O, event loop, coroutines and tasks](https://docs.python.org/3/library/asyncio.html) — Python Software Foundation (2025)
  Publisher: Python Software Foundation
  Python `asyncio`库的官方文档，该库是实现本节讨论的异步和并行LLM调用的基础。

---

[上一节](01-%E6%9F%A5%E6%98%8E%E6%80%A7%E8%83%BD%E7%93%B6%E9%A2%88.md) · [下一节](03-%E6%88%90%E6%9C%AC%E7%AE%A1%E7%90%86%E4%B8%8E%E4%BB%A4%E7%89%8C%E7%94%A8%E9%87%8F%E8%BF%BD%E8%B8%AA.md)
