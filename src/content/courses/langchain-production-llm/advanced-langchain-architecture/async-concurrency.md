---
course: "langchain-production-llm"
chapter: "advanced-langchain-architecture"
lesson: "async-concurrency"
sourceId: 3846
sourceUrl: "https://apxml.com/zh/courses/langchain-production-llm/chapter-1-advanced-langchain-architecture/async-concurrency"
title: "异步操作与并发"
description: "在LangChain中实施异步模式以提升性能和响应速度。"
order: 2
plots: []
sourceHash: "88a6fefef83294c09a4b447c241c0e1d4c92b36b0e723c5b688145cc27efdad3"
sourceCorrections: []
---

随着使用LangChain构建的应用变得更复杂并处理更多流量，同步执行常会成为主要瓶颈。许多核心操作，比如通过网络与大型语言模型（LLMs）交互、查询向量 (vector)数据库或通过工具调用外部API，本质上都受限于I/O。在同步模型中，应用线程在这些操作期间空闲等待，阻碍了它处理其他请求或执行后台任务的能力。这会导致吞吐量 (throughput)下降和用户体验响应变慢。

异步编程，主要是使用Python的`asyncio`库，提供了一种有效处理这些I/O受限操作的机制。异步操作不会阻塞整个线程，而是在遇到等待时间（例如等待网络响应）时，将控制权交还给事件循环。事件循环随后可以运行其他任务，大幅提升资源利用率和整体应用性能。

### 在LangChain中使用异步

LangChain在设计时就考虑了异步操作。大多数涉及潜在I/O等待的核心组件都提供了标准同步方法的异步对应版本。惯例很直接：如果一个组件有像`invoke()`这样的同步方法，它的异步版本通常会命名为`ainvoke()`。同样，你会找到`abatch()`、`astream()`和`atransform_documents()`，它们分别对应`batch()`、`stream()`和`transform_documents()`。

使用这些方法需要在你的应用代码中使用Python的`async`和`await`关键字。

### LangChain中`asyncio`的核心原理

为了有效运用LangChain的异步功能，对`asyncio`有个基本了解是必要的：

1. **`async def`**: 此语法将一个函数声明为协程。协程是可暂停和恢复的特殊函数。
2. **`await`**: 在`async def`函数内部使用，`await`会暂停当前协程的执行，直到所等待的操作（通常是另一个协程或一个可等待对象）完成。暂停期间，事件循环可以运行其他任务。
3. **事件循环**: `asyncio`的核心。它管理并分配不同异步任务的执行。
4. **`asyncio.run(coroutine())`**: 启动`asyncio`事件循环并运行顶层协程直到完成的一种常用方式。
5. **`asyncio.gather(*coroutines)`**: 一个用于并发的重要函数。它接受多个协程（或可等待对象）作为参数 (parameter)并同时运行它们。它会等待所有协程完成并返回它们的R结果列表。

### 实现异步调用

我们来看看一个简单的同步链，并将其转换为异步执行。

```python
# 同步示例
# 假设已完成提示、模型、解析器的设置
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
# from langchain_core.output_parsers import StrOutputParser

# prompt = ChatPromptTemplate.from_template("讲个关于{topic}的笑话")
# model = ChatOpenAI()
# output_parser = StrOutputParser()
# sync_chain = prompt | model | output_parser

# result = sync_chain.invoke({"topic": "程序员"})
# print(result)
```

要使其异步化，你只需在`async def`函数中使用`ainvoke`方法：

```python
import asyncio
# 假设已完成提示、模型、解析器的设置（如上所述）
# sync_chain = prompt | model | output_parser # 如果组件支持异步，LCEL链会自动支持异步

async def run_async_chain():
  print("正在运行异步链...")
  # 使用 ainvoke 而不是 invoke
  result = await sync_chain.ainvoke({"topic": "数据科学家"})
  print(result)
  # 如果模型支持流式传输：
  # print("正在流式传输响应：")
  # async for chunk in sync_chain.astream({"topic": "异步编程"}):
  #   print(chunk, end="", flush=True)
  # print()


async def main():
    await run_async_chain()

if __name__ == "__main__":
    # 启动 asyncio 事件循环
    asyncio.run(main())
```

注意，链定义本身（`prompt | model | output_parser`）保持不变。LangChain表达式语言（LCEL）会自动处理执行流程。如果你在LCEL链上调用`ainvoke`，它将尝试按顺序调用每个组件的`ainvoke`（或等效的异步）方法。如果一个组件只有同步的`invoke`方法，LCEL通常会在一个单独的线程池执行器中运行它，以避免阻塞主`asyncio`事件循环，尽管这会带来一些开销。为获得最佳性能，请确保链中所有I/O受限的组件都具有原生的异步支持。

### 使用`asyncio.gather`实现并发

异步的主要用途在于同时执行多个操作。想象一下需要用几个不同的提示查询LLM，或同时从多个来源获取文档。`asyncio.gather`是实现此目的的工具。

```python
import asyncio
import time
# 假设已完成提示、模型、解析器和sync_chain的设置

async def call_llm(topic):
    # 用睡眠模拟LLM调用
    print(f"开始LLM调用，主题为：{topic}")
    await asyncio.sleep(1.5) # 模拟网络延迟
    # 在实际场景中：
    # result = await sync_chain.ainvoke({"topic": topic})
    result = f"这是一个关于{topic}的笑话。"
    print(f"完成LLM调用，主题为：{topic}")
    return result

async def run_concurrent_chains():
    start_time = time.time()
    topics = ["猫", "狗", "鹦鹉"]

    # 创建协程对象列表
    tasks = [call_llm(topic) for topic in topics]

    # 并发运行它们
    results = await asyncio.gather(*tasks)

    end_time = time.time()
    print(f"\n所有结果: {results}")
    print(f"总耗时: {end_time - start_time:.2f} 秒") # 应该大约是1.5秒

async def main():
    await run_concurrent_chains()

if __name__ == "__main__":
    asyncio.run(main())
```

如果这些调用是同步进行的，总耗时大约是 $1.5\text{s} \times 3 = 4.5$ 秒。使用`asyncio.gather`，由于模拟的`asyncio.sleep`（代表I/O等待）允许其他任务运行，总耗时更接近于最长单个操作的持续时间（大约1.5秒），加上少量开销。

> 比较三个独立的I/O密集型操作（每个耗时1.5秒）在使用同步执行与异步并发执行时的总耗时。

### 异步流式传输

对于支持此功能的组件（主要是LLMs），`astream()`方法提供了一个异步迭代器。这允许你在响应块可用时立即处理它们，而不是等待整个响应生成完毕。这对面向用户的应用（如聊天机器人）特别有用，能提升感知到的响应速度。

```python
import asyncio
# 假设已完成提示、模型、解析器和sync_chain的设置

async def stream_joke():
    print("正在异步流式传输笑话：")
    async for chunk in sync_chain.astream({"topic": "异步编程"}):
        # 处理每个到达的块（例如，打印到控制台，通过websocket发送）
        print(chunk, end="", flush=True)
    print("\n--- 流式传输结束 ---")

async def main():
    await stream_joke()

if __name__ == "__main__":
    asyncio.run(main())
```

### 使用`abatch`进行批处理

类似于`batch()`，`abatch()`方法允许同时处理多个输入。如果底层组件（例如，LLM提供商的API）支持，LangChain会尝试在单个批处理请求中将这些输入传递给它，与使用`asyncio.gather`进行多次`ainvoke`调用相比，这可能会带来效率提升和API调用开销的降低。

```python
import asyncio
import time
# 假设已完成提示、模型、解析器和sync_chain的设置

async def run_batch_chains():
    start_time = time.time()
    topics = ["AI伦理", "量子计算", "无服务器架构"]
    inputs = [{"topic": t} for t in topics]

    print(f"正在运行 {len(inputs)} 个输入的批处理...")
    # 使用 abatch 进行并发处理
    results = await sync_chain.abatch(inputs)

    end_time = time.time()
    print(f"\n批处理结果: {results}")
    print(f"总耗时: {end_time - start_time:.2f} 秒") # 取决于后端批处理效率

async def main():
    await run_batch_chains()

if __name__ == "__main__":
    asyncio.run(main())
```

`abatch`相对于`asyncio.gather`的性能优势，很大程度上取决于底层服务（LLM API、嵌入 (embedding)模型API等）是否实际针对批处理请求进行了优化。

### 生产环境的重要提示

- **错误处理：** 使用`asyncio.gather`时，如果一个协程引发异常，`gather`会立即传播该异常，可能会取消其他正在进行中的任务（取决于`return_exceptions`参数 (parameter)）。你需要对`gather`调用进行错误处理，可能会在每个协程中使用`try...except`块，或者在`return_exceptions=True`时检查结果。
- **并发限制：** 同时运行过多的异步操作可能会使下游服务（LLM API、数据库）不堪重负，或耗尽本地资源（套接字、内存）。使用`asyncio.Semaphore`等机制来限制访问特定资源的并发任务数量。
- **调试：** 由于非线性的执行流程，调试异步代码可能比同步代码更具挑战性。像LangSmith这样的工具对于跟踪异步链和代理步骤的执行路径会很有帮助。Python内置的`asyncio`调试工具也可能提供帮助。
- **同步-异步集成：** 尽管纯异步通常是I/O受限应用的理想选择，但你可能需要与同步库集成。`asyncio`提供了`loop.run_in_executor()`等机制，可在单独的线程池中运行阻塞的同步代码，而不会阻塞主事件循环，但这会增加开销。
- **事件循环策略（进阶）：** 在某些环境（如Jupyter笔记本或某些Web框架）中，你可能需要关注`asyncio`事件循环策略或明确管理循环生命周期。

通过理解并应用异步模式，你可以构建性能更佳、更具伸缩性且响应更快的LangChain应用，这些都是生产环境就绪的关键特点。熟练掌握`ainvoke`、`astream`、`abatch`和`asyncio.gather`为在LangChain框架中有效处理并发操作奠定了基础。

## 参考资料

- [\`asyncio\` - Asynchronous I/O](https://docs.python.org/3/library/asyncio.html) — Python Software Foundation (2025)
  Publisher: Python Software Foundation
  Python `asyncio` 库的官方文档，提供了异步编程、事件循环、协程和并发任务管理的基础信息。
- [Async in LangChain Expression Language (LCEL)](https://python.langchain.com/docs/how_to/streaming/) — LangChain Team (2024)
  LangChain 官方文档，详细介绍了在 LangChain 应用程序和 LCEL 链中使用 `ainvoke`、`astream` 和 `abatch` 等异步方法。
- [Fluent Python: Clear, Concise, and Effective Programming, 2nd Edition](https://www.vitalsource.com/products/fluent-python-2nd-edition-luciano-ramalho-v9781492056300) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media
  一本内容全面的书籍，其中深入解释了 Python 的并发模型，并有专门章节介绍 `asyncio`，用于构建响应迅速且高效的应用程序。
