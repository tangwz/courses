---
course: "building-advanced-llm-agent-tools"
chapter: "developing-custom-python-tools"
lesson: "asynchronous-tool-operations"
sourceId: 6565
sourceUrl: "https://apxml.com/zh/courses/building-advanced-llm-agent-tools/chapter-2-developing-custom-python-tools/asynchronous-tool-operations"
title: "异步工具操作：实现非阻塞任务"
description: "理解并实现LLM工具的异步执行，以处理耗时任务并避免阻塞代理操作。"
order: 6
plots: []
sourceHash: "5813f7914455937442ab529a44e880edf8dbf71dbf2b4d1e931a070edec97ec9"
sourceCorrections: []
---

当你的LLM代理与其环境互动时，无论是从远程API获取数据、查询数据库，还是读取大型文件，这些操作都可能耗时。如果你的工具同步执行这些任务，整个代理可能变得无响应，等待操作完成。这对于交互式代理或预期高效处理多任务的系统来说，是个大问题。异步操作提供了一个解决方案，让你的工具能够执行耗时的I/O密集型任务，同时不阻塞代理的主执行线程。通过使用 Python 的 `asyncio` 库可以构建非阻塞工具，从而提升代理的响应能力和整体表现。

## 工具中异步操作的必要性

设想一个LLM代理，其任务是收集来自多个网络源的信息以回答一个复杂查询。如果每个网络请求都由同步工具处理，代理将发出一个请求，等待响应，处理完毕，然后发出下一个请求，依此类推。在每个等待期间，代理实际上都处于暂停状态。

异步编程，特别是使用Python的`asyncio`框架，允许程序在等待外部操作（如网络请求）完成时暂停某个任务，并切换到其他任务。一旦外部操作完成，暂停的任务即可继续。对于LLM代理而言，这意味着工具可以发起请求，而在等待服务器响应期间，代理（如果其底层框架支持）可能可以为其他步骤做准备，或者事件循环可以管理其他并发的工具操作。

你的工具的主要好处是，它们在I/O等待期间不会独占执行线程。这对于以下工具具有重要意义：

- 与网络服务交互（API、网页抓取）。
- 执行数据库查询。
- 读取或写入文件，特别是大文件。
- 与其他进程通信。

## Python 的 asyncio：快速入门

`asyncio` 的核心是几个基本原理：

- **协程 (Coroutines)：** 这些是使用 `async def` 定义的特殊函数。当你调用一个协程时，它返回一个协程对象，该对象不会立即执行函数的代码。
- **`await`：** 这个关键词在 `async def` 函数内部使用，用于暂停协程的执行，直到被 `await` 的操作（通常是另一个协程或一个可等待对象）完成。暂停期间，事件循环可以运行其他任务。
- **事件循环 (Event Loop)：** `asyncio` 事件循环是运行异步任务和回调、管理网络I/O并实现并发的引擎。

请看这个简单的异步函数：

```python
import asyncio

async def greet_after_delay(name: str, delay: int) -> str:
    print(f"Starting greeting for {name}, will wait {delay} seconds.")
    await asyncio.sleep(delay)  # 在此暂停，允许其他任务运行
    greeting = f"Hello, {name}!"
    print(f"Finished waiting for {name}.")
    return greeting

# 要运行此函数（通常由代理的框架或主异步函数完成）：
# async def main():
#     message = await greet_after_delay("Alice", 2)
#     print(message)
#
# if __name__ == "__main__":
#     asyncio.run(main())
```

在这个例子中，`asyncio.sleep(delay)` 是一个可等待操作。当遇到 `await asyncio.sleep(delay)` 时，`greet_after_delay` 协程暂停执行，允许事件循环处理其他任务。在指定的延迟后，事件循环在 `await` 语句之后立即恢复 `greet_after_delay` 的执行。

## 构建异步工具

要创建异步工具，你需要将其定义为 `async def` 函数。在这个函数内部，任何通常会阻塞的I/O密集型操作都应替换为它的异步版本并使用 `await`。

例如，不再使用流行的 `requests` 库进行HTTP调用（它是同步的），你将使用像 `aiohttp` 或 `httpx` 这样的异步HTTP客户端库。

以下是如何定义一个用于从URL获取数据的异步工具：

```python
import aiohttp
import asyncio

# 这是LLM代理将调用的工具函数
async def fetch_url_content_tool(url: str) -> str:
    """
    异步从给定URL获取内容。
    返回文本内容或错误消息。
    """
    if not url.startswith(('http://', 'https://')):
        return "错误：提供的URL无效。必须以 http:// 或 https:// 开头。"

    try:
        async with aiohttp.ClientSession() as session:
            # 'await' 关键字在此处暂停执行，直到GET请求完成
            async with session.get(url, timeout=10) as response:
                response.raise_for_status()  # 对于不良状态码（4xx或5xx）抛出异常
                # 'await' 也用于读取响应体
                content = await response.text()
                # 为简化起见，如果内容过长，我们只返回前500个字符
                return content[:500] if len(content) > 500 else content
    except aiohttp.ClientError as e:
        return f"获取URL时出错：{str(e)}"
    except asyncio.TimeoutError:
        return f"错误：请求 {url} 在10秒后超时。"
    except Exception as e:
        # 捕获任何其他意外错误
        return f"发生意外错误：{str(e)}"

# 此类工具如何在异步环境中测试或使用的示例
# async def run_example():
#     web_content = await fetch_url_content_tool("https://api.example.com/data")
#     print(f"Fetched content (or error): {web_content}")
#
# if __name__ == "__main__":
#     # 在实际代理中，代理的框架将管理事件循环
#     # 以及如何调用异步工具并处理其结果。
#     asyncio.run(run_example())
```

在 `fetch_url_content_tool` 中，`session.get(url)` 和 `response.text()` 是异步操作。使用 `await` 确保这些操作不会阻塞事件循环。如果LLM代理的框架是基于 `asyncio` 构建的，它就可以高效地管理这些工具的执行。

### 可视化同步与异步流程

执行流程的差异非常明显。同步工具调用会阻塞代理，而异步工具则允许事件循环在等待期间管理其他任务。

> 图示同步与异步工具操作的执行流程。在异步模型中，工具在I/O等待期间让出控制权，从而使事件循环保持响应。

## 提升代理性能的优势

将异步操作整合到你的工具中会带来以下几项好处：

1. **提升响应能力：** 代理在等待慢速I/O操作时不会卡住。这对于需要与用户互动或处理多个请求的代理尤其重要。
2. **提高吞吐量 (throughput)：** 如果代理的架构支持，它可以同时启动多个异步工具操作，或者在工具等待时处理其他内部任务。这可以显著加快需要多个I/O密集型步骤的任务。
3. **资源利用高效：** 与传统的多线程方法相比，异步代码可以用更少的线程处理许多并发的I/O操作，这可能减少内存开销。

## 异步工具的注意事项

异步工具功能强大，但其实现也伴随着一些需要注意的地方：

- **代理框架兼容性：** LLM代理框架或执行代理的环境必须支持 `asyncio`。大多数现代代理框架（如 LangChain 或 LlamaIndex）都提供了与异步工具协作的机制。工具提供 `async def` 接口；框架负责在其事件循环中正确地 `await` 它。
- **异步库：** 你将需要为I/O密集型任务使用异步版本的库。例如，`aiohttp` 用于HTTP请求，`asyncpg` 或 `aiomysql` 用于数据库交互，`aiofiles` 用于文件操作。
- **错误处理：** 异步代码需要仔细的错误处理。`try...except` 块应在 `await` 调用周围使用，以捕获异步操作期间可能发生的异常，包括 `asyncio.TimeoutError`。
- **复杂性：** 异步编程可能引入一种不同的程序流程思考方式，这在初期可能会有更高的学习曲线。调试也可能更复杂，尽管Python的`asyncio`调试工具已经有所改进。
- **“全程异步”：** 要使一个操作真正非阻塞，该操作所涉及的整个调用堆栈，从工具到I/O驱动程序，都需要是异步的。从异步工具中调用同步阻塞函数仍然会在该点阻塞事件循环。

通过周全地纳入异步操作，你可以构建出不仅功能强大，还能促使LLM代理更高效、更具响应能力的工具。对于网络密集型或处理任何形式的外部I/O（等待不可避免）的工具来说，这一点尤为明显。本章后面的实践环节，“构建数据库查询工具”，如果你的数据库驱动支持，会是异步实现的一个很好选择。

## 参考资料

- [asyncio - Asynchronous I/O, event loop, coroutines and tasks](https://docs.python.org/3/library/asyncio.html) — Python Software Foundation (2023)
  Python `asyncio` 库的官方文档，提供了异步编程的完整参考。
- [aiohttp - asynchronous HTTP client/server for asyncio and Python](https://docs.aiohttp.org/en/stable/) — Nikolay Kim and Andrew Svetlov (2023)
  `aiohttp` 的官方文档，这是一个与 `asyncio` 结合使用的异步 HTTP 客户端和服务器库，常用于网络操作。
- [Fluent Python: Clear, Concise, and Effective Programming](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGTbWs41_FqjvUzU-RTsZzTQjIfMEKS7RZv9JALXVzGoWR3vCjUEUvTVs7SGLH7bbRtTo9rwjKosZ06M4muiwbSN9oxect2ElwLktaByE-6tLVHEnKF1nam-kxffyNg5F17YL4o6dwx8_O2TgmYffeMvEgUL0dsy7UVwUD25Nfc8YxfXdKSvWSOfIx_NWXX) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media; Pages: 1012
  一本权威指南，涵盖了 Python 的高级特性，其中包括关于 `asyncio` 和事件循环的并发编程章节。
- [Python Concurrency with asyncio](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE0t6ZevcX-f04Xc01dkQu8lUQN4Pj9Q4A5ExtD9Okrz1y02nM_evI7LYFLkn1vQfNaqx-Ov62ufeAu0f51Vp5ORDpWbuTe9xLLaTA2VqhQ6snQm0fqi6SaX5VX76X35z7e26XhivwrDBPIWAtJYMNfOrlfi3Rrw06YAqjIsg7UXvcC39JVNyEKfFvyc0TYgDrCLueFPYWssoVu6Oji) — Matthew Fowler (2022)
  Publisher: Manning Publications
  一本完全专注于 Python `asyncio` 库的综合指南，涵盖了异步编程的实用模式和最佳实践。
