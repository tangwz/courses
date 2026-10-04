---
course: "fastapi-ml-deployment"
chapter: "async-operations-performance"
lesson: "async-await-fastapi-routes"
sourceId: 5318
sourceUrl: "https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-5-async-operations-performance/async-await-fastapi-routes"
title: "理解 FastAPI 路由中的 async 和 await"
description: "详细阐述如何在 FastAPI 中定义和使用异步路由处理程序。"
order: 1
plots: []
sourceHash: "a6fc2a70ec50fca18ce7e8e627f6742eb29a42c54904275e629a05838e8b7c09"
sourceCorrections: []
---

FastAPI 使用 Python 的 `asyncio` 库来获得出色的性能，特别是在处理涉及等待的操作时，比如网络请求或文件读取。这项功能直接体现在你定义 API 路由函数的方式上。

### 定义异步路由

在 FastAPI 中创建异步路由处理程序时，你需要使用 `async def` 语法来定义函数，而不是标准的 `def`。这会告诉 FastAPI，该函数可能会执行一些可以暂停（通过 `await`）而不会阻塞整个服务器进程的任务。

```python
from fastapi import FastAPI
import asyncio

app = FastAPI()

@app.get("/async-data")
async def get_async_data():
    # 模拟一个 I/O 密集型操作，例如从数据库获取数据或调用另一个 API。asyncio.sleep 常被用作占位符。
    await asyncio.sleep(1) # 在此处暂停执行 1 秒
    return {"message": "Data fetched asynchronously!"}

@app.get("/sync-data")
def get_sync_data():
    # 模拟一个同步操作或一个快速任务。
    # 如果此操作涉及大量 I/O 且不是异步的，它可能会阻塞服务器。
    return {"message": "Data fetched synchronously!"}
```

在上面的例子中，`get_async_data` 函数是一个异步路由处理程序。`await asyncio.sleep(1)` 这一行很关键。当代码运行到此处时，函数不会停止所有进程，而是通知底层的事件循环说：“我需要在这里等待一些事情；在此期间你可以处理其他任务。”

### `await` 和事件循环的作用

Python 的 `async`/`await` 语法依赖于一个**事件循环**。可以将事件循环想象成一个管理多个任务的调度器。当一个 `async` 函数遇到 `await` 表达式（它必须用于另一个可等待对象，比如另一个 `async` 函数调用的结果或某些 I/O 操作）时，它会明确地告诉事件循环：“在这里暂停我，直到这个 `await` 的操作完成。你可以运行其他待处理的任务。”

这种协作式多任务处理使得单个 Python 进程（运行事件循环）能够高效地处理大量并发连接。当一个路由处理程序正在等待数据库响应时，事件循环可以切换去处理一个传入请求，处理另一个刚刚完成 `await` 的路由处理程序，或者管理其他后台活动。

考量 `/async-data` 端点的以下顺序：

1. 一个请求到达 `/async-data`。
2. FastAPI 调用 `get_async_data` 函数。
3. 执行代码直到 `await asyncio.sleep(1)`。
4. `get_async_data` 函数将控制权交还给事件循环。
5. 事件循环现在可以处理其他传入请求，或继续运行其他已准备好的任务。
6. 1 秒后，`asyncio.sleep(1)` 操作完成。
7. 事件循环安排 `get_async_data` 函数的剩余部分继续运行。
8. 执行在 `await` 行之后恢复。
9. 函数准备响应 `{"message": "Data fetched asynchronously!"}`。
10. 响应被发送回客户端。

在 1 秒的 `await` 期间，服务器并未闲置；它有空处理其他任务。这与使用 `time.sleep(1)` 的传统同步函数有着根本的不同，后者会在此期间阻塞整个执行线程，使其无法处理任何其他请求。

> 该图示对比了异步路由交出控制权与同步路由阻塞执行的运行过程。

### I/O 密集型任务的益处

将 `async def` 用于路由处理程序的主要优点在于，当处理程序需要执行**I/O 密集型操作**时，其优势会非常明显。这些任务指的是程序大部分时间都在等待外部资源的那些任务，例如：

- 网络调用（从其他 API、数据库获取数据）。
- 从磁盘或对象存储读写数据。
- 等待来自外部消息队列的响应。

通过对这些操作使用 `await`，你的 FastAPI 应用能够以更少的服务器资源处理多得多的并发请求，相比于纯同步框架而言，因为它不会在主动等待上浪费时间。

### 同步路由如何处理？

FastAPI 很智能。如果你像我们第一个例子中的 `get_sync_data` 那样，使用 `def` 而不是 `async def` 定义一个标准路由处理程序，FastAPI 会知道这个函数可能包含阻塞代码。为了防止这类同步代码阻塞主事件循环，FastAPI 会将其放在一个单独的**线程池**中运行。这使得事件循环能够保持响应，而同步函数则在其自己的线程中执行。

虽然这提供了兼容性，并防止简单的同步代码导致整个应用程序停止运行，但了解其中的权衡是很重要的。管理线程池会带来其自身的额外负担，并且对于真正的 I/O 密集型任务，通常使用 `async def` 配合 `await` 会更有效率。CPU 密集型任务，例如复杂的计算或模型推理 (inference)（我们将在下一节讨论），即使对于线程池方法来说，也带来了不同的难题。

掌握 `async` 和 `await` 如何让你的路由处理程序暂停和恢复，对于使用 FastAPI 构建高性能 API 非常重要，特别是在集成数据获取或预处理等涉及等待外部系统的任务时。

## 参考资料

- [Concurrency and async/await](https://fastapi.tiangolo.com/async/) — Sebastián Ramírez (2024)
  这份官方指南解释了FastAPI如何利用Python的`async`/`await`实现异步操作，并区分了`async def`和`def`路由以及FastAPI如何通过线程池处理同步代码。
- [asyncio - Asynchronous I/O](https://docs.python.org/3/library/asyncio.html) — Python Software Foundation (2024)
  Publisher: Python Software Foundation
  官方Python `asyncio`文档提供了异步编程、协程、`await`able对象和事件循环的基础知识，这些都是FastAPI性能模型的核心。
- [Fluent Python: Clear, Concise, and Effective Programming](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056348/) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media; Pages: 1012
  该书深入介绍了Python的并发特性，包括`asyncio`、事件循环和`async`/`await`模式，提供了全面的理解。第二版（2022年）包含`asyncio`的更新内容。
