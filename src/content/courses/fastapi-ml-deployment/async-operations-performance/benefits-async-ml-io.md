---
course: "fastapi-ml-deployment"
chapter: "async-operations-performance"
lesson: "benefits-async-ml-io"
sourceId: 5330
sourceUrl: "https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-5-async-operations-performance/benefits-async-ml-io"
title: "ML I/O 异步请求的优势"
description: "异步如何在处理与 ML 任务相关的 I/O 操作（例如，获取数据、保存结果）时提高性能。"
order: 5
plots: []
sourceHash: "5c56cc36125816beb7ee3407fd2c073fb4afd6ed7084a2c3f91291ef32ef07ef"
sourceCorrections: []
---

"虽然核心机器学习 (machine learning)推理 (inference)步骤通常受限于 CPU，但许多 ML API 工作流涉及大量的输入/输出 (I/O) 操作。这些操作可能包括从远程数据库获取特征数据、从另一个微服务检索用户配置文件、加载配置文件，或将预测日志保存到存储设备。当同步执行时，这些 I/O 任务会成为主要的性能瓶颈。"

考虑一个典型的 API 请求，它在运行预测前需要获取数据：

1. 接收请求。
2. 查询数据库以获取所需特征（网络 I/O 等待）。
3. 预处理获取的数据（CPU 工作）。
4. 运行模型推理（CPU 工作，可能计算密集）。
5. 后处理结果（CPU 工作）。
6. 将结果或日志保存到存储设备（磁盘/网络 I/O 等待）。
7. 发送响应。

在传统的同步框架中，如果步骤 2 涉及等待数据库 100 毫秒，处理该请求的工作进程将完全阻塞。在该等待期间，它无法处理任何其他传入请求。同样，在步骤 6 中，工作进程再次阻塞，等待存储操作完成。如果您的 API 收到许多并发请求，大多数工作进程可能只是在等待 I/O，导致高延迟和低吞吐量 (throughput)。

这就是异步编程的优势所在。通过使用 `async def` 定义您的路由处理程序，并在调用 I/O 密集型函数时使用 `await`（由兼容异步的库提供，例如用于 HTTP 请求的 `httpx`、用于数据库访问的 `asyncpg` 或 `databases`、用于文件系统操作的 `aiofiles`），您就可以让 FastAPI 的事件循环有效管理这些等待时间。

当 I/O 操作遇到 `await` 时（例如 `await database.fetch_one(...)` 或 `await http_client.get(...)`），函数会暂停在该点的执行。然而，重要的是，工作进程*不会*阻塞。事件循环可以切换上下文 (context)，并使用工作进程处理其他就绪任务，例如处理不同的传入请求，或继续执行已完成 I/O 等待的其他异步函数。一旦原始 I/O 操作完成（例如，数据库返回数据），事件循环就会从暂停的地方恢复已暂停的函数。

> 同步和异步 I/O 处理的对比。同步工作进程按顺序处理请求，并在每次 I/O 等待时阻塞。异步工作进程可以启动多个 I/O 操作，并在 I/O 完成时在处理任务之间切换，从而提高整体吞吐量。

在您的 ML API 中将异步操作用于 I/O 密集型任务的主要优点包括：

1. **并发能力增强：** 应用程序可以处理更多并发请求，因为工作进程不会因等待缓慢的 I/O 而停滞。它们可以被释放去开始或继续处理其他请求。
2. **吞吐量提高：** 通过高效使用工作进程，而不是让它们在 I/O 等待期间空闲，API 可以在单位时间内处理更多请求。
3. **（负载下的）更低延迟：** 虽然单个请求的延迟可能不会明显降低（它仍然需要等待其 I/O），但在负载下，用户感受到的*平均*延迟通常会更低，因为请求不会在因 I/O 而阻塞的其他请求后面排队。
4. **资源利用率更佳：** CPU 资源可以更有效地用于计算，而不是在等待外部系统时处于空闲状态。

需要记住的是，`async`/`await` 主要对 I/O 密集型操作有益。对于像实际模型推理这样的 CPU 密集型任务，单独使用异步定义并不能阻止事件循环的阻塞。正如之前讨论的，`run_in_threadpool` 等技术对于卸载这些密集计算是必要的，通常需要结合用于周边 I/O 操作的异步封装器。通过将异步 I/O 处理与针对 CPU 密集型工作的适当策略结合起来，您可以为您的机器学习模型构建高度响应且可扩展的 FastAPI 应用程序。

## 参考资料

- [asyncio - Asynchronous I/O, event loop, coroutines and tasks](https://docs.python.org/3/library/asyncio.html) — Python Core Developers (2024)
  Publisher: Python Software Foundation
  介绍Python异步编程的基本机制，包括事件循环、协程以及 `async`/`await` 如何管理I/O等待。
- [Concurrency and async/await](https://fastapi.tiangolo.com/async/) — Sebastián Ramírez (2024)
  详细说明FastAPI如何利用Python的异步特性，高效处理I/O密集型操作并提升应用性能，同时讨论CPU密集型任务的处理。
- [Python Concurrency with asyncio](https://www.manning.com/books/python-concurrency-with-asyncio) — Matthew Fowler (2022)
  Publisher: Manning Publications
  提供使用Python `asyncio` 库构建并发应用的全面指南，清晰解释事件循环、协程和I/O管理。
