---
course: "fastapi-ml-deployment"
chapter: "intro-fastapi-api-fundamentals"
lesson: "async-programming-python"
sourceId: 5235
sourceUrl: "https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-1-intro-fastapi-api-fundamentals/async-programming-python"
title: "Python中的异步编程思路"
description: "介绍Python中async/await语法和事件循环思路，用于非阻塞操作。"
order: 4
plots: []
sourceHash: "97ca48238d8da3e9707ef651a2bd47f416c6eaf5b816c2de8d44ddd30912009a"
sourceCorrections: []
---

许多标准Python程序执行时是*同步*的。这意味着当程序遇到一个操作时，它会等待该操作完成，然后才继续执行下一行代码。想象一位厨师在处理下一个订单前，一丝不苟地从头到尾做好一道菜。虽然这种方式易于理解，但效率不高，特别是对于那些需要花费大量时间等待外部事件（如网络请求或磁盘读取）的应用程序。

Web应用程序，包括服务机器学习 (machine learning)模型的API，经常执行这类*I/O密集型*操作。一个请求可能需要从数据库获取用户数据，调用另一个微服务，或者从存储中加载模型特征。在同步情境下，当服务器等待数据库响应时，它无法处理任何其他传入请求，这会导致在负载下性能不佳和资源利用率低。

这就是*异步编程*发挥作用的地方。异步应用程序不再闲置等待，而是在一个任务等待I/O操作完成时，切换去处理其他任务。当操作完成后，应用程序可以从中断处恢复原任务的执行。想想我们的厨师，他开始烧水煮意面，然后趁水加热的时间去切蔬菜，只有当水烧开时才回到意面。这允许更高的并发性，意味着能够用更少的资源同时处理许多操作。

Python通过`async`和`await`语法提供了对异步编程的原生支持，这些语法是基于*协程*的思想构建的。

### 使用 `async def` 定义协程

您可以使用 `async def` 而不是简单的 `def` 来定义一个异步函数，即协程：

```python
async def get_data_from_network():
    # 获取数据的代码...
    print("正在获取数据...")
    # 模拟等待网络响应
    await asyncio.sleep(1) # 核心所在
    print("数据已收到！")
    return {"data": "一些结果"}
```

使用 `async def` 定义的函数在被调用时不会像普通函数那样运行。相反，调用它会返回一个*协程对象*。这个对象代表了函数中定义的工作，但不会立即执行它。

### 使用 `await` 暂停执行

`await` 关键字在 `async def` 函数*内部*使用。它告诉Python被调用的操作（该操作本身必须是*可等待的*，通常是另一个协程或为异步操作设计的对象）可能需要一些时间。当程序遇到 `await` 时，它可以暂停当前协程的执行，从而让应用程序执行其他任务。一旦被等待的操作完成，暂停的协程将从该点恢复执行。

重要的是，`await` 只能在 `async def` 定义的函数中使用。

### 事件循环

应用程序如何知道何时恢复暂停的协程？这由*事件循环*管理。事件循环是任何异步应用程序的核心。它跟踪所有正在运行和暂停的任务。当一个任务使用 `await` 暂停以等待I/O操作时，它会将控制权交还给事件循环。事件循环随后可以运行其他就绪任务。当I/O操作完成后，事件循环会安排原任务恢复执行。Python内置的`asyncio`库提供了事件循环和管理异步任务的工具。

> 涉及事件循环的异步操作基本控制流程。

FastAPI直接构建在这些异步能力之上（具体来说，它使用ASGI标准和Starlette工具包，后者依赖于`asyncio`）。这允许您在API路由处理函数中，使用`async`和`await`编写标准Python代码。当FastAPI收到请求时，它会在事件循环中运行相应的路由处理函数。如果您的处理函数对一个I/O密集型操作执行了`await`（例如异步地与数据库交互或调用外部API），FastAPI会自动处理暂停和恢复，从而使服务器能够高效处理许多并发请求。这种对异步操作的内置支持是FastAPI高性能的一个主要原因，尤其对可能需要在推理 (inference)前获取数据或预处理输入的ML API非常有利。

虽然许多ML模型的核心推理步骤是CPU密集型的（并且在异步代码中需要特殊处理，这将在第5章中讨论），但周边任务，如数据验证、日志记录、特征获取或结果保存，通常涉及I/O，并且从异步模型中获益良多。了解这些基本原理有助于您编写高效、响应迅速的API端点。我们将在后续章节中开始构建简单的同步和异步端点。

## 参考资料

- [asyncio - Asynchronous I/O](https://docs.python.org/3/library/asyncio.html) — Python Software Foundation (2025)
  Publisher: Python Software Foundation
  Python内置异步I/O框架的官方文档，详细说明了`async`/`await`语法、协程和事件循环。
- [Async, Await, and Concurrency](https://fastapi.tiangolo.com/async/) — Sebastián Ramírez and FastAPI Contributors (2024)
  关于FastAPI如何利用Python异步特性高效处理并发请求的文档。
- [Fluent Python: Clear, Concise, and Effective Programming](https://www.oreilly.com/library/view/fluent-python-2nd-edition/9781492056300/) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media; Pages: 629-693 (Chapter 17: Concurrency with Futures and Asyncio)
  一本权威书籍，其中包含关于并发的章节，详细解释了Python的`asyncio`和`async`/`await`。
