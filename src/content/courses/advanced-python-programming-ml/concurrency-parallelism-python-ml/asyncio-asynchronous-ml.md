---
course: "advanced-python-programming-ml"
chapter: "concurrency-parallelism-python-ml"
lesson: "asyncio-asynchronous-ml"
sourceId: 2354
sourceUrl: "https://apxml.com/zh/courses/advanced-python-programming-ml/chapter-5-concurrency-parallelism-python-ml/asyncio-asynchronous-ml"
title: "asyncio 异步机器学习操作简介"
description: "使用 Python 的 asyncio 框架处理分布式机器学习或模型服务中常见的 I/O 密集型操作。"
order: 5
plots: []
sourceHash: "60c5b8ae7da38c9c2c947817f633be25c187cb108d36aba93947a8ab994f113e"
sourceCorrections: []
---

多进程为 CPU 密集型计算提供了绕过全局解释器锁的途径，而线程则提供了并发 I/O 的方法。然而，使用线程管理数千个同时进行的网络连接或文件操作可能由于上下文 (context)切换和内存消耗而导致大量开销。对于 I/O 等待时间占主导的场景，Python 提供了一种不同的并发模式：使用 `asyncio` 库进行异步编程。

`asyncio` 采用一种协作式多任务处理方式，其核心是**事件循环**和**协程**。与操作系统抢占式调度的线程不同，`asyncio` 任务在遇到 I/O 操作（或任何标记 (token)有 `await` 的点）时，会主动将控制权交还给事件循环。这使得事件循环可以在初始任务等待其 I/O 操作完成时运行其他任务，所有这些都在单个线程内进行。

### 主要原理：事件循环、`async` 和 `await`

`asyncio` 的中心是**事件循环**。可以将其看作一个管理多个任务的调度器。当一个任务需要等待某些事情（比如网络数据）时，它会通知事件循环，事件循环随后会暂停该任务并运行另一个已准备好的任务。当数据到达时，事件循环会唤醒原始任务，并从其暂停的地方继续执行。

用于配合事件循环的函数使用 `async def` 语法定义，从而创建**协程**。协程就像一种可以暂停和恢复的函数。调用协程函数并不会立即执行它；它会返回一个协程对象。

```python
import asyncio
import time

async def my_coroutine(name, delay):
    print(f"Coroutine {name}: Starting")
    await asyncio.sleep(delay) # 在此处暂停，让其他任务运行
    print(f"Coroutine {name}: Finished after {delay}s")
    return f"Result from {name}"

# 调用它会返回一个协程对象，但尚未运行
coro_obj = my_coroutine("A", 1)
print(type(coro_obj))
# 输出: <class 'coroutine'>

# 要运行它，你需要一个事件循环
# result = asyncio.run(my_coroutine("A", 1))
# print(result)
```

`await` 关键字用于 `async` 函数内部，指示函数可以暂停的点。你可以 `await` 其他协程或被称为“可等待对象”的特殊对象（如 `asyncio.sleep()` 的结果或来自 `asyncio` 兼容库的网络 I/O 操作）。当一个任务正在 `await` 时，事件循环可以自由执行其他任务。这种协作式让出避免了单个缓慢操作阻塞整个线程。

### 何时在机器学习 (machine learning)中使用 `asyncio`

`asyncio` 在高并发和 I/O 密集型工作负载的应用程序中表现出色。在机器学习场景中，这通常表现为：

1. **模型服务：** 推理 (inference)服务器可能需要处理数百或数千个同时进行的预测请求。每个请求的大部分时间都花费在等待网络 I/O（接收请求、发送响应）上。`asyncio` 允许单个进程高效地处理许多请求，而无需数千个线程的额外负担。FastAPI 和 Sanic 等框架为此目的大量使用 `asyncio`。
2. **分布式系统通信：** 在协调跨多台机器的分布式训练或数据处理时，节点之间频繁通过网络通信。`asyncio` 可以高效地处理这些网络调用，应对潜在的延迟而不会完全堵塞工作进程。Ray 等库在部分通信基础设施中使用了异步方式。
3. **并发数据获取/API：** 机器学习管道通常需要从数据库、Web API 或消息队列等多种来源收集数据。如果数据获取需要等待网络响应，`asyncio` 可以并发运行这些获取操作，与顺序获取相比，可以显著提升数据获取阶段的速度。
4. **实时数据流：** 处理实时数据流通常需要等待来自 Kafka 或 WebSockets 等源的新数据。`asyncio` 很适合处理这些可能数量很多且间歇性的输入源。

### `asyncio` 与线程处理 I/O 的直观对比

假设要从三个缓慢的 Web API 获取数据。线程方式会创建三个线程，每个线程都阻塞直到各自的 API 响应。而 `asyncio` 方式只使用一个线程；当一个协程开始等待某个 API 时，事件循环会切换到另一个协程，发起其请求，以此类推。

> 比较使用线程与 `asyncio` 处理三个并发 I/O 等待。`asyncio` 通过事件循环在单个线程上交错执行操作。

### 一个简单的 `asyncio` 示例：并发任务

我们来模拟并发地从多个“模型端点”获取结果。

```python
import asyncio
import time
import random

async def call_model_endpoint(model_id):
    """模拟调用具有随机延迟的模型端点。"""
    delay = random.uniform(0.5, 2.0)
    print(f"Calling model {model_id}, expecting delay of {delay:.2f}s...")
    await asyncio.sleep(delay) # 模拟网络 I/O 等待
    result = {"model_id": model_id, "prediction": random.random()}
    print(f"Received result from model {model_id}")
    return result

async def main():
    start_time = time.time()
    print("Starting concurrent model calls...")

    # 为每个调用创建任务
    tasks = [
        asyncio.create_task(call_model_endpoint("Alpha")),
        asyncio.create_task(call_model_endpoint("Beta")),
        asyncio.create_task(call_model_endpoint("Gamma"))
    ]

    # 等待所有任务完成
    results = await asyncio.gather(*tasks)

    end_time = time.time()
    print("\n--- 所有模型调用完成 ---")
    print(f"Results: {results}")
    print(f"Total time: {end_time - start_time:.2f}s")

if __name__ == "__main__":
    # 在脚本中，使用 asyncio.run() 运行主异步函数
    asyncio.run(main())

# 示例输出（顺序和具体时间会有所不同）：
# 启动并发模型调用...
# 正在调用模型 Alpha，预计延迟 1.85秒...
# 正在调用模型 Beta，预计延迟 0.65秒...
# 正在调用模型 Gamma，预计延迟 1.20秒...
# 已从模型 Beta 收到结果
# 已从模型 Gamma 收到结果
# 已从模型 Alpha 收到结果
#
# --- 所有模型调用完成 ---
# 结果: [{'model_id': 'Alpha', 'prediction': 0.123...}, {'model_id': 'Beta', 'prediction': 0.987...}, {'model_id': 'Gamma', 'prediction': 0.543...}]
# 总耗时: 1.85秒
```

请注意，总时间接近于最长的单个延迟，而不是所有延迟的总和。这显示了并发执行对 I/O 密集型任务的好处。`asyncio.create_task` 将协程安排在事件循环上运行，而 `asyncio.gather` 则等待所有给定任务/协程完成。

### 注意事项和取舍

尽管 `asyncio` 功能强大，但它也有自己的注意事项：

- **协作特性：** `async` 函数中长时间运行且不 `await` 的 CPU 密集型操作会阻塞整个事件循环，导致其他任务无法运行。CPU 密集型工作理想情况下应在单独的进程中运行（例如，使用 `run_in_executor` 配合 `ProcessPoolExecutor`）。
- **生态系统适配性：** `asyncio` 应用程序中使用的库也必须是异步的，或者经过妥善封装以避免堵塞事件循环。`aiohttp`、`aiopg`、`httpx` 等库就是 `requests` 或 `psycopg2` 的 `asyncio` 适配可选项。
- **调试：** 由于非顺序执行流程，调试异步代码有时可能比同步或线程代码更复杂。

总而言之，`asyncio` 提供了一种高效的机制，用于在单个线程内处理高并发的 I/O 操作。它尤其适用于构建响应迅速的机器学习 (machine learning)模型服务端点、协调分布式系统以及加速受限于网络或磁盘访问的数据管道。它与线程和多进程相辅相成，为现代机器学习应用中常见的 I/O 密集型并发难题提供了一种专用工具。

## 参考资料

- [asyncio - Asynchronous I/O, event loop, coroutines and tasks](https://docs.python.org/3/library/asyncio.html) — Python core developers (2025)
  Publisher: Python Software Foundation
  Python asyncio 库的官方参考资料，提供了详细的 API 规范、概念说明和异步编程示例。
- [Fluent Python: Clear, Concise, and Effective Programming](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056348/) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media; Pages: 1014
  全面介绍 Python 的并发模型，包括 asyncio 的发展历程及其从基础概念到高级模式的应用。
- [FastAPI Documentation](https://fastapi.tiangolo.com/) — Sebastián Ramírez (2024)
  为构建用于机器学习模型服务等应用的高性能异步 Web API 提供实践指导，大量利用了 asyncio。
- [Effective Python: 90 Specific Ways to Write Better Python](https://www.informit.com/store/effective-python-90-specific-ways-to-write-better-9780134853987) — Brett Slatkin (2019)
  Publisher: Addison-Wesley
  包含关于如何以及何时有效使用 asyncio 的具体建议，并比较其与线程和多进程的优缺点。
