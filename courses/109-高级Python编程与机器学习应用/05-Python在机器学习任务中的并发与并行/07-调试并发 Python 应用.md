# 调试并发 Python 应用

来源：[原文](https://apxml.com/zh/courses/advanced-python-programming-ml/chapter-5-concurrency-parallelism-python-ml/debugging-concurrent-python)

[返回章节目录](README.md) · [返回课程目录](../README.md)

编写并发程序会带来在顺序执行代码中通常不存在的复杂性。竞态条件和死锁等问题源于线程和进程调度的不确定性，这使错误难以复现和诊断。对于机器学习 (machine learning)，并行化数据加载、预处理或模型训练的部分环节是常见的做法。在使用线程、多进程和异步IO等技术处理这些任务时，有效的调试策略对于构建可靠且高性能的系统至关重要。

调试并发应用需要改变思考方式。你不仅仅是寻找单一执行流中的逻辑错误，还要考虑多个流之间的有害关联。

### 并发机器学习 (machine learning)代码中的常见问题

认识并发程序特有的错误类型是识别和纠正这些错误的第一步。

#### 竞态条件

当计算结果取决于多个线程或进程访问共享资源（如内存、文件或模型参数 (parameter)）的不可预测的时间或交错时，就会出现竞态条件。其结果通常是数据损坏或应用状态不正确。

考虑一个简化例子，多个进程更新一个共享计数器，用于追踪数据管道中已处理的项目：

```python
# 警告：此代码包含竞态条件！
import multiprocessing

shared_counter = multiprocessing.Value('i', 0) # 共享整数

def worker_process(counter):
    # 模拟处理项目
    local_count = 0
    for _ in range(1000):
        local_count += 1

    # 有问题的数据更新：读取、增加、写入不是原子操作
    # 进程 A 读取计数器（例如，值为 500）
    # 进程 B 读取计数器（例如，值为 500）
    # 进程 A 计算 500 + 1000 = 1500
    # 进程 B 计算 500 + 1000 = 1500
    # 进程 A 将 1500 写入计数器
    # 进程 B 将 1500 写入计数器（覆盖 A 的值）
    # 预期最终值：2000，实际最终值：1500
    current_value = counter.value
    counter.value = current_value + local_count

if __name__ == "__main__":
    processes = []
    for _ in range(2): # 两个进程
        p = multiprocessing.Process(target=worker_process, args=(shared_counter,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print(f"最终计数器值（可能不正确）：{shared_counter.value}")
    # 预期：如果每个进程增加 1000，则为 2000
```

对 `shared_counter.value` 进行的非原子性读-修改-写操作导致了竞态条件。解决方法是使用锁，以确保每次只有一个进程修改计数器，正如在同步原语部分讨论的那样。检测竞态条件可能很困难，因为它们可能只在特定负载条件或时间点下出现。

#### 死锁

当两个或更多线程或进程被无限期阻塞，彼此都在等待对方持有的资源时，就会发生死锁。设想机器学习管道中的两个进程：进程 A 锁定资源 1（例如，一个数据文件）并需要资源 2（例如，一个模型参数服务器），而进程 B 锁定资源 2 并需要资源 1。两者都无法继续。

> 一个涉及两个进程和两个资源的死锁情形的简化示意图。每个进程持有一个资源，同时等待另一个进程持有的资源。

常见原因是在不同线程或进程中以不一致的顺序获取多个锁。预防死锁的主要策略是始终按照相同的预定义全局顺序获取锁。

#### 饥饿

饥饿发生在可运行的进程或线程被长期拒绝获取其前进所需的资源（CPU 时间、锁）时，通常是因为其他进程（例如，优先级更高的或在锁竞争中“更幸运”的进程）独占了这些资源。尽管饥饿不如竞态条件或死锁常见，但它可能导致您的机器学习管道中的某些部分（如特定的数据转换任务）永远无法完成或进展非常缓慢。

### 调试策略和工具

标准调试技术对于并发代码通常不足，这是由于其不确定性。以下是更有效的方法：

#### 1. 日志记录

日志记录可以说是理解并发执行流程最不可或缺的工具。

- **识别线程/进程：** 在每条日志消息中包含线程 ID (`threading.get_ident()`) 或进程 ID (`os.getpid()`)。
- **精确时间戳：** 使用高精度时间戳以理解不同执行上下文 (context)中的事件序列。
- **记录资源获取/释放：** 在获取和释放锁或其他共享资源之前和之后记录消息。这有助于追踪死锁和资源争用。
- **线程/进程安全日志：** 使用 `multiprocessing` 或 `threading` 时，请确保您的日志配置是安全的。标准日志有时可能出问题。使用 `logging.handlers.QueueHandler` 和独立的日志进程或线程是一种可靠的模式，用于从多个并发工作者收集日志而不会损坏。

```python
import logging
import logging.handlers
import multiprocessing
import threading
import time
import os
import queue # Use standard queue for the handler

def logger_thread(log_queue):
    # 配置根日志器或特定日志器
    logger = logging.getLogger('ML_App')
    logger.setLevel(logging.DEBUG)
    # 根据需要添加处理程序（例如，控制台、文件）
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(processName)s/%(threadName)s] - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    while True:
        try:
            record = log_queue.get()
            if record is None: # 停止信号值
                break
            logger.handle(record)
        except Exception:
            import sys, traceback
            print('日志线程错误:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

def worker_task(log_queue):
    # 为此工作者配置处理程序
    queue_handler = logging.handlers.QueueHandler(log_queue)
    worker_logger = logging.getLogger('ML_App')
    worker_logger.addHandler(queue_handler)
    worker_logger.setLevel(logging.DEBUG)

    pid = os.getpid()
    tid = threading.get_ident()
    worker_logger.info(f"工作者已启动 (PID: {pid}, TID: {tid})")
    time.sleep(0.5)
    worker_logger.debug("正在执行一些工作...")
    time.sleep(0.5)
    worker_logger.info("工作者已完成")

if __name__ == "__main__":
    log_queue = multiprocessing.Queue(-1) # 使用多进程队列

    # 启动日志进程/线程
    # 这里为简化使用线程，但进程通常更有效
    listener = threading.Thread(target=logger_thread, args=(log_queue,), daemon=True)
    listener.start()

    # 使用多进程的例子
    processes = []
    for i in range(2):
        p = multiprocessing.Process(target=worker_task, args=(log_queue,), name=f"WorkerProcess-{i}")
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # 通知日志线程停止
    log_queue.put(None)
    listener.join() # 等待日志器完成处理队列
```

#### 2. 代码设计和同步检查

通常，并发错误源于有缺陷的设计。

- **最小化共享状态：** 进程或线程之间共享的状态越少，出现竞态条件的机会就越少。尽可能通过队列或管道传递数据。
- **一致的锁获取顺序：** 如果需要获取多个锁，请始终在代码中的所有地方按照相同的全局顺序获取它们，以预防死锁。清楚地记录这个顺序。
- **使用上下文管理器：** 使用 `with lock:` 语句以确保锁始终被释放，即使发生错误。

#### 3. 使用超时

阻塞操作（获取锁、从队列中获取项目、等待事件）在死锁或饥饿情况下可能无限期挂起。对这些操作应用合理的超时可以将挂起变为可检测的错误。

```python
import threading
import time

lock = threading.Lock()

def worker_waits_too_long():
    print("工作者正在尝试获取锁...")
    acquired = lock.acquire(timeout=2) # 最多等待 2 秒
    if acquired:
        print("工作者已获取锁！")
        try:
            # 使用已锁定资源进行工作
            time.sleep(1)
        finally:
            lock.release()
            print("工作者已释放锁。")
    else:
        print("工作者等待锁超时！")

# 主线程获取锁并持有它
print("主线程正在获取锁...")
lock.acquire()
print("主线程持有锁。")

worker = threading.Thread(target=worker_waits_too_long)
worker.start()

time.sleep(3) # 持有锁足够长时间以使工作者超时
print("主线程正在释放锁。")
lock.release()

worker.join()
```

#### 4. 简化和隔离

并发错误通常依赖于时间。尝试在简化条件下复现问题：

- 减少线程/进程的数量。
- 使用固定、更小的数据集，而不是大型、可变的数据集。
- 在特定代码段引入人工延迟 (`time.sleep()`) 以增加出现问题交错的可能性，但请注意，这会改变时间动态。

#### 5. 调试 `multiprocessing` 特性

- **Pickling 问题：** 使用 `multiprocessing.Queue`、`Pipe` 或作为 `Process` 参数 (parameter)在进程间传递的对象必须是可 pickle 化的。在 pickling/unpickling 期间的错误可能难以理解。使用 `pickle.dumps()` 和 `pickle.loads()` 显式测试有问题的对象的 pickling。
- **子进程调试：** 标准调试器 (`pdb`) 通常只附加到主进程。子进程的打印语句可能会交错或丢失。日志记录（如上所述）是更优选择。对于交互式调试，您可能需要在子进程的目标函数中插入代码，以便有条件地启动调试器（例如，基于环境变量），或使用允许通过 PID 附加到运行中进程的工具。

#### 6. 调试 `asyncio` 特性

- **启用调试模式：** 使用环境变量 `PYTHONASYNCIODEBUG=1` 运行您的应用。这会启用额外的检查，例如记录执行时间过长的协程以及识别未被等待的协程。
- **任务管理：** 追踪已创建的任务。确保任务被等待或正确取消。使用 `asyncio.gather(*tasks, return_exceptions=True)` 以并发运行多个任务并收集结果或异常，而不会因为一个任务失败而提前停止其他任务。
- **阻塞事件循环：** 识别同步阻塞事件循环的代码（例如，长时间运行的 CPU 密集型计算或未使用 `await` 的阻塞 I/O 调用）。使用 `loop.run_in_executor` 来分载阻塞代码。

#### 7. 使用 `faulthandler`

内置的 `faulthandler` 模块可能有用。在应用开始时调用 `faulthandler.enable()`，如果发生致命错误（例如分段错误，这通常由 C 扩展中的问题或导致状态损坏的竞态条件引起），Python 会在所有运行中的线程上转储一个追踪回溯。这即使在错误看起来与您的 Python 代码无关时也能提供线索。

调试并发 Python 代码，尤其是在复杂的机器学习 (machine learning)管道中，需要耐心和系统的方法。优先进行清晰的日志记录、细致的同步设计以及隔离不确定行为的技术。尽管标准调试器存在局限，但组合这些策略会大大提高您诊断和解决并发问题的能力。

## 参考资料

- [\`multiprocessing\` - Process-based parallelism](https://docs.python.org/3/library/multiprocessing.html) — Python Software Foundation (2024)
  Publisher: Python Software Foundation
  解释了Python的基于进程的并行性、共享内存机制和同步原语，这些是理解和调试使用`multiprocessing`的并发应用程序的基础。
- [\`asyncio\` - Asynchronous I/O, event loop, coroutines and tasks](https://docs.python.org/3/library/asyncio.html) — Python Software Foundation (2025)
  Publisher: Python Software Foundation
  提供了Python `asyncio`框架的全面文档，包括其架构、调试模式以及管理协程和事件循环的最佳实践，对于诊断异步代码中的问题至关重要。
- [Operating System Concepts](https://www.wiley.com/en-us/Operating+System+Concepts%2C+10th+Edition-p-9781119320913) — Abraham Silberschatz, Peter B. Galvin, Greg Gagne (2018)
  Publisher: Wiley
  一本基础教材，详细介绍了并发的核心概念，包括竞态条件、死锁和饥饿，以及各种同步机制和策略，以防止并发系统中的这些常见问题。
- [Fluent Python, 2nd Edition](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056381/) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media
  为Python的并发模型（线程、多进程、异步I/O）提供了实用指导，以及编写健壮并发代码的最佳实践，包括管理共享状态和避免常见问题。
- [\`logging\` - Logging facility for Python](https://docs.python.org/3/library/logging.html) — Python Software Foundation (2024)
  涵盖了Python的标准日志系统，特别关注了`QueueHandler`等高级功能，这对于在并发应用程序中实现健壮、线程和进程安全的日志记录至关重要。

---

[上一节](06-%E5%90%8C%E6%AD%A5%E5%8E%9F%E8%AF%AD%EF%BC%88%E9%94%81%E3%80%81%E4%BF%A1%E5%8F%B7%E9%87%8F%E3%80%81%E4%BA%8B%E4%BB%B6%EF%BC%89.md) · [下一节](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%95%B0%E6%8D%AE%E9%A2%84%E5%A4%84%E7%90%86%E5%B9%B6%E8%A1%8C%E5%8C%96.md)
