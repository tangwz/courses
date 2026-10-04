---
course: "advanced-python-programming-ml"
chapter: "concurrency-parallelism-python-ml"
lesson: "multiprocessing-module-parallel"
sourceId: 2345
sourceUrl: "https://apxml.com/zh/courses/advanced-python-programming-ml/chapter-5-concurrency-parallelism-python-ml/multiprocessing-module-parallel"
title: "用于并行执行的 `multiprocessing` 模块"
description: "使用 `multiprocessing` 模块（Pool, Process）在Python中并行化CPU密集型机器学习计算。"
order: 2
plots: []
sourceHash: "e38e559d030d259559364f1b0aa26ec73f3d0a311934964aaa53217c65881f7c"
sourceCorrections: []
---

全局解释器锁（GIL）阻止多个线程在不同CPU核心上同时执行Python字节码。这一限制在标准CPython中处理CPU密集型机器学习 (machine learning)任务时尤为突出。这大大限制了通过线程在计算密集型工作（如机器学习中常见的数值计算）中可获得的性能提升。`multiprocessing` 模块提供了一个很好的替代方案，它通过创建独立的进程来实现，每个进程拥有自己的Python解释器和内存空间，从而绕过GIL，实现多核系统上的真正并行执行。

这种方法特别适合计算量远大于通信开销的任务，例如复杂的数据转换、超参数 (parameter) (hyperparameter)搜索期间的并行模型训练，或者同时评估多种模型变体。

### 使用 `Process` 创建和管理独立进程

`multiprocessing` 中的基本构建块是 `Process` 类。您可以通过实例化 `Process` 来创建一个新进程，通过 `target` 参数 (parameter)指定要在新进程中执行的目标函数，并使用 `args`（元组）或 `kwargs`（字典）参数传递任何必需的参数。

创建 `Process` 对象后，您可以通过调用 `start()` 方法启动其执行。这会生成一个新的子进程，该子进程开始运行指定的目标函数。为确保主程序在继续执行前等待子进程完成工作，您需要在 `Process` 对象上调用 `join()` 方法。这对于同步结果或确保任务在进入后续步骤前完成很重要。

考虑一个简化情况，我们将计算密集型的特征工程步骤应用于数据的不同部分：

```python
import multiprocessing
import time
import os

def compute_intensive_feature(data_chunk):
    """模拟对数据块进行CPU密集型计算。"""
    pid = os.getpid()
    print(f"进程 {pid}: 开始对大小为 {len(data_chunk)} 的数据块进行计算")
    # 模拟工作
    result = sum(x * x for x in data_chunk) 
    time.sleep(1) 
    print(f"进程 {pid}: 完成计算。")
    return result

if __name__ == "__main__": # multiprocessing 的重要保护
    # 示例数据，分成块
    data = list(range(100000)) 
    chunk1 = data[:50000]
    chunk2 = data[50000:]

    print("主进程: 启动子进程...")
    start_time = time.time()

    # 创建 Process 对象
    p1 = multiprocessing.Process(target=compute_intensive_feature, args=(chunk1,))
    p2 = multiprocessing.Process(target=compute_intensive_feature, args=(chunk2,))

    # 启动进程
    p1.start()
    p2.start()

    # 等待进程完成
    p1.join()
    p2.join()

    end_time = time.time()
    print(f"主进程: 所有子进程在 {end_time - start_time:.2f} 秒内完成。")
    # 注意: 像这样直接获取结果需要进程间通信（稍后介绍）。
    # 此示例侧重于并行执行流程。
```

> **重要的:** 在使用 `spawn` 或 `forkserver` 启动方法（如Windows或有时macOS）的平台上，您必须将脚本中创建进程的主体部分置于 `if __name__ == "__main__":` 块中。这可以防止子进程重新导入并重新执行进程创建代码，从而导致无限递归。

### 使用 `Pool` 有效管理工作进程

尽管创建独立的 `Process` 对象提供了细粒度控制，但手动管理大量进程会变得繁琐。`multiprocessing.Pool` 类提供了一种便捷的方式来管理一组工作进程。它会自动处理任务到可用工作进程的分配。

常见的模式是使用 `Pool.map()` 方法。它将给定函数应用于可迭代对象（如列表）中的每个项，将工作分配到池中的工作进程。它收集结果并以列表形式返回它们，同时保持与输入可迭代对象对应的顺序。

让我们修改前面的示例来使用 `Pool`：

```python
import multiprocessing
import time
import os
import math

def compute_intensive_feature_pool(x):
    """模拟对单个项进行CPU密集型计算。"""
    # 模拟单个项的工作 - 更适合 map
    result = math.sqrt(x) * math.sin(x) * math.cos(x) 
    # 添加一个小的延迟来表示计算
    for _ in range(10000): 
        pass 
    return result

if __name__ == "__main__":
    data = list(range(200000)) # 用于 Pool 演示的更大数据集

    print("主进程: 启动 Pool 处理...")
    start_time_serial = time.time()

    # 串行执行以进行比较
    # results_serial = [compute_intensive_feature_pool(item) for item in data] 
    # 取消注释上面一行以运行串行比较，但这会很慢。
    # print(f"主进程: 串行执行完成 (示例 - 已注释)。")

    start_time_parallel = time.time()

    # 确定进程数（通常基于CPU核心数）
    num_processes = multiprocessing.cpu_count() 
    print(f"主进程: 使用 {num_processes} 个进程的池。")

    # 创建一个 Pool 上下文管理器（自动处理关闭/连接）
    with multiprocessing.Pool(processes=num_processes) as pool:
        # 并行应用函数
        results_parallel = pool.map(compute_intensive_feature_pool, data)

    end_time_parallel = time.time()

    print(f"主进程: 并行执行在 {end_time_parallel - start_time_parallel:.2f} 秒内完成。")
    # print(f"结果长度: {len(results_parallel)}") # 验证输出长度
```

`Pool` 会自动将 `data` 可迭代对象分成块，并将每个块分配给一个工作进程。`map` 会阻塞，直到所有结果都被计算并返回。

其他有用的 `Pool` 方法包括：

- `apply_async()`: 为一组参数 (parameter)异步执行一个函数。它会立即返回一个 `AsyncResult` 对象，允许主程序在函数于工作进程中运行时执行其他任务。您可以使用 `get()` 稍后获取结果。
- `imap()` / `imap_unordered()`: `map` 的惰性版本。它们返回迭代器，一旦结果可用就生成结果，这对于非常大的数据集来说可以节省内存。`imap` 保持顺序，而 `imap_unordered` 则按完成的顺序返回结果。

向 `Pool` 提交所有任务后，您通常应调用 `pool.close()` 来表明不再提交任务，随后调用 `pool.join()` 来等待所有已提交的任务完成。将 `Pool` 作为上下文 (context)管理器使用（如示例中 `with multiprocessing.Pool(...) as pool:` 所示）会自动处理 `close()` 和 `join()`，这是推荐的做法。

### 进程间数据传输

由于进程在独立的内存空间中运行，任何传递给子进程（作为参数 (parameter)）或从子进程返回的数据都需要进行传输。`multiprocessing` 主要通过序列化来处理此问题，默认使用 `pickle` 模块。传递给 `target` 函数或通过 `Pool` 方法提交的参数会在主进程中被封装，并在工作进程中解封。返回值会在工作进程中被封装，并在主进程中（或调用 `AsyncResult` 上 `get()` 的进程中）解封。

> 封装会带来开销，特别是对于大型对象。如果管理不当，在进程间传输大型数据集（如NumPy数组或Pandas DataFrame）可能会成为瓶颈。此外，并非所有Python对象都可以被封装（例如，生成器、lambda函数、嵌套函数、一些复杂的自定义对象）。在设计用于并行执行的函数时，此限制很重要。针对需要更复杂数据交换的情况，更高级的进程间通信（IPC）技术，例如 `Queue` 或共享内存，将在后面进行介绍。

### `multiprocessing` 在机器学习 (machine learning)中的常见应用

`multiprocessing` 模块在加速机器学习工作流程中有很多应用：

1. **数据预处理：** 将复杂的特征提取、缩放或转换函数应用于大型数据集的独立数据块。
2. **超参数 (parameter) (hyperparameter)调整（网格搜索/随机搜索）：** 并行训练和评估具有不同超参数组合的模型。每个进程处理一个或多个组合。
3. **交叉验证：** 同时执行交叉验证不同折叠的训练和评估。
4. **集成方法：** 在合并预测之前，并行训练独立的（例如在随机森林或 bagging 集成中）基本模型。
5. **批量推断：** 通过将大批量数据分配到多个进程来运行预测。

### 性能考量

尽管 `multiprocessing` 实现了真正的并行，但请记住以下几点：

- **进程创建开销：** 启动新进程比启动线程需要更多的资源（时间和内存）。使用 `Pool` 有助于将此成本分摊到多个任务中。
- **序列化开销：** 封装和解封数据所需的时间可能相当可观，特别是对于大型或复杂对象。如果这成为瓶颈，请尽量减少数据传输或使用更有效的IPC机制。
- **进程数量：** 创建过多进程（远超可用CPU核心数）可能导致收益递减，甚至由于上下文 (context)切换开销和资源争用而导致性能下降。使用 `multiprocessing.cpu_count()` 是一个常见的起始点，但最佳性能可能需要根据具体任务和硬件进行调整。
- **任务粒度：** 任务应足够大（计算量上），以抵消进程创建和数据序列化的开销。寿命很短的任务串行运行可能更快。

`multiprocessing` 模块是使用多核处理器来加速Python机器学习 (machine learning)管线中CPU密集型计算的重要工具。通过了解如何使用 `Process` 创建进程并使用 `Pool` 有效管理它们，您可以显著减少耗时任务的执行时间，从而加快实验和模型开发周期。在设计并行解决方案时，请记住考虑与进程创建和数据序列化相关的权衡。

## 参考资料

- [multiprocessing - Process-based parallelism](https://docs.python.org/3/library/multiprocessing.html) — Python Software Foundation (2024)
  有关`multiprocessing`模块的官方文档，详细介绍了其类、函数和一般用法。
- [High Performance Python: Practical Performant Programming for Humans with Computers](https://www.oreilly.com/library/view/high-performance-python/9781492055020/) — Micha Gorelick, Ian Ozsvald (2020)
  Publisher: O'Reilly Media
  一本关于优化Python代码的资源，涵盖并行处理策略，适用于数据密集型和科学应用。
- [Effective Python: 90 Specific Ways to Write Better Python](https://www.informit.com/store/effective-python-90-specific-ways-to-write-better-9780134853987) — Brett Slatkin (2019)
  Publisher: Addison-Wesley Professional
  提供编写健壮且高效Python代码的指导，其中一节专门介绍使用`multiprocessing`模块实现并发和并行。
