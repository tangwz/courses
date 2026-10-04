---
course: "advanced-python-programming-ml"
chapter: "concurrency-parallelism-python-ml"
lesson: "concurrent-futures-concurrency"
sourceId: 2351
sourceUrl: "https://apxml.com/zh/courses/advanced-python-programming-ml/chapter-5-concurrency-parallelism-python-ml/concurrent-futures-concurrency"
title: "使用 `concurrent.futures` 实现高级并发"
description: "使用 `concurrent.futures` 模块（ThreadPoolExecutor, ProcessPoolExecutor）简化并发执行。"
order: 4
plots: []
sourceHash: "9cad83c23ed8cf1adaf3a435dd76db5239a112e1269b5e6f50cbbdf68ed6f419"
sourceCorrections: []
---

在 Python 中管理并发和并行，尤其是针对机器学习 (machine learning)工作负载，当直接处理线程或进程时可能会面临挑战。这通常涉及复杂的通信处理和细致的资源清理。Python 3.2 中引入的 `concurrent.futures` 模块提供了一个更高级别的接口，用于异步执行可调用对象，从而实现线程池和进程池的更高效管理。

它抽象化了大量与创建、管理和加入线程或进程相关的样板代码。其主要思想围绕着 `Executor` 抽象基类，该类提供了异步执行调用的方法。它提供了两种具体实现：用于基于线程的并发的 `ThreadPoolExecutor` 和用于基于进程的并行的 `ProcessPoolExecutor`。

### Executor 接口和 Future 对象

`concurrent.futures` 模块的核心是 `Executor` 子类和 `Future` 对象。你实例化一个执行器（可以是 `ThreadPoolExecutor` 或 `ProcessPoolExecutor`），然后向其提交任务。

1. **`Executor.submit(fn, *args, **kwargs)`**：此方法安排可调用对象 `fn` 以 `fn(*args, **kwargs)` 的形式执行，并返回一个表示可调用对象执行的 `Future` 对象。
2. **`Future` 对象**：此对象表示一个可能已完成也可能尚未完成的计算。它充当结果的占位符。你可以查询其状态、添加回调，并获取结果或执行期间引发的任何异常。

> 使用 `concurrent.futures` 的基本流程：向执行器提交任务，立即收到一个 Future 对象，随后从 Future 中获取结果。

`Future` 对象的主要方法包括：

- `done()`: 如果调用已成功取消或完成运行，则返回 `True`。
- `result(timeout=None)`: 返回调用返回的值。如果调用尚未完成，此方法将等待最多 `timeout` 秒。如果调用引发了异常，此方法会引发相同的异常。如果 future 被取消，它会引发 `CancelledError`。
- `exception(timeout=None)`: 返回调用引发的异常对象。如果调用完成时未引发异常，则返回 `None`。如果尚未完成，它会像 `result()` 一样等待。
- `add_done_callback(fn)`: 附加一个可调用对象 `fn`，当 future 被取消或完成运行后，将以 future 对象作为唯一参数 (parameter)调用该 `fn`。

### 用于 I/O 密集型任务的 ThreadPoolExecutor

如前所述，CPython 中的线程适合处理 I/O 密集型任务，因为在阻塞 I/O 操作（如网络请求、磁盘读写）期间，GIL 会被解除。`ThreadPoolExecutor` 管理一个线程池，对提交的任务重复利用这些线程。这避免了为每个任务创建新线程的开销。

考虑从不同 URL 获取多个数据集或配置文件。这主要是一个 I/O 密集型操作。

```python
import concurrent.futures
import requests
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(message)s')

URLS = [
    'https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/iris.csv',
    'https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/diabetes_data.csv.gz',
    'https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/wine_data.csv'
]

def download_data(url):
    """从 URL 下载数据并返回其大小。"""
    try:
        logging.info(f"Starting download: {url.split('/')[-1]}")
        response = requests.get(url, timeout=10)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        logging.info(f"Finished download: {url.split('/')[-1]}, Size: {len(response.content)}")
        return url, len(response.content)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading {url}: {e}")
        return url, None

# 使用上下文管理器进行自动关闭
start_time = time.time()
results = {}

# max_workers 决定池中线程的数量
with concurrent.futures.ThreadPoolExecutor(max_workers=3, thread_name_prefix='Downloader') as executor:
    # 提交任务并存储 Future 对象
    future_to_url = {executor.submit(download_data, url): url for url in URLS}

    # 结果一完成就处理
    for future in concurrent.futures.as_completed(future_to_url):
        url = future_to_url[future]
        try:
            url_res, data_size = future.result()
            if data_size is not None:
                results[url_res] = data_size
                logging.info(f"Result obtained for {url_res.split('/')[-1]}")
            else:
                logging.warning(f"Download failed for {url_res.split('/')[-1]}")
        except Exception as exc:
            logging.error(f'{url} generated an exception: {exc}')

end_time = time.time()

print(f"\nDownloaded data sizes: {results}")
print(f"Total time using ThreadPoolExecutor: {end_time - start_time:.2f} seconds")
```

在此示例中，`ThreadPoolExecutor` 并发管理多个下载任务。由于 `requests.get` 涉及等待网络响应（I/O），线程可以解除 GIL，允许其他线程运行，从而实现比顺序下载更快的整体完成速度。我们使用 `concurrent.futures.as_completed` 在结果可用时立即处理它们。

### 用于 CPU 密集型任务的 ProcessPoolExecutor

对于 CPython 中的 CPU 密集型任务，当 GIL 阻止 Python 字节码在使用线程时在多个 CPU 核上真正并行执行时，`ProcessPoolExecutor` 是更优的选择。它管理一个独立的进程池。每个进程都有自己的 Python 解释器和内存空间，从而绕过了 GIL 的限制。

提交给 `ProcessPoolExecutor` 执行的任务的数据以及从这些任务返回的结果必须是可 pickle 化的，因为进程间通信通常涉及序列化。

我们来修改一个 CPU 密集型特征计算任务。

```python
import concurrent.futures
import time
import math
import os

# 一个模拟的 CPU 密集型函数
def compute_heavy_feature(item_id, data_value):
    """模拟一个复杂的计算。"""
    pid = os.getpid()
    print(f"[PID {pid}] Processing item {item_id}...")
    result = 0
    for i in range(int(data_value * 1e6)): # Simulate work
        result += math.sqrt(i) * math.sin(i)
    print(f"[PID {pid}] Finished item {item_id}.")
    return item_id, result

data_to_process = {
    1: 5.1,
    2: 6.3,
    3: 4.8,
    4: 7.2,
    5: 5.5,
    6: 6.8
}

# 使用上下文管理器进行自动关闭
start_time = time.time()
results = {}

# 默认情况下，使用 os.cpu_count() 个工作进程
# 确保此代码块仅在主脚本中运行
if __name__ == "__main__": # 对于某些操作系统的多进程来说很重要
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        # 提交任务：函数、参数1、参数2、...
        future_to_item = {executor.submit(compute_heavy_feature, item_id, value): item_id
                          for item_id, value in data_to_process.items()}

        # 获取结果
        for future in concurrent.futures.as_completed(future_to_item):
            item_id = future_to_item[future]
            try:
                item_res, computed_value = future.result()
                results[item_res] = computed_value
                print(f"Main: Result received for item {item_res}")
            except Exception as exc:
                print(f'Main: Item {item_id} generated an exception: {exc}')

    end_time = time.time()

    print(f"\nComputed feature results (first 10 chars): { {k: str(v)[:10] + '...' for k, v in results.items()} }")
    print(f"Total time using ProcessPoolExecutor: {end_time - start_time:.2f} seconds")
```

**注意：** 在脚本中使用 `multiprocessing`（`ProcessPoolExecutor` 依赖于它）时，特别是 Windows 上，`if __name__ == "__main__":` 守护代码块很重要，以防止与进程创建相关的问题。

在此，`ProcessPoolExecutor` 创建多个独立的 Python 进程。每个 `compute_heavy_feature` 调用都在一个单独的进程中运行，允许多个计算在不同的 CPU 核上并行进行，从而显著加快了 CPU 密集型工作负载的总计算时间。

### 使用 Executor.map

对于想要将同一函数应用于可迭代对象中多个项的更简单情况，`Executor.map` 提供了一个类似于内置 `map` 函数的接口。它将函数应用于每个项，并返回一个迭代器，该迭代器按照*原始可迭代对象的顺序*生成结果。

```python
import concurrent.futures
import time
import os

def square_number(x):
    pid = os.getpid()
    # print(f"[PID {pid}] 计算 {x} 的平方")
    time.sleep(0.1) # 模拟一些工作
    return x * x

numbers = range(10)

if __name__ == "__main__":
    start_time = time.time()
    # 对于潜在的 CPU 密集型工作，使用 ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        # map 返回一个按顺序生成结果的迭代器
        results_iterator = executor.map(square_number, numbers)

        # 消费迭代器以获取结果
        results = list(results_iterator)

    end_time = time.time()

    print(f"Original numbers: {list(numbers)}")
    print(f"Squared numbers: {results}")
    print(f"Total time using map: {end_time - start_time:.2f} seconds")
```

`map` 很方便，但灵活性不如 `submit`。它按顺序等待结果，处理异常需要遍历结果迭代器，并可能在迭代周围使用 `try...except` 代码块。`submit` 结合 `as_completed` 允许在任务完成时立即处理结果，如果任务持续时间不同，这会更高效。

### 上下文 (context)管理器用法

`ThreadPoolExecutor` 和 `ProcessPoolExecutor` 都实现了上下文管理器协议。使用 `with` 语句可确保在退出代码块时执行器被正确关闭（调用 `executor.shutdown(wait=True)`），即使发生错误也是如此。这是使用执行器的推荐方式，因为它保证资源（线程或进程）得到正确清理。

### 选择合适的执行器

`ThreadPoolExecutor` 和 `ProcessPoolExecutor` 之间的选择主要取决于你的工作负载性质以及 GIL 的限制：

- **`ThreadPoolExecutor`**：最适合 I/O 密集型任务（网络请求、文件操作），因为线程会花费大量时间等待，从而允许 GIL 被解除。它的开销比进程创建要低。
- **`ProcessPoolExecutor`**：对于 CPU 密集型任务（纯 Python 中的大量数值计算、复杂数据转换）是必要的，当你需要跨多个 CPU 核的真正并行时。它克服了 GIL 的限制，但由于进程创建和进程间通信（pickling）会带来更高的开销。

`concurrent.futures` 模块大大简化了 Python 中并发操作的管理。通过 `ThreadPoolExecutor` 和 `ProcessPoolExecutor` 提供统一的接口，它使开发者能够根据任务特点轻松应用线程或多进程，从而加速涉及 I/O 或密集计算的机器学习 (machine learning)工作流程。在选择合适的执行器时，请记住考虑任务类型（I/O 密集型 vs. CPU 密集型）以及每种方法相关的开销。

## 参考资料

- [concurrent.futures - Launching parallel tasks](https://docs.python.org/3/library/concurrent.futures.html) — Python Software Foundation (2024)
  Publisher: Python Software Foundation
  `concurrent.futures`模块的官方权威文档，详细介绍了其API、类和使用方法。
- [Fluent Python, 2nd Edition](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056345/) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media; Pages: Chapter 18-20 (Concurrency)
  一本关于现代Python的优秀综合指南，包含专门的并发部分，特别是关于`concurrent.futures`的讲解。
- [Python Concurrency with asyncio](https://www.manning.com/books/python-concurrency-with-asyncio) — Matthew Fowler (2022)
  Publisher: Manning Publications; Pages: Part 1 (Foundation for Concurrency)
  一本专注于Python并发功能的书籍，深入探讨了`concurrent.futures`及其在线程和进程管理中的作用。
