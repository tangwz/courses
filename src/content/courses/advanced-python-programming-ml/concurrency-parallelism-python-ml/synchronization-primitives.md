---
course: "advanced-python-programming-ml"
chapter: "concurrency-parallelism-python-ml"
lesson: "synchronization-primitives"
sourceId: 2357
sourceUrl: "https://apxml.com/zh/courses/advanced-python-programming-ml/chapter-5-concurrency-parallelism-python-ml/synchronization-primitives"
title: "同步原语（锁、信号量、事件）"
description: "在 Python 机器学习中正确使用同步原语来管理共享资源和协调并发任务。"
order: 6
plots: []
sourceHash: "77a409fe4bc083b434d2e761b39ac49cd6981c570be321240ea7e96c7187e7bd"
sourceCorrections: []
---

当多个线程或进程并发运行时，尤其是在复杂的机器学习 (machine learning)工作流程中，它们通常需要访问共享资源或协调它们的行动。如果没有适当的协调，你可能会遇到竞态条件，即结果不可预测地取决于操作的时序，从而导致数据损坏、模型状态不正确或结果不一致。Python 的 `threading` 和 `multiprocessing` 模块提供了多种同步原语来管理并发访问并确保可预测的行为。掌握这些工具对构建可靠的并行机器学习系统非常重要。

我们将主要关注三种原语：锁、信号量和事件。

## 锁：确保互斥

锁（通常称为互斥锁，mutual exclusion）是最简单的同步原语。它提供了一种机制，用于确保在任何给定时间只有一个线程或进程能执行一段特定的代码，这段代码被称为临界区。

**原理：** 锁有两种主要状态：已锁定和未锁定。它初始状态为未锁定。`acquire()` 方法尝试获取锁。如果锁未锁定，调用线程/进程将获取它，状态变为已锁定。如果锁已被另一个线程/进程持有，`acquire()` 将阻塞，直到通过 `release()` 方法释放锁。

**在机器学习 (machine learning)中的应用场景：**

- **保护共享状态：** 当多个工作进程更新共享模型参数 (parameter)（例如，在某些并行优化算法中）或聚合指标（如损失或准确度）时。
- **访问共享资源：** 控制对文件、数据库连接或非天生线程安全的共享数据结构（如字典或列表）的独占访问，尤其是在复杂操作中。
- **避免竞态条件：** 确保对用于协调的共享计数器或标志进行原子操作。

**Python 实现：**
`threading.Lock` 和 `multiprocessing.Lock` 都提供相同的接口。使用 `with` 语句是处理锁的推荐方式，因为它在进入代码块时自动获取锁，并在退出时（即使发生错误）释放锁。

```python
import threading
import time
import random

# 示例：模拟共享资源更新（例如，结果聚合）
shared_results = {}
results_lock = threading.Lock()
num_workers = 5
num_items_per_worker = 10

def worker_task(worker_id):
    """模拟工作进程处理项目并更新共享结果。"""
    local_sum = 0
    for i in range(num_items_per_worker):
        # 模拟工作
        time.sleep(random.uniform(0.01, 0.05))
        local_sum += i

    # 临界区：更新共享结果
    print(f"工作进程 {worker_id} 等待获取锁...")
    with results_lock:
        print(f"工作进程 {worker_id} 已获取锁。")
        # 模拟需要原子性的更新逻辑
        current_total = shared_results.get('total_sum', 0)
        time.sleep(random.uniform(0.02, 0.06)) # 模拟持有锁的时间
        shared_results['total_sum'] = current_total + local_sum
        shared_results[f'worker_{worker_id}_sum'] = local_sum
        print(f"工作进程 {worker_id} 已释放锁。新总和：{shared_results['total_sum']}")
    print(f"工作进程 {worker_id} 完成。")

threads = []
for i in range(num_workers):
    thread = threading.Thread(target=worker_task, args=(i,))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

print("\n最终共享结果：")
print(shared_results)

# 计算预期总和
expected_sum = num_workers * sum(range(num_items_per_worker))
print(f"\n预期总和：{expected_sum}")
print(f"实际总和：{shared_results.get('total_sum', '未找到')}")

# 验证单个工作进程的总和是否正确
calculated_total = sum(v for k, v in shared_results.items() if k.startswith('worker_'))
print(f"单个工作进程总和：{calculated_total}")
assert calculated_total == shared_results.get('total_sum'), "总和不匹配！"
```

**可重入锁 (`RLock`)：** 标准 `Lock` 不能被同一线程/进程多次获取，即使该线程/进程已持有该锁。尝试这样做会导致死锁。`threading.RLock` 或 `multiprocessing.RLock`（可重入锁）可以被同一线程/进程多次获取。它会跟踪获取计数，并且只有当 `release()` 被调用的次数与 `acquire()` 相同次数时，锁才会被完全释放。这在复杂的函数中很有用，这些函数可能会调用其他也需要获取同一锁的函数。

**死锁：** 锁的常见问题发生在两个或多个线程/进程永远阻塞，每个都在等待对方持有的锁。例如，线程 A 持有锁 1 并等待锁 2，而线程 B 持有锁 2 并等待锁 1。仔细的设计，例如始终按一致的全局顺序获取锁，可以避免死锁。

> 一个简单的死锁场景，其中线程 A 持有锁 1 并需要锁 2，而线程 B 持有锁 2 并需要锁 1。

## 信号量：限制并发访问

信号量是比锁更通用的同步原语。它维护一个内部计数器，每次 `acquire()` 调用时递减，每次 `release()` 调用时递增。如果计数器为零，`acquire()` 将阻塞，直到另一个线程/进程调用 `release()`。

**原理：** 信号量通常用大于零的值初始化。这个值代表可以同时访问有限资源或执行特定代码段的线程/进程数量。一个初始化值为 1 的信号量与锁的行为完全相同。

**在机器学习 (machine learning)中的应用场景：**

- **资源池化：** 限制用于数据获取的数据库或 API 端点的连接数量。
- **控制并行度：** 限制执行计算密集型任务（例如，在多个 CPU 核心上进行特征提取）的并发线程数量，以避免系统过载或管理 GPU 内存使用。
- **生产者-消费者场景：** 管理对有界缓冲区或队列的访问，其中生产者添加项目，消费者移除项目。

**Python 实现：**
`threading.Semaphore` 和 `multiprocessing.Semaphore` 接受可选的初始 `value` 参数 (parameter)。

```python
import threading
import time
import random

# 示例：限制并发下载
max_concurrent_downloads = 3
download_semaphore = threading.Semaphore(value=max_concurrent_downloads)
num_files_to_download = 10

def download_file(file_id):
    """模拟文件下载，受信号量限制。"""
    print(f"文件 {file_id}：等待获取信号量...")
    with download_semaphore: # 获取信号量，如果计数为 0 则阻塞
        print(f"文件 {file_id}：已获取信号量。开始下载...")
        # 模拟下载时间
        download_duration = random.uniform(0.5, 1.5)
        time.sleep(download_duration)
        print(f"文件 {file_id}：下载完成，耗时 {download_duration:.2f}s。释放信号量。")
    # 信号量在退出 'with' 块时自动释放

threads = []
print(f"尝试下载 {num_files_to_download} 个文件，最多 {max_concurrent_downloads} 个并发下载。")
for i in range(num_files_to_download):
    thread = threading.Thread(target=download_file, args=(i,))
    threads.append(thread)
    thread.start()
    time.sleep(0.1) # 稍微错开线程启动时间

for thread in threads:
    thread.join()

print("\n所有文件下载任务已完成。")
```

在这个示例中，尽管我们快速启动了 10 个线程，但最多只有 3 个线程会并发执行“下载”部分（受信号量保护的临界区）。

**有界信号量：** `threading.BoundedSemaphore` 与 `Semaphore` 类似，但如果 `release()` 被调用的次数多于 `acquire()`，它会引发 `ValueError`，这可能会使计数器超出其初始值。这有助于发现资源释放不正确的编程错误。

## 事件：线程/进程间信号通知

事件对象管理一个内部标志，该标志可以被设置 (`set()`) 或清除 (`clear()`)。线程/进程可以等待此标志被设置 (`wait()`)。事件是一种简单但有效的方法，用于协调并发任务之间的行动。

**原理：** 事件初始时内部标志为清除状态（False）。调用 `wait()` 会阻塞，直到标志变为 True（由另一个线程/进程调用 `set()` 设置）。如果标志已为 True 时调用 `wait()`，它会立即返回。`is_set()` 检查标志状态，不会阻塞。`clear()` 将标志重置为 False。

**在机器学习 (machine learning)中的应用场景：**

- **管道协调：** 发出数据预处理阶段完成的信号，从而允许模型训练阶段开始。
- **任务完成通知：** 主线程等待多个工作线程完成初始化或特定子任务。
- **暂停/恢复控制：** 控制线程可以使用事件来通知工作线程暂停或恢复处理。
- **终止信号：** 通过设置一个工作线程/进程周期性检查的事件来优雅地关闭它们。

**Python 实现：**
`threading.Event` 和 `multiprocessing.Event` 提供核心方法：`set()`、`clear()`、`wait(timeout=None)`、`is_set()`。

```python
import threading
import time
import random

# 示例：协调数据加载和处理
data_ready_event = threading.Event()
processing_finished_event = threading.Event()
shared_data = None

def data_loader():
    """模拟数据加载。"""
    global shared_data
    print("加载器：开始加载数据...")
    load_time = random.uniform(1, 3)
    time.sleep(load_time)
    shared_data = list(range(10)) # 模拟已加载数据
    print(f"加载器：数据加载完成，耗时 {load_time:.2f}s。通知数据就绪。")
    data_ready_event.set() # 发出数据就绪信号

def data_processor():
    """等待数据，然后处理数据。"""
    print("处理器：等待数据...")
    data_ready_event.wait() # 阻塞直到 data_ready_event 被设置
    print("处理器：接收到数据。开始处理...")

    if shared_data is not None:
        # 模拟处理
        processed_sum = sum(item * 2 for item in shared_data)
        process_time = random.uniform(0.5, 1.5)
        time.sleep(process_time)
        print(f"处理器：处理完成，耗时 {process_time:.2f}s。结果：{processed_sum}")
    else:
        print("处理器：错误！共享数据为 None。")

    print("处理器：通知处理完成。")
    processing_finished_event.set()


loader_thread = threading.Thread(target=data_loader)
processor_thread = threading.Thread(target=data_processor)

loader_thread.start()
processor_thread.start()

# 主线程可以等待处理完成
print("主线程：等待处理完成...")
processing_finished_event.wait(timeout=10) # 带超时等待

if processing_finished_event.is_set():
    print("主线程：处理器已完成。")
else:
    print("主线程：等待处理器超时。")

loader_thread.join()
processor_thread.join()

print("主线程：所有线程已加入。工作流完成。")
```

在此示例中，`data_processor` 线程使用 `data_ready_event.wait()` 有效地等待，而不是轮询，只有当 `data_loader` 通过 `data_ready_event.set()` 发出完成信号后才继续。主线程也类似地使用另一个事件来等待处理器。

## 选择合适的原语

- 当你需要确保在任何给定时间只有一个线程/进程可以访问临界区或共享资源（互斥）时，使用**锁**（`threading.Lock`、`multiprocessing.Lock`）。
- 当你需要限制并发线程/进程访问资源或执行代码的数量，允许最多 N 个并发访问（其中 N 是信号量的初始值）时，使用**信号量**（`threading.Semaphore`、`multiprocessing.Semaphore`）。
- 当你需要一种简单机制，让一个或多个线程/进程等待由另一个线程/进程触发的信号或条件时，使用**事件**（`threading.Event`、`multiprocessing.Event`）。

## 最佳实践与注意事项

- **缩短锁持有时间：** 使临界区（受锁或信号量保护）内的代码尽可能短和快，以减少争用并提高并行度。在锁定区域之外执行非关键工作。
- **避免复杂的锁定：** 复杂的锁定逻辑会增加死锁的风险，并使代码难以理解。尽可能简化同步逻辑。
- **注意性能开销：** 同步原语本身会引入一些开销。对你的应用程序进行性能分析，以确保并发带来的好处大于同步成本。
- **`threading` 与 `multiprocessing` 的比较：** 请记住，`threading` 中的原语用于协调*同一*进程内的线程，而 `multiprocessing` 中的原语则用于协调*不同*进程，因为它们处理必要的进程间通信机制。

同步原语是管理共享状态和协调并发 Python 应用程序中任务的基本工具。正确应用它们对发挥机器学习 (machine learning)管道中并行计算的威力，同时保持正确性并避免竞态条件和死锁等问题非常重要。

## 参考资料

- [threading - Thread-based parallelism](https://docs.python.org/3/library/threading.html) — Python Software Foundation (2024)
  Publisher: Python Software Foundation
  Python `threading` 模块的官方文档，详细介绍了锁、信号量、事件和其他同步原语。
- [multiprocessing - Process-based parallelism](https://docs.python.org/3/library/multiprocessing.html) — Python Software Foundation (2024)
  Python `multiprocessing` 模块的官方文档，详细说明了如何使用基于进程的同步原语，例如锁、信号量和事件。
- [Fluent Python: Clear, Concise, and Effective Programming, 2nd Edition](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGASZYm-fq3BldlHQYAyq553kn7IYkmfSfQpn-wE022Kht6j5Lm9Wi-m38Iq5t-hAeCrnQOm63b_n82Bxt_tYZA6eiMHVvsVFzqhUrDWVrEmkIcNVK0Y0we_o3FpTsTv_xi4TsaSsBLB2xbnGSQT0ikraWTEMe-Oi8tH6tTWbOPByES4JSw3A4y1oiUoC8tkhA==) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media
  一本权威的进阶 Python 指南，涵盖了使用 `threading`、`multiprocessing` 和 `asyncio` 实现并发模式的实际应用，并提供了详细示例和最佳实践。
