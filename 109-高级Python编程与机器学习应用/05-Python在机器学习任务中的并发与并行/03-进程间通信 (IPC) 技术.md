# 进程间通信 (IPC) 技术

来源：[原文](https://apxml.com/zh/courses/advanced-python-programming-ml/chapter-5-concurrency-parallelism-python-ml/inter-process-communication-ipc)

[返回章节目录](README.md) · [返回课程目录](../README.md)

当您使用 Python 的 `multiprocessing` 模块启动多个进程时，每个进程都在其独立的内存空间中运行。这种隔离有助于避免数据损坏，但也带来一个难题：进程如何协调和交换信息？进程间的协调和信息交换在机器学习 (machine learning)工作流程中经常是必要的。例如，数据批次可能需要分发给工作进程进行并行预处理或训练，然后收集计算结果或更新的模型参数 (parameter)。进程间通信 (IPC) 技术为这种必要的数据交换提供了机制。

本节介绍 Python 的 `multiprocessing` 库中主要的 IPC 方法，重点介绍它们在计算密集型机器学习场景中的应用。我们将分析队列（Queues）、管道（Pipes）以及最近添加的共享内存（Shared Memory），评估它们在易用性、性能以及对不同通信模式的适用性方面的权衡。

### 使用 `multiprocessing.Queue` 进行灵活通信

`multiprocessing.Queue` 是最常用和多功能的一种 IPC 机制。它提供了一个线程安全且进程安全的先进先出（FIFO）队列，允许多个生产者进程添加项，多个消费者进程检索项。

在后台，放入 `Queue` 的对象会由发送进程进行序列化（pickled），并由接收进程进行反序列化（unpickled）。这使得 `Queue` 非常灵活，因为它可以处理大多数可序列化的 Python 对象。然而，这种序列化/反序列化步骤会引入开销，当频繁传输大型对象（如 NumPy 数组或 Pandas DataFrames）时，这种开销会变得明显。

考虑一个典型的机器学习 (machine learning)场景，其中主进程将数据块分发给工作进程进行并行特征提取：

```python
import multiprocessing
import time
import numpy as np

# 工作进程执行的示例函数
def worker_process_data(data_queue, result_queue):
    pid = multiprocessing.current_process().pid
    while True:
        try:
            # 从队列获取数据（如果为空则阻塞）
            # 如果需要，使用超时防止无限期阻塞
            data_chunk = data_queue.get(timeout=1)
            if data_chunk is None: # 终止信号的哨兵值
                print(f"工作进程 {pid} 收到终止信号。")
                break

            # 模拟处理（例如，特征工程）
            processed_result = np.sum(data_chunk ** 2) # 示例计算
            print(f"工作进程 {pid} 处理块和：{processed_result}")

            # 将结果放入结果队列
            result_queue.put((pid, processed_result))

        except multiprocessing.queues.Empty:
            # 队列在超时时间内为空
            print(f"工作进程 {pid} 发现数据队列为空，继续...")
            continue # 或者如果没有更多数据，则中断
        except Exception as e:
            print(f"工作进程 {pid} 遇到错误：{e}")
            break # 出错时退出

# 主进程设置
if __name__ == '__main__':
    # 如果需要，在某些平台/设置中使用 Manager 上下文管理队列
    # manager = multiprocessing.Manager()
    # data_queue = manager.Queue()
    # result_queue = manager.Queue()

    # 通常标准队列直接工作
    data_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    num_workers = 4
    processes = []

    # 启动工作进程
    for _ in range(num_workers):
        p = multiprocessing.Process(target=worker_process_data, args=(data_queue, result_queue))
        processes.append(p)
        p.start()

    # 模拟加载和分发数据块
    num_chunks = 20
    chunk_size = 10000
    print(f"主进程正在分发 {num_chunks} 个数据块...")
    for i in range(num_chunks):
        data = np.random.rand(chunk_size)
        data_queue.put(data)
        time.sleep(0.05) # 模拟加载/准备数据所需时间

    # 通过发送 None 信号通知工作进程终止
    print("主进程正在发送终止信号...")
    for _ in range(num_workers):
        data_queue.put(None)

    # 收集结果
    results = []
    print("主进程正在收集结果...")
    # 确保收集到终止信号发送后可能生成的结果
    # 需要一种方法来知道所有工作进程何时完成数据处理
    # 这里我们假设预期会有 num_chunks 个结果
    for _ in range(num_chunks):
         try:
            worker_id, result = result_queue.get(timeout=5) # 等待结果
            results.append(result)
            print(f"主进程收到工作进程 {worker_id} 的结果")
         except multiprocessing.queues.Empty:
            print("结果队列为空，可能结果丢失或提前退出。")
            break

    # 等待所有工作进程完成
    print("主进程等待工作进程加入...")
    for p in processes:
        p.join()

    print(f"\n收集到 {len(results)} 个结果。示例：{results[:5]}")
    print("主进程完成。")
```

**`Queue` 的使用场景：**

- 将任务或数据项分发给工作池。
- 从多个工作进程收集结果。
- 在进程间实现生产者-消费者模式。

**注意事项：**

- **性能：** 序列化开销对于非常大或数量众多的项来说可能成为瓶颈。
- **缓冲：** 队列有内部缓冲区；如果队列已满，`put()` 调用可能会阻塞。
- **复杂性：** 对于许多常见模式来说，使用起来相对简单。

### 使用 `multiprocessing.Pipe` 进行双进程通信

虽然 `Queue` 支持多对多通信，但 `multiprocessing.Pipe` 专为在*恰好两个*进程之间建立连接而设计。管道返回一对 `Connection` 对象，管道的每一端各一个。默认情况下，它是双工（双向）的，这意味着两端都可以发送和接收数据。

```python
import multiprocessing
import time

def child_process_task(conn):
    pid = multiprocessing.current_process().pid
    print(f"子进程 {pid} 启动。")
    while True:
        try:
            # 等待接收来自父进程的消息
            msg = conn.recv()
            print(f"子进程 {pid} 收到：{msg}")

            if msg == "EXIT":
                print(f"子进程 {pid} 退出。")
                conn.close()
                break
            elif isinstance(msg, dict) and 'data' in msg:
                # 模拟处理
                result = f"Processed data ID {msg.get('id', 'N/A')}"
                time.sleep(0.5)
                # 将结果发送回父进程
                conn.send(result)
            else:
                conn.send("Unknown command")

        except EOFError:
            print(f"子进程 {pid}：连接被父进程关闭。")
            break
        except Exception as e:
            print(f"子进程 {pid} 错误：{e}")
            conn.close()
            break

if __name__ == '__main__':
    # 创建一个双向管道
    parent_conn, child_conn = multiprocessing.Pipe()

    # 创建并启动子进程
    p = multiprocessing.Process(target=child_process_task, args=(child_conn,))
    p.start()

    # 父进程与子进程交互
    print("父进程正在发送任务...")
    for i in range(3):
        task_data = {'id': i, 'data': [i]*5}
        print(f"父进程发送：{task_data}")
        parent_conn.send(task_data)

        # 等待子进程的响应
        response = parent_conn.recv()
        print(f"父进程收到：{response}\n")
        time.sleep(0.1)

    # 发送信号让子进程退出
    print("父进程正在发送退出信号。")
    parent_conn.send("EXIT")

    # 关闭管道的父进程端
    parent_conn.close()

    # 等待子进程完成
    p.join()
    print("父进程完成。")
```

与 `Queue` 类似，`Pipe` 也依赖序列化进行对象传输，对大型数据具有类似的性能影响。

**`Pipe` 的使用场景：**

- 在父进程和单个子进程之间建立专用通信通道。
- 两个特定进程之间的双向控制和数据交换。

**注意事项：**

- **范围受限：** 只连接两个进程。
- **性能：** 序列化开销与 `Queue` 类似。
- **潜在死锁：** 在双向通信中必须小心；如果两个进程同时尝试 `recv()` 而对方没有发送，它们可能会死锁。同样，如果两个进程都尝试 `send()` 到一个已满的管道缓冲区，也会发生死锁。

### 使用 `multiprocessing.shared_memory` 进行高性能数据共享 (Python 3.8+)

`multiprocessing.shared_memory` 模块在 Python 3.8 中引入，它提供了一种让进程直接访问同一块内存的方法，绕过序列化和复制数据的需要。这对于大型数值数据集（如 NumPy 数组）尤其有利，这些数据集在机器学习 (machine learning)中很常见。

使用共享内存涉及以下几个步骤：

1. 一个进程创建一个特定大小的 `SharedMemory` 块。
2. 该进程（或另一个进程）可以访问这块内存，通常是创建一个直接使用共享内存缓冲区的 NumPy 数组。
3. 共享内存块的*名称*必须传达给其他进程（例如，使用 `Queue` 或 `Pipe`）。数组的数据类型和形状等元数据也必须传达。
4. 其他进程使用该名称连接到现有的 `SharedMemory` 块，并创建自己的 NumPy 数组，这些数组映射到相同的缓冲区。
5. 进程现在可以并发地读写此内存块。**重要：** 并发写入需要仔细的同步（使用锁、信号量等）以防止竞态条件和数据损坏。
6. 当共享内存块不再需要时，一个进程必须调用 `unlink()` 来标记 (token)它进行销毁。所有进程也应该在其 `SharedMemory` 实例上调用 `close()` 以释放映射。未能调用 `unlink()` 可能导致内存资源泄漏。

```python
import multiprocessing
import numpy as np
from multiprocessing import shared_memory, Process, Lock, Queue
import time

# 访问共享内存的工作函数
def worker_modify_shared_array(shm_name, array_shape, array_dtype, start_index, end_index, lock, result_queue):
    pid = multiprocessing.current_process().pid
    existing_shm = None
    try:
        # 连接到现有共享内存块
        existing_shm = shared_memory.SharedMemory(name=shm_name)

        # 创建一个由共享内存支持的 NumPy 数组
        shared_array = np.ndarray(array_shape, dtype=array_dtype, buffer=existing_shm.buf)

        print(f"工作进程 {pid} 连接到共享内存 '{shm_name}'。正在处理切片 [{start_index}:{end_index}]")

        # 修改共享数组的一个切片（如果并发写入，则获取锁）
        partial_sum = 0
        with lock: # 在修改共享状态之前获取锁
             print(f"工作进程 {pid} 获取锁。")
             for i in range(start_index, end_index):
                 # 示例修改：将元素平方
                 shared_array[i] = shared_array[i] ** 2
                 partial_sum += shared_array[i] # 在可能被其他进程修改后读取（如果锁的粒度更细）
             print(f"工作进程 {pid} 释放锁。")
             # 注意：在计算期间持有锁可能会使执行串行化。
             # 通常最好在本地计算，然后只在更新时获取锁。

        print(f"工作进程 {pid} 完成切片处理。")
        result_queue.put((pid, partial_sum)) # 将结果发送回

    except Exception as e:
        print(f"工作进程 {pid} 错误：{e}")
        result_queue.put((pid, None)) # 表示错误
    finally:
        # 始终在每个进程中关闭共享内存实例
        if existing_shm:
            existing_shm.close()
            print(f"工作进程 {pid} 关闭共享内存句柄。")

if __name__ == '__main__':
    array_size = 20
    array_shape = (array_size,)
    array_dtype = np.float64
    num_workers = 4

    shm = None
    try:
        # 1. 创建共享内存块
        # 计算大小：元素数量 * 每个元素的大小
        nbytes = np.prod(array_shape) * np.dtype(array_dtype).itemsize
        shm = shared_memory.SharedMemory(create=True, size=nbytes)
        print(f"主进程创建了共享内存块 '{shm.name}'，大小为 {shm.size} 字节。")

        # 2. 创建一个链接到共享内存的 NumPy 数组
        master_array = np.ndarray(array_shape, dtype=array_dtype, buffer=shm.buf)

        # 初始化数组
        master_array[:] = np.arange(array_size, dtype=array_dtype)
        print(f"主进程初始化了共享数组：{master_array}")

        # 创建同步锁和结果队列
        lock = Lock()
        result_queue = Queue()

        processes = []
        chunk_size = array_size // num_workers
        # 3. 启动工作进程，传递共享内存名称和元数据
        for i in range(num_workers):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < num_workers - 1 else array_size
            p = Process(target=worker_modify_shared_array,
                        args=(shm.name, array_shape, array_dtype, start, end, lock, result_queue))
            processes.append(p)
            p.start()

        # 4. 等待工作进程完成并收集结果
        total_sum = 0
        for i in range(num_workers):
             worker_id, partial_res = result_queue.get()
             if partial_res is not None:
                 print(f"主进程收到工作进程 {worker_id} 的部分和 {partial_res}")
                 total_sum += partial_res
             else:
                 print(f"主进程收到工作进程 {worker_id} 的错误信号")

        print("主进程等待工作进程加入...")
        for p in processes:
            p.join()

        # 5. 在主进程中访问修改后的数组
        print(f"\n主进程正在访问修改后的共享数组：{master_array}")
        print(f"从工作进程计算的总和：{total_sum}")
        print(f"修改后数组的直接和：{np.sum(master_array)}") # 验证结果

    except Exception as e:
        print(f"主进程错误：{e}")
    finally:
        # 6. 清理：关闭并解除共享内存块的链接
        if shm:
            print(f"主进程正在关闭共享内存句柄 '{shm.name}'。")
            shm.close() # 关闭主进程中的实例
            print(f"主进程正在解除共享内存块 '{shm.name}' 的链接。")
            shm.unlink() # 标记该块进行删除

        print("主进程完成。")
```

**`shared_memory` 的使用场景：**

- 在进程间高效共享大型 NumPy 数组或其他数据缓冲区。
- 多个进程对数据进行就地修改（需要仔细同步）。
- 使用队列/管道进行序列化/反序列化开销过大的情况。

**注意事项：**

- **复杂性：** 需要手动管理共享块的名称、元数据（形状、类型）、生命周期（`unlink`）和同步。
- **同步：** 对于并发写入以防止竞态条件至关重要。需要使用 `multiprocessing.Lock`、`Semaphore` 或其他同步原语（在下一节介绍）。
- **数据类型：** 最适合原始字节缓冲区，主要与 NumPy 等可以将数组映射到缓冲区的库一起使用。不直接适用于任意 Python 对象，除非手动将其序列化到缓冲区中。
- **平台可用性：** 需要 Python 3.8+。

### 选择合适的 IPC 技术

最好的 IPC 方法很大程度上取决于您的具体需求：

- **在以下情况下使用 `multiprocessing.Queue`：**
  - 您需要灵活的多对多通信。
  - 易用性很重要。
  - 您正在传输相对较小或不频繁的数据项，或者序列化开销可以接受。
  - 实现标准的生产者-消费者模式。
- **在以下情况下使用 `multiprocessing.Pipe`：**
  - 您需要在恰好两个进程（例如，父子进程）之间建立专用的双向通道。
  - 通信模式符合这种双端模型。
- **在以下情况下使用 `multiprocessing.shared_memory`：**
  - 您需要最高性能来传输大量数值数据块（如 NumPy 数组）。
  - 您能够管理资源生命周期并实现必要的同步。
  - 您正在使用 Python 3.8 或更高版本。

在许多复杂的机器学习 (machine learning)应用中，您可能会使用这些技术的组合。例如，您可以使用 `Queue` 分发任务指令（这些指令很小），并使用 `shared_memory` 让工作进程高效地访问大型只读数据集。无论选择哪种方法，了解进程如何交换信息对于在 Python 中构建高效的并行机器学习系统非常重要。请记住，管理共享状态，特别是可写共享状态，需要仔细的同步，这将在下一节讨论。

## 参考资料

- [\`multiprocessing\` - Process-based parallelism](https://docs.python.org/3/library/multiprocessing.html) — Python Software Foundation (2023)
  官方文档提供了Python `multiprocessing`模块的全面概述，包含队列和管道的详细解释及示例，以及进程管理的基本概念。
- [\`multiprocessing.shared_memory\` - Share data between processes](https://docs.python.org/3/library/multiprocessing.shared_memory.html) — Python Software Foundation (2023)
  Publisher: Python Software Foundation
  `shared_memory`模块的官方文档，详细说明了其在高性能数据共享（特别是大型数值数组）中的应用，并概述了同步和资源生命周期管理的注意事项。
- [High Performance Python: Practical Performant Programming for Humans with Machines](https://www.oreilly.com/library/view/high-performance-python/9781492055013/) — Micha Gorelick and Ian Ozsvald (2020)
  Publisher: O'Reilly Media
  本书提供了优化Python代码的实用策略，包括关于并行数据处理的章节，为在机器学习工作负载中使用 `multiprocessing` 提供了相关背景和技术。

---

[上一节](02-%E7%94%A8%E4%BA%8E%E5%B9%B6%E8%A1%8C%E6%89%A7%E8%A1%8C%E7%9A%84%20%60multiprocessing%60%20%E6%A8%A1%E5%9D%97.md) · [下一节](04-%E4%BD%BF%E7%94%A8%20%60concurrent.futures%60%20%E5%AE%9E%E7%8E%B0%E9%AB%98%E7%BA%A7%E5%B9%B6%E5%8F%91.md)
