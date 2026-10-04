---
course: "advanced-python-programming-ml"
chapter: "concurrency-parallelism-python-ml"
lesson: "practice-parallelizing-preprocessing"
sourceId: 2361
sourceUrl: "https://apxml.com/zh/courses/advanced-python-programming-ml/chapter-5-concurrency-parallelism-python-ml/practice-parallelizing-preprocessing"
title: "动手实践：数据预处理并行化"
description: "使用多进程或 concurrent.futures 并行化数据预处理流程，以加快执行速度。"
order: 8
plots: ["plots/2361-0.json"]
sourceHash: "5e587d765cf52f6e5e43adbda7b64839138ece4315d4b84c8e64c6bbdc858424"
sourceCorrections: []
---

数据预处理通常是机器学习 (machine learning)流程中计算密集型的部分，尤其是在处理大规模数据集或复杂特征工程步骤时。许多预处理任务，例如逐行或逐列应用转换，本身就可以并行处理。应用前面讨论的并发技术可以大幅缩短此阶段耗时。

选取一个典型的数据预处理任务，并使用 Python 的 `concurrent.futures` 模块来加速，特别是运用 `ProcessPoolExecutor` 处理 CPU 密集型操作。

### 场景：复杂特征工程

假设我们有一个表示为 Pandas DataFrame 的数据集，我们需要对其一列应用一个计算密集型函数，以生成一个新特征。我们的函数将模拟一些复杂计算。

```python
import pandas as pd
import numpy as np
import time
import math
from concurrent.futures import ProcessPoolExecutor

# 生成一些样本数据
data_size = 1_000_000
df = pd.DataFrame({
    'id': range(data_size),
    'value_to_process': np.random.rand(data_size) * 100
})

# 定义一个模拟的复杂处理函数
def complex_feature_engineering(value):
    """模拟一个CPU密集型计算。"""
    result = 0
    for i in range(100): # 模拟计算工作
        result += math.sqrt(abs(math.sin(value) * math.cos(value / (i + 1)))) * math.log(value + 1)
    # 添加短暂休眠以模拟潜在的I/O或外部调用
    # time.sleep(0.0001)
    return result

print("样本 DataFrame 前几行：")
print(df.head())
print(f"\nDataFrame 大小：{len(df)} 行")
```

### 基准：顺序执行

首先，我们使用 Pandas 的 `apply` 方法来实现标准的顺序方法。我们将计时以查看此过程需要多长时间。

```python
# 使用pandas apply进行顺序执行
print("\n开始顺序处理...")
start_time_seq = time.time()

df['new_feature_seq'] = df['value_to_process'].apply(complex_feature_engineering)

end_time_seq = time.time()
sequential_duration = end_time_seq - start_time_seq

print(f"顺序处理完成。")
print(f"耗时：{sequential_duration:.2f} 秒")
print("\n包含顺序结果的 DataFrame 前几行：")
print(df[['id', 'value_to_process', 'new_feature_seq']].head())
```

在典型的多核机器上，对一百万行数据运行此函数将耗费相当长的时间，因为它在单个 CPU 核心上逐个处理每个值。

### 使用 ProcessPoolExecutor 进行并行执行

现在，让我们使用 `ProcessPoolExecutor` 来并行化此任务。该执行器将工作分配到多个进程，使 Python 能够绕过全局解释器锁 (GIL)，处理像 `complex_feature_engineering` 函数这样的 CPU 密集型任务。

我们可以使用 `executor.map` 方法，它的工作方式类似于内置的 `map` 函数，但会并发执行调用。它将给定函数应用于可迭代对象（我们的 DataFrame 列）中的每个项，并返回一个迭代器，按输入顺序生成结果。

以块（分块）方式处理数据通常比将单独的行发送到工作进程更有效，因为这可以减少进程间通信 (IPC) 的开销。我们可以将 DataFrame 列分成块，并提交每个块进行处理。然而，为了 `executor.map` 的简单性，我们将直接在 Series 上映射。Pandas Series（和 NumPy 数组）通常是可序列化的，这使得它们可以被发送到工作进程。

```python
# 使用 ProcessPoolExecutor 进行并行执行
# 确定最优工作进程数（通常与CPU核心数有关）
# 使用 None 通常默认为机器的处理器数量
num_workers = None # 或者设置为特定数字，例如 4

print(f"\n开始并行处理，使用 {num_workers or '默认'} 个工作进程...")
start_time_par = time.time()

# 将列转换为列表或NumPy数组，以获得可能更好的分块/序列化效果
# 这取决于执行器的实现细节。对于map，Series通常可以直接使用。
values_to_process = df['value_to_process'].values

results_par = []
# 使用 ProcessPoolExecutor 并行应用函数
# chunksize 可以根据任务持续时间与开销进行性能调整
# 较大的 chunksize 会减少开销，但可能导致负载不平衡。
# 让我们估算一个 chunksize。如果每个任务都很短，则更大的块更好。
# 如果 data_size 是 1M 且我们有 8 个工作进程，每个块可能是 1000-10000 个项。
estimated_chunksize = max(1, min(len(values_to_process) // (num_workers or os.cpu_count() or 1) // 4, 10000))

with ProcessPoolExecutor(max_workers=num_workers) as executor:
    # map 并发地将函数应用于可迭代对象中的每个项
    # 结果按输入可迭代对象的顺序返回
    results_par = list(executor.map(complex_feature_engineering, values_to_process, chunksize=estimated_chunksize))

# 将结果重新赋值给 DataFrame
df['new_feature_par'] = results_par

end_time_par = time.time()
parallel_duration = end_time_par - start_time_par

print(f"并行处理完成。")
print(f"耗时：{parallel_duration:.2f} 秒")
print("\n包含并行结果的 DataFrame 前几行：")
# 验证结果是否一致（如果存在浮点精度差异则允许）
print(df[['id', 'new_feature_seq', 'new_feature_par']].head())
# 检查结果是否接近匹配
if 'new_feature_seq' in df.columns:
    print("\n验证结果是否匹配（近似）：")
    print(np.allclose(df['new_feature_seq'], df['new_feature_par']))

# 计算加速比
speedup = sequential_duration / parallel_duration
print(f"\n加速因子：{speedup:.2f}x")

# 如果需要，清除顺序列
# del df['new_feature_seq']
```

### 性能比较

让我们可视化这两种方法的耗时。



![数据预处理执行时间](plots/2361-0.json)



> 顺序数据预处理与并行数据预处理的执行时间比较。

### 讨论

您应该会发现并行方法能大幅减少执行时间，特别是在具有多个 CPU 核心的机器上。具体的加速效果取决于：

1. **CPU 核心数：** `ProcessPoolExecutor` 利用多个核心。可用的核心越多（在一定程度上），潜在的加速效果越好。
2. **任务粒度：** `complex_feature_engineering` 函数需要足够计算密集，才能使并行化开销（IPC、进程创建）变得值得。如果函数执行速度极快，则开销可能会抵消收益。
3. **开销：** 创建进程并在它们之间传输数据（通过 pickling 进行序列化/反序列化）会产生开销。对于非常大的数据集或复杂对象，这可能成为瓶颈。使用共享内存或以更大块处理数据（`executor.map` 中的 `chunksize`）等技术可以帮助缓解此问题。
4. **GIL 限制：** 由于我们的任务是 CPU 密集型的，`ProcessPoolExecutor` 非常有效，因为每个进程都有自己的 Python 解释器和内存空间，从而绕开了 GIL。对于 I/O 密集型任务，`ThreadPoolExecutor` 或 `asyncio` 可能更适合，正如本章前面讨论的。

### 管道的考量

- **分块：** 对于无法舒适地适应内存的超大数据集，或者函数应用开销较低时，显式地将数据分成更大的块（例如，分割 DataFrame 并对每个块使用 `executor.submit`）比映射单个元素更有效率。
- **内存使用：** 请注意，创建多个进程比线程消耗更多内存，因为每个进程都有自己的内存空间。确保您的机器有足够的内存。
- **库支持：** 许多现代数据科学库（如 Scikit-learn、Dask、Ray、Polars）都内置了并行执行支持（Scikit-learn 中的 `n_jobs` 参数 (parameter)，Dask/Ray 中的分布式计算，Polars 中的多线程）。在实现自定义解决方案之前，请调查您的工具是否已提供优化的并行处理能力。
- **错误处理：** 在生产代码中，在并行执行逻辑周围添加错误处理，以管理工作进程中潜在的故障。

本实践演示了如何应用 `ProcessPoolExecutor` 来加速常见的 CPU 密集型数据预处理任务。通过了解您的工作负载性质（CPU 密集型 vs. I/O 密集型）和可用的工具，您可以有效地运用并发来加速您的机器学习 (machine learning)流程。

## 参考资料

- [\`concurrent.futures\` - Launching parallel tasks](https://docs.python.org/3/library/concurrent.futures.html) — Python Software Foundation (2024)
  Publisher: Python Software Foundation
  `concurrent.futures`模块的官方文档，详细介绍了`ProcessPoolExecutor`和高级异步执行。
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本关于使用Python库（包括Pandas）进行数据操作和分析的指南，由其创建者撰写。
- [High Performance Python: Practical Performant Programming for Speed, Scale, and Efficiency](https://www.oreilly.com/library/view/high-performance-python/9781492055013/) — Micha Gorelick and Ian Ozsvald (2020)
  Publisher: O'Reilly Media
  涵盖优化Python代码的策略和技术，包含关于科学计算并行和并发的部分。
