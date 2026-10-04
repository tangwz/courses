---
course: "advanced-python-programming-ml"
chapter: "python-performance-optimization-ml"
lesson: "memory-profiling-optimization"
sourceId: 2287
sourceUrl: "https://apxml.com/zh/courses/advanced-python-programming-ml/chapter-2-python-performance-optimization-ml/memory-profiling-optimization"
title: "内存分析与优化方法"
description: "使用内存分析工具和方法来减少Python机器学习应用的内存占用。"
order: 7
plots: ["plots/2287-0.json"]
sourceHash: "a9871209280e8762194c5a7d4fedebb5d6d9d223fc6aacf2b6a1e0350737faf8"
sourceCorrections: []
---

CPU性能优化虽然必不可少，但内存使用管理同样重要，特别是在机器学习 (machine learning)中，数据集很容易超过可用的RAM。低效的内存管理不仅会导致 `MemoryError` 异常，还可能因交换增加或垃圾回收开销而严重降低性能。有效的策略可用于找出内存热点并应用优化，以构建更内存高效的机器学习应用程序。

在使用大型特征集、处理庞大数据集或在资源受限环境中部署模型时，理解和控制内存消耗非常重要。我们将介绍检查内存使用的工具，以及减少Python代码内存占用的实用方法。

## 识别内存瓶颈：性能分析工具

正如CPU性能分析器有助于定位耗时代码段一样，内存分析器有助于识别程序中消耗大量内存或可能随时间发生内存泄漏的部分。

### 使用 `memory_profiler` 进行逐行性能分析

`memory_profiler` 包是一个有用的工具，用于逐行监控Python进程的内存消耗。它有助于精确找出导致大量内存分配的具体代码行。

要使用它，你通常需要安装它（`pip install memory_profiler`），然后用 `@profile` 装饰器修饰要进行性能分析的函数。接着，你可以使用该包提供的特殊解释器或通过 `mprof` 命令行工具来运行你的脚本。

```python
# pip install memory_profiler psutil
# (psutil is often needed for more accurate measurements)

import numpy as np
from memory_profiler import profile

@profile
def create_large_matrices(size):
    print(f"Creating matrix A ({size}x{size})...")
    matrix_a = np.random.rand(size, size)
    print(f"Creating matrix B ({size}x{size})...")
    matrix_b = np.random.rand(size, size)
    # 此操作可能会创建一个临时的、较大的矩阵
    print("Multiplying matrices...")
    result = matrix_a @ matrix_b
    print("Calculation complete.")
    # 如果内存紧张，明确删除大型对象
    # del matrix_a
    # del matrix_b
    # import gc
    # gc.collect() # 强制进行垃圾回收（谨慎使用）
    return result

if __name__ == '__main__':
    # 示例：分析创建1000x1000矩阵的内存使用情况
    # 运行方式: python -m memory_profiler your_script_name.py
    large_result = create_large_matrices(1000)
    print(f"Result shape: {large_result.shape}")
```

使用 `python -m memory_profiler your_script.py` 运行此脚本会产生类似于以下内容的输出（数值为示意性质）：

```
Filename: your_script.py

Line #    Mem usage    Increment   Line Contents
================================================
     6     45.1 MiB     45.1 MiB   @profile
     7                             def create_large_matrices(size):
     8     45.1 MiB      0.0 MiB       print(f"Creating matrix A ({size}x{size})...")
     9    121.4 MiB     76.3 MiB       matrix_a = np.random.rand(size, size)
    10    121.4 MiB      0.0 MiB       print(f"Creating matrix B ({size}x{size})...")
    11    197.7 MiB     76.3 MiB       matrix_b = np.random.rand(size, size)
    12    197.7 MiB      0.0 MiB       print("Multiplying matrices...")
    13    274.0 MiB     76.3 MiB       result = matrix_a @ matrix_b # Potential peak here
    14    274.0 MiB      0.0 MiB       print("Calculation complete.")
    15    274.0 MiB      0.0 MiB       return result
```

`Increment` 列显示了执行该行代码分配的内存量，有助于识别最耗内存的操作。`mprof` 工具（`mprof run your_script.py`，然后 `mprof plot`）可以生成显示内存随时间变化的图表，这对于可视化趋势和峰值很有用。

### 使用 `tracemalloc` 追踪对象分配

Python的内置 `tracemalloc` 模块提供了一个不同的视角。它不按逐行增量来追踪，而是追踪Python分配的内存块，并按分配位置对其进行分组。这尤其有助于检测内存泄漏或理解大量小对象是如何产生的。

```python
import tracemalloc
import numpy as np
import pandas as pd
import linecache # Required by tracemalloc for detailed output

def create_and_process_data():
    # 模拟创建可能累积的对象
    data_list = []
    for _ in range(10000):
        # 例如：创建许多小型DataFrames或复杂对象
        df = pd.DataFrame(np.random.randn(10, 5), columns=list('ABCDE'))
        data_list.append(df) # 保持引用
    return data_list # 假设此列表意外增长

# 开始追踪内存分配
tracemalloc.start()

# 在操作前截取快照
snap1 = tracemalloc.take_snapshot()

# 运行可能导致内存问题的函数
processed_data = create_and_process_data()
# 保留一个引用以防止演示时立即进行垃圾回收
print(f"Processed {len(processed_data)} items.")

# 操作后截取快照
snap2 = tracemalloc.take_snapshot()

# 停止追踪
tracemalloc.stop()

# 比较快照以查看差异
top_stats = snap2.compare_to(snap1, 'lineno')

print("\nTop 10 memory allocation differences:")
for stat in top_stats[:10]:
    print(stat)

# 获取特定分配位置回溯的示例
# print("\nTraceback for the top allocation:")
# for line in stat.traceback.format():
#      print(line)
```

`tracemalloc` 输出显示了分配块的文件、行号、大小和数量。比较快照有助于识别在两个时间点之间分配了大量内存的代码路径。虽然它会增加一些开销，但通常比 `memory_profiler` 小，并且对于长时间运行的应用程序或内存泄漏检测很有用。

### 基本对象尺寸检查

对于单个对象的快速检查，`sys.getsizeof` 可以提供基础内存使用量（字节）。但是，请注意，这不包括容器对象（如列表或字典）*内容*的递归内存使用量。对于Pandas对象，请使用 `memory_usage(deep=True)` 方法来更准确地估算DataFrame的大小，包括像字符串这样的对象数据类型。

```python
import sys
import numpy as np
import pandas as pd

my_list = list(range(10000))
my_array = np.arange(10000, dtype=np.int64)
my_df = pd.DataFrame({'col': my_array})

print(f"Size of list object itself: {sys.getsizeof(my_list)} bytes")
# 注意：这不包括列表中整数的尺寸
list_content_size = sum(sys.getsizeof(i) for i in my_list)
print(f"Approximate size of integers in list: {list_content_size} bytes")

print(f"Size of NumPy array: {sys.getsizeof(my_array)} bytes (includes data buffer)")
print(f"NumPy array memory usage (.nbytes): {my_array.nbytes} bytes")

print(f"\nPandas DataFrame memory usage:")
print(my_df.memory_usage()) # 每列
print(f"\nPandas DataFrame deep memory usage:")
print(my_df.memory_usage(deep=True)) # 包括对象内存（例如字符串）
print(f"Total deep usage: {my_df.memory_usage(deep=True).sum()} bytes")
```

## ML代码中的常见内存问题

性能分析常显示导致高内存使用的常见模式：

1. **大型中间对象：** 操作，特别是链式Pandas或NumPy操作，可能创建大型临时对象，即使最终结果很小，也会消耗大量峰值内存。例如：`result = df['col_a'] + df['col_b'] * 2` 可能会为 `df['col_b'] * 2` 和最终求和创建临时数组。
2. **加载整个数据集：** 直接将大型CSV或数据库表读入Pandas DataFrame通常是最简单的方法，但当数据大小超过可用RAM时会失败。
3. **低效数据结构：** 当NumPy数组或经过仔细类型处理的Pandas DataFrame能更节省内存时，却使用了Python字典列表。重复存储字符串而非使用分类类型。
4. **意外复制：** 意外地创建大型数据结构完整副本而非视图的操作，导致内存需求翻倍。
5. **引用循环（在纯Python中较少见）：** 在复杂应用程序中，特别是涉及自定义类或C扩展的应用程序，如果对象循环引用，Python的垃圾回收器可能无法回收内存。`tracemalloc` 和 `gc` 模块（`gc.get_referrers`，`gc.get_referents`）可以帮助调试这些问题。

## 内存优化方法

一旦识别出内存瓶颈，请应用这些方法：

### 1. 数据流式处理与分块

将大型数据集分批处理，而不是一次性加载所有数据。

- **生成器：** 使用生成器函数（`yield`）惰性处理数据，只读取当前步骤所需的部分（第一章中已涵盖）。
- **Pandas分块：** 在 `pd.read_csv`、`pd.read_sql` 等函数中使用 `chunksize` 参数 (parameter)，以逐块迭代文件或查询结果。

```python
import pandas as pd

# 分块处理大型CSV文件
chunk_iter = pd.read_csv('large_dataset.csv', chunksize=100000) # 一次读取10万行

results = []
for chunk_df in chunk_iter:
    # 在较小的分块DataFrame上执行处理
    processed_chunk = chunk_df[chunk_df['value'] > 0].groupby('category').size()
    results.append(processed_chunk)

# 如果需要，合并所有分块的结果
final_result = pd.concat(results).groupby(level=0).sum()
print(final_result.head())
```

- **适用于核外计算的库：** 对于远大于RAM的数据集，Dask（`dask.dataframe`）等库模仿Pandas API，但以惰性方式运行并执行核外计算（必要时将中间结果溢出到磁盘）。

### 2. 选择合适的数据类型

使用内存占用较少的数据类型可以显著节省内存，特别是在大型NumPy数组或Pandas DataFrames中。

- **数值类型缩减：** 如果数据范围和精度允许，使用更小的整数或浮点类型。一个 `float32` 使用的内存是默认 `float64` 的一半。一个 `int8` 使用的内存是 `int64` 的八分之一。

```python
import numpy as np
import pandas as pd

# 默认 float64 数组
large_float_array_64 = np.random.rand(1_000_000)
print(f"float64 array size: {large_float_array_64.nbytes / (1024**2):.2f} MiB")

# 缩减为 float32
large_float_array_32 = large_float_array_64.astype(np.float32)
print(f"float32 array size: {large_float_array_32.nbytes / (1024**2):.2f} MiB")

# Pandas中整数的处理类似
df = pd.DataFrame({'user_id': np.random.randint(0, 10000, size=1_000_000),
                   'small_int_col': np.random.randint(0, 100, size=1_000_000)})

print(f"\nOriginal DataFrame memory:\n{df.memory_usage(deep=True).sum() / (1024**2):.2f} MiB")
df['user_id'] = pd.to_numeric(df['user_id'], downcast='unsigned')
df['small_int_col'] = pd.to_numeric(df['small_int_col'], downcast='integer')
print(f"Downcasted DataFrame memory:\n{df.memory_usage(deep=True).sum() / (1024**2):.2f} MiB")
print(df.dtypes)
```

- **Pandas分类类型：** 对于唯一值相对较少（基数低）的字符串列，将其转换为 `category` 数据类型可以大幅减少内存。Pandas 内部只存储一次唯一的字符串并使用整数编码。

```python
# 分类数据示例
data = {'country': ['USA', 'Canada', 'USA', 'Mexico', 'Canada'] * 100000}
df_str = pd.DataFrame(data)
print(f"\nString DataFrame memory: {df_str.memory_usage(deep=True).sum() / (1024**2):.2f} MiB")

df_cat = df_str.copy()
df_cat['country'] = df_cat['country'].astype('category')
print(f"Categorical DataFrame memory: {df_cat.memory_usage(deep=True).sum() / (1024**2):.2f} MiB")
```

- **稀疏数据结构：** 如果数据包含大量零值（在TF-IDF表示、独热编码中常见），请使用 `scipy.sparse` 中的稀疏矩阵（例如 `csr_matrix`、`csc_matrix`），它们只存储非零元素及其位置。

### 3. 使用原地操作（谨慎）

直接修改数据（原地）的操作有时可以避免分配大型中间副本。NumPy（`+=`，`*=`）和一些Pandas方法（`dropna(inplace=True)`，`fillna(inplace=True)`）支持此功能。

```python
import numpy as np

# 避免为结果创建新数组
a = np.ones((1000, 1000))
b = np.ones((1000, 1000))

# 取代：c = a + b （为c分配内存）
a += b # 直接修改'a'，可能节省内存
```

**注意：** 请谨慎使用原地操作。它们可能使代码更难理解，尤其是在Pandas DataFrame中，视图和副本的行为很复杂。修改一个被其他变量引用（或作为另一个对象视图）的对象可能导致意外的副作用。通常，除非内存压力很大并且分析确认有益，否则优先考虑代码清晰度而非使用 `inplace=True` 进行微优化会更好。

### 4. 理解视图与副本

注意操作是返回视图（与原始对象共享内存）还是副本（分配新内存）。不必要的复制是内存使用过多的常见原因。NumPy中的基本切片通常创建视图，而布尔索引或非连续索引切片通常创建副本。Pandas的行为可能更复杂；使用 `np.shares_memory(array1, array2)` 检查NumPy数组，或依赖性能分析来理解Pandas操作。

### 5. 显式垃圾回收与删除

Python的垃圾回收器通常会自动处理内存回收。但是，你可以给予它提示：

- **`del`：** 使用 `del` 可以移除对象的一个名称绑定。如果这是最后一个引用，该对象就符合垃圾回收的条件。对不再需要的大型对象显式使用 `del`，尤其是在处理大型数据的循环或函数中，*有时*可以帮助更快地释放内存。
- **`gc.collect()`：** 你可以手动触发垃圾回收。通常，这*不*建议用于性能优化，因为它会暂停执行，而且可能不会比自动回收释放更多的内存。其主要应用场景是调试内存泄漏，或者在非常特殊、对内存要求严格的情况下（例如，在一个大型对象被删除后，在分配另一个对象之前）立即释放内存。

```python
import gc
import numpy as np

def process_large_item(item_data):
    large_intermediate = np.ones((item_data * 100, item_data * 100))
    # ... 处理 large_intermediate ...
    result = large_intermediate.sum()

    # 提示这个大对象不再需要
    del large_intermediate
    # 可选地，如果内存非常紧张，可在下一次循环迭代前触发GC
    # gc.collect() # 谨慎使用！
    return result

# 示例循环
# for item in large_dataset_iterator:
#     process_large_item(item)
```

过度依赖 `gc.collect()` 可能会掩盖潜在的设计问题，并可能降低应用程序的速度。首先应侧重于通过更好的数据结构、分块和适当的类型来减少分配。

## 示例：优化数据加载

让我们比较一下简单加载与优化加载在内存使用上的差异。假设 `large_sales.csv` 文件包含 `product_id`（整数）、`category`（字符串，少量唯一值）、`timestamp`（字符串）和 `sales`（浮点数）等列。

**简单方法：**

```python
# 简单：加载所有数据，使用默认类型
import pandas as pd
from memory_profiler import profile

@profile
def load_naive(filename='large_sales.csv'):
    df = pd.read_csv(filename)
    # 进一步处理... 想象这里有内存密集型步骤
    peak_memory_usage = df.memory_usage(deep=True).sum()
    print(f"Naive load peak DF memory: {peak_memory_usage / (1024**2):.2f} MiB")
    return df # 保留引用以进行性能分析

# 运行方式: python -m memory_profiler your_script.py
# if __name__ == '__main__':
#     df_naive = load_naive()
```

**优化方法：**

```python
# 优化：分块、类型指定、分类类型
import pandas as pd
from memory_profiler import profile

@profile
def load_optimized(filename='large_sales.csv', chunk_size=100000):
    chunks = []
    # 定义优化类型
    dtype_spec = {
        'product_id': 'uint32', # 假设ID为正整数且适用
        'category': 'category', # 对于低基数字符串使用分类类型
        'sales': 'float32'      # 如果精度允许，使用更小的浮点数
    }
    # 在读取时指定日期解析
    date_cols = ['timestamp']

    total_memory = 0
    for chunk in pd.read_csv(filename,
                             chunksize=chunk_size,
                             dtype=dtype_spec,
                             parse_dates=date_cols):
        # 处理每个分块（示例：仅计算内存）
        chunk_mem = chunk.memory_usage(deep=True).sum()
        total_memory += chunk_mem # 注意：这仅为示意，峰值使用量更重要
        chunks.append(chunk) # 实际上，你会处理并丢弃/聚合

    # 如果需要则合并（此步骤本身会使用内存）
    # df_optimized = pd.concat(chunks, ignore_index=True)
    # peak_memory_usage = df_optimized.memory_usage(deep=True).sum()
    # print(f"Optimized combined DF memory: {peak_memory_usage / (1024**2):.2f} MiB")
    # 更实际的做法是，聚合分块结果而不存储所有数据
    print(f"Processed in chunks. Peak memory per chunk is lower.")
    # 性能分析将显示循环*期间*的内存峰值低于 load_naive

# 运行方式: python -m memory_profiler your_script.py
# if __name__ == '__main__':
#     load_optimized()
```

对 `load_naive` 进行性能分析可能会显示，当 `pd.read_csv` 完成时，内存会一次性大幅增加。对 `load_optimized` 进行性能分析（尤其是使用 `mprof plot` 时）将显示，随着每个分块的处理（取决于处理逻辑和垃圾回收），内存使用量会上升并可能略微下降，总体内存峰值将显著低于简单方法，即使最终合并结果（如果创建的话）大小相似。主要益处在于避免了大规模的初始分配。



![内存使用示意：简单加载与分块加载对比](plots/2287-0.json)



> 此图表说明了简单加载如何导致内存急剧飙升，而分块处理通过增量处理数据来保持较低的内存峰值。如果所有数据最终合并，最终内存可能相似，但处理过程中的峰值降低了。

内存优化通常是一个迭代过程。对代码进行性能分析，找出内存占用最大的部分，应用分块、类型优化或使用内存高效结构等相关方法，然后再次分析以衡量效果。在Python中构建高效且可扩展的机器学习 (machine learning)系统，平衡内存使用、CPU性能和代码可维护性非常重要。

## 参考资料

- [Pandas User Guide: Memory Usage and Optimized Data Storage](https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html#memory-usage) — The Pandas Development Team (2024)
  Pandas 官方文档中关于内存使用、分类数据类型和高效 I/O 操作（如分块处理）的章节，对于优化机器学习中的数据结构和数据加载至关重要。
- [High Performance Python: Practical Performant Programming for Data Science](https://www.oreilly.com/library/view/high-performance-python-2nd/9781492055018/) — Micha Gorelick, Ian Ozsvald (2020)
  Publisher: O'Reilly Media
  一本广受认可的书籍，提供了优化 Python 代码的实用策略，包括对内存管理、数据结构和与数据科学及机器学习应用相关的分析技术的详细讨论。
- [NumPy documentation: The N-dimensional array (ndarray)](https://numpy.org/doc/stable/reference/arrays.ndarray.html) — The NumPy Developers (2024)
  NumPy 官方文档，解释了 `ndarray` 对象、其内存布局 (`nbytes`) 以及数据类型 (`astype`) 如何影响内存消耗，这对于机器学习中高效的数值计算来说是基础。
