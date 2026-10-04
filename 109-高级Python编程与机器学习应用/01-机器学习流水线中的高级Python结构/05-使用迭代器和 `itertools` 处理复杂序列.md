# 使用迭代器和 `itertools` 处理复杂序列

来源：[原文](https://apxml.com/zh/courses/advanced-python-programming-ml/chapter-1-advanced-python-constructs-ml-pipelines/iterators-itertools-ml)

[返回章节目录](README.md) · [返回课程目录](../README.md)

机器学习 (machine learning)经常涉及处理数据序列，这些序列可能过大而无法完全载入内存，或者是在运行时生成，或者需要复杂的转换。尽管基本循环和列表推导式适用于简单情况，但高效处理复杂数据流需要更专业的工具。Python 的迭代协议和标准库的 `itertools` 模块提供了这些工具，使得处理复杂序列时能够节省内存并采用延迟计算的方式。

### 迭代器协议：延迟计算的基本机制

回想一下我们对生成器的讨论，延迟计算对于内存效率非常重要。迭代器是实现这一点的机制。如果一个对象在传递给内置函数 `iter()` 时能产生一个 *迭代器*，那么它就是一个 *可迭代对象*。*迭代器* 是一个对象，当传递给内置函数 `next()` 时，它会产生序列中的下一个值，并在序列耗尽时引发 `StopIteration` 异常。

此协议意味着数据是按需逐项处理的，而不是将整个序列实例化到内存中。对于读取大型日志文件、流式传输传感器数据或处理来自庞大数据集的批次等任务，迭代器不仅方便，而且常常是必需的。

### `itertools` 模块介绍：迭代器工具集

`itertools` 模块是一个高度优化的函数集合，它们作用于迭代器或返回迭代器。可以将其视为一组构建块，用于直接在迭代器层面构建精巧的数据处理流程。与手动实现相比，使用 `itertools` 通常能使代码更具可读性和更高性能，特别是对于复杂的逻辑。

接下来，我们考察一些在机器学习 (machine learning)场景中非常有用的 `itertools` 函数。

#### 组合与拆分迭代器

数据通常来自多个源，或者需要拆分以进行并行处理。

- **`itertools.chain(*iterables)`**：按顺序从第一个可迭代对象中取出项，直到其耗尽，然后继续处理下一个，依此类推。这对于将多个数据集或特征文件视为单个序列处理非常有用。

  ```python
  import itertools

  # 模拟来自两个不同源的特征
  features_source1 = [(1, 0.5), (2, 0.6)]
  features_source2 = [(3, 0.7), (4, 0.8)]

  combined_features = itertools.chain(features_source1, features_source2)

  # 按顺序处理所有特征
  for feature_vec in combined_features:
      print(f"正在处理特征: {feature_vec}")
      # 输出:
      # 正在处理特征: (1, 0.5)
      # 正在处理特征: (2, 0.6)
      # 正在处理特征: (3, 0.7)
      # 正在处理特征: (4, 0.8)
  ```
- **`itertools.tee(iterable, n=2)`**：将单个迭代器拆分为 `n` 个独立的迭代器。一旦使用了 `tee`，就不应再使用原始迭代器。当您需要将相同的数据流发送到多个处理路径时，这很有用。请注意，如果一个分支比另一个分支更快地使用项，`tee` 需要存储中间值，如果迭代器差异很大，这可能会消耗内存。

  ```python
  import itertools

  data_stream = iter(range(5)) # 模拟数据流

  # 从原始流创建两个独立流
  stream1, stream2 = itertools.tee(data_stream, 2)

  # 处理 stream1（例如，计算均值）
  vals1 = list(stream1)
  mean_val = sum(vals1) / len(vals1) if vals1 else 0
  print(f"流 1 的值: {vals1}, 均值: {mean_val}")
  # 输出: 流 1 的值: [0, 1, 2, 3, 4], 均值: 2.0

  # 独立处理 stream2（例如，过滤偶数）
  even_vals = [x for x in stream2 if x % 2 == 0]
  print(f"流 2 的偶数值: {even_vals}")
  # 输出: 流 2 的偶数值: [0, 2, 4]
  ```

  > 使用 `itertools.tee` 复制迭代器，以用于不同的处理路径。

#### 切片与选择项

处理序列时，您通常只需要特定部分或符合特定条件的项。

- **`itertools.islice(iterable, start, stop[, step])`**：返回一个迭代器，它从输入的可迭代对象中生成选定的项，类似于列表切片但不复制。这对于从流中批量处理数据非常重要。

  ```python
  import itertools

  # 模拟一个大型数据集迭代器
  large_dataset = iter(range(1000))

  # 获取批次号 3（索引 20-29），批次大小为 10
  batch_size = 10
  batch_num = 3
  start_index = (batch_num - 1) * batch_size
  stop_index = batch_num * batch_size

  batch = list(itertools.islice(large_dataset, start_index, stop_index))
  print(f"批次 {batch_num}: {batch}")
  # 输出: 批次 3: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
  ```
- **`itertools.takewhile(predicate, iterable)`**：只要 `predicate` 函数返回真，就从 `iterable` 中返回元素。一旦 `predicate` 首次返回假，就完全停止迭代。
- **`itertools.dropwhile(predicate, iterable)`**：只要 `predicate` 为真，就跳过 `iterable` 中的元素，然后返回所有剩余元素。
- **`itertools.filterfalse(predicate, iterable)`**：返回 `iterable` 中 `predicate` 为假的元素。这是内置 `filter` 函数的补充。

  ```python
  import itertools

  # 模拟带有异常阈值的传感器读数
  sensor_readings = [1.2, 1.1, 1.3, 1.0, 5.6, 1.4, 1.5, 6.1, 0.9]
  threshold = 5.0

  # 获取低于阈值的初始读数
  initial_normal = list(itertools.takewhile(lambda x: x < threshold, sensor_readings))
  print(f"初始正常读数: {initial_normal}")
  # 输出: 初始正常读数: [1.2, 1.1, 1.3, 1.0]

  # 获取第一个异常点（包括该点）之后的读数
  post_anomaly = list(itertools.dropwhile(lambda x: x < threshold, sensor_readings))
  print(f"从第一个异常点开始的读数: {post_anomaly}")
  # 输出: 从第一个异常点开始的读数: [5.6, 1.4, 1.5, 6.1, 0.9]

  # 获取所有异常读数
  anomalies = list(filter(lambda x: x >= threshold, sensor_readings)) # 使用 filter
  print(f"所有异常点（使用 filter）: {anomalies}")
  # 输出: 所有异常点（使用 filter）: [5.6, 6.1]

  # 使用 filterfalse 获取所有非异常读数
  all_normal = list(itertools.filterfalse(lambda x: x >= threshold, sensor_readings))
  print(f"所有正常读数（使用 filterfalse）: {all_normal}")
  # 输出: 所有正常读数（使用 filterfalse）: [1.2, 1.1, 1.3, 1.0, 1.4, 1.5, 0.9]
  ```

#### 生成组合与笛卡尔积

在机器学习中，生成特征组合或超参数 (parameter) (hyperparameter)设置很常见。

- **`itertools.product(*iterables, repeat=1)`**：输入可迭代对象的笛卡尔积。等同于嵌套的 for 循环。对于生成超参数网格很有用。
- **`itertools.combinations(iterable, r)`**：从输入 `iterable` 中生成长度为 `r` 的元素子序列。顺序无关紧要，并且组合中的元素是唯一的。
- **`itertools.permutations(iterable, r=None)`**：生成长度为 `r` 的元素排列。顺序很重要。

  ```python
  import itertools

  # 超参数网格搜索空间
  learning_rates = [0.1, 0.01]
  batch_sizes = [32, 64]
  optimizers = ['adam', 'sgd']

  param_grid = itertools.product(learning_rates, batch_sizes, optimizers)

  print("超参数组合:")
  for lr, bs, opt in param_grid:
      print(f"  LR={lr}, BatchSize={bs}, Optimizer={opt}")
  # 输出:
  # 超参数组合:
  #   LR=0.1, BatchSize=32, Optimizer=adam
  #   LR=0.1, BatchSize=32, Optimizer=sgd
  #   LR=0.1, BatchSize=64, Optimizer=adam
  #   LR=0.1, BatchSize=64, Optimizer=sgd
  #   LR=0.01, BatchSize=32, Optimizer=adam
  #   LR=0.01, BatchSize=32, Optimizer=sgd
  #   LR=0.01, BatchSize=64, Optimizer=adam
  #   LR=0.01, BatchSize=64, Optimizer=sgd

  # 特征交互组合（顺序无关）
  features = ['A', 'B', 'C', 'D']
  feature_pairs = itertools.combinations(features, 2)
  print("\n用于交互项的特征对:")
  print(f"  {list(feature_pairs)}")
  # 输出:
  # 用于交互项的特征对:
  #   [('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'C'), ('B', 'D'), ('C', 'D')]
  ```

#### 其他实用工具

- **`itertools.accumulate(iterable[, func, *, initial=None])`**：返回累积和或二元函数的结果。可用于计算运行指标，例如强化学习 (reinforcement learning)中的累积奖励或累积特征值。

  ```python
  import itertools
  import operator

  rewards = [1, 0, 1, -1, 1, 1, 0]
  cumulative_rewards = list(itertools.accumulate(rewards))
  print(f"奖励: {rewards}")
  print(f"累积奖励: {cumulative_rewards}")
  # 输出:
  # 奖励: [1, 0, 1, -1, 1, 1, 0]
  # 累积奖励: [1, 1, 2, 1, 2, 3, 3]

  # 可以使用其他函数，例如，运行乘积
  values = [1, 2, 3, 4, 5]
  running_product = list(itertools.accumulate(values, operator.mul))
  print(f"\n数值: {values}")
  print(f"运行乘积: {running_product}")
  # 输出:
  # 数值: [1, 2, 3, 4, 5]
  # 运行乘积: [1, 2, 6, 24, 120]
  ```

  
  
  [交互图表：随时间变化的累积奖励](https://apxml.com/zh/courses/advanced-python-programming-ml/chapter-1-advanced-python-constructs-ml-pipelines/iterators-itertools-ml#plot-1ddylt2)
  
  <details>
  <summary>图表数据（JSON）</summary>
  
  ```json
  {
    "layout": {
      "title": "随时间变化的累积奖励",
      "xaxis": {
        "title": "时间步"
      },
      "yaxis": {
        "title": "累积奖励"
      },
      "template": "plotly_white",
      "margin": {
        "l": 40,
        "r": 20,
        "t": 40,
        "b": 40
      },
      "width": 600,
      "height": 350
    },
    "data": [
      {
        "x": [
          0,
          1,
          2,
          3,
          4,
          5,
          6
        ],
        "y": [
          1,
          1,
          2,
          1,
          2,
          3,
          3
        ],
        "type": "scatter",
        "mode": "lines+markers",
        "name": "累积奖励",
        "marker": {
          "color": "#1c7ed6"
        },
        "line": {
          "color": "#1c7ed6"
        }
      }
    ]
  }
  ```
  
  </details>
  
  

  > 使用 `itertools.accumulate` 计算的累积奖励总和。
- **`itertools.groupby(iterable, key=None)`**：将 `iterable` 中具有相同键的连续元素进行分组。为了使其按预期工作，`iterable` 需要按分组键排序。这对于处理分段数据（如时间序列事件）很有用。
- **`itertools.zip_longest(*iterables, fillvalue=None)`**：类似于内置的 `zip`，但会持续到最长的可迭代对象耗尽，用 `fillvalue` 填充缺失的值。这在配对可能不同长度的序列时很重要，例如某些特征可能缺失的特征向量 (vector)或带有缺失的时间序列数据。

  ```python
  import itertools

  features = [[1, 2, 3], [4, 5], [6, 7, 8, 9]] # 不规则特征
  labels = [0, 1, 0]

  # 配对特征和标签，填充较短的特征列表
  paired_data = itertools.zip_longest(features, labels, fillvalue=None)

  print("配对的特征（已填充）和标签:")
  for feat, lbl in paired_data:
      # 在此处，您可以在模型输入逻辑中处理填充 (None)
      print(f"  特征: {feat}, 标签: {lbl}")
  # 输出:
  # 配对的特征（已填充）和标签:
  #   特征: [1, 2, 3], 标签: 0
  #   特征: [4, 5], 标签: 1
  #   特征: [6, 7, 8, 9], 标签: 0 # 注意标签迭代器已耗尽

  # 使用特定值（例如，特征的 0）填充的示例
  feature_iter = iter([[1, 2, 3], [4, 5], [6, 7, 8, 9]])
  metadata_iter = iter(['A', 'B']) # 较短的元数据流

  aligned = itertools.zip_longest(feature_iter, metadata_iter, fillvalue='<PAD>')
  print("\n对齐特征和元数据:")
  print(list(aligned))
  # 输出:
  # 对齐特征和元数据:
  # [([1, 2, 3], 'A'), ([4, 5], 'B'), ([6, 7, 8, 9], '<PAD>')]
  ```

### 组合 `itertools` 用于复杂流程

`itertools` 真正的优势在于其可组合性。您可以将这些函数链接在一起，以创建表达力强且高效的数据处理流程，而无需中间列表。

考虑从时间序列流中创建重叠窗口以进行序列建模：

```python
import itertools
from collections import deque

def sliding_window(iterable, size):
    """在可迭代对象上创建滑动窗口（元组）的迭代器。"""
    it = iter(iterable)
    window = deque(itertools.islice(it, size), maxlen=size)
    if len(window) == size:
        yield tuple(window)
    for element in it:
        window.append(element)
        yield tuple(window)

# 模拟时间序列数据
time_series = range(10)
window_size = 3

windows = sliding_window(time_series, window_size)

print(f"大小为 {window_size} 的滑动窗口:")
for w in windows:
    print(f"  {w}")
# 输出:
# 大小为 3 的滑动窗口:
#   (0, 1, 2)
#   (1, 2, 3)
#   (2, 3, 4)
#   (3, 4, 5)
#   (4, 5, 6)
#   (5, 6, 7)
#   (6, 7, 8)
#   (7, 8, 9)
```

这个 `sliding_window` 函数使用 `iter`、`itertools.islice` 和 `collections.deque` 高效地生成窗口，而无需存储整个序列或冗余副本。

熟练使用迭代器和 `itertools` 提供了一种强有力的方法来处理机器学习 (machine learning)中常见的复杂数据序列。通过采用延迟计算并借助这些专业工具，您可以构建节省内存、高性能且出人意料地易读的数据流程。这是我们迈向性能优化和构建更高级机器学习组件的重要一步。

## 参考资料

- [itertools - Functions creating iterators for efficient looping](https://docs.python.org/3/library/itertools.html) — Python Software Foundation (2024)
  官方文档，详细介绍了 itertools 模块中的函数，提供了每个迭代器函数的语法、示例和行为。
- [Iterator Types](https://docs.python.org/3/library/stdtypes.html#iterator-types) — Python Software Foundation (2024)
  Publisher: Python Software Foundation
  官方文档，描述了 Python 的迭代器协议，包括可迭代对象、迭代器以及 `iter()` 和 `next()` 内置函数的定义。
- [Fluent Python: Clear, Concise, and Effective Programming](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056346/) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media; Pages: 1012
  一本关于高级 Python 特性的综合书籍，其中有专门的章节对迭代器、生成器和 itertools 模块进行了深入的解释和示例。

---

[上一节](04-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E4%B8%AD%E7%9A%84%E9%AB%98%E9%98%B6%E5%87%BD%E6%95%B0%E5%92%8C%E9%97%AD%E5%8C%85.md) · [下一节](06-%E5%AE%9E%E8%B7%B5%E7%8E%AF%E8%8A%82%EF%BC%9A%E6%9E%84%E5%BB%BA%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86%E7%AE%A1%E9%81%93%E7%BB%84%E4%BB%B6.md)
