# Python 代码性能分析

来源：[原文](https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-6-efficient-maintainable-python-ml/python-code-profiling)

[返回章节目录](README.md) · [返回课程目录](../README.md)

编写能够正常运行的 Python 代码是第一步。然而，在机器学习 (machine learning)中，尤其是在处理大型数据集或计算密集型算法时，*性能*成为一项主要考量。运行缓慢的代码可能会拖慢你的整个工作流程，延迟实验并增加计算成本。猜测代码大部分时间花在哪里通常是不准确的。性能分析提供了一种客观方法来衡量代码执行，并发现其中的性能瓶颈。

性能分析是对程序执行的系统性分析，旨在确定代码的不同部分消耗了多少时间或其他资源（如内存）。通过了解时间都花在了哪里，你可以有效地集中优化工作，而不是仅凭直觉进行更改。

### 使用 `cProfile` 进行函数级分析

Python 的标准库包含一个强大的内置分析器，名为 `cProfile`。它提供确定性时间信息，这意味着它测量执行代码所花费的实际 CPU 时间，使结果可重复（不同于采样执行的统计分析器）。`cProfile` 追踪每个函数内部花费的时间以及每个函数被调用的次数。

你可以轻松地从命令行或直接在你的脚本中调用 `cProfile`。

**示例：从命令行分析脚本：**

```bash
python -m cProfile -s cumulative your_script.py
```

`-s cumulative` 选项根据每个函数中累积花费的时间对输出进行排序，有助于快速发现最耗时的调用堆栈。

**示例：分析代码中的特定函数：**

```python
import cProfile
import pstats
import io
import numpy as np

def expensive_data_processing(data):
    """模拟一些数据处理步骤。"""
    # 步骤 1：元素级操作（相对较快）
    processed_data = np.log(data + 1)

    # 步骤 2：模拟一个较慢的逐行操作
    result = []
    for row in processed_data:
        # 模拟每行的复杂计算
        row_sum = np.sum(np.sin(row) * np.cos(row))
        result.append(row_sum * np.mean(row))
    return np.array(result)

# 生成一些样本数据
sample_data = np.random.rand(500, 100)

# 创建一个性能分析器对象
profiler = cProfile.Profile()

# 在性能分析器的控制下运行函数
profiler.enable()
processed_result = expensive_data_processing(sample_data)
profiler.disable()

# 分析结果
s = io.StringIO()
# 按累积时间 ('cumtime') 排序统计信息
sortby = pstats.SortKey.CUMULATIVE
ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
ps.print_stats()

print(s.getvalue())

# 你也可以限制输出，例如，打印前 10 个函数
# ps.print_stats(10)
```

**`cProfile` 输出解释：**

输出通常看起来像这样（简化）：

```
 ncalls  tottime  percall  cumtime  percall 文件名:行号(函数)
      1    0.001    0.001    0.520    0.520 <string>:1(<module>)
      1    0.015    0.015    0.519    0.519 script.py:5(expensive_data_processing)
    500    0.450    0.001    0.480    0.001 script.py:12(<listcomp> or loop body)
    500    0.010    0.000    0.010    0.000 {method 'reduce' of 'numpy.ufunc' objects} (sum)
      1    0.002    0.002    0.002    0.002 {method 'log' of 'numpy.ufunc' objects}
    ... 更多行 ...
```

- `ncalls`：函数被调用的次数。
- `tottime`：此函数内部花费的总时间（不包括此函数调用的其他函数所花费的时间）。这有助于找出本质上较慢的函数。
- `percall`：`tottime` 除以 `ncalls`。
- `cumtime`：此函数及其调用的所有函数中累积花费的时间。这有助于发现启动长时间运行操作的高级函数。
- `percall`：`cumtime` 除以 `ncalls`。
- `filename:lineno(function)`：函数标识符。

在上面的例子中，`expensive_data_processing` 具有较高的 `cumtime`，但其 `tottime` 相对较低。与循环体（`script.py:12`）相关联的高 `tottime` 表明循环本身是主要的性能瓶颈。

虽然 `cProfile` 非常适合查看函数调用的全貌，但它不会告诉你一个慢函数中是*哪一行*导致了延迟。

### 使用 `line_profiler` 精确找到瓶颈

对于更细粒度的视图，第三方 `line_profiler` 包非常有用。它测量你指定函数中每行代码的执行时间。

**安装：**

```bash
pip install line_profiler
```

**用法：**

1. **装饰：** 将 `@profile` 装饰器添加到要分析的函数。注意：这个装饰器不是内置的；它被 `kernprof` 命令识别。你不需要为装饰器本身导入任何东西，除非你的代码检查工具报错，在这种情况下可能需要一个占位符。
2. **运行：** 使用 `kernprof` 命令执行你的脚本，该命令随 `line_profiler` 一起提供。

```python
# script_with_line_profiler.py
import numpy as np
# No import needed for @profile unless needed for linting/IDE
# If needed: try: from line_profiler import profile except ImportError: def profile(func): return func

@profile
def expensive_data_processing_line(data):
    """模拟一些数据处理步骤。"""
    # 步骤 1：元素级操作
    processed_data = np.log(data + 1) # line 9

    # 步骤 2：模拟一个较慢的逐行操作
    result = [] # line 12
    for row in processed_data: # line 13
        # 模拟每行的复杂计算
        row_sum = np.sum(np.sin(row) * np.cos(row)) # line 15
        result.append(row_sum * np.mean(row)) # line 16
    return np.array(result) # line 17

# 生成一些样本数据
sample_data = np.random.rand(500, 100)

# 正常调用函数
processed_result = expensive_data_processing_line(sample_data)
print("处理完成。")
```

**从命令行运行：**

```bash
kernprof -l -v script_with_line_profiler.py
```

- `-l`：告诉 `kernprof` 注入 `@profile` 装饰器并运行逐行性能分析。
- `-v`：告诉 `kernprof` 在脚本完成后立即显示计时结果。

**`line_profiler` 输出解释：**

输出提供装饰函数中每行的计时信息：

```
计时单位: 1e-06 秒

总时间: 0.584321 秒
文件: script_with_line_profiler.py
函数: expensive_data_processing_line 在行 7

行号      命中         时间  每次命中   % 时间  行内容
==============================================================
     7                                           @profile
     8                                           def expensive_data_processing_line(data):
     9                                               """模拟一些数据处理步骤。"""
    10         1       2105.0   2105.0      0.4      processed_data = np.log(data + 1) # line 9
    11
    12         1          1.0      1.0      0.0      result = [] # line 12
    13       501        380.0      0.8      0.1      for row in processed_data: # line 13
    14                                               # 模拟每行的复杂计算
    15       500     450120.0    900.2     77.0          row_sum = np.sum(np.sin(row) * np.cos(row)) # line 15
    16       500     131710.0    263.4     22.5          result.append(row_sum * np.mean(row)) # line 16
    17         1          5.0      5.0      0.0      return np.array(result) # line 17

处理完成。
```

- `Line #`：文件中的行号。
- `Hits`：该行被执行的次数。
- `Time`：执行该行所花费的总时间（以计时单位计，通常是微秒）。
- `Per Hit`：每次执行的平均时间（`Time / Hits`）。
- `% Time`：函数内部在该行上花费的总时间的百分比。这通常是最重要的列。
- `Line Contents`：实际的代码。

在此输出中，循环内的第 15 行和第 16 行消耗了函数执行时间中的绝大部分（77.0% 和 22.5%）。这明确地指导优化工作针对这些特定的计算，也许可以通过寻找向量 (vector)化的 NumPy 替代方案而不是逐行迭代来实现。

### 内存分析

虽然优化速度很常见，但过度的内存使用也可能损害你的机器学习 (machine learning)应用程序，特别是在处理可能无法完全载入 RAM 的大数据集时。`memory_profiler` 包与 `line_profiler` 类似，但追踪内存消耗。

**安装：**

```bash
pip install memory_profiler
# 可能还需要 psutil: pip install psutil
```

**用法：**

添加 `@profile` 装饰器（与 `line_profiler` 相同，但追踪内存）并使用 Python 的 `-m` 标志运行：

```python
# script_with_memory_profiler.py
import numpy as np
# memory_profiler 需要明确导入
from memory_profiler import profile

@profile
def memory_intensive_task(size):
    """创建大型中间结构。"""
    print(f"正在创建初始数组 ({size}x{size})...")
    initial_data = np.ones((size, size)) # line 9
    print("正在创建中间副本...")
    intermediate_copy = initial_data * 2 # line 11
    print("正在计算最终结果...")
    final_result = np.sqrt(intermediate_copy) # line 13
    print("任务完成。")
    return final_result # line 15

result = memory_intensive_task(2000) # 使用适中大小
```

**从命令行运行：**

```bash
python -m memory_profiler script_with_memory_profiler.py
```

输出将显示执行每行*之前*的内存使用量、该行导致的内存*增量*以及该行的内容。这有助于发现分配大量内存的行。

### 性能分析策略

1. **从宏观入手：** 首先使用 `cProfile` 来大致了解哪些高级函数消耗的时间最多（`cumtime`）。
2. **精准定位：** 对 `cProfile` 确定为耗时的函数使用 `line_profiler`，以精确指出导致延迟的具体行（`% Time`）。
3. **考虑内存：** 如果怀疑内存使用存在问题，对相关函数使用 `memory_profiler`。
4. **优化目标区域：** 仅将优化工作（例如，向量 (vector)化、算法更改、缓存）集中在通过性能分析发现的瓶颈上。避免优化耗时可忽略不计的代码。
5. **再次分析：** 更改后再次进行性能分析，以确认瓶颈已解决，并且没有无意中创建新的瓶颈。

"性能分析是一个迭代过程。它提供所需数据，以便就代码优化投入时间做出明智决策，从而带来更快、更高效的机器学习 (machine learning)工作流程。请记住，目标不仅仅是功能正常的代码，更是能在数据和计算需求下表现良好的代码。"

## 参考资料

- [cProfile - Profile Python code](https://docs.python.org/3/library/profile.html) — Python Software Foundation (2023)
  Publisher: Python Software Foundation
  Python内置分析器（`cProfile`）及其统计模块（`pstats`）的官方文档，为性能分析提供基础信息。
- [line_profiler: Line-by-line profiling for Python](https://github.com/pyutils/line_profiler) — Robert Kern (2024)
  `line_profiler` 项目的存储库和使用指南，详细说明如何在Python函数中逐行测量执行时间。
- [High Performance Python: Practical Performant Programming for Scientists and Engineers](https://www.oreilly.com/library/view/high-performance-python-2nd/9781492055015/) — Micha Gorelick and Ian Ozsvald (2020)
  Publisher: O'Reilly Media
  一份优化Python代码的实用指南，涵盖多种性能技术，包括分析、数据结构和高效计算方法。

---

[上一节](04-%E8%99%9A%E6%8B%9F%E7%8E%AF%E5%A2%83%E7%AE%80%E4%BB%8B.md) · [下一节](06-%E4%BC%98%E5%8C%96%20NumPy%20%E5%92%8C%20Pandas%20%E7%9A%84%E6%96%B9%E6%B3%95.md)
