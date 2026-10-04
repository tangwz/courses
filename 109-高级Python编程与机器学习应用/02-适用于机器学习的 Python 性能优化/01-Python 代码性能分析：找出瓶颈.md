# Python 代码性能分析：找出瓶颈

来源：[原文](https://apxml.com/zh/courses/advanced-python-programming-ml/chapter-2-python-performance-optimization-ml/profiling-python-code)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在尝试任何优化之前，你首先需要明白你的程序时间花在*哪里*。直觉常常不准；你认为慢的代码部分可能完全没问题，而那些看起来没问题的代码行可能是真正的性能消耗点。性能分析提供所需的数据，有效指引你的优化工作，确保你专注于代码中能带来最大改进的部分。正如章节介绍中提到的，找到这些瓶颈是迈向更快、更可扩展机器学习 (machine learning)应用的重要第一步。

### 什么是性能分析？

性能分析是对程序执行情况进行分析的过程，目的是确定代码的不同部分花费了多少时间（有时也包括内存或其他资源）。对于性能优化，我们主要关注*时间分析*：找出哪些函数、方法，甚至具体的代码行消耗了最多的执行时间。

核心是衡量。我们不靠猜测，而是使用工具在实际条件下观察程序的行为。这种数据驱动的方法避免了浪费精力去优化那些对整体性能影响微乎其微的代码部分。帕累托法则在软件性能方面常常适用：大约80%的执行时间经常只花费在20%的代码中。性能分析帮助你找到那重要的20%。

### Python 生态系统中的性能分析工具

Python 提供了一些内置和第三方工具用于性能分析。我们将重点介绍用于找出机器学习 (machine learning)任务中典型的 CPU 密集型瓶颈的常用工具。

#### 使用 `timeit` 衡量小段代码时间

为了衡量非常小段代码的执行时间，Python 内置的 `timeit` 模块很方便。它会多次运行代码片段，以最大限度地减少外部因素（如系统上运行的其他进程）的影响，并提供更稳定的执行时间估计。

你可以直接从命令行使用 `timeit`：

```bash
# 比较使用循环和列表推导式创建列表
python -m timeit -s "data = range(1000)" "result = []" "for x in data: result.append(x*x)"
python -m timeit -s "data = range(1000)" "result = [x*x for x in data]"
```

或者在脚本中以编程方式使用：

```python
import timeit

setup_code = "import numpy as np; data = np.random.rand(100)"
# 示例 1: 简单循环
stmt1 = """
total = 0
for x in data:
    total += x
"""
# 示例 2: NumPy 内置的求和
stmt2 = "np.sum(data)"

# 执行每个语句 10000 次
time1 = timeit.timeit(stmt=stmt1, setup=setup_code, number=10000)
time2 = timeit.timeit(stmt=stmt2, setup=setup_code, number=10000)

print(f"Loop sum time: {time1:.6f} seconds")
print(f"NumPy sum time: {time2:.6f} seconds")
```

虽然 `timeit` 非常适合微基准测试和比较简洁的替代方案，但它不适合对复杂应用或大型工作流中的整个函数进行性能分析。

#### 使用 `cProfile` 进行函数级性能分析

为了宏观地查看时间在不同函数之间如何分配，Python 内置的 `cProfile` 模块是标准工具。它记录每次函数调用并衡量每个函数*内部*花费的时间（不包括其*调用*的函数所花费的时间，即 `tottime`），以及函数中花费的*总*时间（包括所有子调用，即 `cumtime`）。

**运行 `cProfile`：**

你可以从命令行对整个脚本运行 `cProfile`，并将结果保存到文件以便后续分析：

```bash
python -m cProfile -o my_program.prof my_ml_script.py
```

另外，你也可以对代码中的特定函数进行性能分析：

```python
import cProfile
import pstats
import io
import numpy as np

def calculate_distances(points):
    """低效地计算点对距离。"""
    n = len(points)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((points[i] - points[j])**2))
            distances[i, j] = dist
            distances[j, i] = dist # 对称性
    return distances

def main_task():
    # 生成一些示例数据
    data_points = np.random.rand(50, 3) # 50个三维点
    # 对距离计算进行性能分析
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = calculate_distances(data_points)
    
    profiler.disable()
    
    # 打印统计信息
    s = io.StringIO()
    # 按累计时间排序统计信息
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative') 
    ps.print_stats(10) # 打印前10个函数
    print(s.getvalue())

if __name__ == "__main__":
    main_task()
```

**解读 `cProfile` 输出：**

输出通常如下（简化版）：

```
         1253 function calls in 0.015 seconds

   Ordered by: cumulative time
   List reduced from ... to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.015    0.015 script.py:22(main_task)
        1    0.002    0.002    0.015    0.015 script.py:7(calculate_distances)
     1225    0.012    0.000    0.012    0.000 {method 'reduce' of 'numpy.ufunc' objects}
        ... etc ...
```

- `ncalls`：函数被调用的次数。
- `tottime`：函数自身（不包括子调用）花费的总时间。高 `tottime` 表明函数自身正在进行大量工作。
- `percall`：`tottime` 除以 `ncalls`。
- `cumtime`：此函数以及所有被其调用的函数花费的累计时间。高 `cumtime` 通常指向调用栈中的重要函数，即使它们自身的 `tottime` 很低。
- `percall`：`cumtime` 除以 `ncalls`。
- `filename:lineno(function)`：函数标识符。

把注意力放在 `cumtime` 和 `tottime` 都高的函数上。高 `cumtime` 表明一个函数正在组织大量工作（或调用了慢速的子函数），而高 `tottime` 则表示函数自身的代码计算开销大。

**可视化 `cProfile` 数据：**

对于复杂的程序来说，原始文本输出可能难以理解。像 `snakeviz` 这样的工具可以从 `.prof` 文件创建交互式可视化，使得理解调用链和时间分配变得容易得多。

```bash
# 首先，安装 snakeviz
pip install snakeviz

# 然后在你的性能分析输出文件上运行它
snakeviz my_program.prof 
```

这通常会打开一个网页浏览器视图，显示哪些函数调用了其他函数以及时间是如何累积的。

> 示例 `cProfile` 累计时间细分以柱状图形式可视化。高的柱状图表示函数消耗了更多的总执行时间，包括在子调用中花费的时间。



[交互图表：cProfile 累计时间示例 (前 5)](https://apxml.com/zh/courses/advanced-python-programming-ml/chapter-2-python-performance-optimization-ml/profiling-python-code#plot-j6h1t4)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "cProfile 累计时间示例 (前 5)",
    "xaxis": {
      "title": "函数"
    },
    "yaxis": {
      "title": "累计时间 (秒)"
    },
    "bargap": 0.2
  },
  "data": [
    {
      "type": "bar",
      "x": [
        "calculate_distances",
        "numpy.sum",
        "numpy.sqrt",
        "numpy.subtract",
        "main_task"
      ],
      "y": [
        0.015,
        0.01,
        0.008,
        0.007,
        0.015
      ],
      "marker": {
        "color": [
          "#f03e3e",
          "#fa5252",
          "#ff6b6b",
          "#ff8787",
          "#ffa8a8"
        ]
      }
    }
  ]
}
```

</details>



#### 使用 `line_profiler` 进行逐行性能分析

有时，`cProfile` 会告诉你*哪个*函数很慢，但函数可能很长，你需要知道里面*具体哪些行*是罪魁祸首。这时就可以使用 `line_profiler`。它会衡量你指定函数中每行代码的执行时间。

**安装：** `line_profiler` 是一个第三方包。

```bash
pip install line_profiler
```

**用法：**

1. **添加装饰器：** 为你想要逐行分析的函数添加 `@profile` 装饰器。注意：此装饰器默认不活跃；它只是 `kernprof` 工具的一个标记 (token)。

   ```python
   import numpy as np
   # 注意：这里不需要导入 line_profiler，
   # 装饰器只是一个标记。

   @profile 
   def calculate_distances_line_prof(points):
       """低效地计算点对距离。"""
       n = len(points)
       distances = np.zeros((n, n)) # 第 1 行
       for i in range(n):           # 第 2 行
           for j in range(i + 1, n): # 第 3 行
               diff = points[i] - points[j] # 第 4 行
               sq_diff = diff**2          # 第 5 行
               sum_sq_diff = np.sum(sq_diff) # 第 6 行
               dist = np.sqrt(sum_sq_diff)   # 第 7 行
               distances[i, j] = dist      # 第 8 行
               distances[j, i] = dist      # 第 9 行
       return distances

   # ... (脚本其余部分调用函数) ...
   if __name__ == "__main__":
       data_points = np.random.rand(50, 3)
       calculate_distances_line_prof(data_points)
   ```
2. **使用 `kernprof` 运行：** 使用 `kernprof` 命令行工具（它随 `line_profiler` 一起提供）执行你的脚本。`-l` 标志表示执行行级性能分析，`-v` 表示运行后立即查看结果。

   ```bash
   kernprof -l -v your_script_containing_profile_decorator.py 
   ```

**解读 `line_profiler` 输出：**

输出显示了被装饰函数中每行的计时信息：

```
Timer unit: 1e-06 s

Total time: 0.028532 s
File: your_script_containing_profile_decorator.py
Function: calculate_distances_line_prof at line 6

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
     6                                           @profile
     7                                           def calculate_distances_line_prof(points):
     8                                               """Inefficiently calculates pairwise distances."""
     9         1         25.0     25.0      0.1      n = len(points)
    10         1        135.0    135.0      0.5      distances = np.zeros((n, n))
    11        51         65.0      1.3      0.2      for i in range(n):
    12      1225       1450.0      1.2      5.1          for j in range(i + 1, n):
    13      1225       3480.0      2.8     12.2              diff = points[i] - points[j]
    14      1225       3150.0      2.6     11.0              sq_diff = diff**2
    15      1225      11550.0      9.4     40.5              sum_sq_diff = np.sum(sq_diff)
    16      1225       6352.0      5.2     22.3              dist = np.sqrt(sum_sq_diff)
    17      1225        850.0      0.7      3.0              distances[i, j] = dist
    18      1225        475.0      0.4      1.7              distances[j, i] = dist
    19         1          0.0      0.0      0.0      return distances 
```

- `Line #`：文件中的行号。
- `Hits`：该行被执行的次数。
- `Time`：执行该行所花费的总时间（以计时器单位，通常是微秒）。
- `Per Hit`：每次执行的平均时间（`Time` / `Hits`）。
- `% Time`：函数内总时间中该行所占的百分比。**这通常是最重要的一列。** 寻找 `% Time` 高的行。
- `Line Contents`：实际代码。

在此示例中，内循环中的第 15 行 (`np.sum`) 和第 16 行 (`np.sqrt`) 显然消耗了大部分时间（分别为 40.5% 和 22.3%）。这种详细视图立即将优化工作指向那些特定的计算或包含它们的循环。

### 性能分析策略与注意事项

- **分析实际工作负载：** 始终使用与你的生产环境紧密相似的数据和条件来分析你的代码。性能特点会随着数据大小或分布而发生显著变化。
- **性能分析开销：** 性能分析工具，特别是 `line_profiler`，会引入开销。衡量行为可能会稍微减慢你的代码。请记住这一点，但所获得的见解通常远超衡量成本。
- **迭代过程：** 性能分析并非一次性任务。典型的工作流程是：分析 -> 找到瓶颈 -> 优化 -> 再次分析。确认你的更改确实提高了性能并且没有引入新的瓶颈。
- **专注于热点：** 不要试图优化所有地方。把精力集中在那些被认为是消耗时间最主要部分的函数和代码行上。与优化主要瓶颈相比，在很少执行的代码中获得的小幅提升通常不值得付出努力。

通过系统地使用 `cProfile` 和 `line_profiler` 等工具，你将用数据取代猜测，从而精准地定位你的优化工作。在应用后续章节中讨论的 NumPy、Pandas、Cython 和 Numba 的具体优化技术之前，这一根本步骤很重要。

## 参考资料

- [The Python Standard Library: Performance Measurement](https://docs.python.org/3/library/profile.html) — Python Software Foundation (2024)
  针对 Python 内置性能分析器（`cProfile`、`pstats`）的官方文档，解释了它们的使用方法和输出。
- [The Python Standard Library: \`timeit\` - Measure execution time of small code snippets](https://docs.python.org/3/library/timeit.html) — Python Software Foundation (2024)
  针对 Python `timeit` 模块的官方文档，详细介绍了其在微基准测试中的应用。
- [\`line_profiler\` GitHub Repository](https://github.com/pyutils/line_profiler) — pyutils (2024)
  `line_profiler` 的官方存储库，提供了安装说明、使用示例和逐行代码分析的详细文档。
- [\`snakeviz\` GitHub Repository](https://github.com/jiffyclub/snakeviz) — jiffyclub (2024)
  `snakeviz` 的官方存储库，一个用于可视化 Python `cProfile` 统计数据的工具，有助于更轻松地识别性能瓶颈。
- [High Performance Python: Practical Performant Programming for Scientists and Engineers](https://www.oreilly.com/library/view/high-performance-python/9781492055013/) — Micha Gorelick and Ian Ozsvald (2020)
  Publisher: O'Reilly Media
  一本优化 Python 代码的综合指南，其中包含有关性能分析工具和科学与机器学习应用策略的专门章节。

---

[上一节](../01-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%B5%81%E6%B0%B4%E7%BA%BF%E4%B8%AD%E7%9A%84%E9%AB%98%E7%BA%A7Python%E7%BB%93%E6%9E%84/06-%E5%AE%9E%E8%B7%B5%E7%8E%AF%E8%8A%82%EF%BC%9A%E6%9E%84%E5%BB%BA%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86%E7%AE%A1%E9%81%93%E7%BB%84%E4%BB%B6.md) · [下一节](02-%E4%BC%98%E5%8C%96%20NumPy%20%E6%93%8D%E4%BD%9C.md)
