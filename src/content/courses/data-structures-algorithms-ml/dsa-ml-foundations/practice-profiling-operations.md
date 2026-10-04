---
course: "data-structures-algorithms-ml"
chapter: "dsa-ml-foundations"
lesson: "practice-profiling-operations"
sourceId: 5247
sourceUrl: "https://apxml.com/zh/courses/data-structures-algorithms-ml/chapter-1-dsa-ml-foundations/practice-profiling-operations"
title: "实践：分析基本数据操作"
description: "动手实践测量不同Python数据结构上常见操作的性能。"
order: 7
plots: ["plots/5247-0.json"]
sourceHash: "fb55b785cac266b3e1fe103537e10a3b7e8c1e64013f4c505859614b3b0a08c8"
sourceCorrections: []
---

理论和复杂度分析提供了重要的方向，但实际观察性能差异更能加深理解。在本节中，我们将实际测量所讨论数据结构上常见操作的执行时间。这种称为性能分析的做法，有助于感受理论复杂度如何影响执行速度，尤其是在机器学习 (machine learning)任务中。

Python内置的`timeit`模块是准确测量小型代码片段时间的常用工具。它会多次运行代码，以减少系统负载等外部因素的干扰，从而提供比简单起始/结束时间检查更可靠的测量结果。

### 使用`timeit`测量执行时间

`timeit`模块提供了一种简单的方法来测量Python代码的时间。它的主要功能`timeit.timeit()`接受要执行的代码片段、设置代码（如导入或数据创建）以及运行片段的次数。

```python
import timeit

# 例子：测量列表添加操作的时间
setup_code = "my_list = list(range(100))"
code_to_time = "my_list.append(100)"

# 执行 code_to_time 100万次并返回总时间
execution_time = timeit.timeit(stmt=code_to_time, setup=setup_code, number=1000000)

print(f"1,000,000次追加操作的时间： {execution_time:.4f} seconds")
```

现在，我们将其应用于比较不同的数据结构。

### 实验1：元素查找（列表与集合）

查找元素是一项基本操作。我们预计列表（可能需要顺序扫描元素）与使用哈希进行更快查找的集合或字典会有不同的表现。让我们测试在大结构中查找一个存在的元素。

**设置：** 创建一个包含从0到`n-1`的整数的大型列表和集合。我们将查找元素`n-1`。

```python
import timeit
import random

# 数据规模
n = 100000

# 设置代码创建列表和集合
setup_code = f"""
import random
data_list = list(range({n}))
data_set = set(data_list)
search_element = {n-1}
# 确保元素实际存在
# random.shuffle(data_list) # 打乱列表会增加最坏情况的可能性
"""

# 测量列表搜索时间的代码（最坏情况：接近末尾）
list_search_code = "search_element in data_list"

# 测量集合搜索时间的代码
set_search_code = "search_element in data_set"

# 计时重复次数
num_executions = 1000

# 测量操作时间
list_time = timeit.timeit(stmt=list_search_code, setup=setup_code, number=num_executions)
set_time = timeit.timeit(stmt=set_search_code, setup=setup_code, number=num_executions)

print(f"列表搜索时间（运行{num_executions}次）： {list_time:.6f} seconds")
print(f"集合搜索时间（运行{num_executions}次）： {set_time:.6f} seconds")

if set_time > 0: # 如果计时极快，避免除以零
    print(f"集合查找大约快{list_time / set_time:.1f}倍。")
else:
    print("集合查找显著更快（已达到计时精度限制）。")
```

**分析：**
您应该会注意到，搜索列表比搜索集合耗时得多。这与我们对复杂度的认知一致。列表搜索（使用`in`）的平均时间复杂度为$O(n)$，因为平均而言，它可能需要检查一半的元素。在最坏情况下（例如搜索最后一个元素或不存在的元素），它会检查所有$n$个元素。集合搜索使用哈希表，平均时间复杂度为$O(1)$，这意味着所需时间大致固定，与集合大小无关。测量的时间提供了这种差异的实际依据。

### 实验2：数值运算（Python列表与NumPy数组）

机器学习 (machine learning)大量涉及数值计算。NumPy在此方面旨在提高效率。我们来比较使用Python循环给列表中的每个元素添加一个常量值，与使用NumPy的向量 (vector)化操作之间的差异。

**设置：** 创建一个大型数字列表和一个对应的NumPy数组。

```python
import timeit
import numpy as np

# 数据规模
n = 1000000
constant_value = 5

# 设置代码创建列表和NumPy数组
setup_code = f"""
import numpy as np
py_list = list(range({n}))
np_array = np.arange({n})
constant = {constant_value}
"""

# 测量使用循环进行列表加法的时间（列表推导式）
list_op_code = "[x + constant for x in py_list]"

# 测量NumPy向量化加法的时间
numpy_op_code = "np_array + constant"

# 计时重复次数
num_executions = 100

# 测量操作时间
list_time = timeit.timeit(stmt=list_op_code, setup=setup_code, number=num_executions)
numpy_time = timeit.timeit(stmt=numpy_op_code, setup=setup_code, number=num_executions)

print(f"Python列表操作时间（运行{num_executions}次）： {list_time:.6f} seconds")
print(f"NumPy数组操作时间（运行{num_executions}次）： {numpy_time:.6f} seconds")

if numpy_time > 0:
    print(f"NumPy操作大约快{list_time / numpy_time:.1f}倍。")
else:
    print("NumPy操作显著更快。")
```

**分析：**
NumPy操作会快得多。原因何在？Python列表需要使用解释型Python代码逐个元素迭代，每次加法都会产生可观的额外开销。另一方面，NumPy数组使用高度优化的预编译C代码进行操作，这些代码在连续的内存块上运行。这个过程称为向量化 (quantization)，它避免了Python循环开销，并利用了底层硬件优化，从而在机器学习中常见的数值任务中实现显著加速。

### 实验3：数据聚合（Pandas Series与Python循环）

数据聚合，如计算总和或平均值，在数据准备中很常见。Pandas为这些任务提供了优化的方法。我们来比较使用Pandas Series的`.sum()`方法与手动迭代求和元素之间的差异。

**设置：** 创建一个Pandas Series。

```python
import timeit
import pandas as pd
import numpy as np # 通常与pandas一起使用

# 数据规模
n = 1000000

# 设置代码创建Pandas Series
setup_code = f"""
import pandas as pd
import numpy as np
pd_series = pd.Series(np.arange({n}))
total = 0 # 用于循环版本
"""

# 测量使用Python循环手动求和的时间
loop_sum_code = """
total = 0
for x in pd_series:
    total += x
"""

# 测量Pandas内置sum方法的时间
pandas_sum_code = "pd_series.sum()"

# 计时重复次数
num_executions = 50

# 测量操作时间
loop_time = timeit.timeit(stmt=loop_sum_code, setup=setup_code, number=num_executions)
pandas_time = timeit.timeit(stmt=pandas_sum_code, setup=setup_code, number=num_executions)

print(f"手动循环求和时间（运行{num_executions}次）： {loop_time:.6f} seconds")
print(f"Pandas .sum()方法时间（运行{num_executions}次）： {pandas_time:.6f} seconds")

if pandas_time > 0:
    print(f"Pandas .sum()方法大约快{loop_time / pandas_time:.1f}倍。")
else:
     print("Pandas .sum()方法显著更快。")
```

**分析：**
与NumPy示例类似，Pandas的`.sum()`方法将明显优于手动Python循环。Pandas方法使用优化的C或Cython代码实现，在底层数据（通常是NumPy数组）上高效运行。使用标准Python `for`循环遍历Pandas Series通常效率不高，因为它涉及获取每个元素的额外开销。为了提高性能，通常优先使用Pandas的内置聚合和操作函数。

### 性能差异可视化

有时，图表有助于更清楚地显示性能差距。让我们可视化实验1中的查找时间比较。



![查找时间比较（元素存在）](plots/5247-0.json)



> 在大型数据集（N=100,000）中，查找列表中现有元素与集合中现有元素的平均时间。请注意y轴上的对数刻度，它显示了显著的差异。实际时间取决于硬件和具体的Python版本。

这些简单的实验表明了一个重要事实：数据结构的选择极大地影响性能。虽然Python内置的列表和字典用途广泛，但NumPy和Pandas等专业库提供了高度优化的结构和操作，这对于高效的机器学习 (machine learning)工作流程非常重要。通过分析和实际测试来把握这些权衡，有助于开发更快、更具扩展性的机器学习应用。

## 参考资料

- [timeit - Measure execution time of small code snippets](https://docs.python.org/3/library/timeit.html) — Python Documentation (2024)
  使用Python内置工具进行精确代码计时和微基准测试的重要指南，如本节所示。
- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html) — NumPy Developers (2023)
  解释了NumPy高性能背后的原理，包括向量化和数组导向计算，以实现机器学习中高效的数值操作。
- [User Guide for Pandas](https://pandas.pydata.org/docs/user_guide/index.html) — Pandas Development Team (2024)
  深入了解Pandas如何通过向量化操作优化数据处理和聚合，解释了其相对于纯Python循环的性能优势。
- [Introduction to Algorithms](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition) — Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein (2022)
  Publisher: MIT Press
  一本经典的教材，涵盖了基本数据结构和算法，包括详细的复杂度分析，为性能分析奠定了理论基础。
