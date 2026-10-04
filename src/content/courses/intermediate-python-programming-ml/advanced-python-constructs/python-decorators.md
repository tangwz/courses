---
course: "intermediate-python-programming-ml"
chapter: "advanced-python-constructs"
lesson: "python-decorators"
sourceId: 2093
sourceUrl: "https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-1-advanced-python-constructs/python-decorators"
title: "代码重用性中的装饰器"
description: "学习如何在 Python 中使用装饰器来修改或加强函数和方法。"
order: 5
plots: []
sourceHash: "8baa952b4c0bd161eee7532d12ff74e5683d3cfa8c58ebebf58724bc20637a99"
sourceCorrections: []
---

装饰器提供了一种强大且Python风格的方式来修改或加强函数和方法。它们允许您在现有代码周围添加额外功能，而无需永久改变原始函数的定义。这有助于提升代码重用性和关注点分离，这在构建数据处理管道或机器学习 (machine learning)工作流时是非常有益的做法。

本质上，装饰器是一个可调用对象（通常是函数），它接受另一个函数作为输入并返回一个新函数。直接放置在函数定义上方的 `@decorator_name` 语法是一种语法糖，简化了此过程。

来看一下这个基本结构：

```python
import functools

def my_decorator(func):
    @functools.wraps(func) # 保留原始函数元数据
    def wrapper(*args, **kwargs):
        # 在原始函数运行之前执行的代码
        print(f"在调用 {func.__name__} 之前发生了一些事情。")
        result = func(*args, **kwargs) # 调用原始函数
        # 在原始函数运行之后执行的代码
        print(f"在 {func.__name__} 完成之后发生了一些事情。")
        return result
    return wrapper

@my_decorator
def say_hello(name):
    """问候用户。"""
    print(f"你好, {name}!")

# 调用被装饰函数
say_hello("Data Scientist")

# 输出:
# 在调用 say_hello 之前发生了一些事情。
# 你好, Data Scientist!
# 在 say_hello 完成之后发生了一些事情。
```

在这里，`my_decorator` 是装饰器函数。它定义了一个内部函数 `wrapper`，其中包含额外逻辑。`wrapper` 函数会调用传递给装饰器的原始函数 (`func`)。`say_hello` 上方的 `@my_decorator` 语法等同于在函数定义之后编写 `say_hello = my_decorator(say_hello)`。

注意在装饰器内部使用 `@functools.wraps(func)`。这是一个辅助装饰器，它通过复制 `__name__`、`__doc__` 和参数 (parameter)签名等属性，使 `wrapper` 函数看起来像原始函数 (`func`)。如果没有 `@functools.wraps`，内省工具（以及其他潜在代码）将看到 `wrapper` 函数的信息，而不是 `say_hello` 函数的信息。

### 装饰器工作原理图解

您可以将装饰器看作是在原始函数周围应用了一个层：

> 装饰器 (`my_decorator`) 定义了一个 `wrapper`。当被装饰函数 (`say_hello`) 被调用时，`wrapper` 执行，在调用原始函数 (`func`) 之前和之后运行代码。

### 数据科学中的常见应用场景

装饰器对于添加与数据分析和机器学习 (machine learning)任务相关的跨领域功能特别有用：

1. **函数执行计时：** 衡量特定数据处理步骤或计算的耗时对于优化很重要。

   ```python
   import time
   import functools
   import pandas as pd
   import numpy as np

   def timer(func):
       @functools.wraps(func)
       def wrapper_timer(*args, **kwargs):
           start_time = time.perf_counter() # 比 time.time() 更精确
           value = func(*args, **kwargs)
           end_time = time.perf_counter()
           run_time = end_time - start_time
           print(f"' {func.__name__!r} ' 完成于 {run_time:.4f} 秒")
           return value
       return wrapper_timer

   @timer
   def simulate_data_processing(rows=1000000):
       """模拟一个可能耗时的数据操作。"""
       df = pd.DataFrame(np.random.rand(rows, 5), columns=list('ABCDE'))
       # 模拟一些计算
       result = df['A'] * np.sin(df['B']) - df['C'] * np.cos(df['D'])
       time.sleep(0.5) # 模拟 I/O 或其他延迟
       return result.mean()

   mean_value = simulate_data_processing(rows=500000)
   print(f"平均结果: {mean_value}")

   # 示例输出:
   # 'simulate_data_processing' 完成于 0.6123 秒
   # 平均结果: 0.001...
   ```
2. **日志记录：** 追踪函数调用、参数 (parameter)或结果对于调试复杂管道非常有帮助。

   ```python
   import logging
   import functools

   logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

   def logger(func):
       @functools.wraps(func)
       def wrapper_logger(*args, **kwargs):
           logging.info(f"调用 {func.__name__}，参数: {args}，关键字参数: {kwargs}")
           try:
               result = func(*args, **kwargs)
               logging.info(f"{func.__name__} 返回: {type(result)}")
               return result
           except Exception as e:
               logging.error(f" {func.__name__} 中发生异常: {e}", exc_info=True)
               raise # 记录后重新引发异常
       return wrapper_logger

   @logger
   def load_data(filepath):
       """加载数据，可能引发错误。"""
       if not filepath.endswith(".csv"):
           raise ValueError("文件类型无效，预期为 .csv")
       # 模拟加载数据
       print(f"从 {filepath} 加载数据...")
       return pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]}) # 虚拟 DataFrame

   try:
       df = load_data("my_data.csv")
       # df_error = load_data("my_data.txt") # 取消注释以查看错误日志
   except ValueError as e:
       print(f"捕获到预期错误: {e}")

   # 示例日志输出:
   # 2023-10-27 10:30:00,123 - INFO - 调用 load_data，参数: ('my_data.csv',)，关键字参数: {}
   # 从 my_data.csv 加载数据...
   # 2023-10-27 10:30:00,124 - INFO - load_data 返回: <class 'pandas.core.frame.DataFrame'>
   # (如果错误情况取消注释)
   # 2023-10-27 10:30:00,125 - INFO - 调用 load_data，参数: ('my_data.txt',)，关键字参数: {}
   # 2023-10-27 10:30:00,126 - ERROR - load_data 中发生异常: 文件类型无效，预期为 .csv
   # Traceback (most recent call last): ...
   # 捕获到预期错误: 文件类型无效，预期为 .csv
   ```
3. **输入验证：** 在继续之前，确保函数接收的数据格式正确（例如，具有特定列的 DataFrame）。

   ```python
   import functools
   import pandas as pd

   def requires_columns(required_cols):
       def decorator(func):
           @functools.wraps(func)
           def wrapper_validator(*args, **kwargs):
               # 假设 DataFrame 是第一个位置参数
               if args and isinstance(args[0], pd.DataFrame):
                   df = args[0]
                   missing_cols = set(required_cols) - set(df.columns)
                   if missing_cols:
                       raise ValueError(f" {func.__name__} 的 DataFrame 中缺少所需列: {missing_cols}")
               else:
                   # 可以为 kwargs 或其他位置添加更复杂的检查
                   pass # 或者如果未在预期位置找到 DF 则引发错误
               return func(*args, **kwargs)
           return wrapper_validator
       return decorator

   @requires_columns(['feature1', 'target'])
   def process_features(df):
       """处理 DataFrame 中的特定特征。"""
       print("正在处理特征...")
       # 实际处理逻辑在此处
       return df['feature1'] * 2

   data_ok = pd.DataFrame({'feature1': [1, 2, 3], 'target': [0, 1, 0], 'extra': [5, 6, 7]})
   data_bad = pd.DataFrame({'feature_typo': [1, 2, 3], 'target': [0, 1, 0]})

   result = process_features(data_ok) # 正常运行

   try:
       process_features(data_bad) # 引发 ValueError
   except ValueError as e:
       print(f"验证失败: {e}")

   # 输出:
   # 正在处理特征...
   # 验证失败: process_features 的 DataFrame 中缺少所需列: {'feature1'}
   ```

   此示例还展示了**带参数的装饰器**。`requires_columns` 是一个工厂函数，它接受所需列列表并*返回*实际的装饰器函数。这允许定制装饰器的行为。
4. **记忆化（缓存）：** 存储计算开销大的函数调用的结果，并在相同输入再次出现时返回缓存结果。Python 的 `functools` 模块为此提供了 `lru_cache`（最近最少使用缓存）。

   ```python
   import functools
   import time

   @functools.lru_cache(maxsize=None) # None 表示缓存大小无限制
   def expensive_calculation(a, b):
       """模拟一个开销大的计算。"""
       print(f"正在对 ({a}, {b}) 执行开销大的计算...")
       time.sleep(1) # 模拟工作
       return a + b * b

   print(expensive_calculation(2, 3)) # 运行计算
   print(expensive_calculation(5, 2)) # 运行计算
   print(expensive_calculation(2, 3)) # 立即返回缓存结果
   print(expensive_calculation(5, 2)) # 立即返回缓存结果

   # 输出:
   # 正在对 (2, 3) 执行开销大的计算...
   # 11
   # 正在对 (5, 2) 执行开销大的计算...
   # 9
   # 11
   # 9
   ```

   尽管 NumPy 和 Pandas 操作通常在内部高度优化，但 `lru_cache` 对于您的工作流中重复对相同输入执行大量计算的自定义 Python 函数很有益处。

### 装饰器堆叠

您可以将多个装饰器应用于单个函数。它们在语法上从下到上应用，但执行时从上到下（最外层包装器首先运行）。

```python
@timer
@logger
# @requires_columns(['input']) # 示例：添加验证
def complex_step(data):
    # ... 处理逻辑 ...
    print("正在执行复杂步骤...")
    time.sleep(0.2)
    return "Done"

complex_step("Some Input Data")

# 日志输出将显示计时器在日志消息前后开始/结束。
# 执行顺序：计时器包装器 -> 日志记录器包装器 -> 原始 complex_step
```

装饰器是一种灵活的工具，可以在不使核心逻辑变得混乱的情况下，为您的函数添加日志记录、计时、验证或缓存等行为。熟练使用它们能帮助你编写更模块化、可重用和易于维护的 Python 代码，这在工作流可能变得复杂的数据科学和机器学习 (machine learning)项目中非常有优势。

## 参考资料

- [functools - Higher-order functions and operations on callable objects](https://docs.python.org/3/library/functools.html) — Python Software Foundation (2023)
  functools 模块的官方文档，包含 @functools.wraps 和 @functools.lru_cache 等核心装饰器实用工具。
- [Fluent Python, 2nd Edition](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056375/) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media
  全面涵盖了 Python 装饰器，包括其实现细节、使用场景以及与闭包和一流函数的关系。
- [Python Cookbook, 3rd Edition](https://books.google.com/books/about/Python_Cookbook_3rd_Edition.html?id=L8YqBAAAQBAJ) — David Beazley and Brian K. Jones (2013)
  Publisher: O'Reilly Media
  提供了实现自定义装饰器和理解高级函数包装技术的实用示例和代码方案。
