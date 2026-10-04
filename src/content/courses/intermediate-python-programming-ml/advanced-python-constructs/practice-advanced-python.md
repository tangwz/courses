---
course: "intermediate-python-programming-ml"
chapter: "advanced-python-constructs"
lesson: "practice-advanced-python"
sourceId: 2107
sourceUrl: "https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-1-advanced-python-constructs/practice-advanced-python"
title: "实践：应用高级Python技术"
description: "通过与数据科学相关的实际编码问题应用高级Python概念。"
order: 9
plots: []
sourceHash: "56a1639d533fc3a3bb1c1c92c84759c011b3a3d29ff9e38fa6eeb65cd5497fd2"
sourceCorrections: []
---

这些练习旨在帮助你巩固对列表推导、生成器、高级函数参数 (parameter)、装饰器、上下文 (context)管理器、基本面向对象编程以及与数据准备和分析相关的错误处理的掌握。完成这些示例将帮助你将这些技术融入到自己的数据科学工作流程中。

## 练习 1：使用推导式进行数据筛选

你有一个字典列表，每个字典代表一个传感器读数。每个读数包含'sensor\_id'、'timestamp'和'value'。你的目标是创建一个新列表，其中只包含'sensor\_id'为`S1`且'value'大于50的读数。为此，请使用列表推导式。

**数据：**

```python
sensor_data = [
    {'sensor_id': 'S1', 'timestamp': 1678886400, 'value': 45.6},
    {'sensor_id': 'S2', 'timestamp': 1678886401, 'value': 60.1},
    {'sensor_id': 'S1', 'timestamp': 1678886402, 'value': 55.9},
    {'sensor_id': 'S3', 'timestamp': 1678886403, 'value': 32.0},
    {'sensor_id': 'S1', 'timestamp': 1678886404, 'value': 62.3},
    {'sensor_id': 'S2', 'timestamp': 1678886405, 'value': 58.7},
    {'sensor_id': 'S1', 'timestamp': 1678886406, 'value': 49.8},
]
```

**解决方案：**

```python
# 使用列表推导式筛选数据
filtered_readings = [
    reading for reading in sensor_data
    if reading['sensor_id'] == 'S1' and reading['value'] > 50
]

# 打印结果进行验证
print(filtered_readings)
# 预期输出：
# [{'sensor_id': 'S1', 'timestamp': 1678886402, 'value': 55.9},
#  {'sensor_id': 'S1', 'timestamp': 1678886404, 'value': 62.3}]
```

这个解决方案以简洁易读的一行代码对列表进行了筛选。与带有`if`条件和`append`的传统`for`循环相比，列表推导式对于此类任务通常效率更高且更符合Python风格。

## 练习 2：使用生成器处理大型文件

假设你有一个非常大的日志文件，其中每一行代表一个事件。一次性处理整个文件可能会占用太多内存。编写一个生成器函数`get_error_lines(filepath)`，它接受文件路径，逐行读取文件，并仅返回包含“ERROR”一词的行。

**提示：** 在遍历文件行的循环中使用`yield`语句。

**解决方案：**

```python
import os # 用于创建虚拟文件

# 创建一个用于演示的虚拟日志文件
log_content = """
INFO: Process started
DEBUG: Connection established
ERROR: Failed to read record 102
INFO: Processing record 103
WARN: Disk space low
ERROR: Timeout connecting to service X
INFO: Process finished
"""
dummy_filepath = 'sample.log'
with open(dummy_filepath, 'w') as f:
    f.write(log_content)

# 生成器函数
def get_error_lines(filepath):
    """
    逐行读取文件，并返回包含“ERROR”的行。
    """
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if "ERROR" in line:
                    yield line.strip() # strip() 移除行首/行尾的空白字符
    except FileNotFoundError:
        print(f"错误：未在 {filepath} 找到文件")
        # 可选：不返回任何内容或抛出异常

# 使用生成器
error_lines_generator = get_error_lines(dummy_filepath)

print("找到的错误行：")
for error_line in error_lines_generator:
    print(error_line)

# 清理虚拟文件
os.remove(dummy_filepath)

# 预期输出：
# Error lines found:
# ERROR: Failed to read record 102
# ERROR: Timeout connecting to service X
```

生成器在这里非常适合，因为它们以惰性方式处理文件。在迭代过程中，内存中一次只保留一行，这使得这种方法适用于大型文件。`try-except`块也展示了文件不存在情况下的基本错误处理。

## 练习 3：灵活的数据聚合函数

编写一个函数`aggregate_data(agg_func, *args)`，它接受一个聚合函数（`agg_func`，例如`sum`、`max`、`min`）和可变数量的数值参数 (parameter)（`*args`）。该函数应将聚合函数应用于这些参数并返回结果。处理未提供数值参数的情况。

**解决方案：**

```python
def aggregate_data(agg_func, *args):
    """
    将聚合函数应用于可变数量的参数。

    参数：
        agg_func: 要应用的函数（例如，sum, max, min）。
        *args: 可变数量的数值参数。

    返回：
        聚合结果，如果未提供参数则返回 None。
    """
    if not args:
        print("警告：未提供用于聚合的数据。")
        return None
    # 检查所有参数是否为数值（整数或浮点数）
    if not all(isinstance(arg, (int, float)) for arg in args):
        print("错误：所有参数都必须是数值。")
        return None
    try:
        return agg_func(args)
    except Exception as e:
        print(f"聚合过程中出错：{e}")
        return None

# 示例用法
numbers = [10, 5, 25, 15, 8]

total = aggregate_data(sum, *numbers)
maximum = aggregate_data(max, *numbers)
minimum = aggregate_data(min, 1, 2, 3, 0.5) # 直接传递参数
no_data = aggregate_data(sum)
mixed_data = aggregate_data(sum, 10, 'a', 30)

print(f"Sum: {total}")
print(f"Max: {maximum}")
print(f"Min: {minimum}")
print(f"No Data Result: {no_data}")
print(f"Mixed Data Result: {mixed_data}")

# 预期输出：
# Sum: 63
# Max: 25
# Min: 0.5
# Warning: No data provided for aggregation.
# No Data Result: None
# Error: All arguments must be numerical.
# Mixed Data Result: None
```

此函数使用`*args`接受任意数量的位置参数，使其具有灵活性。它包含对空输入和非数值类型的检查，体现了基本的验证和错误反馈。

## 练习 4：使用装饰器计时数据操作

编写一个装饰器`time_it`，用于测量其所包装的任何函数的执行时间并打印时长。将此装饰器应用于一个模拟数据处理任务（例如，创建一个大型列表）的函数。

**解决方案：**

```python
import time

def time_it(func):
    """
    一个打印被包装函数执行时间的装饰器。
    """
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter() # 比 time.time() 更精确
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"函数 '{func.__name__}' 在 {duration:.4f} 秒内执行完毕")
        return result
    return wrapper

@time_it
def simulate_data_processing(n_records):
    """
    通过创建平方列表来模拟数据处理。
    """
    print(f"正在处理 {n_records} 条记录...")
    processed_data = [i*i for i in range(n_records)]
    # 模拟更多工作
    time.sleep(0.1)
    print("处理完成。")
    return len(processed_data) # 返回已处理项的数量

# 示例用法
num_records = 1_000_000
count = simulate_data_processing(num_records)
print(f"已处理 {count} 项。")

# 示例输出（具体时间会有所不同）：
# Processing 1000000 records...
# Processing complete.
# Function 'simulate_data_processing' executed in 0.1578 seconds
# Processed 1000000 items.
```

装饰器提供了一种简洁的方式，可以在不修改函数核心逻辑的情况下，为函数添加日志记录、计时或访问控制等横切关注点。`time_it`装饰器截取函数调用，记录调用前后的时间，打印时间差，然后返回原始函数的结果。

## 练习 5：使用上下文 (context)管理器进行资源管理

创建一个简单的上下文管理器类`TempFileHandler`，它在进入`with`块时创建一个临时文件，并在退出时自动删除该文件，即使块内发生错误也不例外。

**提示：** 实现`__enter__`和`__exit__`特殊方法。`__enter__`应创建并返回文件对象（或路径），而`__exit__`应处理清理工作。

**解决方案：**

```python
import os
import tempfile

class TempFileHandler:
    """
    一个用于创建和自动删除临时文件的上下文管理器。
    """
    def __init__(self, mode='w+t', suffix='.tmp', prefix='my_temp_'):
        self.mode = mode
        self.suffix = suffix
        self.prefix = prefix
        self.temp_file = None
        self.temp_filepath = ""

    def __enter__(self):
        # 创建一个命名的临时文件
        self.temp_file = tempfile.NamedTemporaryFile(
            mode=self.mode,
            suffix=self.suffix,
            prefix=self.prefix,
            delete=False # 我们在 __exit__ 中处理删除
        )
        self.temp_filepath = self.temp_file.name
        print(f"进入上下文：已创建临时文件 '{self.temp_filepath}'")
        return self.temp_file # 返回将在 'with' 块中使用的文件对象

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 此方法在退出 'with' 块时调用
        print(f"正在退出 '{self.temp_filepath}' 的上下文...")
        if self.temp_file:
            self.temp_file.close()
            try:
                os.remove(self.temp_filepath)
                print(f"成功删除临时文件 '{self.temp_filepath}'")
            except OSError as e:
                print(f"删除临时文件 '{self.temp_filepath}' 时出错：{e}")

        # 如果在 'with' 块内发生异常，exc_type、exc_val、
        # 和 exc_tb 将包含有关信息。
        if exc_type:
            print(f"发生了 {exc_type.__name__} 类型的异常：{exc_val}")
            # 返回 False（或不返回任何内容）以传播异常，
            # 返回 True 以抑制它。我们将让它传播。
            return False
        return True # 未发生异常或我们已处理

# 示例用法 1：成功操作
print("--- 示例 1：成功的文件操作 ---")
try:
    with TempFileHandler(suffix='.csv') as tmp_f:
        print(f"'with' 块内的文件对象：{tmp_f}")
        tmp_f.write("header1,header2\n")
        tmp_f.write("value1,value2\n")
        # 文件仍在此处打开并存在
        print(f"'with' 块期间文件存在：{os.path.exists(tmp_f.name)}")
    # 在 'with' 块之外：
    print("在 'with' 块之后。")
    # 检查文件是否存在（不应存在）
    print(f"'with' 块后文件存在：{os.path.exists(tmp_f.name)}") # 访问 tmp_f.name 是可以的
except Exception as e:
    print(f"捕获到意外错误：{e}")

print("\n--- 示例 2：带有错误的操作 ---")
try:
    with TempFileHandler(suffix='.log') as tmp_f:
        filepath = tmp_f.name # 在可能发生错误之前存储文件路径
        print(f"'with' 块内的文件对象：{tmp_f}")
        tmp_f.write("Log entry 1\n")
        # 模拟错误
        result = 10 / 0
        tmp_f.write("这将不会被写入\n")
except ZeroDivisionError:
    print("捕获到预期的 ZeroDivisionError。")
    # 检查文件是否存在（它应该已经被 __exit__ 删除）
    print(f"'with' 块中发生错误后文件是否存在：{os.path.exists(filepath)}")
except Exception as e:
    print(f"捕获到意外错误：{e}")

# 预期输出：
# --- Example 1: Successful file operation ---
# Entering context: Created temporary file '.../my_temp_....csv'
# File object inside 'with': <_io.TextIOWrapper name='.../my_temp_....csv' mode='w+t' encoding='...'>
# File exists during 'with' block: True
# Exiting context for '.../my_temp_....csv'...
# Successfully deleted temporary file '.../my_temp_....csv'
# After 'with' block.
# File exists after 'with' block: False
#
# --- Example 2: Operation with an error ---
# Entering context: Created temporary file '.../my_temp_....log'
# File object inside 'with': <_io.TextIOWrapper name='.../my_temp_....log' mode='w+t' encoding='...'>
# Exiting context for '.../my_temp_....log'...
# An exception of type ZeroDivisionError occurred: division by zero
# Successfully deleted temporary file '.../my_temp_....log'
# Caught expected ZeroDivisionError.
# File exists after error in 'with' block: False
```

上下文管理器确保清理代码（如关闭文件、释放锁或删除临时资源）可靠运行。`__exit__`方法接收有关任何发生异常的详细信息，从而可以进行条件性清理或错误日志记录。

这些练习提供了实际场景，其中本章讨论的高级Python结构是有益的。当你遇到数据加载、清理和转换任务时，请思考这些技术如何使你的代码更高效、可读且易于维护。

## 参考资料

- [The Python Language Reference](https://docs.python.org/3/reference/index.html) — Guido van Rossum and the Python development team (2025)
  Publisher: Python Software Foundation
  Python语言的官方权威文档，提供所有核心功能的详细规范，包括列表推导式、生成器、函数参数、装饰器和上下文管理器。
- [Fluent Python: Clear, Concise, and Effective Programming, 2nd Edition](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056345/) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media
  一本强烈推荐给中高级Python开发者的书籍，深入探讨了Python的对象模型、数据结构、迭代器、生成器、装饰器和上下文管理器，并强调了Pythonic编程风格。
- [Learning Python, 5th Edition](https://www.oreilly.com/library/view/learning-python-5th/9781449355739/) — Mark Lutz (2013)
  Publisher: O'Reilly Media
  一本涵盖Python语言从基础到高级功能的综合指南，包括对函数、模块、类、异常和装饰器的详细讨论，适合进行扎实的学习。
