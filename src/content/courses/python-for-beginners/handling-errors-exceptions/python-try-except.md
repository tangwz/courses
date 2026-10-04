---
course: "python-for-beginners"
chapter: "handling-errors-exceptions"
lesson: "python-try-except"
sourceId: 2383
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-9-handling-errors-exceptions/python-try-except"
title: "处理异常：try 和 except 块"
description: "了解基本的 `try...except` 结构，用于捕获潜在错误并执行备用代码。"
order: 3
plots: []
sourceHash: "35d6eef2b78a2dc1905d8d174aa53c87cb7af450d9038d6cac7ac8800a666949"
sourceCorrections: []
---

Python 程序在执行时可能会遇到错误，这些错误被称为异常。当异常发生且未被处理时，程序通常会停止运行并显示错误信息（一个追溯信息）。这种突然中止对用户或需要保持稳定的应用程序来说并不理想。

为了管理这些情况，Python 提供了 `try` 和 `except` 语句。这个结构允许您隔离可能引发异常的代码，并定义在异常发生时要采取的具体操作。

### 基本结构：try...except

其基本语法涉及两个代码块：

1. **`try` 块：** 您将可能引发异常的代码放在此块中。Python 会尝试正常执行这段代码。
2. **`except` 块：** 此块紧随 `try` 块之后。`except` 块中的代码*只有在*前面的 `try` 块中发生异常时才会执行。如果 `try` 块在没有发生任何异常的情况下完成，那么 `except` 块将完全跳过。

以下是代码示例：

```python
try:
    # 可能引发异常的代码
    # 例如，获取用户输入并进行转换
    age_str = input("请输入您的年龄：")
    age = int(age_str)
    print(f"明年您将是 {age + 1}。")

except:
    # 如果 try 块中发生*任何*异常，则执行此处的代码
    print("出错了。请输入一个有效的年龄数字。")

print("程序在 try-except 块之后继续执行。")
```

### 工作原理

我们来追踪上面示例的执行流程：

1. **开始：** Python 进入 `try` 块。
2. **执行 `try` 代码：** 它执行 `age_str = input(...)`。假设用户输入“30”。
3. **执行 `try` 代码：** 它执行 `age = int(age_str)`。这运行良好，将“30”转换为整数 `30`。
4. **执行 `try` 代码：** 它执行 `print(...)`。这也运行良好，打印“明年您将是 31。”。
5. **无异常：** 由于 `try` 块内没有发生异常，整个 `except` 块被跳过。
6. **继续：** 程序执行到 `try...except` 结构之后的 `print("程序继续执行...")` 行。

现在，考虑如果用户输入“thirty”而不是“30”会发生什么情况：

1. **开始：** Python 进入 `try` 块。
2. **执行 `try` 代码：** 它执行 `age_str = input(...)`。变量 `age_str` 现在的值是“thirty”。
3. **执行 `try` 代码：** 它尝试执行 `age = int(age_str)`。这会失败，因为“thirty”无法直接转换为整数。此时会引发 `ValueError` 异常。
4. **异常发生：** 由于发生了异常，Python 会立即停止执行 `try` 块内的任何后续代码（`print(f"明年...")` 行永远不会被执行到）。
5. **查找 `except`：** Python 会寻找 `try` 块后的 `except` 块。它找到了一个。
6. **执行 `except` 代码：** `except` 块内的代码被执行：`print("出错了...")`。
7. **继续：** `except` 块完成执行后，程序执行到 `try...except` 结构之后的 `print("程序继续执行...")` 行。

请注意其显著差异：程序没有因为 `ValueError` 追溯信息而崩溃，而是打印了一条有用的消息并继续运行。这就是异常处理的主要目的：捕获潜在错误并优雅地响应它们。

### 捕获所有异常的 `except`

上面的示例使用了裸 `except:` 子句。这充当了一个“全捕获”机制——它将处理 `try` 块中发生的*任何*类型的异常，无论是 `ValueError`、`ZeroDivisionError`、`TypeError` 还是完全不同的其他异常。

```python
numerator = 10
denominator = 0

try:
    result = numerator / denominator # 这将导致 ZeroDivisionError
    print("结果是：", result)
except:
    print("除法运算中发生错误。")

print("程序结束。")
```

尽管简单，但在大型程序中通常不推荐使用裸 `except:`。为什么呢？因为它可能捕获您未预料到的错误，潜在地隐藏 bug 或使其难以准确了解*具体*哪里出了问题。如果您的 `try` 块可能引发几种不同类型的错误，一个通用的 `except:` 可能会以相同的方式处理它们，这可能不合适。通常最好处理特定类型的异常，我们将在接下来进行讨论。

## 参考资料

- [Errors and Exceptions](https://docs.python.org/3/tutorial/errors.html) — Python Software Foundation (2024)
  Python 错误和异常处理机制的官方指南，包含 `try...except` 语句，提供了基础性理解。
- [Python Crash Course, 2nd Edition: A Hands-On, Project-Based Introduction to Programming](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH28VcemecjNifd1LiZ8b51naobRQLGdgH5t-WzDMiqw5zX4JVieQ5SgxwF723trYeIwMx6ihBgZZC7JEaq4EOb-0bbbTcZE1qiPdyjYD3W0Id7tgr1FmvRhDo_X91OwXo4UwK3lPu5kmNIqegapUmbuVdzw3BiBHzKsbtQIW_nTjMexuK0458iX4UUkdpQK976Nw==) — Eric Matthes (2019)
  Publisher: No Starch Press
  一本面向初学者的 Python 实用入门书，其中包含专门介绍使用 `try...except` 块处理异常的部分。
