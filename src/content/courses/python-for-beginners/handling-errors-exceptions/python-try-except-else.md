---
course: "python-for-beginners"
chapter: "handling-errors-exceptions"
lesson: "python-try-except-else"
sourceId: 2385
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-9-handling-errors-exceptions/python-try-except-else"
title: "异常处理中的else块"
description: "了解如何在`try...except`中使用可选的`else`块，以便仅当`try`块中未引发任何异常时才运行代码。"
order: 5
plots: []
sourceHash: "90577a82549865c997d0508671bff52f196decf46935f2f172aa7ae15753a359"
sourceCorrections: []
---

有时，你希望只有当`try`块执行完毕且未引发任何异常时，才运行特定的代码块。将这段代码直接放在`try`块内部并非总是理想选择，因为如果*那段*代码引发了异常，它可能会被旨在处理原始操作的`except`块捕获。同样，将其放在整个`try...except`结构*之后*意味着即使捕获并处理了异常，它也会运行。

这就是可选的`else`块在`try...except`结构中发挥作用的地方。

### 带有`else`的结构

你可以在所有`except`子句之后添加一个`else`子句。`else`块中的代码当且仅当`try`块在没有引发异常的情况下完成时才会执行。

以下是通用语法：

```python
try:
    # 可能引发异常的代码
    # 示例：尝试可能失败的操作
    print("正在尝试操作...")
    result = 10 / 2 # 这会成功
    # result = 10 / 0 # 这会引发 ZeroDivisionError
    print("操作可能会成功。") # 只有当上面的操作没有立即失败时，此行才会运行

except ZeroDivisionError:
    # 处理特定 ZeroDivisionError 的代码
    print("错误：不能除以零！")

except TypeError:
    # 处理潜在 TypeError 的代码
    print("错误：操作的类型无效！")

else:
    # 此块仅在 try 块成功完成时运行
    # （即，没有引发异常）
    print("操作成功！没有发生异常。")
    print(f"结果是：{result}") # 在这里使用 'result' 是安全的

print("执行在 try/except/else 块之后继续。")
```

### 为何使用`else`？

使用`else`块的主要好处是提高了清晰度并减少了歧义。

1. **职责分离：** 它将可能引发异常的代码（在`try`块中）与只有在潜在风险代码成功完成后才应运行的代码（在`else`块中）清晰地分开。这使得每个块的用途更加明确。
2. **避免意外捕获：** 考虑将依赖于成功的代码直接放在`try`块的末尾。如果*那段*代码（成功代码）本身引发了与你某个`except`子句匹配的异常，它就会被捕获，这可能不是你所期望的。`else`块避免了这个问题，因为它超出了`except`子句直接监控初始`try`操作的范围。

### 示例：文件处理

让我们回顾一下文件处理。一种常见模式是尝试打开文件，处理文件不存在等潜在错误，并且*如果*文件成功打开，则继续读取或处理它。

```python
file_path = 'my_data.txt' # 假设此文件存在以演示成功情况
# file_path = 'non_existent_file.txt' # 使用此文件测试异常情况

try:
    print(f"尝试打开 '{file_path}'...")
    # 对于文件，通常更推荐使用 'with' 语句，
    # 但我们这里使用显式的 open/close 来展示 else。
    f = open(file_path, 'r')
    print("文件成功打开。") # 如果 open() 成功，则此行运行

except FileNotFoundError:
    print(f"错误：未找到文件 '{file_path}'。")

except PermissionError:
    print(f"错误：没有权限读取 '{file_path}'。")

else:
    # 只有在文件无错误打开时，此块才运行。
    print("正在处理文件内容...")
    content = f.read()
    print("文件内容已读取。")
    f.close() # 记住关闭文件
    print("文件已关闭。")
    # 如果需要，进一步处理内容
    # print(f"内容：\n{content[:50]}...") # 显示前 50 个字符

print("\n文件处理部分已完成。")
```

在此示例中：

- `try`块尝试`open()`操作。
- `except`块处理特定的文件相关错误（`FileNotFoundError`，`PermissionError`）。
- `else`块包含读取、处理和关闭文件的代码。此代码仅在`try`块中的`open()`调用成功时运行。如果`f.read()`或`f.close()`*本身*引发了意料之外的异常，它们将不会被其上方的`FileNotFoundError`或`PermissionError`处理程序捕获。

使用`else`块使你的错误处理逻辑更清晰、更精确，仅当“成功”路径实际无错误地采用时，才执行为此路径专门指定的代码。

## 参考资料

- [The try statement](https://docs.python.org/3/reference/compound_stmts.html#the-try-statement) — Python Software Foundation (2023)
  官方语言参考，解释了 try-except-else-finally 语句的语法和行为。
- [Handling Exceptions](https://docs.python.org/3/tutorial/errors.html#handling-exceptions) — Python Software Foundation (2023)
  Python 官方教程的一部分，提供了异常处理的入门解释，包括 else 子句的用法。
- [Fluent Python, 2nd Edition](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056340/) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media
  一本关于编写惯用 Python 代码的指南，涵盖了错误处理和 else 子句有效使用等高级特性。
- [Learning Python, 5th Edition](https://learning.oreilly.com/library/view/learning-python-5th/9781449355739/) — Mark Lutz (2013)
  Publisher: O'Reilly Media; Pages: 701-702
  一本全面的参考书，详细介绍了 try-except-else 语句及其在控制执行流中的作用。
