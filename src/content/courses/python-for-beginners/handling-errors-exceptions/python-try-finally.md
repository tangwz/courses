---
course: "python-for-beginners"
chapter: "handling-errors-exceptions"
lesson: "python-try-finally"
sourceId: 2386
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-9-handling-errors-exceptions/python-try-finally"
title: "`finally` 块：清理操作"
description: "理解 `finally` 块，它包含的代码无论是否发生异常都将始终执行，对于清理任务非常有用。"
order: 6
plots: []
sourceHash: "99d9d036b9a15b3b0663bfe43e6630ed4d854711e5e7508ba05e0a6a3e9244d3"
sourceCorrections: []
---

有时，无论是否发生错误，您都绝对需要执行某些操作。比如关闭已打开的文件、释放网络连接或清理临时资源。这些清理操作对于防止资源泄漏或数据损坏很重要。Python 提供了 `finally` 块，正是为此：它保证了代码的执行。

`finally` 子句被添加到 `try...except` 结构中。无论 `try` 和 `except` 块中发生什么，`finally` 块中的代码都将始终运行。

- 如果 `try` 块在没有发生任何异常的情况下完成，`finally` 块将在 `try` 块（如果存在 `else` 块，则在 `else` 块之后）之后运行。
- 如果 `try` 块中发生异常并被 `except` 块捕获，`finally` 块将在 `except` 块完成之后运行。
- 如果 `try` 块中发生异常但 *未* 被任何 `except` 块捕获，`finally` 块仍会在 Python 将异常沿调用栈传播之前运行（这通常会导致程序停止并显示错误消息）。
- 即使 `try` 或 `except` 块执行 `return`、`break` 或 `continue` 语句，`finally` 块也将在控制权实际移交出去之前执行。

### 语法

您可以将 `finally` 单独与 `try` 使用，或者与 `except` 和可选的 `else` 块一起使用：

```python
# 示例 1: try...finally
try:
   # 可能需要清理的代码
   print("尝试执行某些操作...")
   # result = 10 / 0 # 取消注释以模拟错误
finally:
   print("这个 finally 块总是执行清理操作。")

# 示例 2: try...except...finally
try:
   print("\n尝试文件操作...")
   file = open("my_temporary_file.txt", "w")
   file.write("写入一些数据。\n")
   # 在处理过程中模拟错误
   # int("not a number")
   print("文件写入成功（可能）。")
except ValueError:
   print("捕获到 ValueError！")
finally:
   print("正在执行 finally 块。")
   # 确保文件已关闭，即使发生了错误
   if 'file' in locals() and not file.closed:
      file.close()
      print("文件已关闭。")
   else:
       print("文件对象不存在或已关闭。")
```

### 为何使用 `finally`？

主要的用例是资源管理。当你获取资源（比如打开文件）时，通常需要在之后释放它。如果在获取资源之后但在释放之前发生错误，你的程序可能会导致资源泄漏。将释放代码放在 `finally` 块中可以保证它运行。

考虑上面的文件示例。我们打开 `my_temporary_file.txt` 进行写入。

1. **无错误：** `try` 块成功完成。`finally` 块运行，关闭文件。
2. **捕获的错误：** 如果我们取消注释 `int("not a number")`，将会发生 `ValueError`。`except ValueError` 块会运行。之后，`finally` 块 *仍然* 会运行，关闭文件。
3. **未捕获的错误：** 如果我们取消注释 `result = 10 / 0`，将会发生 `ZeroDivisionError`。因为没有 `except ZeroDivisionError` 块，所以异常未被捕获。然而，`finally` 块 *仍然* 会执行（关闭文件），然后程序才会因未处理的 `ZeroDivisionError` 而终止。

### `finally` 与 `with` 语句对比

您可能会注意到，文件关闭的示例似乎与使用 `with` 语句类似，我们在文件处理章节中已经看过：

```python
try:
   with open("another_temp_file.txt", "w") as file:
      print("\n尝试使用 'with' 执行文件操作...")
      file.write("使用 with 语句写入数据。\n")
      # 模拟错误
      # result = 1 / 0
      print("文件写入成功（使用 'with'）。")
   print("已退出 'with' 块。")
except ZeroDivisionError:
    print("在 'with' 外部捕获到 ZeroDivisionError。")

print("代码在 'with' 块或异常处理后继续执行。")
```

`with` 语句实质上是常见 `try...finally` 清理模式的语法糖，特别是对于那些具有特定设置 (`__enter__`) 和清理 (`__exit__`) 方法的对象，例如文件对象。当你使用 `with open(...)` 时，Python 会自动确保在块退出时调用文件的 `close()` 方法，无论是正常退出还是因为异常。对于文件处理以及上下文 (context)管理器支持的类似资源管理情况，通常优先使用 `with` 语句，因为它比手动编写 `try...finally` 块更简洁，且不易出错。

然而，`finally` 块仍然是保证执行的基本构造，在 `with` 语句不适用或需要比简单资源释放更复杂的清理逻辑的情况下。它是提供错误处理和资源管理所需执行保证的底层机制。

## 参考资料

- [The \`try\` statement](https://docs.python.org/3/reference/compound_stmts.html#the-try-statement) — Python Software Foundation (2024)
  Publisher: Python Software Foundation
  Python 官方语言参考，详细说明了 `try` 语句的语法和行为，包括 `except`、`else` 和 `finally` 子句。
- [Fluent Python, 2nd Edition](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056322/) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media
  讲解了 Python 的异常处理机制和资源管理模式，包括对上下文管理器（`with` 语句）的讨论，作为 `try...finally` 在特定场景下的惯用替代方案。
- [Python Cookbook, 3rd Edition](https://www.oreilly.com/library/view/python-cookbook-3rd/9781449357993/) — David Beazley, Brian K. Jones (2013)
  Publisher: O'Reilly Media
  提供了使用 `try...finally` 和上下文管理器在 Python 中进行健壮错误处理和有效资源管理的实用技巧和示例。
