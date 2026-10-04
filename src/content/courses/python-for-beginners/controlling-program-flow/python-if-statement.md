---
course: "python-for-beginners"
chapter: "controlling-program-flow"
lesson: "python-if-statement"
sourceId: 2289
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-3-controlling-program-flow/python-if-statement"
title: "做出判断：`if` 语句"
description: "了解如何使用 `if` 语句，根据特定条件是否为真来有条件地执行代码块。"
order: 1
plots: []
sourceHash: "7faaf68093e4b5c82b83b116ce62cf3c1595481fbb2b22c0cdfb55782d27d432"
sourceCorrections: []
---

通常，你希望代码的某部分只在特定情况下运行。就像在现实生活中你可能会决定“如果下雨，我就带伞”一样，程序也需要一种根据条件做出决定的方式。Python 用来做这件事的基本工具是 `if` 语句。

`if` 语句允许你*有条件地*执行一段代码块。它会检查某个条件是否为真。如果条件为真，与 `if` 语句关联的代码块就会运行。如果条件为假，该代码块将被完全跳过，程序会继续执行 `if` 块之后的下一条指令。

### `if` 语句的结构

基本语法如下：

```python
if condition:
    # 如果条件为真，则执行此处的代码
    statement1
    statement2
    # ... 更多语句 ...

# 此处的代码无论条件如何，都会在 if 块之后运行
next_statement
```

我们来分解一下：

1. **`if` 关键字：** 语句以关键字 `if` 开始。
2. **`condition`：** 这是一个求值为 `True` 或 `False` 的表达式。这通常被称为布尔表达式。你通常会在这里使用比较运算符（如 `==`、`!=`、`<`、`>`、`<=`、`>=`）或逻辑运算符（`and`、`or`、`not`），这些你在上一章中已经遇到。你也可以直接使用布尔变量。
3. **冒号 `:`：** 条件后必须跟一个冒号。这标志着依赖于该条件的代码块的开始。
4. **缩进块：** 紧跟在 `if condition:` 行之后、并向右进一步缩进的代码行构成 `if` 块。Python 使用缩进（通常是 4 个空格）来定义代码块。所有在 `if` 下方同一缩进级别的行都属于该块。第一行*没有*达到这个缩进级别的行标志着 `if` 块的结束。

### 运作方式：一个例子

想象你正在编写一个程序，检查用户是否达到投票年龄。投票年龄通常是 18 岁。

```python
age = 20

print("检查投票资格...")

if age >= 18:
    print("你已达到投票年龄。")
    print("如果你还没注册，请注册。")

print("资格检查完成。")
```

我们来追踪执行过程：

1. 变量 `age` 被赋值为 `20`。
2. 程序打印“检查投票资格...”。
3. `if` 语句检查条件 `age >= 18`。由于 `20 >= 18` 为 `True`，条件成立。
4. 因为条件为 `True`，缩进块被执行：
   - 打印“你已达到投票年龄。”
   - 打印“如果你还没注册，请注册。”
5. `if` 块执行完毕后，程序继续执行下一条没有缩进的行，打印“资格检查完成。”

那么，如果 `age` 不同呢？

```python
age = 15

print("检查投票资格...")

if age >= 18:
    # 该块将被跳过
    print("你已达到投票年龄。")
    print("如果你还没注册，请注册。")

print("资格检查完成。")
```

执行追踪：

1. `age` 被赋值为 `15`。
2. 打印“检查投票资格...”。
3. `if` 语句检查 `age >= 18`。由于 `15 >= 18` 为 `False`，条件不成立。
4. 因为条件为 `False`，`if` 下方的整个缩进块被跳过。
5. 程序直接跳到 `if` 块*之后*的下一行，并打印“资格检查完成。”

### 缩进的作用

在 Python 中，缩进不仅仅是为了可读性；它还是语法的一部分。它告诉解释器哪些语句属于 `if` 块。不正确的缩进会导致错误或意料之外的行为。

```python
# 正确的缩进
temperature = 30
if temperature > 25:
    print("今天天气很暖和。") # 缩进正确
    print("穿轻便的衣服。") # 缩进正确

# 不正确的缩进（将导致 IndentationError）
# if temperature > 25:
# print("今天天气很暖和。") # 缺少缩进
```

始终使用一致的缩进，通常每级缩进 4 个空格。大多数代码编辑器都会自动帮你处理这一点。

### 流程可视化

我们可以使用流程图来表示 `if` 语句的决策过程：

> 流程图展示了 `if` 语句的执行路径。如果条件求值为真，`if` 块代码在继续执行之前被运行。如果为假，`if` 块被跳过。

`if` 语句是将条件逻辑引入程序的最简单方法。它允许你的代码根据特定条件做出不同反应，让你的程序更具动态性和智能性。接下来你将看到，你可以在此基础上使用 `elif` 和 `else` 来处理其他条件。

## 参考资料

- [4. More Control Flow Tools](https://docs.python.org/3/tutorial/controlflow.html) — Python Development Team (2024)
  Publisher: Python Software Foundation
  官方Python教程中关于条件语句（包括“if”语句）的基础解释，涵盖了其语法和使用场景。
- [Python Crash Course, 3rd Edition: A Hands-On, Project-Based Introduction to Programming](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG5KyZjcXelZ21bb0eQckEw-IezIvaCrYflfQy5ZOkeHtgf5Elfe95S3I_O-T5eQWmfeaLL2xsf0h0QJj7vNQHIsgw6x3p9Oqo_e7Yhui9Wqt_x43Chb_NQaDf4F0h3rr2bF7mMqUHOyNaxs2FfopOYbWZ4PR0tdP89_70qU12mQ==) — Eric Matthes (2023)
  Publisher: No Starch Press; Pages: Chapter 5: If Statements
  一本广受欢迎的入门级书籍，用一个章节清晰地教授条件测试和“if”语句，并提供了示例。
