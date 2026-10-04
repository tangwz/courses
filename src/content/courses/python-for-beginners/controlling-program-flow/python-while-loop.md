---
course: "python-for-beginners"
chapter: "controlling-program-flow"
lesson: "python-while-loop"
sourceId: 2295
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-3-controlling-program-flow/python-while-loop"
title: "重复动作：while 循环"
description: "学习如何使用 while 循环，只要特定条件保持为真，就重复执行代码块。"
order: 3
plots: []
sourceHash: "0b89735e51a8e7b9f9b77b6a67ef92463b7139af8469cd34686b970e7d105bfd"
sourceCorrections: []
---

有时，您需要一段代码只要某个条件保持为真就重复运行。Python 的 `while` 循环恰好提供了此功能。当您不提前知道需要重复操作多少次时，此功能特别有用。

可以这样理解：您告诉 Python：“只要此条件成立，就一直这样做。”

### `while` 循环的结构

`while` 循环的基本语法如下所示：

```python
while condition:
    # 将重复执行的代码块
    # 此代码块必须缩进
    statement1
    statement2
    # ... 可能会更新影响条件的变量 ...
```

我们来分析一下：

1. **`while` 关键词**：这表示循环的开始。
2. **`condition`**：这是一个布尔表达式（计算结果为 `True` 或 `False` 的表达式），就像您在 `if` 语句中使用的条件一样。例如：`count < 5`、`user_input != 'quit'`、`is_valid == False`。
3. **冒号 `:`**：表示 `while` 语句头部的结束。
4. **缩进代码块**：这是循环体。所有在 `while` 行下方缩进的语句都属于此循环。此代码块会重复执行。非常重要的一点是，此代码块中必须有内容最终使 `condition` 变为 `False`，否则循环将无限运行！

### Python 如何执行 `while` 循环

`while` 循环的执行流程很简单：

1. **检查条件**：Python 首先判断 `condition`。
2. **执行或跳过**：
   - 如果 `condition` 为 `True`，Python 会执行整个缩进的代码块。
   - 如果 `condition` 为 `False`，Python 会完全跳过缩进的代码块，并从循环 *之后* 的第一条语句继续执行。
3. **重复**：如果代码块被执行（因为条件为 `True`），Python 会回到第 1 步并重新判断 `condition`。

这种检查-执行-重复的循环会一直持续，直到 `condition` 的判断结果为 `False`。

> 流程图说明了 `while` 循环的执行逻辑。

### 示例：一个简单的计数器

我们创建一个循环，打印从 1 开始到（但不包括）5 的数字。

```python
# 在循环前初始化一个计数器变量
count = 1

# 只要 'count' 小于 5，循环就会继续
while count < 5:
    print(f"Current count is: {count}")
    # 重要：在循环内更新计数器！
    count = count + 1 # 或者使用简写：count += 1

print("循环结束！")
```

输出：

```text
Current count is: 1
Current count is: 2
Current count is: 3
Current count is: 4
Loop finished!
```

注意控制此循环的三个重要部分：

- **初始化**：`count = 1` 在循环开始前设置了起始状态。
- **条件**：`count < 5` 在每次可能的迭代前都会被检查。
- **更新**：`count = count + 1` 修改循环 *内部* 用于条件判断的变量。如果没有此更新，`count` 将始终为 1，`count < 5` 将始终为 `True`，循环将永远不会结束。

### 危险：无限循环

如果 `while` 循环中的条件 *永不* 变为 `False`，循环将永远运行。这被称为**无限循环**。这是常见的初学者错误，通常是由于忘记在循环内部包含最终将条件变为 `False` 的逻辑。

请看这个例子（除非您知道如何停止它，否则不要运行！）：

```python
# 警告：这将创建一个无限循环！
counter = 0
while counter >= 0:
    print(f"Counter is {counter}. Still positive!")
    # 糟糕！我们忘记了以某种方式更改 'counter'
    # 以使条件 (counter >= 0) 变为假。
    # 也许我们是想递减它？或者检查不同的条件？
    # 如果不进行更改，它将永远运行。
    counter = counter + 1 # 这实际上使其 *更* 有可能保持 >= 0！
```

如果您不小心在终端或交互式解释器中运行了带有无限循环的代码，通常可以通过按 **Ctrl+C** 来停止它。

请务必仔细检查 `while` 循环内部的逻辑是否最终会导致条件变为 `False`。

### 示例：等待特定的用户输入

`while` 循环非常适用您需要重复某些操作直到特定事件发生时，例如从用户那里获得有效输入。

```python
# 初始化响应变量，确保循环至少运行一次
response = ""

# 一直询问，直到用户输入 'quit'
while response.lower() != 'quit':
    response = input("Enter some text (or type 'quit' to exit): ")
    print(f"您输入了：{response}")

print("好的，现在退出。再见！")
```

在这种情况下，我们不知道用户在输入 'quit' 之前会输入多少次文本。`while` 循环完美处理了这种不确定性，在每次输入后检查 `response`。

掌握 `while` 循环使您能够编写根据动态条件重复执行任务的程序，使您的代码更具灵活性和功能性。您经常会用它来处理数据直到满足某个条件、运行模拟或处理用户交互。

## 参考资料

- [The while statement](https://docs.python.org/3/reference/compound_stmts.html#the-while-statement) — Python Documentation (2023)
  官方文档，描述了Python中`while`循环的语法和执行方式。
- [Python Crash Course: A Hands-On, Project-Based Introduction to Programming, 3rd Edition](https://nostarch.com/python-crash-course-3rd-edition/) — Eric Matthes (2022)
  Publisher: No Starch Press
  一本广受初学者欢迎的书，通过清晰的解释和实际示例介绍了`while`循环。
- [6.0001 Introduction to Computer Science and Programming in Python](https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/) — Ana Bell, Eric Grimson, John Guttag (2016)
  Journal: MIT OpenCourseWare; Publisher: Massachusetts Institute of Technology
  麻省理工学院的入门课程，使用Python教授基本编程概念，包括使用`while`语句的控制流。
