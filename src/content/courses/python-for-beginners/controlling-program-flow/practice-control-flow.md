---
course: "python-for-beginners"
chapter: "controlling-program-flow"
lesson: "practice-control-flow"
sourceId: 2306
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-3-controlling-program-flow/practice-control-flow"
title: "练习：条件逻辑与循环的实现"
description: "通过实际编码问题应用您对 if 语句和循环的理解。"
order: 7
plots: []
sourceHash: "3b4f1ac759103b06ca92b3ce2b25d7fdbbb4d12a2d156f0b2cdc29925af4c8a9"
sourceCorrections: []
---

实践在 Python 中实现条件逻辑和循环。这些练习将帮助您巩固对 `if`、`elif`、`else`、`while`、`for`、`break` 和 `continue` 的掌握。完成这些问题有助于编写更具动态性和实用性 Python 程序。

请记住，目标不仅仅是得到正确答案，更要了解代码 *为什么* 这样运行。请大胆尝试，修改示例，并尝试不同的方法。

### 练习 1：年龄段分类器

编写一个 Python 脚本，询问用户年龄，然后打印一条消息，指明他们是“未成年人”（18岁以下）、“成年人”（18到64岁）还是“老年人”（65岁及以上）。

**指导：**

1. 使用 `input()` 函数从用户获取年龄。请记住 `input()` 返回一个字符串。
2. 使用 `int()` 将输入字符串转换为整数。您可能希望稍后（在第9章中学习）将其包装在 `try-except` 块中，以优雅地处理非数字输入，但目前请假定输入有效。
3. 使用 `if-elif-else` 结构检查年龄是否符合定义的范围。
4. 打印相应的类别。

```python
# 获取输入（记住转换为整数）
# age_str = input("请输入您的年龄: ")
# age = int(age_str)

# 使用 if/elif/else 检查年龄范围
# if age < 18:
#     print(...)
# elif age >= 18 and age <= 64: # 或者简单地：elif age <= 64: 因为第一个 'if' 已经失败
#     print(...)
# else: # 必须是 65 岁或以上
#     print(...)
```

### 练习 2：使用 `while` 循环倒计时

创建一个脚本，使用 `while` 循环从 5 倒数到 1，并打印每个数字。循环结束后，打印“发射！”。

**指导：**

1. 初始化一个变量来存储起始计数（例如，`count = 5`）。
2. 设置一个 `while` 循环，只要计数大于 0 就继续。
3. 在循环内部，打印当前计数的值。
4. 在每次迭代中将计数变量减 1（`count = count - 1` 或 `count -= 1`）。
5. 循环终止后（当条件 `count > 0` 变为假时），打印最终消息。

```python
# 初始化计数器
# counter = 5

# 当计数器为正时循环
# while counter > 0:
    # 打印计数器
    # print(counter)
    # 递减计数器
    # counter -= 1

# 循环结束后打印最终消息
# print("Blast off!")
```

### 练习 3：使用 `for` 循环求和

给定列表 `numbers = [12, 7, 9, 21, 15]`，编写一个脚本，使用 `for` 循环计算并打印列表中所有数字的总和。

**指导：**

1. 初始化一个变量来存储总和，从 0 开始（例如，`total_sum = 0`）。
2. 使用 `for` 循环遍历 `numbers` 列表中的每个 `number`。
3. 在循环内部，将当前 `number` 添加到 `total_sum`。
4. 循环处理完所有数字后，打印最终的 `total_sum`。

```python
# 给定列表
# numbers = [12, 7, 9, 21, 15]

# 初始化求和累加器
# current_sum = 0

# 遍历列表
# for num in numbers:
    # 将当前数字添加到总和
    # current_sum += num # current_sum = current_sum + num 的简写

# 打印最终总和
# print("The sum is:", current_sum)
```

### 练习 4：使用循环控制的简单猜数字游戏

编写一个程序来模拟一个简单的数字猜谜游戏。

1. 选择一个“秘密”数字（例如，`secret_number = 8`）。
2. 使用 `for` 循环为用户提供有限次的猜测机会（例如，3 次尝试）。`range()` 函数在此处可能有用。
3. 在循环内部，使用 `input()` 提示用户进行猜测，并将其转换为整数。
4. 检查猜测是否正确：
   - 如果正确，打印“正确！你猜对了！”并使用 `break` 立即退出循环。
   - 如果错误，打印“抱歉，不是这个。”
5. 循环结束后（无论是猜对还是用完尝试次数），检查用户是否猜对。您可能需要一个单独的变量（一个标志）来跟踪是否猜对了。如果他们用完尝试次数仍未猜对，打印一条消息，例如“抱歉，您的尝试次数已用完。数字是 [秘密数字]。”

**指导：**

- 您可以使用 `range(3)` 精确地循环三次。
- 一个布尔变量，例如在循环前设置为 `guessed_correctly = False` 并在猜对时更改为 `True`，可以帮助确定最终消息。

```python
# secret_number = 8
# guessed_correctly = False
# max_attempts = 3

# print("猜一个 1 到 10 之间的数字。您有 3 次尝试机会。")

# for attempt in range(max_attempts):
    # guess_str = input(f"第 {attempt + 1} 次尝试：输入您的猜测: ")
    # guess = int(guess_str)

    # if guess == secret_number:
        # print("正确！你猜对了！")
        # guessed_correctly = True
        # break # 退出循环
    # else:
        # print("抱歉，不是这个。")

# 循环结束后，检查用户是否成功
# if not guessed_correctly:
    # print(f"抱歉，您的尝试次数已用完。数字是 {secret_number}。")
```

**（练习 4 的可选增强）：** 修改猜数字游戏以使用 `continue`。如果用户输入的猜测超出了有效范围（例如，小于 1 或大于 10），打印错误消息并使用 `continue` 跳过当前循环迭代的其余部分，直接进入下一次尝试，而不惩罚他们的无效输入（或者根据您的游戏规则，可能仍将其计为一次尝试）。

### 练习 5：打印图案（嵌套循环）

使用嵌套的 `for` 循环打印以下图案：

```
*
**
***
****
*****
```

**指导：**

1. 外层循环将控制行数（本例中为 5 行）。
2. 内层循环将控制当前行中打印的星号数量。请注意，第 `i` 行的星号数量是 `i`。
3. 在内层循环中使用 `print()` 函数的 `end` 参数 (parameter)（`print("*", end="")`）在同一行上打印星号。
4. 内层循环完成给定行后，使用一个简单的 `print()` 语句将光标移动到下一行，然后外层循环开始其下一次迭代。

```python
# number_of_rows = 5

# 外层循环用于控制行
# for i in range(1, number_of_rows + 1): # range(1, 6) 产生 1, 2, 3, 4, 5
    # 内层循环用于控制列（当前行中的星号）
    # for j in range(i): # range(i) 产生从 0 到 i-1 的数字，因此它运行 'i' 次
        # print("*", end="")
    # 打印完当前行的所有星号后，移动到下一行
    # print()
```

这些练习涵盖了条件逻辑和循环的核心内容。尝试不同的变体，组合各类内容（例如，在循环内部使用 `if` 语句，就像在猜数字游戏中那样），并增强对 Python 程序执行流程的控制能力。

## 参考资料

- [The Python Tutorial - Control Flow Tools](https://docs.python.org/3/tutorial/controlflow.html) — Guido van Rossum and the Python development team (2024)
  Python控制流语句的官方文档，提供了关于`if`、`elif`、`else`、`for`、`while`、`break`和`continue`的核心信息。
- [Python Crash Course, 3rd Edition: A Hands-On, Project-Based Introduction to Programming](https://nostarch.com/python-crash-course-3rd-edition) — Eric Matthes (2022)
  Publisher: No Starch Press
  Python编程的实用性介绍，以易于初学者理解的方式涵盖了条件逻辑和循环等基础概念。
- [Introduction to Computer Science and Programming in Python](https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/) — Ana Bell, Eric Grimson, John Guttag (2016)
  Publisher: MIT OpenCourseWare
  这门大学入门课程为Python概念提供了结构化的学习路径，包含了对条件执行和迭代的全面介绍。
