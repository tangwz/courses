---
course: "python-for-beginners"
chapter: "python-basics-variables-data-types-operators"
lesson: "python-variables"
sourceId: 2261
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-2-python-basics-variables-data-types-operators/python-variables"
title: "存储信息：变量"
description: "学习如何在Python中创建和使用变量，以便在程序中存储和引用数据。"
order: 1
plots: []
sourceHash: "3cbf5384881e681599dc46ed13de38de9522459c790bb7a2707224cda957766d"
sourceCorrections: []
---

在编写程序时，你经常需要处理各种信息，例如数字、文本或计算结果。与其直接在各处使用原始数据，不如给数据一个名称并通过该名称来引用它，这样做会方便得多。这时就需要用到变量了。你可以将变量视为一个带标签的容器或一个命名占位符，在计算机内存中存储一个值。

### 什么是变量？

变量是 Python 中指代某个值的符号名称。程序使用变量存储数据，以便稍后引用和操作。“变量”一词意味着存储的值在程序执行期间可以改变或变化。

想象一下你在盒子里整理物品。你不会把东西随意放入没有标签的盒子；你会给它们贴上标签（“冬季衣物”、“书籍”、“厨房用具”）。变量在编程中也起着类似的作用：它们为存储数据的内存位置提供描述性名称。

### 创建变量：赋值语句

在 Python 中，使用赋值运算符（即等号 `=`）创建变量很简单。结构如下：

`variable_name = value`

工作方式如下：

1. 你为变量选择一个名称（需遵循以下讨论的特定规则）。
2. 使用 `=` 符号。
3. 提供你希望存储在变量中的值。

我们来看几个例子：

```python
message = "Hello, Python Learner!"
count = 10
price = 49.99
is_active = True
```

在这些例子中：

- `message` 是一个存储文本字符串的变量。
- `count` 是一个存储整数（一个完整数字）的变量。
- `price` 是一个存储浮点数（带小数点的数字）的变量。
- `is_active` 是一个存储布尔值（`True`）的变量。

Python 是动态类型的，这意味着你无需明确声明变量将持有的数据*类型*。Python 会根据你赋给它的值自动确定类型。当你将 `"Hello, Python Learner!"` 赋给 `message` 时，Python 会识别出 `message` 现在持有一个字符串。

### 使用变量

一旦你创建（或*声明*并*初始化*）了一个变量，你就可以在代码中使用它的名称来访问它存储的值。

```python
message = ""
print(message) # 输出：

items = 5
cost_per_item = 2.50
total_cost = items * cost_per_item # 在计算中使用变量
print(total_cost) # 输出：12.5
```

使用变量使你的代码更具可读性且更易于维护。如果 `cost_per_item` 发生变化，你只需在一个地方（变量被赋值的地方）更新它，而不是在代码中搜索每个 `2.50` 的实例。

### 变量命名：规则和约定

选择好的变量名对于编写清晰易懂的代码很重要。Python 有特定的规则和普遍接受的约定来命名变量：

**规则（必须遵循）：**

- 变量名必须以字母（a-z，A-Z）或下划线（`_`）开头。它们不能以数字开头。
- 名称的其余部分可以包含字母、数字（0-9）和下划线。
- 变量名区分大小写。`myVariable`、`myvariable` 和 `MYVARIABLE` 是不同的变量。
- 变量名不能与Python的关键字相同。关键字是Python中具有特殊含义的保留字（如 `if`、`else`、`for`、`while`、`def`、`class`、`True`、`False`、`None`）。

**约定（为提高可读性应遵循）：**

- 使用描述性名称来表明变量的用途（例如，`user_name` 而不是 `un`，`total_price` 而不是 `tp`）。
- 对于由多个词组成的变量名，使用 `snake_case` 风格。这意味着所有字母都小写，并且词与词之间用下划线分隔（例如，`first_name`，`number_of_items`）。
- 避免使用像 `l`、`O` 或 `I` 这样的单字母名称，因为它们容易与数字 `1` 和 `0` 混淆。像 `x`、`y` 或 `i` 这样的短名称在特定情境中通常可以接受，例如循环计数器或数学坐标，但描述性名称通常更受推荐。

以下是有效和无效变量名的例子：

- **有效：** `name`、`age`、`user_input`、`_internal_counter`、`price_2024`
- **无效：** `2nd_place`（以数字开头）、`user-name`（包含连字符）、`class`（是关键字）

### 更改变量值（重新赋值）

你可以通过再次使用赋值运算符（`=`）为其赋新值来更改变量中存储的值。

```python
# 赋初始值
score = 100
print("Initial score:", score) # 输出：初始分数：100

# 更新值
score = 150
print("Updated score:", score) # 输出：更新分数：150

# 基于当前值进行赋值
score = score + 20 # 读取当前值（150），加20，将结果（170）存储回去
print("Final score:", score)   # 输出：最终分数：170
```

每次你赋新值时，与该变量名关联的旧值都会被替换。

变量是Python中基础组成部分。它们允许你标记 (token)和存储数据，使你的程序灵活且更易于理解。随着学习的推进，你会看到变量持续用于管理代码中的信息流。

## 参考资料

- [The Python Tutorial - 3. An Informal Introduction to Python](https://docs.python.org/3/tutorial/introduction.html) — Guido van Rossum and the Python Development Team (2023)
  Publisher: Python Software Foundation
  提供Python基本概念的官方简介，包括变量赋值和基本数据类型。
- [Python Crash Course, 3rd Edition: A Hands-On, Project-Based Introduction to Programming](https://nostarch.com/python-crash-course-3rd-edition/) — Eric Matthes (2022)
  Publisher: No Starch Press
  一本对初学者友好的教材，系统地介绍Python编程，其中有专门的章节解释变量、数据类型和命名约定。
- [Introduction to Computer Science and Programming in Python, Fall 2016](https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/) — John Guttag, Ana Bell, Eric Grimson (2016)
  Publisher: MIT OpenCourseWare
  一门基础大学课程，通过视频讲座和习题集涵盖Python的基本概念，如变量、数据类型和表达式。
