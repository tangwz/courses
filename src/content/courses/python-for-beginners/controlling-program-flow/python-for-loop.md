---
course: "python-for-beginners"
chapter: "controlling-program-flow"
lesson: "python-for-loop"
sourceId: 2298
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-3-controlling-program-flow/python-for-loop"
title: "遍历序列：`for` 循环"
description: "学习如何使用 `for` 循环遍历序列（如列表或字符串）中的项目。"
order: 4
plots: []
sourceHash: "0bc738c45d29d4d960a0bc5a784115729bf63e1aef5d2903d8a0c581554a0ed7"
sourceCorrections: []
---

Python 提供了多种控制程序流程的构造。`for` 循环提供了一种直接且高效的方式来对序列或集合中的*每个*项目执行操作。当你逐一处理已知项目集合时，它特别有用。常见用途包括迭代列表中的所有元素、字符串中的所有字符，或重复执行某个操作特定次数。

### 基本 `for` 循环语法

一个 `for` 循环的基本结构如下所示：

```python
for variable_name in sequence:
    # 对每个项目执行的代码块
    # 使用 variable_name 来访问当前项目
    statement1
    statement2
    # ...
```

我们来分解一下：

1. `for`：开始循环的关键词。
2. `variable_name`：你选择的一个变量（使用描述性名称），它将在循环的每次迭代（通过）中保存序列中的当前项目。
3. `in`：将循环变量与序列分隔开的关键词。
4. `sequence`：你想要遍历的项目集合。这可以是一个列表、一个字符串，或者我们之后会遇到的其他可迭代类型。
5. `:`：冒号标记 (token) `for` 语句行的结束。
6. **缩进代码块**：与 `if` 和 `while` 类似，应在循环*内部*运行的代码必须缩进。此代码块对 `sequence` 中的每个项目执行一次。

### 遍历列表

列表是有序集合，它们与 `for` 循环非常适配。假设你有一个数字列表，并想打印每个数字。

```python
temperatures = [19.5, 22.1, 18.0, 25.3]

print("每日温度：")
for temp in temperatures:
    print(f"记录温度：{temp}°C")

print("\n循环结束。")
```

**输出：**

```
每日温度：
记录温度：19.5°C
记录温度：22.1°C
记录温度：18.0°C
记录温度：25.3°C

循环结束。
```

在此示例中：

- `temperatures` 是序列（一个列表）。
- `temp` 是循环变量。
- 在第一次迭代中，`temp` 值为 `19.5`。缩进的 `print` 语句执行。
- 在第二次迭代中，`temp` 值为 `22.1`。`print` 语句再次执行。
- 这持续到最后一个项目（`25.3`）被处理。
- 循环结束后，程序继续执行下一个未缩进的行（`print("\n循环结束。")`）。

### 遍历字符串

字符串是字符序列。`for` 循环可以自动遍历字符串中的每个字符。

```python
user_name = "Alice"

print(f"名称 {user_name} 中的字符：")
for character in user_name:
    print(f"- {character}")
```

**输出：**

```
名称 Alice 中的字符：
- A
- l
- i
- c
- e
```

这里，循环变量 `character` 在每次通过时，获取字符串 `user_name` 中每个字符的值（先是 'A'，然后是 'l'，然后是 'i' 等）。

### 使用 `range()` 重复代码特定次数

有时，你没有现成的列表或字符串来循环，而只是想重复执行一个操作固定次数。Python 的内置 `range()` 函数在此非常有用。它生成一个数字序列，`for` 循环可以对其进行遍历。

`range()` 可以通过几种方式使用：

1. **`range(stop)`**：生成从 0 开始直到（但不*包含*）`stop` 值的数字。

   ```python
   print("从 0 数到 4：")
   for i in range(5): # 生成 0, 1, 2, 3, 4
       print(i)
   ```

   **输出：**

   ```
   从 0 数到 4：
   0
   1
   2
   3
   4
   ```

   请注意，`range(5)` 产生从 0 到 4 的数字，总共是 5 个数字。这种从零开始的行为在编程中很常见。变量 `i` 通常用于简单的循环计数器，但你可以使用任何有效的变量名。
2. **`range(start, stop)`**：生成从 `start` 开始直到（但不*包含*）`stop` 值的数字。

   ```python
   print("\n从 2 到 5 的数字：")
   for num in range(2, 6): # 生成 2, 3, 4, 5
       print(num)
   ```

   **输出：**

   ```
   从 2 到 5 的数字：
   2
   3
   4
   5
   ```
3. **`range(start, stop, step)`**：生成从 `start` 开始、直到（但不*包含*）`stop`、每次按 `step` 递增的数字。

   ```python
   print("\n小于 10 的奇数：")
   for odd_num in range(1, 10, 2): # 生成 1, 3, 5, 7, 9
       print(odd_num)
   ```

   **输出：**

   ```
   小于 10 的奇数：
   1
   3
   5
   7
   9
   ```

将 `range()` 与 `for` 循环结合使用是执行诸如运行 100 步模拟、处理数据集中的前 10 个项目或简单地多次打印消息等操作的标准方式。

### 实用示例：列表元素求和

我们将 `for` 循环与我们所学的变量和运算符结合起来，计算列表中数字的总和。

```python
scores = [88, 92, 75, 98, 85]
total_score = 0 # 初始化累加变量

for score in scores:
    total_score = total_score + score # 将当前分数加到总和中

print(f"分数列表为：{scores}")
print(f"总分数为：{total_score}")
```

**输出：**

```
分数列表为：[88, 92, 75, 98, 85]
总分数为：438
```

在此示例中，`total_score` 用作累加器。它从 0 开始，在每次迭代中，`scores` 列表中的当前 `score` 都被加到其中。

### 何时使用 `for` 与 `while`

- 当你想遍历已知序列（如列表、字符串或 `range`）中的每个项目，或者当你需要重复执行某个操作特定次数时，请使用 **`for` 循环**。它为你处理迭代过程的管理（获取下一个项目，在末尾停止）。
- 当你需要只要某个条件保持为真就重复执行一个操作，并且不一定预先知道循环将运行多少次时，请使用 **`while` 循环**。你通常需要在循环内明确管理该条件。

`for` 循环提供了一种简洁且可读的方式来处理序列中的项目，使其成为 Python 中最常用的控制流结构之一。当你学习下一章中更复杂的数据结构（如字典和元组）时，你会发现 `for` 循环也能很好地与它们配合使用。

## 参考资料

- [The Python Tutorial - 4. More Control Flow Tools](https://docs.python.org/3/tutorial/controlflow.html#for-statements) — Guido van Rossum and the Python Development Team (2024)
  Python官方教程的这一部分解释了`for`语句、序列迭代和`range()`函数，并提供了清晰的示例。
- [Learning Python](https://www.oreilly.com/library/view/learning-python-6th/9781098171308/) — Mark Lutz (2013)
  Publisher: O'Reilly Media
  一本详尽的教科书，其中包含控制流语句的章节，深入探讨了Python中的`for`循环、迭代器和序列。
- [Introduction to Computation and Programming Using Python, with Application to Understanding Data](https://books.google.com/books/about/Introduction_to_Computation_and_Programm.html?id=oB-ODAAAQBAJ) — John V. Guttag (2016)
  Publisher: MIT Press; Pages: Chapter 4: Functions and Abstraction by Specification; DOI: [10.7551/mitpress/10390.001.0001](https://doi.org/10.7551/mitpress/10390.001.0001)
  麻省理工学院入门编程课程所用的教科书，为`for`循环等迭代结构提供了清晰的解释和示例。
