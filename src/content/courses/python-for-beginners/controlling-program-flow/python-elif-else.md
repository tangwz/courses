---
course: "python-for-beginners"
chapter: "controlling-program-flow"
lesson: "python-elif-else"
sourceId: 2292
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-3-controlling-program-flow/python-elif-else"
title: "处理不同情况：elif 和 else"
description: "扩展条件逻辑，使用 elif (else if) 和 else 处理多种备选执行路径。"
order: 2
plots: []
sourceHash: "06fc163ece5948aa7f45943e51cec55558f402cc5d6d0600d1bc89d270e8292d"
sourceCorrections: []
---

Python 程序中的 `if` 语句允许代码块仅在特定条件评估为 `True` 时执行。然而，当存在多个需要检查的条件，或者当初始条件不满足时需要一个默认操作时，就需要更灵活的控制。`elif` 和 `else` 子句扩展了 `if` 语句，以处理这些更复杂的决策情况。

### 检查多重条件：`elif`

假设你想检查不止一个条件。例如，判断一个数字是正数、负数还是零，需要检查多种可能性。`elif` 语句是 "else if" 的缩写，它允许你仅在前面的 `if`（或 `elif`）条件为 `False` 时测试其他条件。

你可以在初始 `if` 语句后连接多个 `elif` 语句。Python 会从上到下逐一评估这些条件。一旦找到一个为 `True` 的条件，它就会执行与该条件关联的代码块，然后 *跳过* 整个 `if-elif-else` 结构中其余的部分。

下面是通用结构：

```python
if condition_1:
    # 块 1: 如果 condition_1 为 True 则执行
    statement(s)
elif condition_2:
    # 块 2: 如果 condition_1 为 False 且 condition_2 为 True 则执行
    statement(s)
elif condition_3:
    # 块 3: 如果 condition_1 和 condition_2 都为 False，
    # 并且 condition_3 为 True 则执行
    statement(s)
# ... 你可以有更多的 elif 块
```

让我们修改数字示例：

```python
number = -10

if number > 0:
    print("The number is positive.")
elif number < 0:
    print("The number is negative.")

# 如果 number 是 0，则上面两个条件都不为 True，因此目前什么也不会发生。
```

在这种情况下，如果 `number` 是 -10：

1. `number > 0` 为 `False`，因此跳过第一个块。
2. `number < 0` 为 `True`，因此打印消息“The number is negative.”。程序随后跳过此结构中所有后续的 `elif` 或 `else` 子句。

如果数字是 0 呢？`number > 0` 和 `number < 0` 都不为真。在上面的代码中，将不会打印任何内容。为了处理 *所有* `if` 或 `elif` 条件都不满足的情况，我们使用 `else` 子句。

### 默认情况：`else`

`else` 子句提供了一个默认的代码块，仅当结构中所有前面的 `if` 和 `elif` 条件都评估为 `False` 时才执行。它起到一个包罗万象的作用。如果使用 `else` 子句，它必须始终放在最后，在所有 `if` 和 `elif` 子句之后，并且它没有关联的条件。

完整的结构如下所示：

```python
if condition_1:
    # 块 1: 如果 condition_1 为 True 则执行
    statement(s)
elif condition_2:
    # 块 2: 如果 condition_1 为 False 且 condition_2 为 True 则执行
    statement(s)
# ... 更多 elif 块（可选）...
else:
    # 块 N: 如果所有前面的条件 (condition_1, condition_2, ...) 都为 False 则执行
    statement(s)
```

让我们使用 `else` 完成数字符号的示例：

```python
number = 0

print(f"检查数字: {number}")

if number > 0:
    print("该数字是正数。")
elif number < 0:
    print("该数字是负数。")
else:
    # 只有当 number 不大于 0 且不小于 0 时，此块才运行
    print("该数字是零。")

print("检查完毕。")
```

如果你用 `number = 0` 运行这段代码：

1. `number > 0` 为 `False`。
2. `number < 0` 为 `False`。
3. 由于所有前面的条件都为假，`else` 块执行，打印“The number is zero.”

下面的图表展示了此示例的控制流程：

> 流程图展示了使用 if-elif-else 判断数字是正数、负数还是零的决策过程。

### 顺序和排他性

关于 `if-elif-else` 结构，有两点需要记住：

1. **顺序很重要：** 条件会按照它们出现的顺序进行检查。如果多个条件可能为真，只有与 *第一个* 遇到的真条件关联的代码块才会执行。考虑一个评分示例：

   ```python
   score = 85

   if score >= 70: # 这是第一个为 True 的
       grade = 'C'
   elif score >= 80: # 这也为 True，但会在后面检查
       grade = 'B'
   else:
       grade = 'Not C or B'

   print(grade) # 输出: C
   ```

   因为 `score >= 70` 首先被检查，并且对于 85 来说为真，所以等级被设置为 'C'，`elif score >= 80` 的检查永远不会到达。为了获得正确的等级，你应该按照从最具体（最高分数）到最不具体（最低分数）的顺序来排列检查。

   ```python
   score = 85

   if score >= 90:
       grade = 'A'
   elif score >= 80: # 第二个检查，对于 85 为 True
       grade = 'B'
   elif score >= 70: # 第三个检查
       grade = 'C'
   # ... 其他等级
   else:
       grade = 'F'

   print(grade) # 输出: B
   ```
2. **只执行一个代码块：** 在单个 `if-elif-...-else` 链中，最多只执行一个代码块。即使后面的 `elif` 条件可能也为真，一旦前面的 `if` 或 `elif` 条件评估为 `True`，它们就会被跳过。`else` 块仅在其上方 *所有* 条件都不为真时才会执行。

通过结合使用 `if`、`elif` 和 `else`，你可以构建清晰有效的逻辑来处理程序中的各种情况，使它们对不同的输入和情形做出适当的响应。

## 参考资料

- [The \`if\` statement](https://docs.python.org/3/reference/compound_stmts.html#the-if-statement) — Python Software Foundation (2024)
  提供 Python 中 `if`、`elif` 和 `else` 语句的权威规范和详细信息。
- [Python Crash Course, 3rd Edition: A Hands-On, Project-Based Introduction to Programming](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH51N2QpVje2kTG4I-KogZw_iGzGDUrfBuj1e8FtMJU4FKiipxN-zGo4Bt7z7qtXCqePtFsfbS9kLvyd05zlQw4fqSq12nRSJBQ8wYvB1uznXadI2QNOtt3tkwcAdLiP6Y1SQQo5YyJw5z3dl3RhD8ffsSLTdBwLivtX1IieS9xJCHKNXw_gKEvQGXi_BbQzMOuqpU=) — Eric Matthes (2023)
  Publisher: No Starch Press
  一本广受好评的入门书籍，通过实用示例为初学者清晰地解释了条件语句和控制流。
- [Lecture Notes: Branching and Iteration (Lecture 3)](https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/resources/mit6_0001f16_lec3/) — Ana Bell, Eric Grimson, and John Guttag (2016)
  Journal: MIT OpenCourseWare; Publisher: Massachusetts Institute of Technology
  这些来自知名大学课程的讲义为理解条件语句和控制流提供了学术基础。
