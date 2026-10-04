# 控制循环：break 和 continue

来源：[原文](https://apxml.com/zh/courses/python-for-beginners/chapter-3-controlling-program-flow/python-loop-control)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然 `for` 循环和 `while` 循环提供了重复执行代码块的强大方式，但有时你需要对其执行有更精细的控制。你可能希望在循环完成之前完全退出它，或者跳过当前迭代的剩余部分并直接进入下一次迭代。Python 为这些具体情况提供了两个语句：`break` 和 `continue`。

### 提前退出：`break` 语句

`break` 语句会提前终止当前循环。Python 一旦在 `for` 或 `while` 循环中遇到 `break`，就会立即退出该循环，程序执行将继续在循环块 *之后* 的下一个语句处。

当你在寻找某个东西并希望一旦找到就停止时，或者当出现错误情况使得后续迭代没有意义时，这特别有用。

考虑搜索列表中某个特定数字的第一次出现：

```python
numbers = [1, 5, 12, 8, 9, 21, 5, 3]
target = 9
found_at_index = -1 # 初始化一个表示未找到的值

print(f"Searching for {target} in {numbers}...")

for index, num in enumerate(numbers):
    print(f"Checking index {index}: value {num}")
    if num == target:
        found_at_index = index
        print(f"Found {target}!")
        break # 立即退出循环

# 循环完成或中断后，这里的代码会运行
if found_at_index != -1:
    print(f"Target {target} first found at index {found_at_index}.")
else:
    print(f"Target {target} not found in the list.")

print("Search complete.")
```

Output:

```
Searching for 9 in [1, 5, 12, 8, 9, 21, 5, 3]...
Checking index 0: value 1
Checking index 1: value 5
Checking index 2: value 12
Checking index 3: value 8
Checking index 4: value 9
Found 9!
Target 9 first found at index 4.
Search complete.
```

注意循环在索引 4 处停止了。值 `21`、`5` 和 `3` 从未被检查，因为一旦 `num == target` 为真，`break` 语句就导致了立即退出。

以下是 `break` 如何改变流程的视图：

> 当在循环中遇到 `break` 语句时的控制流程。执行会直接跳转到循环之后的代码。

### 跳过一次迭代：`continue` 语句

`continue` 语句与 `break` 不同。它不是终止整个循环，而是跳过 *当前迭代内部* 剩余的代码，并立即进入循环的 *下一次* 迭代。

对于 `while` 循环，这意味着跳回再次检查循环条件。对于 `for` 循环，这意味着移至序列中的下一个项目。

当你在迭代中遇到需要忽略或以不同方式处理的数据或条件时，这很有用，而且不会停止对后续项目的处理。

假设你只想对列表中的正数求和，忽略任何负数或零：

```python
data = [10, -5, 20, 0, -15, 30, 5]
positive_sum = 0

print(f"Processing data: {data}")

for num in data:
    print(f"Current number: {num}")
    if num <= 0:
        print("  跳过非正数。")
        continue # 跳过当前迭代的剩余部分
    # 只有在 continue 未执行时，此代码才会运行
    print(f"  Adding {num} to sum.")
    positive_sum += num

print(f"\nSum of positive numbers: {positive_sum}")
```

Output:

```
Processing data: [10, -5, 20, 0, -15, 30, 5]
Current number: 10
  Adding 10 to sum.
Current number: -5
  Skipping non-positive number.
Current number: 20
  Adding 20 to sum.
Current number: 0
  Skipping non-positive number.
Current number: -15
  Skipping non-positive number.
Current number: 30
  Adding 30 to sum.
Current number: 5
  Adding 5 to sum.

Sum of positive numbers: 65
```

在这里，每当 `num <= 0` 为真时，`continue` 语句就被执行。这阻止了 `positive_sum += num` 行对 `-5`、`0` 和 `-15` 运行，但循环继续处理了剩余的项目（`30` 和 `5`）。

以下是 `continue` 如何影响循环流程的视图：

> 当遇到 `continue` 语句时的控制流程。执行会跳回到下一次循环迭代的开始（或条件检查处），跳过当前迭代代码的剩余部分。

### `break` 与 `continue` 的对比

- **`break`**：完全退出循环。执行在循环 *之后* 继续。
- **`continue`**：跳过当前迭代。执行在下一次迭代的 *开始* 处（或循环条件检查处）继续。

### 嵌套循环

记住一点很重要，`break` 和 `continue` 只影响它们所在的 *最内层* 循环。如果你有嵌套循环（一个循环在另一个循环内部），内层循环中的 `break` 或 `continue` 不会影响外层循环。

```python
for i in range(1, 4): # 外层循环
    print(f"Outer loop iteration {i}:")
    for j in ['a', 'b', 'c', 'STOP', 'd']: # 内层循环
        if j == 'STOP':
            print("  Inner loop breaking!")
            break # 只中断内层循环
        if j == 'b':
            print(f"  在内层循环中跳过 '{j}'...")
            continue # 继续到内层循环的下一个项目
        print(f"  Processing inner item: {j}")
    # 内层循环完成或中断后，会执行到这一行
    print(f"Outer loop iteration {i} finished.")
print("All loops complete.")
```

Output:

```
Outer loop iteration 1:
  Processing inner item: a
  Skipping 'b' in inner loop...
  Processing inner item: c
  Inner loop breaking!
Outer loop iteration 1 finished.
Outer loop iteration 2:
  Processing inner item: a
  Skipping 'b' in inner loop...
  Processing inner item: c
  Inner loop breaking!
Outer loop iteration 2 finished.
Outer loop iteration 3:
  Processing inner item: a
  Skipping 'b' in inner loop...
  Processing inner item: c
  Inner loop breaking!
Outer loop iteration 3 finished.
All loops complete.
```

注意 `break` 只停止了对 `['a', 'b', 'c', 'STOP', 'd']` 的处理，而外层循环继续其迭代。类似地，`continue` 只在每次内层循环运行时跳过了打印 `b`。

熟练掌握 `break` 和 `continue` 让你能精确控制循环的执行方式，从而高效地处理特定条件，并使你的代码对所处理的数据更具响应性。虽然它们功能强大，但要谨慎使用；有时重构循环条件或使用 `if` 语句可以使代码比过度依赖 `break` 和 `continue` 更清晰。

## 参考资料

- [6. Control Flow Tools](https://docs.python.org/3/tutorial/controlflow.html#break-and-continue-statements-and-else-clauses-on-loops) — Python Software Foundation (2024)
  Python官方语言参考文档，说明了 `break` 和 `continue` 语句及其在循环中的用法。
- [Python Crash Course, 2nd Edition: A Hands-On, Project-Based Introduction to Programming](https://www.oreilly.com/library/view/python-crash-course/9781593279288/) — Eric Matthes (2019)
  Publisher: No Starch Press; Pages: Chapter 7
  一本介绍Python编程基础知识的入门教材，包含循环控制语句的说明和实用示例。

---

[上一节](04-%E9%81%8D%E5%8E%86%E5%BA%8F%E5%88%97%EF%BC%9A%60for%60%20%E5%BE%AA%E7%8E%AF.md) · [下一节](06-%E5%B5%8C%E5%A5%97%E6%8E%A7%E5%88%B6%E7%BB%93%E6%9E%84.md)
