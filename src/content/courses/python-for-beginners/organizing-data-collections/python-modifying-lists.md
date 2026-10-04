---
course: "python-for-beginners"
chapter: "organizing-data-collections"
lesson: "python-modifying-lists"
sourceId: 2312
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-4-organizing-data-collections/python-modifying-lists"
title: "修改列表：添加、删除、改变元素"
description: "了解用于在 Python 列表中添加、删除和改变现有元素的常用方法。"
order: 2
plots: []
sourceHash: "1cdf0c3fdb260eb1382dcb2032e0be5a9cbc53c5ab2e0e0044574c396087551d"
sourceCorrections: []
---

Python列表的一个重要特点是它们的可变性。与字符串或元组（我们很快就会介绍）不同，列表创建后，可以改变、添加或删除其中的元素。这种灵活性使得列表在存储程序运行过程中可能需要变化的数据集合时非常有用。

### 改变特定索引处的元素

如果你知道要改变的元素的位置（索引），可以直接为该索引赋值新内容。请记住，列表索引从0开始。

```python
# 一个颜色列表
colors = ["red", "green", "blue"]
print(f"Original list: {colors}")

# 将索引1处的元素（第二个元素）从“green”改为“yellow”
colors[1] = "yellow"
print(f"After changing index 1: {colors}")

# 使用负数索引改变最后一个元素
colors[-1] = "purple"
print(f"After changing index -1: {colors}")
```

这种直接赋值会用新值替换指定索引处的现有值。如果你使用的索引在列表中不存在，Python会引发 `IndexError`。

### 向列表添加元素

Python提供了几种向列表添加新元素的方法。

#### 在末尾添加一个元素：`append()`

添加单个元素最常见的方法是使用 `append()` 方法。这会将元素添加到列表的末尾。

```python
# 今日待办事项列表
tasks = ["email team", "review report"]
print(f"Tasks to do: {tasks}")

# 在末尾添加一个新任务
tasks.append("buy groceries")
print(f"After append: {tasks}")
```

`append()` 方法简单明了，并且在每次添加一个元素来增长列表时很有效率。

#### 在特定位置添加一个元素：`insert()`

如果你需要在非末尾的特定位置添加元素，请使用 `insert()` 方法。它接受两个参数 (parameter)：你希望插入元素的位置索引，以及元素本身。从该索引开始的现有元素都会向右移动一个位置。

```python
# 在开头（索引0）插入一个高优先级任务
tasks.insert(0, "call client")
print(f"After insert at index 0: {tasks}")

# 在“buy groceries”之前（它现在在索引3）插入另一个任务
tasks.insert(3, "prepare meeting notes")
print(f"After insert at index 3: {tasks}")
```

对于大型列表，使用 `insert()` 可能不如 `append()` 高效，因为移动元素需要时间。

#### 从另一个序列添加多个元素：`extend()`

要将另一个可迭代对象（如另一个列表、元组或字符串）中的所有元素添加到当前列表的末尾，请使用 `extend()` 方法。

```python
# 从另一个列表添加更多任务
more_tasks = ["schedule follow-up", "submit expenses"]
tasks.extend(more_tasks)
print(f"After extend: {tasks}")
```

重要的是要注意 `extend()` 与 `append()` 的区别。如果你 `append()` 一个列表，整个列表会作为一个单独的元素添加。`extend()` 则会添加可迭代对象中的每个独立元素。

```python
numbers = [1, 2, 3]
extra_numbers = [4, 5]

# 使用 extend（添加单个元素的正确方法）
numbers.extend(extra_numbers)
print(f"After extend: {numbers}") # 输出：[1, 2, 3, 4, 5]

# 重置 numbers 并尝试 append
numbers = [1, 2, 3]
numbers.append(extra_numbers)
print(f"After append: {numbers}") # 输出：[1, 2, 3, [4, 5]] - 注意嵌套列表！
```

### 从列表中删除元素

正如存在添加元素的方法一样，也有几种删除元素的方法。

#### 删除值的第一次出现：`remove()`

如果你知道要删除的元素的值，但不知道它的索引，请使用 `remove()` 方法。它会在列表中查找该值的第一次出现并将其删除。

```python
# 让我们回到颜色列表
colors = ["red", "yellow", "purple", "yellow"]
print(f"Current colors: {colors}")

# 删除第一个“yellow”
colors.remove("yellow")
print(f"After remove('yellow'): {colors}")
```

如果你要删除的值在列表中未找到，Python会引发 `ValueError`。

#### 按索引删除元素：`pop()`

如果你想根据元素的位置（索引）删除它，请使用 `pop()` 方法。这个方法还会返回被删除的元素，这会很有用。如果不指定索引，`pop()` 会删除并返回列表中的*最后一个*元素。

```python
print(f"Current colors: {colors}") # 应该是 ['red', 'purple', 'yellow']

# 删除并获取索引1处的元素（“purple”）
removed_color = colors.pop(1)
print(f"Removed color: {removed_color}")
print(f"After pop(1): {colors}")

# 删除并获取最后一个元素（默认行为）
last_color = colors.pop()
print(f"Removed last color: {last_color}")
print(f"After pop(): {colors}")
```

使用无效索引调用 `pop()` 会导致 `IndexError`。

#### 按索引或切片删除元素：`del` 语句

`del` 语句是使用索引从列表中删除元素或切片的一种更通用的方式。与 `pop()` 不同，`del` 不会返回被删除的值。

```python
numbers = [10, 20, 30, 40, 50, 60]
print(f"Original numbers: {numbers}")

# 删除索引0处的元素
del numbers[0]
print(f"After del numbers[0]: {numbers}")

# 删除从索引2开始（不包括索引4）的元素
# 这会删除当前索引2和3处的元素（原来是30和40，现在是40和50）
del numbers[2:4]
print(f"After del numbers[2:4]: {numbers}")

# del 也可以删除整个列表变量，但这与清空列表不同
# del numbers
# print(numbers) # 这现在会导致 NameError
```

#### 删除所有元素：`clear()`

如果你想从列表中删除所有元素，使其变为空列表，请使用 `clear()` 方法。

```python
numbers = [10, 60] # 来自上一个示例
print(f"Numbers before clear: {numbers}")

numbers.clear()
print(f"Numbers after clear: {numbers}")
```

这会就地修改列表，使其变为空。列表变量仍然存在，只是它不包含任何元素了。

> 列表修改的可视化表示，从 `['A', 'B', 'C']` 开始。每一步都显示了应用修改方法后的结果。

了解这些改变、添加和删除元素的方法，会为你有效管理使用Python列表的动态数据集合提供所需工具。请根据你是否知道元素的值或位置，以及是要添加/删除单个还是多个元素来选择最适合的方法。

## 参考资料

- [5. Data Structures](https://docs.python.org/3/tutorial/datastructures.html) — Python Software Foundation (2023)
  Python 官方教程提供了关于 Python 内置数据结构的全面指南，包含列表操作和方法的详细说明。
- [Python Crash Course, 3rd Edition: A Hands-On, Project-Based Introduction to Programming](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHUi3jzP0MjYOyB9T_nvX_V8_EGDfnSD7X_1HW2F08Y29uZENzE2Kmaz5DR-qwV6LQkF2GCh-GkTATCbLiwbVJKWzRhtQkMPGrujZbhtVNklYlHRHRHN9jFe4Ml4S8PFs7y9AoL9JEgZK4Cprpob_J1kNnICH4cKSSZgDYJLhaSOnAZcDH-GGz2bwxVxua-vN9XEm8=) — Eric Matthes (2023)
  Publisher: No Starch Press
  一本流行且易于理解的入门书籍，通过实际示例清晰地解释了包括列表修改方法在内的 Python 基本概念。
- [Fluent Python, 2nd Edition: Clear, Concise, and Effective Programming](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056348/) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media; Pages: 1014
  深入探讨了 Python 的数据模型和特殊方法，对列表操作的行为和性能影响提供了详细见解。
