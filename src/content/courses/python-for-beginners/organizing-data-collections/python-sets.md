---
course: "python-for-beginners"
chapter: "organizing-data-collections"
lesson: "python-sets"
sourceId: 2321
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-4-organizing-data-collections/python-sets"
title: "唯一项：集合"
description: "了解集合，Python 的无序集合类型，它只存储独一无二的元素，有助于成员资格检查和消除重复项。"
order: 6
plots: []
sourceHash: "dfbeca0a2a751c3730d3700b4bdd3de7f6f7c06ba058718fa7f4ea4c50df5d7d"
sourceCorrections: []
---

Python 提供了多种集合类型，例如列表、元组和字典。`set` 类型是另一种集合，专门用于存储独一无二的项，并且不考虑元素的顺序。

可以把集合想象成一个数学上的集合：一组不同的对象。如果你尝试添加一个集合中已有的项，什么都不会发生；集合保持不变，因为它只存储每个独一无二项的一个副本。这一特性使得集合在从其他集合中移除重复项，或快速检查某项是否属于某个组等任务中非常有用。

集合的另一个特点是它们是无序的。与列表或元组不同，集合不按任何特定顺序保存元素。这意味着你不能使用索引（例如 `my_list[0]`）访问元素，因为没有定义“第一个”或“第二个”元素。

### 创建集合

你可以通过将逗号分隔的元素放在花括号 `{}` 中来创建一个集合。

```python
# 创建一个包含独一无二元素的集合
unique_numbers = {1, 2, 3, 4, 5}
print(unique_numbers)
# 输出可能是 {1, 2, 3, 4, 5} 或其他顺序

# 创建一个包含重复项的集合 - 重复项会自动移除
colors = {'red', 'green', 'blue', 'red'}
print(colors)
# 输出: {'blue', 'red', 'green'} (顺序可能不同)
```

请注意，即使我们最初将 `'red'` 包含两次，它在 `colors` 集合中也只出现一次。

一个虽小但重要的地方：创建一个*空*集合需要使用不带任何参数 (parameter)的 `set()` 函数。使用空花括号 `{}` 实际上会创建一个空*字典*，我们之前已经讲过。

```python
# 创建空集合的正确方法
empty_s = set()
print(type(empty_s))
# Output: <class 'set'>
print(empty_s)
# Output: set()

# 这会创建一个空字典，而不是集合
empty_d = {}
print(type(empty_d))
# Output: <class 'dict'>
```

你也可以使用 `set()` 函数从任何可迭代对象（如列表、元组或字符串）创建集合。这是从序列中移除重复项的常用方法。

```python
# 从包含重复项的列表中创建集合
numbers_list = [1, 2, 2, 3, 4, 4, 4, 5]
unique_numbers_set = set(numbers_list)
print(unique_numbers_set)
# 输出: {1, 2, 3, 4, 5} (顺序可能不同)

# 从字符串创建集合
letters_set = set('hello')
print(letters_set)
# 输出: {'h', 'l', 'o', 'e'} (顺序可能不同)
```

### 修改集合

像列表一样，集合是可变的，这意味着你可以在创建后改变它们的内容。

#### 添加元素

要向集合中添加单个元素，请使用 `.add()` 方法。

```python
permissions = {'read', 'write'}
print(permissions)
# 输出: {'write', 'read'} (顺序可能不同)

# 添加一个新权限
permissions.add('execute')
print(permissions)
# 输出: {'write', 'read', 'execute'} (顺序可能不同)

# 尝试添加一个已有的元素 - 无变化
permissions.add('read')
print(permissions)
# 输出: {'write', 'read', 'execute'} (顺序可能不同)
```

#### 移除元素

有两种主要方法可以移除元素：

1. `.remove(element)`: 此方法移除指定元素。但是，如果集合中找不到该元素，它会引发 `KeyError`。
2. `.discard(element)`: 此方法也移除指定元素，但如果找不到该元素，它*不会*引发错误。在这种情况下，它只是什么都不做。

选择 `.remove()` 还是 `.discard()` 取决于你是否希望在尝试移除不存在的元素时（使用 `.remove()`）程序停止，还是希望它默默地继续执行（使用 `.discard()`）。

```python
data_points = {10, 20, 30, 40}

# 使用 remove() 移除一个已有的元素
data_points.remove(20)
print(data_points)
# 输出: {10, 40, 30} (顺序可能不同)

# 尝试使用 remove() 移除一个不存在的元素 - 这将导致错误
# data_points.remove(99) # 引发 KeyError: 99

# 使用 discard() 移除一个已有的元素
data_points.discard(30)
print(data_points)
# 输出: {10, 40} (顺序可能不同)

# 尝试使用 discard() 移除一个不存在的元素 - 无错误
data_points.discard(99)
print(data_points)
# 输出: {10, 40} (顺序可能不同)
```

### 检查成员资格

集合中最有效率的操作之一是使用 `in` 关键字检查元素是否存在。这比检查列表中成员资格要快得多，特别是对于大型集合。

```python
allowed_users = {'alice', 'bob', 'charlie'}

# 检查用户是否被允许
print('bob' in allowed_users)
# Output: True

print('eve' in allowed_users)
# Output: False

# 检查用户是否未被允许
print('david' not in allowed_users)
# Output: True
```

### 何时使用集合

在以下情况中，集合是理想的选择：

1. **要求独一无二：** 你需要存储一组项，其中不允许有重复项或需要自动移除重复项。
2. **频繁进行成员资格检查：** 你的程序经常需要检查集合中是否存在某个项。集合提供非常快速的查找。
3. **顺序不重要：** 元素的顺序或位置对你的应用不重要。
4. **需要进行数学集合运算：** 你计划执行诸如查找共同元素（交集）、组合元素（并集）或查找集合间差异等操作。我们将在下一节讨论这些操作。

例如，如果你有一个从日志条目中获取的用户 ID 列表，并且想要一份访问过系统的独一无二的用户列表：

```python
log_entries = ['user1', 'user2', 'user1', 'user3', 'user2', 'user1']

# 使用集合查找独一无二的用户
unique_users_set = set(log_entries)
print(unique_users_set)
# 输出: {'user2', 'user3', 'user1'} (顺序可能不同)

# 如果你需要一个独一无二的用户列表（顺序可能与原始顺序不同）
unique_users_list = list(unique_users_set)
print(unique_users_list)
# 输出: ['user2', 'user3', 'user1'] (顺序可能不同)
```

集合提供了一种强大且高效的方法来处理集合，其中独一无二性和快速成员资格检查是主要要求。接下来，我们将学习你可以对集合执行的特定数学操作。

## 参考资料

- [Set Types (set, frozenset)](https://docs.python.org/3/library/stdtypes.html#set-types-set-frozenset) — Python Software Foundation (2023)
  Publisher: Python Software Foundation
  对 Python 内置集合类型、其操作和行为提供全面详细说明。
- [Python Crash Course, 3rd Edition: A Hands-On, Project-Based Introduction to Programming](https://nostarch.com/python-crash-course-3rd-edition/) — Eric Matthes (2022)
  Publisher: No Starch Press
  在适合初学者的背景下，介绍基本的 Python 编程概念，包括集合的应用。
- [Lecture 8: Tuples, Sets, and Efficiency, MIT 6.0001 Introduction to Computer Science and Programming in Python](https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/) — Ana Bell, Eric Grimson, John Guttag (2016)
  在学术背景下，提供关于集合作为 Python 数据结构的基础教学，讨论其特性和效率。
