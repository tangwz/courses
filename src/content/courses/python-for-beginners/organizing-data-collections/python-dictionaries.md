---
course: "python-for-beginners"
chapter: "organizing-data-collections"
lesson: "python-dictionaries"
sourceId: 2317
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-4-organizing-data-collections/python-dictionaries"
title: "键值对：字典"
description: "了解字典，Python 中用于将数据存储为键值对的映射类型，支持按键快速查找。"
order: 4
plots: []
sourceHash: "c296d44bec4182cbe5d2934d457d339b8a4c4e608cc3d2b2dfa1a96cbcaf8590"
sourceCorrections: []
---

在组织数据时，常常需要将不同的信息片段关联起来，而不需要依赖特定的顺序。想象一本字典：你查找一个词（“键”）来找到它的定义（“值”）。Python 的字典也类似，它以**键值对**的形式存储数据。

字典在当你需要根据特定标识符而不是序列中的位置来获取数据时非常有用。它们使用花括号 `{}` 定义。

### 创建字典

你可以通过将逗号分隔的键值对放在花括号 `{}` 中来创建一个字典。每个键值对由一个键、一个冒号 `:` 和对应的值组成。

```python
# 一个空字典
empty_dict = {}

# 存储学生成绩的字典
student_grades = {
    "Alice": 85,
    "Bob": 92,
    "Charlie": 78
}

print(student_grades)
# 输出: {'Alice': 85, 'Bob': 92, 'Charlie': 78}

# 键可以是不同的不可变类型（字符串、数字、元组）
mixed_keys = {
    "name": "Example Item",
    101: "Item ID",
    ("x", "y"): "Coordinates"
}
print(mixed_keys)
# 输出: {'name': 'Example Item', 101: 'Item ID', ('x', 'y'): 'Coordinates'}
```

**键的重要规则：**

1. **唯一性：** 同一个字典中的键必须是唯一的。如果你尝试添加一个已存在的键，与该键关联的旧值将被覆盖。
2. **不可变性：** 键必须是不可变类型。这表示你可以使用字符串、数字（整数、浮点数）或元组作为键（只要元组本身只包含不可变元素）。你不能使用列表或其他字典作为键，因为它们是可变的（可以改变）。然而，值可以是任何数据类型，包括列表或其他字典。

### 访问值

你可以使用方括号 `[]` 来访问与特定键关联的值，类似于你通过索引访问列表元素的方式，但这里使用键。

```python
student_grades = {"Alice": 85, "Bob": 92, "Charlie": 78}

# 获取 Bob 的成绩
bobs_grade = student_grades["Bob"]
print(f"Bob 的成绩是: {bobs_grade}")
# 输出: Bob 的成绩是: 92

# 尝试访问不存在的键会导致错误
# print(student_grades["David"]) # 这会引发 KeyError
```

如果你尝试访问字典中不存在的键，Python 会引发 `KeyError`。我们将在稍后的异常处理章节中了解如何处理这类潜在错误。如果你不确定键是否存在，一种更安全地获取值的方法是使用 `get()` 方法，如果找不到键，它会返回 `None`（或你指定的默认值），而不是引发错误。

```python
student_grades = {"Alice": 85, "Bob": 92, "Charlie": 78}

# 使用 get() 获取 Alice 的成绩
alice_grade = student_grades.get("Alice")
print(f"Alice 的成绩 (使用 get): {alice_grade}")
# 输出: Alice 的成绩 (使用 get): 85

# 尝试获取 David 的成绩 (不存在)
david_grade = student_grades.get("David")
print(f"David 的成绩 (使用 get): {david_grade}")
# 输出: David 的成绩 (使用 get): None

# 如果键缺失，提供一个默认值
eve_grade = student_grades.get("Eve", "Not Found")
print(f"Eve 的成绩 (使用带默认值的 get): {eve_grade}")
# 输出: Eve 的成绩 (使用带默认值的 get): Not Found
```

### 添加和修改键值对

添加新的键值对或修改现有键关联的值都很直接。你在赋值语句的左侧使用方括号表示法。

```python
student_grades = {"Alice": 85, "Bob": 92}
print(f"原始成绩: {student_grades}")

# 添加新学生
student_grades["Charlie"] = 78
print(f"添加 Charlie 后: {student_grades}")

# 更新 Bob 的成绩
student_grades["Bob"] = 95
print(f"更新 Bob 后: {student_grades}")
```

输出:

```
原始成绩: {'Alice': 85, 'Bob': 92}
添加 Charlie 后: {'Alice': 85, 'Bob': 92, 'Charlie': 78}
更新 Bob 后: {'Alice': 85, 'Bob': 95, 'Charlie': 78}
```

如果 (“Charlie”) 不存在，则添加一个新的键值对。如果 (“Bob”) 已经存在，则更新其对应的值。

### 移除键值对

你可以使用 `del` 关键字或 `pop()` 方法移除键值对。

```python
student_grades = {"Alice": 85, "Bob": 95, "Charlie": 78}
print(f"初始字典: {student_grades}")

# 使用 del 移除 Charlie
del student_grades["Charlie"]
print(f"删除 Charlie 后: {student_grades}")

# 使用 pop() 移除 Bob 并获取被移除的值
bobs_removed_grade = student_grades.pop("Bob")
print(f"移除的 Bob 成绩: {bobs_removed_grade}")
print(f"弹出 Bob 后字典: {student_grades}")
```

输出:

```
初始字典: {'Alice': 85, 'Bob': 95, 'Charlie': 78}
删除 Charlie 后: {'Alice': 85, 'Bob': 95}
移除的 Bob 成绩: 95
弹出 Bob 后字典: {'Alice': 85}
```

使用 `del` 只是移除项。`pop()` 方法移除项*并*返回其值，这会很有用。如果你尝试 `del` 或 `pop` 一个不存在的键，你会得到一个 `KeyError`。

### 常用字典操作

字典带有一些有用的方法：

- `keys()`: 返回一个视图对象，显示所有键的列表。
- `values()`: 返回一个视图对象，显示所有值的列表。
- `items()`: 返回一个视图对象，显示所有键值元组对的列表。
- `len()`: （不是方法，而是内置函数）返回键值对的数量。
- `in`: （操作符）检查字典中是否存在某个键。

```python
student_grades = {"Alice": 85, "Bob": 95, "Charlie": 78}

print(f"键: {student_grades.keys()}")
print(f"值: {student_grades.values()}")
print(f"项: {student_grades.items()}")
print(f"学生数量: {len(student_grades)}")

# 检查 'Alice' 是否在字典中
if "Alice" in student_grades:
    print("Alice 在成绩字典中。")

# 检查 'David' 是否在字典中
if "David" not in student_grades:
    print("David 不在成绩字典中。")
```

输出:

```
键: dict_keys(['Alice', 'Bob', 'Charlie'])
值: dict_values([85, 95, 78])
项: dict_items([('Alice', 85), ('Bob', 95), ('Charlie', 78)])
学生数量: 3
Alice 在成绩字典中。
David 不在成绩字典中。
```

注意 `keys()`、`values()` 和 `items()` 返回的是“视图对象”。这些视图会反映字典的修改。你可以直接在循环中遍历这些视图（我们在上一章中讲过），或者如果需要，可以使用 `list()` 将它们转换为列表。

> 字典将唯一的不可变键映射到对应的值。

当顺序不是主要考虑因素，而基于标识符的快速查找是主要需求时，字典提供了一种灵活的数据组织方式。它们是 Python 中最常用的数据结构之一。

## 参考资料

- [Mapping Types - dict](https://docs.python.org/3/library/stdtypes.html#mapping-types-dict) — Python Core Team (2023)
  Python字典类型的权威指南，包含所有方法和行为。
- [Fluent Python: Clear, Concise, and Effective Programming](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056345/) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media; Pages: Chapter 3: Dictionaries and Sets
  全面介绍Python字典，包括其实现细节和性能特点。
- [Introduction to Algorithms](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition) — Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein (2022)
  Publisher: MIT Press; Pages: Chapter 11: Hash Tables
  计算机科学基础教材，解释了散列表的理论，Python字典即基于此实现。
- [Python Crash Course, 3rd Edition: A Hands-On, Project-Based Introduction to Programming](https://ehmatthes.com/pcc_3e/) — Eric Matthes (2023)
  Publisher: No Starch Press; Pages: Chapter 6: Dictionaries
  一本实用且适合初学者的指南，通过清晰的示例介绍如何使用Python字典。
