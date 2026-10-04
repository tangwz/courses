---
course: "python-for-beginners"
chapter: "organizing-data-collections"
lesson: "python-lists"
sourceId: 2309
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-4-organizing-data-collections/python-lists"
title: "序列简介：列表"
description: "了解列表，这是 Python 中用途广、有序且可变的序列类型，用于存储项目集合。"
order: 1
plots: []
sourceHash: "c5a8cb340c73bd092697c16c5a113c77563744ded07c655e45b17f6def7ecc04"
sourceCorrections: []
---

编程经常需要处理项目集合。比如，在一个班级中记录学生姓名，购物清单上的商品，或者游戏中的高分。虽然你可以为每个项目创建一个单独的变量（例如 `item1 = "milk"`，`item2 = "eggs"`），但当项目数量增加时，这种做法很快就会变得难以管理。

Python 提供了一些内置的数据结构，通常被称为集合或容器，它们专门用于存储数据组。其中最基本、用途最广的一种是**列表**。

可以将列表看作是一系列有序的项目，很像购物清单或编号的待办事项列表。列表中的每个项目都有一个特定的位置。重要的是，Python 中的列表是**可变的**，这意味着它们在创建后可以更改内容——你可以添加新项目、移除现有项目，或者更改列表中已有项目的值。

### 创建列表

你可以通过将逗号分隔的一系列项目放在方括号 `[]` 中来创建 Python 列表。

```python
# 一个空列表
empty_list = []
print(empty_list)

# 一个整数列表
numbers = [1, 2, 3, 5, 8]
print(numbers)

# 一个字符串列表
fruits = ["apple", "banana", "cherry"]
print(fruits)

# 一个包含混合数据类型的列表
mixed_list = [10, "hello", 3.14, True]
print(mixed_list)
```

如你所见，列表很灵活，可以保存不同数据类型的项目，包括数字、字符串、布尔值，甚至其他列表（我们稍后会介绍嵌套列表）。

### 访问项目：索引

由于列表是有序的，你可以使用它们的位置（称为**索引**）访问单个项目。Python 使用从零开始的索引，这意味着第一个项目位于索引 0，第二个项目位于索引 1，以此类推。

要访问一个项目，你使用列表名称，后面跟上方括号 `[]` 中的索引。

```python
fruits = ["apple", "banana", "cherry", "date"]

# 访问第一个项目（索引 0）
first_fruit = fruits[0]
print(first_fruit)  # 输出：apple

# 访问第三个项目（索引 2）
third_fruit = fruits[2]
print(third_fruit)  # 输出：cherry
```

尝试访问一个不存在的索引（例如，上面列表中 `fruits[4]`，它只有索引 0、1、2、3）将导致 `IndexError`。

### 负数索引

Python 也支持负数索引，这是一种从列表末尾开始访问项目的便捷方式。索引 `-1` 指的是最后一个项目，`-2` 指的是倒数第二个项目，以此类推。

```python
fruits = ["apple", "banana", "cherry", "date"]

# 访问最后一个项目
last_fruit = fruits[-1]
print(last_fruit)  # 输出：date

# 访问倒数第二个项目
second_last_fruit = fruits[-2]
print(second_last_fruit) # 输出：cherry
```

### 获取列表长度

你可以使用内置的 `len()` 函数轻松地查明列表中有多少个项目。

```python
numbers = [10, 20, 30, 40, 50]
fruits = ["apple", "banana"]
empty_list = []

print(len(numbers))      # 输出：5
print(len(fruits))       # 输出：2
print(len(empty_list))   # 输出：0
```

列表是 Python 中组织数据的重要工具。它们的主要特点是：

1. **有序：** 项目根据添加顺序保持特定次序。
2. **可变：** 你可以在列表创建后更改它（添加、移除、修改项目）。
3. **可索引：** 可以直接使用数字索引（从 0 开始）访问项目。
4. **异构：** 它们可以包含不同数据类型的项目。

在下一节中，我们将进一步了解列表的*可变*特性，并学习如何在其中添加、移除和更改项目。

## 参考资料

- [Built-in Types - list](https://docs.python.org/3/library/stdtypes.html#list) — Python Software Foundation (2024)
  Python内置列表类型的官方文档，其中详细介绍了其创建、方法和属性。
- [Fluent Python: Clear, Concise, and Effective Programming](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056348/) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media
  一本关于编写高效Python的权威指南，其中包含了对列表等内置数据结构的全面审查。
- [Python Crash Course, 3rd Edition: A Hands-On, Project-Based Introduction to Programming](https://nostarch.com/python-crash-course-3rd-edition) — Eric Matthes (2023)
  Publisher: No Starch Press
  一本面向初学者的Python编程实践指南，其中包含专门介绍列表及其操作的章节。
- [Introduction to Computer Science and Programming in Python](https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/) — Ana Bell, Eric Grimson, and John Guttag (2016)
  Publisher: MIT OpenCourseWare
  一门基础在线课程，涵盖基本的Python数据类型和结构，包括列表，作为计算机科学基础的一部分。
