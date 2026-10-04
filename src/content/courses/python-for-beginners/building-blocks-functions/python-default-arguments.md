---
course: "python-for-beginners"
chapter: "building-blocks-functions"
lesson: "python-default-arguments"
sourceId: 2335
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-5-building-blocks-functions/python-default-arguments"
title: "默认参数值"
description: "学习如何为函数参数提供默认值，使其在函数调用时变为可选。"
order: 5
plots: []
sourceHash: "236403b7de4a7787e51b10f7fe94092c530734cec4a333ce5f88987700446fc3"
sourceCorrections: []
---

在定义函数时，您学习了如何指定在函数调用时接受参数 (parameter)的形参。然而，有时您希望函数形参在调用方未提供值时，能有一个“标准”或“默认”值。这使得函数更为灵活，因为调用方只需为那些与默认值不同的形参提供实参即可。

Python 允许您直接在函数定义中为形参设定默认值。

### 定义默认参数 (parameter)值

要为一个形参赋予默认值，您可以在 `def` 语句中使用赋值运算符（`=`）为其赋值。

```python
def greet(name, greeting="Hello"):
  """打印问候语。

  Args:
    name: 要问候的人的名字。
    greeting: 要使用的问候语（默认为“Hello”）。
  """
  print(f"{greeting}, {name}!")

# 调用函数
greet("Alice")             # 使用默认问候语
greet("Bob", "Good morning") # 覆盖默认问候语
```

**输出:**

```
Hello, Alice!
Good morning, Bob!
```

在此示例中，`greeting` 形参的默认值为 `"Hello"`。

- 当我们调用 `greet("Alice")` 时，我们只提供了 `name` 实参。Python 检测到 `greeting` 实参缺失，并自动使用其默认值 `"Hello"`。
- 当我们调用 `greet("Bob", "Good morning")` 时，我们提供了两个实参。值 `"Good morning"` 会覆盖 `greeting` 的默认值。

### 形参排序规则

定义带有默认实参的函数时，一个重要规则是：**所有带有默认值的形参必须位于所有不带默认值的形参*之后***。

为什么？当您使用位置实参（不指定形参名称，按顺序传递的实参）调用函数时，Python 从左到右将值与形参匹配。如果一个非默认形参位于默认形参之后，Python 将无法判断所提供的实参是针对非默认形参的，还是旨在覆盖前面形参的默认值。

**正确示例:**

```python
def create_user(username, is_active=True, permissions="read"):
  print(f"创建用户: {username}")
  print(f"活跃状态: {is_active}")
  print(f"权限: {permissions}")
```

**错误示例:** 这将导致 `SyntaxError`。

```python
# SyntaxError: 非默认参数位于默认参数之后
# def create_user_incorrect(is_active=True, username, permissions="read"):
#   pass
```

### 调用带默认值的函数

您可以以多种方式调用带有默认实参的函数：

1. **只提供必需实参：** Python 会为其余实参使用默认值。

   ```python
   create_user("charlie")
   # 输出:
   # 创建用户: charlie
   # 活跃状态: True
   # 权限: read
   ```
2. **按位置提供实参：** 值会按顺序覆盖默认值。

   ```python
   create_user("david", False) # 覆盖 is_active，使用默认权限
   # 输出:
   # 创建用户: david
   # 活跃状态: False
   # 权限: read

   create_user("eve", False, "admin") # 覆盖两个默认值
   # 输出:
   # 创建用户: eve
   # 活跃状态: False
   # 权限: admin
   ```
3. **使用关键字提供实参：** 这允许您覆盖特定的默认值，而不考虑它们的位置（只要满足必需实参）。这种方式通常更清晰。

   ```python
   create_user("frank", permissions="write") # 只覆盖权限
   # 输出:
   # 创建用户: frank
   # 活跃状态: True
   # 权限: write

   create_user(username="grace", is_active=False) # 使用关键字实参
   # 输出:
   # 创建用户: grace
   # 活跃状态: False
   # 权限: read
   ```

### 常见用途

默认实参常用于：

- **标志或选项：** 控制可选行为的形参，例如 `verbose=False` 标志用于启用额外输出。
- **配置：** 设定可能偶尔需要更改的常用值，例如网络操作的 `timeout=30` 秒。
- **提供合理默认值：** 提供适用于大多数情况的标准值，例如数值比较中的 `tolerance=0.001`。

### 一点注意事项：可变默认实参

有一个不明显但重要的细节需要注意：默认实参值**只在函数定义时评估一次**，而不是每次函数调用时都评估。对于数字、字符串或元组等不可变类型，这没有问题。然而，如果您使用列表或字典等可变类型作为默认值，那么所有依赖默认值的调用都将使用**同一个对象**。这可能导致意料之外的结果。

考虑这个示例：

```python
def add_item(item, my_list=[]):
  """向列表中添加一个项。（演示可变默认值问题）"""
  my_list.append(item)
  print(f"列表当前为: {my_list}")

print("第一次调用:")
add_item("apple")

print("\n第二次调用:")
add_item("banana") # 意外地添加到第一次调用时的列表！

print("\n第三次调用，提供一个列表:")
add_item("cherry", ["start"]) # 当提供一个列表时，按预期工作
```

**输出:**

```
第一次调用:
列表当前为: ['apple']

第二次调用:
列表当前为: ['apple', 'banana']

第三次调用，提供一个列表:
列表当前为: ['start', 'cherry']
```

请注意第二次调用 `add_item("banana")` 并没有从一个空列表开始。它使用了在第一次调用中被修改的*同一个列表*。这很少是预期的行为。

**标准解决方案:**

处理可变默认值的常规方法是使用 `None` 作为默认值，然后如果形参为 `None`，则在函数内部创建一个*新的*可变对象。

```python
def add_item_safe(item, my_list=None):
  """安全地向列表中添加一个项。"""
  if my_list is None:
    my_list = [] # 如果未提供列表，则创建一个新列表
  my_list.append(item)
  print(f"列表当前为: {my_list}")

print("第一次调用（安全）:")
add_item_safe("apple")

print("\n第二次调用（安全）:")
add_item_safe("banana") # 从一个新的空列表开始

print("\n第三次调用，提供一个列表（安全）:")
add_item_safe("cherry", ["start"])
```

**输出:**

```
第一次调用（安全）:
列表当前为: ['apple']

第二次调用（安全）:
列表当前为: ['banana']

第三次调用，提供一个列表（安全）:
列表当前为: ['start', 'cherry']
```

使用 `None` 作为占位符默认值，并在函数内部创建可变对象，能确保每次依赖默认值的调用都获得一个全新的实例，从而避免调用之间产生意外的副作用。

默认实参值是一项让您的函数更方便、更具适应性的功能，但请记住可变默认值和不可变默认值之间的区分，以避免常见问题。

## 参考资料

- [Fluent Python: Clear, Concise, and Effective Programming](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056348/) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media
  深入讲解Python惯用法，包括对默认参数值的详细讨论、其正确使用方法以及与可变默认参数相关的关键问题，并提供实用的解决方案。第二版。
- [Python Cookbook](https://www.oreilly.com/library/view/python-cookbook-3rd/9781449357368/) — David Beazley, Brian K. Jones (2013)
  Publisher: O'Reilly Media
  为常见的Python编程任务提供实用技巧和解决方案，包括定义和使用带默认参数的函数的有效策略，以及避免可变默认值常见问题的指导。第三版。
