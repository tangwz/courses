---
course: "python-for-beginners"
chapter: "building-blocks-functions"
lesson: "python-function-parameters-arguments"
sourceId: 2330
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-5-building-blocks-functions/python-function-parameters-arguments"
title: "函数参数与实参"
description: "了解如何在函数定义中设定参数以及在调用函数时传递实参。"
order: 2
plots: []
sourceHash: "d7115057e0d00b5feba328c987f52af6e53959f35c27db14334ef2d066449c6a"
sourceCorrections: []
---

使用 `def` 关键字定义基本函数可以打包代码。但当你能向函数*传递*信息时，它们会变得非常实用。想象一个问候用户的函数；它需要知道用户的名字。或者一个将两个数字相加的函数；它需要知道要加哪两个数字。这就是参数 (parameter)和实参的作用。

- **参数：** 它们是在函数定义中括号内列出的名称。可以把它们看作是占位符或变量，在函数被调用时会接收值。
- **实参：** 它们是你在调用函数时实际传递给函数的值。这些值会被赋值给对应的参数。

让我们来看一个简单示例：

```python
# 定义一个带有一个名为 'user_name' 的参数的函数
def greet(user_name):
  """打印一个简单的问候语。"""
  print(f"Hello, {user_name}!")

# 调用函数，传入实参 "Alice"
greet("Alice")

# 再次调用函数，传入不同的实参
greet("Bob")
```

在这段代码中：

1. `def greet(user_name):` 中的 `user_name` 是**参数**。它是函数内部的一个占位符，代表我们想要问候的名字。
2. `"Alice"` 和 `"Bob"` 是**实参**。它们是我们在*调用* `greet` 函数时提供的具体值。当 `greet("Alice")` 运行时，值 `"Alice"` 会被赋值给函数内部的 `user_name` 参数。

### 传递多个实参

函数可以接受多个参数 (parameter)，在定义时用逗号分隔。调用函数时，你按照相同的顺序提供对应的实参。这被称为使用**位置实参**，因为实参的位置决定了哪个参数会接收哪个实参。

```python
def describe_pet(animal_type, pet_name):
  """显示宠物的信息。"""
  print(f"I have a {animal_type}.")
  print(f"My {animal_type}'s name is {pet_name}.")

# 使用位置实参调用
describe_pet("hamster", "Harry")
print("---") # 仅用于分隔输出
describe_pet("dog", "Willie")
```

输出：

```
I have a hamster.
My hamster's name is Harry.
---
I have a dog.
My dog's name is Willie.
```

在这里，首次调用 `describe_pet("hamster", "Harry")` 时：

- `"hamster"`（第一个实参）被赋值给 `animal_type`（第一个参数）。
- `"Harry"`（第二个实参）被赋值给 `pet_name`（第二个参数）。

顺序很重要。如果你调用 `describe_pet("Max", "cat")`，输出将不正确，因为 `"Max"` 会被赋值给 `animal_type`，而 `"cat"` 会被赋值给 `pet_name`。

> 位置实参到参数的对应关系。第一个实参对应第一个参数，第二个实参对应第二个参数，依此类推。

### 关键字实参

Python 也支持使用**关键字实参**。这需要指定参数 (parameter)名，后跟等号（`=`）和你想要传递的值。

```python
def describe_pet(animal_type, pet_name):
  """显示宠物的信息。"""
  print(f"I have a {animal_type}.")
  print(f"My {animal_type}'s name is {pet_name}.")

# 使用关键字实参调用
describe_pet(animal_type="hamster", pet_name="Harry")
print("---")
# 关键字实参的顺序不重要
describe_pet(pet_name="Willie", animal_type="dog")
```

输出：

```
I have a hamster.
My hamster's name is Harry.
---
I have a dog.
My dog's name is Willie.
```

使用关键字实参时：

- 你明确地将实参值与参数名关联起来（`pet_name="Willie"`）。
- 你提供关键字实参的顺序不重要，因为 Python 会按名称匹配它们。

> 关键字实参到参数的对应关系。实参根据指定的名称与参数匹配，无论顺序如何。

### 结合位置实参和关键字实参

调用函数时，你可以混合使用位置实参和关键字实参。但有一个规则：**位置实参必须在关键字实参之前。**

```python
def describe_pet(animal_type, pet_name):
  """显示宠物的信息。"""
  print(f"I have a {animal_type}.")
  print(f"My {animal_type}'s name is {pet_name}.")

# 有效：位置实参在前
describe_pet("cat", pet_name="Whiskers")

# 无效：关键字实参后不能跟位置实参
# describe_pet(animal_type="dog", "Buddy") # 这会导致 SyntaxError
```

### 为什么要两者都用？

- **位置实参** 对于参数 (parameter)较少、顺序明确的函数来说，通常更简洁。
- **关键字实参** 可以提高可读性，特别是对于参数很多或参数含义从位置上不明确的函数。它们还允许你跳过具有默认值的参数（我们接下来会讲到）。

理解如何使用参数和实参将数据传递给函数，是编写灵活可复用代码的根本。你现在可以编写每次调用时处理不同数据的函数了。

## 参考资料

- [Python Crash Course, 3rd Edition: A Hands-On, Project-Based Introduction to Programming](https://nostarch.com/python-crash-course-3rd-edition) — Eric Matthes (2022)
  Publisher: No Starch Press
  一本广泛使用的入门书籍，提供了定义函数和处理各种参数类型的实用示例。
- [Python Functions](https://developers.google.com/edu/python/) — Nick Parlante (2024)
  Publisher: Google
  一个在线教程，系统介绍了Python函数，包括参数传递。
