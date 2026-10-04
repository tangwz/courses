---
course: "python-for-beginners"
chapter: "introduction-object-oriented-concepts"
lesson: "python-init-method"
sourceId: 2375
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-8-introduction-object-oriented-concepts/python-init-method"
title: "__init__ 方法（构造函数）"
description: "了解特殊的 `__init__` 方法，它作为构造函数在对象创建时用于初始化其状态。"
order: 6
plots: []
sourceHash: "a4659c03713cbdb1fced88c5dd882fab8093a5cdbb6b4da5de54375841d802b1"
sourceCorrections: []
---

当你从一个类创建对象时，通常会希望立即设定它的初始状态。例如，如果你有一个 `Dog` 类，你可能希望在创建 `Dog` 对象时就指定它的名字和品种，而不是先创建一个普通的狗，然后再设定其属性。

Python 为此提供了一个特殊方法：`__init__`。每当你创建一个类的新实例（对象）时，此方法都会自动被调用。因为它初始化了对象的状态，所以常被称为类的构造函数，尽管从技术上讲，它初始化的是一个已经创建好的对象。`__init__` 这个名字前后都有双下划线，表明它是 Python 识别的特殊方法。

### 定义和使用 `__init__`

你在类定义中定义 `__init__`，就像定义其他方法一样。第一个参数 (parameter)必须始终是 `self`，它指向正在创建的类的具体实例。在 `self` 之后，你可以定义额外参数来接收初始化对象所需的数据。

在 `__init__` 方法内部，你通常使用 `self.attribute_name = value` 语法将这些参数传入的值赋给对象的属性。

让我们修改 `Dog` 类示例，加入 `__init__` 方法：

```python
class Dog:
    def __init__(self, name, breed):
        # 当创建 Dog 对象时，此方法会自动被调用
        print(f"正在创建名为 {name} 的新 Dog 对象...")

        # 将提供的名字和品种赋值给 'self' 对象的属性
        self.name = name
        self.breed = breed
        self.tricks = [] # 为技巧初始化一个空列表

    def add_trick(self, trick):
        self.tricks.append(trick)
        print(f"{self.name} 学会了一个新技巧：{trick}！")

# 现在，当我们创建 Dog 对象时，必须提供名字和品种
dog1 = Dog("Buddy", "Golden Retriever")
dog2 = Dog("Lucy", "Beagle")

# 属性在创建时由 __init__ 设定
print(f"{dog1.name} 是一只 {dog1.breed}。") # 输出：Buddy is a Golden Retriever.
print(f"{dog2.name} 是一只 {dog2.breed}。") # 输出：Lucy is a Beagle.

# 我们仍然可以调用其他方法
dog1.add_trick("fetch")
print(f"{dog1.name} 的技巧：{dog1.tricks}") # 输出：Buddy's tricks: ['fetch']
```

### 工作原理

1. **实例化：** 当你写 `dog1 = Dog("Buddy", "Golden Retriever")` 时，Python 首先创建一个新的空 `Dog` 对象。
2. **调用 `__init__`：** Python 接着自动在这个新创建的对象上调用 `__init__` 方法。
3. **参数 (parameter)传递：** 对象本身作为第一个参数 (`self`) 传递，你在括号中提供的任何参数（`"Buddy"`、`"Golden Retriever"`）则作为后续参数（`name`、`breed`）传递。
4. **属性赋值：** 在 `__init__` 内部，代码 `self.name = name` 将值 `"Buddy"` 赋给 `dog1` 对象的 `name` 属性。类似地，`self.breed = breed` 将 `"Golden Retriever"` 赋给 `breed` 属性。`self.tricks = []` 这一行则为此特定的狗实例将 `tricks` 属性初始化为空列表。
5. **返回：** `__init__` 方法没有显式 `return` 值（它隐式返回 `None`）。它的作用是修改 `self` 对象。实例化过程（`Dog(...)`）会返回完全初始化好的对象，然后将其赋值给变量 `dog1`。

使用 `__init__` 是确保对象在创建时拥有有效且明确的初始状态的常用方式。它让你的类更容易被正确使用，因为它清楚地说明了在对象创建时需要哪些信息。你也可以直接在 `__init__` 中为属性设置默认值，就像我们为 `self.tricks = []` 所做的那样。

## 参考资料

- [Classes](https://docs.python.org/3/tutorial/classes.html) — Python Software Foundation (2024)
  官方文档，涵盖Python的类语法、对象实例化以及`__init__`方法在对象初始化中的作用。
- [Python Crash Course, 3rd Edition: A Hands-On, Project-Based Introduction to Programming](https://nostarch.com/python-crash-course-3rd-edition/) — Eric Matthes (2022)
  Publisher: No Starch Press
  一本实用的Python入门书籍，包含对类、对象和`__init__`方法的清晰解释，适合初学者。
- [Fluent Python, 2nd Edition: Clear, Concise, and Effective Programming](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056340/) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media
  本书对Python的对象模型和特殊方法进行了全面检查，在对象创建和数据模型的背景下详细说明了`__init__`。
