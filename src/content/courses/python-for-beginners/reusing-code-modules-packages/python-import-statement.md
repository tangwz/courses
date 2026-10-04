---
course: "python-for-beginners"
chapter: "reusing-code-modules-packages"
lesson: "python-import-statement"
sourceId: 2355
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-7-reusing-code-modules-packages/python-import-statement"
title: "导入模块：import 语句"
description: "学习使用 `import` 语句将其他模块中的代码引入当前脚本的基本方法。"
order: 2
plots: []
sourceHash: "f1b93629a72d3296724605db480a59d4964152f7244b8d9bac165e83b5c9f1c2"
sourceCorrections: []
---

将代码组织成独立文件（模块）是一种有益的实践。要在另一个文件中使用一个文件的代码，Python 提供了 `import` 语句。这个语句是 Python 用于加载和使用其他模块中代码的基本机制。

### 基本 `import` 语句

使用模块最简单的方法是使用 `import` 关键字，后跟模块文件名（不带 `.py` 扩展名）。

假设你想使用一些数学函数。Python 带有一个名为 `math` 的内置模块，其中包含许多实用的数学运算和常量。要使用它，你的脚本开头是：

```python
import math
```

这行代码实际做了什么？

1. **查找模块：** Python 会在它所知道的特定目录列表中查找名为 `math.py` 的文件（或其它类型的模块，但目前主要针对 `.py` 文件）。此列表包括包含你当前脚本的目录和标准库位置。
2. **运行模块（如果需要）：** 如果模块在当前程序执行中尚未加载，Python 会运行 `math.py` 内部的代码。这使得 `math.py` 中定义的所有函数、变量和类都变得可用。
3. **创建命名空间：** 这点非常重要。`import math` 语句*不会*直接将 `math` 模块中的所有函数（如 `sqrt`）或常量（如 `pi`）复制到你当前脚本的主要工作区。相反，它在你的脚本命名空间中创建一个名称：`math`。这个 `math` 名称现在指向模块对象本身。

### 访问模块内容：点记法

因为 `import math` 创建了一个名为 `math` 的命名空间，你需要告诉 Python *在哪里*查找该模块中定义的函数或变量。你通过使用**点记法**来实现这一点：`module_name.item_name`。

要使用 `math` 模块中的平方根函数 (`sqrt`)，你会这样写：

```python
math.sqrt(16)
```

要访问常量 pi (`pi`)，你会这样写：

```python
math.pi
```

这种显式的 `module_name.` 前缀很有好处，因为它能避免命名冲突。如果你在脚本中定义了自己的变量 `pi`，它不会与 `math.pi` 冲突，因为它们存在于不同的命名空间中。你的 `pi` 在主脚本的命名空间中，而数学常量则通过 `math` 命名空间访问。

### 示例：使用 `math` 模块

下面是一个完整、简单的脚本，演示了导入和使用方法：

```python
# 导入整个 math 模块
import math

# 计算 25 的平方根
number = 25
square_root = math.sqrt(number)
print(f"The square root of {number} is {square_root}") # 输出：25 的平方根是 5.0

# 计算半径为 3 的圆的面积
radius = 3
area = math.pi * (radius ** 2) # 使用 math.pi 和幂运算符
print(f"The area of a circle with radius {radius} is {area}") # 输出：半径为 3 的圆的面积是 28.274333882308138
```

在此示例中：

- `import math` 使 `math` 模块可用。
- `math.sqrt()` 调用 `math` 模块*内部*的 `sqrt` 函数。
- `math.pi` 访问 `math` 模块*中的* `pi` 常量。

### 导入多个模块

如果你需要来自几个不同模块的函数或变量，可以单独导入它们。标准做法是将每个 import 语句放在文件的顶部，单独一行：

```python
import math
import random # 另一个用于生成随机数的标准库模块
import os     # 用于与操作系统交互的模块

print(math.sqrt(100))
print(random.randint(1, 10)) # 获取一个 1 到 10 之间的随机整数
print(os.getcwd())           # 获取当前工作目录
```

这种简单的 `import module_name` 语句，结合点记法，是将外部代码引入 Python 脚本最常用和推荐的方式，它能确保代码清晰并避免名称冲突。在接下来的章节中，我们将了解 import 语句的变体，并进一步了解 Python 的标准库。

## 参考资料

- [The import system](https://docs.python.org/3/reference/import.html) — Python Software Foundation (2024)
  这份官方文档权威地解释了 Python 的模块导入机制，包括模块搜索路径、执行和命名空间创建。
- [Python Crash Course: A Hands-On, Project-Based Introduction to Programming](https://nostarch.com/products/python-crash-course-3rd-edition) — Eric Matthes (2023)
  Publisher: No Starch Press; Pages: 552
  提供对 Python 模块和 `import` 语句的实践性介绍，通过清晰的示例展示其用法。
- [Learning Python](https://www.oreilly.com/library/view/learning-python-5th/9781449355722/) — Mark Lutz (2013)
  Publisher: O'Reilly Media
  Python 基础知识的综合指南，详细阐述模块、导入形式和底层命名空间模型，并提供广泛的解释。
