---
course: "python-for-beginners"
chapter: "reusing-code-modules-packages"
lesson: "python-what-are-modules"
sourceId: 2353
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-7-reusing-code-modules-packages/python-what-are-modules"
title: "什么是模块？"
description: "理解 Python 模块的原理，它们是包含 Python 定义和语句的文件，有助于代码组织和复用。"
order: 1
plots: []
sourceHash: "cd62f30459bf4bf9fd892fa5d4ef6d49a54231119b57291e327fa0e797674789"
sourceCorrections: []
---

随着您的Python程序代码量增多，将所有代码放入一个文件中会很快变得杂乱且难以管理。设想一下，您要在一个几千行的文件中查找某个函数定义，或者想在完全不同的项目中复用您为一个项目编写的有用的函数，而无需复制粘贴。这时，Python的模块就派上用场了。

本质上，Python 中的**模块**只是一个包含 Python 定义和语句的文件。文件名即是模块名加上 `.py` 后缀。例如，如果您创建一个名为 `utilities.py` 的文件并在其中放置多个函数定义，您就创建了一个名为 `utilities` 的模块。

模块有以下几个重要作用：

- **组织性：** 它们让您可以逻辑地将相关代码归类。您可以将所有数学辅助函数放在一个模块中，而将处理用户输入的函数放在另一个模块中。这使得您的代码库更容易理解和查阅。
- **可复用性：** 一旦您在一个模块中编写并测试了代码，您就可以轻松地在其他程序中使用它，而无需重写。这能提高开发效率并减少错误。就像您可以用工具箱中的特定工具完成各种任务一样，您也可以在不同项目中使用模块中的函数。
- **命名空间隔离：** 模块创建独立的命名空间（变量和函数等名称存在的地方）。这意味着您的 `utilities` 模块中名为 `calculate` 的函数不会与另一个模块或主程序文件中同样名为 `calculate` 的函数冲突。每个模块都管理自己的名称，从而避免冲突。

可以把模块看作是积木。您不是构建一个庞大、单一的程序，而是构建更小、独立的模块来处理特定任务。然后，您可以组装这些模块来创建您的最终应用程序。

例如，考虑我们的 `utilities.py` 文件：

```python
# utilities.py

def calculate_area(length, width):
  """计算矩形面积。"""
  return length * width

def format_greeting(name):
  """创建一个简单的问候字符串。"""
  return f"Hello, {name}!"

# ... 其他辅助函数 ...
```

这个文件，`utilities.py`，就是一个模块。它包含相关函数（`calculate_area`、`format_greeting`）。我们如何从另一个 Python 脚本实际访问和使用这些函数，是接下来几节的内容，届时我们将介绍 `import` 语句。理解模块对于编写有组织、可维护的 Python 代码非常重要，尤其是在您开始处理大型项目时。

## 参考资料

- [The Python Tutorial - Modules](https://docs.python.org/3/tutorial/modules.html) — Python Software Foundation (2024)
  Publisher: Python Software Foundation
  这份官方资料提供了Python模块的入门介绍，阐明了它们的定义、用途以及如何在语言中使用。
- [Python Crash Course, 3rd Edition](https://nostarch.com/python-crash-course-3rd-edition/) — Eric Matthes (2022)
  Publisher: No Starch Press
  这份资料为Python提供了实用且面向初学者的介绍，包含关于创建和使用模块以实现高效代码结构和重用的清晰解释。
