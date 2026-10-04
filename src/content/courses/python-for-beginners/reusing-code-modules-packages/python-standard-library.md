---
course: "python-for-beginners"
chapter: "reusing-code-modules-packages"
lesson: "python-standard-library"
sourceId: 2358
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-7-reusing-code-modules-packages/python-standard-library"
title: "Python 标准库概述"
description: "了解 Python 捆绑的丰富模块集合，即标准库。"
order: 4
plots: []
sourceHash: "8fff39139bd59dd21afd38a7900fec3e36226c0f0b48a363191df35d374b943a"
sourceCorrections: []
---

Python 的一个重要优点在于它自带“电池”。这意味着当你安装 Python 时，你还会获得一个捆绑了大量有用模块的集合，这些模块统称为 **Python 标准库**。你可以把它看作是随 Python 安装一同提供的一个工具箱，里面装满了预构建好的工具，可供你在程序中使用。

这个库为你省去了大量的功夫。与其从零开始编写代码以处理数学计算、文本操作、与操作系统交互或日期时间处理等常见操作，你通常可以在标准库中找到一个已经能满足你需求的模块。这些模块由 Python 核心开发者和其他贡献者编写，经过良好测试，并被设计为高效且可靠。

标准库的内容丰富，涵盖了各种各样的任务。你不需要一次性学完所有内容，但了解它的存在以及大致有哪些可用功能会非常有帮助。当你遇到新的编程问题时，你通常可以问自己：“Python 标准库里是否可能存在对应的模块呢？”

以下是标准库中模块提供的功能类型的一些示例：

- **数学运算：** `math` 模块提供对三角函数、对数、幂运算以及其他常见数学操作的访问。例如，`math.sqrt()` 计算平方根。
- **随机数生成：** `random` 模块允许你生成伪随机数、打乱序列或进行随机选择。对模拟、游戏或统计抽样很有用。
- **操作系统交互：** `os` 模块提供与底层操作系统交互的方式，例如导航文件系统 (`os.path`)、管理目录 (`os.mkdir`、`os.listdir`) 和访问环境变量。
- **日期和时间处理：** `datetime` 模块提供用于处理日期、时间以及时间间隔的类。你可以获取当前日期和时间，对日期进行计算，并将它们格式化为字符串。
- **文本处理：** 诸如 `re`（正则表达式）之类的模块提供用于字符串内模式匹配和操作的复杂工具。
- **读取和写入数据格式：** 诸如 `csv` 和 `json` 之类的模块帮助你处理常见数据文件格式。

要使用这些工具，你只需 `import` 相关的模块，就像导入你自己编写的模块一样。例如，要使用平方根函数，你可以这样写：

```python
import math

number = 16
square_root = math.sqrt(number)
print(f"The square root of {number} is {square_root}")
# 输出: 16 的平方根是 4.0
```

> 你的 Python 脚本可以使用 `import` 语句访问标准库中各种模块的功能。

查阅标准库文档（可在 Python 官方网站上查阅）是了解可用工具广度的好方法。尽管你不会使用每个模块，但了解主要模块将大大提升你编写高效 Python 代码的能力，而无需重复造轮子。在后续章节和未来的学习中，你将更频繁地遇到并使用具体的标准库模块。目前，请理解这个库是 Python 开发的核心部分，为构建应用程序提供了丰富的功能支撑。

## 参考资料

- [The Python Standard Library](https://docs.python.org/3/library/index.html) — Python Software Foundation (2023)
  Publisher: Python Software Foundation
  Python标准库中所有模块的官方和最全面的文档。
- [Python Crash Course, 3rd Edition](https://nostarch.com/python-crash-course-3rd-edition/) — Eric Matthes (2022)
  Publisher: No Starch Press
  一本介绍性教材，涵盖了Python的基础概念，包括关键标准库模块的概述和实际使用。
- [Introduction to Computer Science and Programming in Python](https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/) — Ana Bell, Eric Grimson, and John Guttag (2016)
  Publisher: MIT OpenCourseWare
  一门大学课程，介绍使用Python进行编程，并演示其标准库在各种计算任务中的应用。
