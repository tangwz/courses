---
course: "python-for-beginners"
chapter: "getting-started-with-python"
lesson: "what-is-python-programming"
sourceId: 2236
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-1-getting-started-with-python/what-is-python-programming"
title: "什么是Python编程？"
description: "了解Python语言的基础知识、其历史及其作为编程工具的主要特点。"
order: 1
plots: []
sourceHash: "9ffeea28c8e74697f57e1c93d4a42d01d385cd42451157aa60a3e2f3746ddcb9"
sourceCorrections: []
---

编程让我们能够向计算机发出指令。计算机的核心是只能理解由零和一组成的序列，通常称为机器码。直接用机器码编写非常困难且耗时。编程语言充当中间媒介，提供一种更易于人类阅读的方式来编写这些指令，然后这些指令会被转换成计算机可执行的形式。

Python是一种以其**高级**、**解释型**和**通用**的特性而闻名的编程语言。

- **高级：** 这表示Python的语法（编写代码的规则）设计得更接近人类语言，它将计算机硬件许多复杂的基础细节进行抽象处理。可以将其想象成给出驾驶路线：您不是指定方向盘的每一个转动和踏板的每一次踩踏（低级），而是给出“在下一个路口左转”之类的指令（高级）。这使得Python代码与C或汇编等低级语言相比，通常更容易阅读、编写和维护。
- **解释型：** Python代码通常由一个称为解释器的程序逐行执行。当你运行一个Python脚本时，解释器会读取你的代码并几乎立即执行指定的动作。这与编译型语言不同，编译型语言需要先将整个代码翻译成机器码才能运行。解释型特性通常能加快开发过程，尤其是在测试和调试时，因为你不需要单独的编译步骤。我们将在本章稍后对这一区别进行更详细的说明。
- **通用：** Python并非只为一项特定任务而设计。它用途广泛，应用于各类场景，包括Web开发（如网站的后端系统）、数据分析、人工智能（AI）、科学计算、任务自动化（脚本编写）等等。

Python由Guido van Rossum在1980年代后期创建。其设计的一个主要目标是代码的可读性。Python的语法刻意保持简洁，并强调使用空白（缩进）来构造代码块，这极大地提升了其清晰度。通常，编写良好的Python代码读起来几乎就像普通的英文句子。

考虑一个显示消息的简单指令：

```python
print("Hello, Python learner!")
```

这一行代码清晰地表达了其意图。在低级语言中实现同样的结果可能需要更多的设置和不那么直观的代码。

### Python的主要特性

有几个特点促成了Python的普及，尤其是对初学者而言：

- **可读性和简洁性：** 如前所述，清晰的语法使得学习编程基础知识变得更容易，而不会被复杂的规则所困扰。
- **庞大的标准库：** Python捆绑了大量预编写的代码模块（“标准库”），这些模块可以处理常见任务，例如处理文本、数字、文件、网络等。这意味着对于基本操作，你通常不必从头开始编写代码。
- **活跃的社区和生态系统：** Python受益于一个庞大而活跃的全球社区。这意味着有丰富的学习资源、求助论坛，以及通过`pip`等工具（我们稍后会介绍）提供的海量第三方包（库）。这些包将Python的功能扩展到几乎所有可想象的方面。
- **动态类型：** 在Python中，你通常不需要明确声明变量将保存的数据*类型*（例如，它是数字还是文本）。Python会在程序运行时自动识别这一点。虽然这提供了灵活性，但了解数据类型（我们将在下一章中介绍）仍然非常有用。

Python的简洁性、强大功能和多功能性相结合，使其成为第一门编程语言的绝佳选择，它为软件开发和数据科学的许多方面提供了稳固的基础。本课程旨在从头开始指导你，假设你没有预先的编程经验。我们将发挥Python的优势来帮助你有效掌握编程知识。

## 参考资料

- [The Python Tutorial](https://docs.python.org/3/tutorial/) — Guido van Rossum and the Python Software Foundation (2023)
  Publisher: Python Software Foundation
  学习 Python 的官方指南，阐释其基本特性和设计理念。
- [Introduction to Computer Science and Programming in Python](https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/) — Ana Bell, Eric Grimson, John Guttag (2016)
  Journal: MIT OpenCourseWare; Publisher: Massachusetts Institute of Technology
  一门学术课程，提供使用 Python 进行编程的基础知识。
