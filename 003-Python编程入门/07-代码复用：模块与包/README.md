# 第 7 章：代码复用：模块与包

来源：[原章节](https://apxml.com/zh/courses/python-for-beginners/chapter-7-reusing-code-modules-packages)

[返回课程目录](../README.md)

随着程序规模的增大，仅在一个文件中编写所有代码将变得难以管理。将相关函数和数据组织在一起、避免命名冲突以及在不同项目间复用代码，对高效开发来说非常重要。本章将介绍Python实现这种代码组织的机制：模块和包。

你将学习如何将代码组织成独立的文件（即模块），以及如何使用`import`语句访问其他模块中定义的代码。我们会介绍Python庞大的标准库，它是一个包含许多可直接使用的预置模块的集合。你还将学习如何使用`pip`工具从Python包索引 (PyPI) 查找、安装和使用第三方包。最后，我们将了解如何将自己的模块组织成包的基本结构。这些基本知识对于构建更大、更易于维护的Python应用程序非常重要。

## 小节

- 1. [什么是模块？](01-%E4%BB%80%E4%B9%88%E6%98%AF%E6%A8%A1%E5%9D%97%EF%BC%9F.md)
- 2. [导入模块：import 语句](02-%E5%AF%BC%E5%85%A5%E6%A8%A1%E5%9D%97%EF%BC%9Aimport%20%E8%AF%AD%E5%8F%A5.md)
- 3. [导入特定名称：\`from ... import\`](03-%E5%AF%BC%E5%85%A5%E7%89%B9%E5%AE%9A%E5%90%8D%E7%A7%B0%EF%BC%9A%60from%20...%20import%60.md)
- 4. [Python 标准库概述](04-Python%20%E6%A0%87%E5%87%86%E5%BA%93%E6%A6%82%E8%BF%B0.md)
- 5. [使用pip查找和安装外部包](05-%E4%BD%BF%E7%94%A8pip%E6%9F%A5%E6%89%BE%E5%92%8C%E5%AE%89%E8%A3%85%E5%A4%96%E9%83%A8%E5%8C%85.md)
- 6. [常用标准模块](06-%E5%B8%B8%E7%94%A8%E6%A0%87%E5%87%86%E6%A8%A1%E5%9D%97.md)
- 7. [将代码组织成包（基本结构）](07-%E5%B0%86%E4%BB%A3%E7%A0%81%E7%BB%84%E7%BB%87%E6%88%90%E5%8C%85%EF%BC%88%E5%9F%BA%E6%9C%AC%E7%BB%93%E6%9E%84%EF%BC%89.md)
- 8. [实践：使用标准模块和外部模块](08-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BD%BF%E7%94%A8%E6%A0%87%E5%87%86%E6%A8%A1%E5%9D%97%E5%92%8C%E5%A4%96%E9%83%A8%E6%A8%A1%E5%9D%97.md)

章节测验：[在线测验](https://apxml.com/zh/courses/python-for-beginners/chapter-7-reusing-code-modules-packages/quiz)
