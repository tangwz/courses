# 第 6 章：为机器学习编写高效且可维护的 Python 代码

来源：[原章节](https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-6-efficient-maintainable-python-ml)

[返回课程目录](../README.md)

为机器学习任务编写可运行的 Python 代码是一个主要目标。然而，随着项目规模扩大和涉及协作，代码的*质量*也变得同样重要。难以阅读、执行缓慢或难以修改的代码会极大地妨碍项目进展。

本章侧重于那些有助于您编写机器学习 Python 代码的实践和工具，使这些代码不仅正确，而且高效、可读且可维护。我们将涵盖建立清晰的代码风格、逻辑地组织项目，以及编写有效的函数和模块。您将了解如何使用虚拟环境管理项目依赖、使用性能分析工具识别性能瓶颈，以及优化 NumPy 和 Pandas 等常用库的特定方法。此外，我们将介绍用于验证代码组件的单元测试基础知识，以及使用 Git 进行版本控制的基本用法，以有效管理您的代码库。这些技能对于构建可靠且可扩展的机器学习系统是必要的。

## 小节

- 1. [代码风格与可读性](01-%E4%BB%A3%E7%A0%81%E9%A3%8E%E6%A0%BC%E4%B8%8E%E5%8F%AF%E8%AF%BB%E6%80%A7.md)
- 2. [机器学习项目结构化](02-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E9%A1%B9%E7%9B%AE%E7%BB%93%E6%9E%84%E5%8C%96.md)
- 3. [编写高效的函数和模块](03-%E7%BC%96%E5%86%99%E9%AB%98%E6%95%88%E7%9A%84%E5%87%BD%E6%95%B0%E5%92%8C%E6%A8%A1%E5%9D%97.md)
- 4. [虚拟环境简介](04-%E8%99%9A%E6%8B%9F%E7%8E%AF%E5%A2%83%E7%AE%80%E4%BB%8B.md)
- 5. [Python 代码性能分析](05-Python%20%E4%BB%A3%E7%A0%81%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90.md)
- 6. [优化 NumPy 和 Pandas 的方法](06-%E4%BC%98%E5%8C%96%20NumPy%20%E5%92%8C%20Pandas%20%E7%9A%84%E6%96%B9%E6%B3%95.md)
- 7. [机器学习单元测试介绍](07-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%8D%95%E5%85%83%E6%B5%8B%E8%AF%95%E4%BB%8B%E7%BB%8D.md)
- 8. [Git 版本控制基础](08-Git%20%E7%89%88%E6%9C%AC%E6%8E%A7%E5%88%B6%E5%9F%BA%E7%A1%80.md)
- 9. [实践：重构与优化机器学习代码片段](09-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E9%87%8D%E6%9E%84%E4%B8%8E%E4%BC%98%E5%8C%96%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E4%BB%A3%E7%A0%81%E7%89%87%E6%AE%B5.md)

章节测验：[在线测验](https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-6-efficient-maintainable-python-ml/quiz)
