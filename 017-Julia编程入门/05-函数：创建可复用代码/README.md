# 第 5 章：函数：创建可复用代码

来源：[原章节](https://apxml.com/zh/courses/getting-started-julia-programming/chapter-5-functions-creating-reusable-code)

[返回课程目录](../README.md)

随着程序的增长，管理重复的代码段会变得有难度，导致效率低下并增加出错的可能性。函数通过允许您将代码分组为命名的、可复用的代码块来解决这个问题。之后，您可以在需要时执行这些代码块，即使输入不同，也无需复制代码。这种做法提升了程序的组织性、可读性和可维护性。

本章将引导您学习如何在 Julia 中创建和使用函数。我们将介绍如何定义和调用函数。您将学习处理函数参数，包括位置参数和关键字参数类型，以及如何分配默认值。本章还讲解了函数如何返回值，以及变量作用域如何影响数据访问。此外，还将向您介绍编写简洁的匿名函数，Julia 的独特之处——多重派发，以及为函数编写文档字符串的重要性。

## 小节

- 1. [在 Julia 中定义和调用函数](01-%E5%9C%A8%20Julia%20%E4%B8%AD%E5%AE%9A%E4%B9%89%E5%92%8C%E8%B0%83%E7%94%A8%E5%87%BD%E6%95%B0.md)
- 2. [理解函数实参与形参](02-%E7%90%86%E8%A7%A3%E5%87%BD%E6%95%B0%E5%AE%9E%E5%8F%82%E4%B8%8E%E5%BD%A2%E5%8F%82.md)
- 3. [为参数指定默认值](03-%E4%B8%BA%E5%8F%82%E6%95%B0%E6%8C%87%E5%AE%9A%E9%BB%98%E8%AE%A4%E5%80%BC.md)
- 4. [函数返回值](04-%E5%87%BD%E6%95%B0%E8%BF%94%E5%9B%9E%E5%80%BC.md)
- 5. [管理变量作用域和生命周期](05-%E7%AE%A1%E7%90%86%E5%8F%98%E9%87%8F%E4%BD%9C%E7%94%A8%E5%9F%9F%E5%92%8C%E7%94%9F%E5%91%BD%E5%91%A8%E6%9C%9F.md)
- 6. [编写简洁的匿名函数](06-%E7%BC%96%E5%86%99%E7%AE%80%E6%B4%81%E7%9A%84%E5%8C%BF%E5%90%8D%E5%87%BD%E6%95%B0.md)
- 7. [多重派发：Julia 的一个独特功能](07-%E5%A4%9A%E9%87%8D%E6%B4%BE%E5%8F%91%EF%BC%9AJulia%20%E7%9A%84%E4%B8%80%E4%B8%AA%E7%8B%AC%E7%89%B9%E5%8A%9F%E8%83%BD.md)
- 8. [使用文档字符串为函数编写文档](08-%E4%BD%BF%E7%94%A8%E6%96%87%E6%A1%A3%E5%AD%97%E7%AC%A6%E4%B8%B2%E4%B8%BA%E5%87%BD%E6%95%B0%E7%BC%96%E5%86%99%E6%96%87%E6%A1%A3.md)
- 9. [动手实践：构建和使用自定义函数](09-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E5%92%8C%E4%BD%BF%E7%94%A8%E8%87%AA%E5%AE%9A%E4%B9%89%E5%87%BD%E6%95%B0.md)

章节测验：[在线测验](https://apxml.com/zh/courses/getting-started-julia-programming/chapter-5-functions-creating-reusable-code/quiz)
