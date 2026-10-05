# 第 9 章：处理错误和异常

来源：[原章节](https://apxml.com/zh/courses/python-for-beginners/chapter-9-handling-errors-exceptions)

[返回课程目录](../README.md)

程序很少能在所有条件下完美运行。意料之外的情况，从无效的用户输入到文件缺失，都可能导致运行时错误。Python 使用一种称为“异常”的机制来提示和处理这些错误。正确处理异常对于构建稳定且用户友好的应用程序很重要。

本章介绍 Python 处理错误的方法。你将学会如何预见可能出现的问题，并编写在问题发生时能够妥善响应的代码。我们会涵盖：

*   理解语法错误和运行时异常之间的区别。
*   使用 `try` 和 `except` 块来捕获和处理异常。
*   处理特定类型的异常，以实现更有针对性的错误管理。
*   使用可选的 `else` 块，在没有异常发生时执行代码。
*   运用 `finally` 块，确保清理操作始终运行。
*   有意地使用 `raise` 语句引发异常。

## 小节

- 1. [理解 Python 中的错误](01-%E7%90%86%E8%A7%A3%20Python%20%E4%B8%AD%E7%9A%84%E9%94%99%E8%AF%AF.md)
- 2. [异常介绍](02-%E5%BC%82%E5%B8%B8%E4%BB%8B%E7%BB%8D.md)
- 3. [处理异常：try 和 except 块](03-%E5%A4%84%E7%90%86%E5%BC%82%E5%B8%B8%EF%BC%9Atry%20%E5%92%8C%20except%20%E5%9D%97.md)
- 4. [处理特定异常类型](04-%E5%A4%84%E7%90%86%E7%89%B9%E5%AE%9A%E5%BC%82%E5%B8%B8%E7%B1%BB%E5%9E%8B.md)
- 5. [异常处理中的else块](05-%E5%BC%82%E5%B8%B8%E5%A4%84%E7%90%86%E4%B8%AD%E7%9A%84else%E5%9D%97.md)
- 6. [\`finally\` 块：清理操作](06-%60finally%60%20%E5%9D%97%EF%BC%9A%E6%B8%85%E7%90%86%E6%93%8D%E4%BD%9C.md)
- 7. [手动引发异常](07-%E6%89%8B%E5%8A%A8%E5%BC%95%E5%8F%91%E5%BC%82%E5%B8%B8.md)
- 8. [实践：实现错误处理](08-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%E9%94%99%E8%AF%AF%E5%A4%84%E7%90%86.md)

章节测验：[在线测验](https://apxml.com/zh/courses/python-for-beginners/chapter-9-handling-errors-exceptions/quiz)
