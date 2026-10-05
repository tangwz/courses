# 第 8 章：错误处理与基本调试

来源：[原章节](https://apxml.com/zh/courses/getting-started-julia-programming/chapter-8-handling-errors-basic-debugging)

[返回课程目录](../README.md)

即使精心编写的程序在运行时也可能遇到问题。本章讲解如何在Julia中应对这些情况。你将学会识别常见的错误来源，并运用方法妥善处理它们。我们将了解如何使用 `try-catch` 语句块进行异常管理，`finally` 子句在确保清理操作中的作用，以及定义自定义错误类型的方法。此外，本章还会介绍基本的调试策略，帮助你诊断并解决问题，从而编写出更可靠的代码。

## 小节

- 1. [Julia 程序中的错误来源](01-Julia%20%E7%A8%8B%E5%BA%8F%E4%B8%AD%E7%9A%84%E9%94%99%E8%AF%AF%E6%9D%A5%E6%BA%90.md)
- 2. [使用 \`try-catch\` 进行异常处理](02-%E4%BD%BF%E7%94%A8%20%60try-catch%60%20%E8%BF%9B%E8%A1%8C%E5%BC%82%E5%B8%B8%E5%A4%84%E7%90%86.md)
- 3. [使用 \`finally\` 保证代码执行](03-%E4%BD%BF%E7%94%A8%20%60finally%60%20%E4%BF%9D%E8%AF%81%E4%BB%A3%E7%A0%81%E6%89%A7%E8%A1%8C.md)
- 4. [定义和抛出自定义错误](04-%E5%AE%9A%E4%B9%89%E5%92%8C%E6%8A%9B%E5%87%BA%E8%87%AA%E5%AE%9A%E4%B9%89%E9%94%99%E8%AF%AF.md)
- 5. [常见错误情况和处理方法](05-%E5%B8%B8%E8%A7%81%E9%94%99%E8%AF%AF%E6%83%85%E5%86%B5%E5%92%8C%E5%A4%84%E7%90%86%E6%96%B9%E6%B3%95.md)
- 6. [Julia 代码调试策略](06-Julia%20%E4%BB%A3%E7%A0%81%E8%B0%83%E8%AF%95%E7%AD%96%E7%95%A5.md)
- 7. [Julia 调试工具简介](07-Julia%20%E8%B0%83%E8%AF%95%E5%B7%A5%E5%85%B7%E7%AE%80%E4%BB%8B.md)
- 8. [练习：编写健壮代码与错误处理](08-%E7%BB%83%E4%B9%A0%EF%BC%9A%E7%BC%96%E5%86%99%E5%81%A5%E5%A3%AE%E4%BB%A3%E7%A0%81%E4%B8%8E%E9%94%99%E8%AF%AF%E5%A4%84%E7%90%86.md)

章节测验：[在线测验](https://apxml.com/zh/courses/getting-started-julia-programming/chapter-8-handling-errors-basic-debugging/quiz)
