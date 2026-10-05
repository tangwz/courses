# 第 3 章：表达式与控制流结构

来源：[原章节](https://apxml.com/zh/courses/getting-started-julia-programming/chapter-3-expressions-control-flow-structures)

[返回课程目录](../README.md)

具备变量和数据类型知识后，我们现在来讨论程序如何执行操作和做出判断。本章介绍表达式，即Julia组合值和运算符以生成新值的方式，以及控制流结构，它规定了语句的执行顺序。

您将学会：
*   编写表达式，使用算术、比较（例如，`$x > y$`）和逻辑运算符。
*   使用 `if`、`elseif` 和 `else` 语句实现条件执行路径。
*   创建用于重复任务的循环，使用 `for` 和 `while` 构造。
*   管理循环行为，使用 `break` 和 `continue`，并编写简洁的条件表达式，使用三元运算符。

完成本章后，您将能够编写程序，这些程序能够判断条件、执行多种计算并自动化重复过程。

## 小节

- 1. [使用运算符构建表达式](01-%E4%BD%BF%E7%94%A8%E8%BF%90%E7%AE%97%E7%AC%A6%E6%9E%84%E5%BB%BA%E8%A1%A8%E8%BE%BE%E5%BC%8F.md)
- 2. [算术与赋值运算](02-%E7%AE%97%E6%9C%AF%E4%B8%8E%E8%B5%8B%E5%80%BC%E8%BF%90%E7%AE%97.md)
- 3. [比较运算符和逻辑运算符](03-%E6%AF%94%E8%BE%83%E8%BF%90%E7%AE%97%E7%AC%A6%E5%92%8C%E9%80%BB%E8%BE%91%E8%BF%90%E7%AE%97%E7%AC%A6.md)
- 4. [使用 if-elseif-else 控制程序流程](04-%E4%BD%BF%E7%94%A8%20if-elseif-else%20%E6%8E%A7%E5%88%B6%E7%A8%8B%E5%BA%8F%E6%B5%81%E7%A8%8B.md)
- 5. [使用 for 循环重复任务](05-%E4%BD%BF%E7%94%A8%20for%20%E5%BE%AA%E7%8E%AF%E9%87%8D%E5%A4%8D%E4%BB%BB%E5%8A%A1.md)
- 6. [使用 \`while\` 循环进行条件迭代](06-%E4%BD%BF%E7%94%A8%20%60while%60%20%E5%BE%AA%E7%8E%AF%E8%BF%9B%E8%A1%8C%E6%9D%A1%E4%BB%B6%E8%BF%AD%E4%BB%A3.md)
- 7. [修改循环行为：\`break\` 和 \`continue\`](07-%E4%BF%AE%E6%94%B9%E5%BE%AA%E7%8E%AF%E8%A1%8C%E4%B8%BA%EF%BC%9A%60break%60%20%E5%92%8C%20%60continue%60.md)
- 8. [三元运算符：简洁的条件语句](08-%E4%B8%89%E5%85%83%E8%BF%90%E7%AE%97%E7%AC%A6%EF%BC%9A%E7%AE%80%E6%B4%81%E7%9A%84%E6%9D%A1%E4%BB%B6%E8%AF%AD%E5%8F%A5.md)
- 9. [动手实践：实现决策逻辑](09-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%E5%86%B3%E7%AD%96%E9%80%BB%E8%BE%91.md)

章节测验：[在线测验](https://apxml.com/zh/courses/getting-started-julia-programming/chapter-3-expressions-control-flow-structures/quiz)
