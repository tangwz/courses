# Julia 程序中的错误来源

来源：[原文](https://apxml.com/zh/courses/getting-started-julia-programming/chapter-8-handling-errors-basic-debugging/sources-errors-julia-programs)

[返回章节目录](README.md) · [返回课程目录](../README.md)

无论你多么仔细地编写 Julia 代码，错误都是编程过程中不可避免的一部分。不要将它们视为失败，而应看作是引导你写出更完善程序的指引。弄清楚这些错误通常的来源，是学习如何有效管理它们的第一步。Julia 中的错误，与许多编程语言一样，通常可以归为几大类。

> Julia 编程中常见错误类别的分类说明。

### 语法错误

语法错误就像句子中的语法错误。Julia 的解释器会读取你的代码，如果遇到不符合语言规则（其语法）的内容，它就会停止并报告一个语法错误。这些错误会阻止你的程序启动运行。

常见原因包括：

- **拼写错误：** 关键词（例如，将 `function` 拼写为 `functon`）、变量名或函数名拼写错误。
- **标点错误：** 缺少括号 `()`、方括号 `[]`、花括号 `{}`、逗号 `,`，或将其用在错误的位置。
- **`begin`/`end` 块不匹配：** 忘记循环、条件块或函数定义中的 `end` 语句。
- **无效的运算符或表达式：** 使用 Julia 不识别为有效操作的符号。

例如，如果你编写：

```julia
println("Hello, Julia" # 缺少右括号
x = 5 +
```

Julia 会在尝试执行它们之前立即将这些标记 (token)为错误。好消息是，Julia 通常会提供明确的信息来指出它认为语法错误发生的位置，使得这些错误相对容易修复。在这些错误得到修正之前，你的代码根本无法运行。

### 运行时错误 (异常)

运行时错误，在 Julia 中通常被称为异常，发生在程序实际运行期间。你的代码语法可能完全正确，但在执行过程中出现了程序未准备好处理的意外情况。当发生运行时错误时，Julia 通常会暂停程序执行并打印一条错误消息。此消息通常包含“堆栈跟踪”，这是一个报告，显示了导致错误发生的函数调用序列，有助于你诊断问题。

下面我们来看几种你可能遇到的常见运行时错误类型：

- **`TypeError`**：当你尝试对不兼容类型的数据执行操作时会发生此错误。例如，Julia 知道如何将两个数字相加（例如，`2 + 3`），但它本身不知道如何将数字与文本字符串相加（例如，`2 + "hello"`）。尝试此类操作将导致 `TypeError`。
- **`MethodError`**：Julia 的函数可以根据它们接收的参数 (parameter)类型而有不同的行为（方法）。这是一个称为多重派发的特性。如果你使用没有定义特定方法的参数类型组合调用函数，就会发生 `MethodError`。例如，如果你有一个专门定义为接受整数的函数 `process_data(x::Int)`，那么使用浮点数（如 `process_data(3.14)`）调用它将触发 `MethodError`，因为没有 `process_data` 的方法与该输入类型匹配。
- **`UndefVarError`**：此错误表示你尝试使用尚未赋值的变量，或者可能拼错了它的名称。Julia 需要知道变量的值才能在计算或操作中使用它。例如：

  ```julia
  # println(my_variable) # 如果 my_variable 未定义，这将导致 UndefVarError
  # y = x + 5            # 如果 x 未定义，这也会导致 UndefVarError
  ```
- **`BoundsError`**：当处理像数组这样的集合时，你使用索引来访问元素（例如，`my_array[1]`）。如果你尝试使用超出该集合有效索引范围的索引来访问元素，就会发生 `BoundsError`。如果数组 `my_numbers` 有 3 个元素，有效的索引通常是 `1`、`2` 和 `3`（因为 Julia 数组默认从 1 开始索引）。尝试访问 `my_numbers[4]` 或 `my_numbers[0]` 将导致 `BoundsError`。
- **`DomainError`**：某些数学函数仅在特定范围（定义域）的输入值下定义。如果你使用超出此有效定义域的参数调用函数，就会发生 `DomainError`。一个典型例子是尝试使用 `sqrt()` 计算负数的平方根，`sqrt()` 期望非负实数：`sqrt(-4.0)` 会引发 `DomainError`，因为 -4.0 不在标准实数平方根函数的有效定义域内。
- **`DivideError`**：顾名思义，当尝试将一个数除以零时会发生此错误。例如，`10 / 0` 这样的表达式将导致 `DivideError`。
- **输入/输出错误（例如，`IOError`，`SystemError`）**：当你的程序在与外部资源交互时遇到问题时，会发生这些错误，最常见的是处理文件时。例子包括尝试打开一个不存在的文件，尝试写入程序没有修改权限的文件，或者在写入文件时磁盘空间不足。

本章将重点介绍如何使用 `try-catch` 块来预期并优雅地处理这些运行时错误，防止你的程序意外崩溃，并使你能够管理此类情况。

### 逻辑错误

逻辑错误通常是最难检测和修复的。当存在逻辑错误时，你的程序在没有语法错误或运行时异常的情况下运行，但它不会产生正确的结果或按预期运行。代码是 Julia 可以执行的有效 Julia 代码，但程序员实现的底层逻辑或算法存在缺陷。

逻辑错误的例子包括：

- **使用错误的比较运算符**（例如，用 `temperature > 0` 检查水是否结冰，而你的本意是 `temperature < 0`）。
- **计算中错误的数学公式**（例如，将矩形面积计算为 `length + width` 而不是 `length * width`）。
- **有缺陷的算法设计**，未能正确解决所有可能的输入或边界情况下的问题。
- **循环中的差一错误**，即循环多运行一次或少运行一次，导致遗漏元素或计数不正确。

Julia 本身无法告诉你你的逻辑是错误的，因为从它的角度来看，这些指令是有效且可执行的。识别逻辑错误需要用各种输入仔细测试你的程序，进行细致的调试（我们将在本章后面提及），并清晰了解你的程序应该实现什么。你可能会发现自己需要单步执行代码，打印出变量的中间值，或者使用调试工具来跟踪执行流程，并精确找出程序行为与预期不符之处。

理解这些不同类别的错误非常重要。语法错误由 Julia 及早捕获。运行时错误是我们通常旨在“处理”的事件，以使我们的程序更具弹性。逻辑错误则需要一种不同的侦察工作，通常涉及对代码行为更细致的分析。随着经验的增加，你将更擅长识别这些错误的模式，并更快地解决它们。

## 参考资料

- [Error Handling](https://docs.julialang.org/en/v1/manual/control-flow/#Error-Handling) — The Julia Language (2024)
  Publisher: The Julia Language
  提供了Julia错误处理机制的全面指南，包括`try-catch`块和异常的工作原理，与理解运行时错误直接相关。
- [The Julia Language Manual: Types and Methods](https://docs.julialang.org/en/stable/manual/) — The Julia Language (2024)
  Publisher: The Julia Language
  涵盖了Julia的类型系统和多重分派，这些是理解`TypeError`和`MethodError`的核心概念。
- [Think Julia: How to Think Like a Computer Scientist](https://benlauwens.github.io/ThinkJulia.jl/latest/book.html) — Ben Lauwens and Allen B. Downey (2019)
  Publisher: O'Reilly Media
  一本介绍性的教科书，涵盖了Julia中的核心编程概念，包括常见的错误和针对初学者的调试策略。

---

[上一节](../07-%E8%BE%93%E5%85%A5%E3%80%81%E8%BE%93%E5%87%BA%E4%B8%8E%E6%96%87%E4%BB%B6%E7%B3%BB%E7%BB%9F%E6%93%8D%E4%BD%9C/08-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E7%BC%96%E5%86%99%E6%96%87%E4%BB%B6%E6%93%8D%E4%BD%9C%E8%84%9A%E6%9C%AC.md) · [下一节](02-%E4%BD%BF%E7%94%A8%20%60try-catch%60%20%E8%BF%9B%E8%A1%8C%E5%BC%82%E5%B8%B8%E5%A4%84%E7%90%86.md)
