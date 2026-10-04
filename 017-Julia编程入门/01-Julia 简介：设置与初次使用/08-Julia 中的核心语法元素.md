# Julia 中的核心语法元素

来源：[原文](https://apxml.com/zh/courses/getting-started-julia-programming/chapter-1-introducing-julia-setup-first-steps/core-syntax-elements-julia)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Julia 代码的编写受基本规则和约定支配。理解这些核心语法元素将帮助你从一开始就编写出清晰、正确且易读的程序。

### 语句：你给 Julia 的指示

在 Julia 中，程序是**语句**的序列。每个语句都是一个指示，告诉计算机执行一个操作。通常，你每行编写一个语句。例如，给变量赋值或在屏幕上打印内容都是常见语句：

```julia
x = 10
println("Hello, Julia!")
```

Julia 通过换行符确定语句的结束。如果你想在一行中编写多个语句，可以用分号（`;`）分隔它们。

```julia
a = 5; b = 10; c = a + b
println(c) # 这将打印 15
```

虽然可以在一行中使用分号分隔多个语句，但通常建议每行写一个语句以提高可读性，特别是当你刚开始学习时。

### 注释：为你的代码添加说明

随着你的程序变大，你会想添加笔记来解释你的代码做了什么、为什么这样做或它是如何工作的。这些笔记称为**注释**。Julia 在运行代码时会忽略注释，因此它们纯粹是为了人类读者（包括未来的你自己！）而存在。

Julia 有两种主要方式来编写注释：

1. **单行注释**：行中井号（`#`）后面的任何内容都被视为注释。

   ```julia
   # 这是一个单行注释。
   radius = 5 # 将值 5 赋给变量 'radius'
   area = 3.14159 * radius^2 # 计算面积
   ```
2. **多行注释**：你可以将跨多行的注释括在 `#=` 和 `=#` 之间。

   ```julia
   #=
   这是一个多行注释。
   它可以跨多行，对较长的解释
   或者暂时禁用一段代码很有用。
   =#
   x = 100 # 这一行不属于上面的多行注释
   ```

良好的注释习惯能让你的代码更容易理解和维护。尽量多解释“为什么”，而不是“做什么”，特别是如果代码本身很简单。

### 空格和缩进

空格指空格符、制表符和换行符。Julia 对于代码元素之间的空格通常很灵活，但一致地使用它会使你的代码更易读。例如，围绕运算符（如 `+`、`-`、`=` 等）放置空格是常见做法：

```julia
# 好的做法：
result = (value1 + value2) * factor

# 可读性较差：
result=(value1+value2)*factor
```

**缩进**指行首的空格。虽然 Julia 对代码运行不强制要求缩进规则（不像 Python 等语言中缩进定义了代码块），但一致的缩进对于可读性极其重要。它有助于在视觉上将相关的代码行分组，特别是在你之后会学到的循环或条件语句等控制结构中。大多数 Julia 程序员对每个缩进级别使用 4 个空格。

```julia
# 良好缩进的例子（稍后会更多介绍 'if'）
if x > 10
    println("x is greater than 10")
    # 此处的更多操作也应缩进
end 
```

你的代码编辑器通常可以帮助你自动缩进。

### 标识符：在 Julia 中命名事物

**标识符**是你给变量、函数和代码中其他实体起的名字。选择好的、有描述性的名字是编写易懂程序的基础。

以下是 Julia 中标识符的基本规则：

- 它们区分大小写，意味着 `myVariable` 和 `myvariable` 是不同的名字。
- 它们可以由字母 (A-Z, a-z)、数字 (0-9)、下划线 (`_`) 以及各种 Unicode 字符（如 `π` 或 `α`）组成。
- 第一个字符必须是字母、下划线或被归类为字母的 Unicode 字符。它不能是数字。
- 有一些保留词，称为**关键字**（如 `if`、`else`、`while`、`function`），它们在 Julia 中有特殊含义，不能用作标识符。

常见的 Julia 命名约定包括：

- 变量和函数名通常是小写，单词之间用下划线分隔（`snake_case`），例如 `user_age`、`calculate_mean`。
- 另外，`camelCase`（例如 `userAge`、`calculateMean`）也常见，尽管 `snake_case` 在变量和函数中更普遍。
- 类型和模块的名称通常以大写字母开头，并使用 `CamelCase`（例如 `MyCustomType`、`GraphicsModule`）。

```julia
# 有效标识符
count = 0
student_name = "Alice"
_temp_value = 25.5
π_val = 3.14159 # Julia 支持 Unicode 字符

# 无效标识符（以数字开头）
# 1st_place = "Gold" # 这会引起错误
```

尽管 Julia 对标识符中 Unicode 的支持功能强大，特别是对于科学计算（例如，直接在代码中使用 `δ` 或 `Σ`），但如果你与可能拥有不同键盘设置或编辑器支持的其他人协作，或者如果你刚开始学习，最好坚持使用标准的 ASCII 字符（字母、数字、下划线）。

### 关键字：Julia 的保留字

Julia 有一组**关键字**，它们具有预定义含义并构成语言的基本结构。你不能将这些关键字用作变量或函数的名称。一些例子包括：

`if`、`else`、`elseif`、`while`、`for`、`function`、`struct`、`module`、`begin`、`end`、`try`、`catch`、`finally`、`return`、`true`、`false`、`const`。

你现在不需要记住所有这些关键字。随着你学习更多 Julia 构造，你会自然而然地熟悉它们。如果你不小心尝试将关键字用作变量名，Julia 会给你一个错误消息。例如，`if = 10` 将是无效的。

### 代码块和 `end` 关键字

Julia 中的许多结构，例如条件语句（`if`）、循环（`for`、`while`）、函数定义（`function`）和模块定义（`module`），都定义了代码**块**。这些代码块通常由 `end` 关键字终止。这是 Julia 语法的一个基本方面，它将相关语句组合在一起。

```julia
# 一个你稍后会更清楚理解的简单例子
x = 5
if x < 10            # 'if' 块的开始
    println("x is less than 10")
    y = x * 2        # 这个语句是 'if' 块的一部分
    println(y)
end                  # 标记 'if' 块的结束

println("这一行在 if 块之外。")
```

始终使用 `end` 来关闭代码块显著有助于代码清晰度。

通过牢记这些核心语法元素，编写清晰的语句，有效使用注释，保持一致的缩进，选择合理的名字，以及理解代码块的结构方式，你将很好地开始编写 Julia 代码，这些代码不仅功能正常，而且易于他人（和你自己）阅读和理解。随着你的进步，这些规则将变得熟练掌握。

## 参考资料

- [Julia Documentation](https://docs.julialang.org/en/v1/) — The Julia Language Developers (2024)
  Julia编程语言的官方和最权威来源，涵盖所有核心语法元素。
- [Think Julia: How to Think Like a Computer Scientist](https://www.oreilly.com/library/view/think-julia/9781492045038/) — Ben Lauwens and Allen B. Downey (2019)
  Publisher: O'Reilly Media; Pages: 295
  一本优秀的入门书籍，使用Julia教授基础编程概念，包括基本语法和结构。
- [Julia Programming for Engineers and Scientists](https://press.siam.org/store/sa24) — Alan Edelman, David P. Sanders, and Charles F. Van Loan (2021)
  Publisher: Society for Industrial and Applied Mathematics (SIAM)
  由Julia的联合创始人之一撰写，本书对该语言进行了严谨的介绍，从其基本语法开始。

---

[上一节](07-%E7%BC%96%E5%86%99%E5%92%8C%E8%BF%90%E8%A1%8C%E4%BD%A0%E7%9A%84%E7%AC%AC%E4%B8%80%E4%B8%AA%20Julia%20%E8%84%9A%E6%9C%AC.md) · [下一节](09-%E8%8E%B7%E5%8F%96Julia%E6%96%87%E6%A1%A3%E4%B8%8E%E6%94%AF%E6%8C%81.md)
