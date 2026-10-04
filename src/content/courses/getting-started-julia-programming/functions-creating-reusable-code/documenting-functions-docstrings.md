---
course: "getting-started-julia-programming"
chapter: "functions-creating-reusable-code"
lesson: "documenting-functions-docstrings"
sourceId: 6800
sourceUrl: "https://apxml.com/zh/courses/getting-started-julia-programming/chapter-5-functions-creating-reusable-code/documenting-functions-docstrings"
title: "使用文档字符串为函数编写文档"
description: "学习如何编写文档字符串来为你的 Julia 函数编写文档，使它们更易于理解和使用。"
order: 8
plots: []
sourceHash: "2f8da73e0b2462c0cf89f0d5a82413d6de285d54a0c1147624336b9857ba69cd"
sourceCorrections: []
---

编写能运行的代码是一回事；编写让其他人（以及未来的自己）能有效理解和使用的代码则是另一回事。随着你的函数变得更复杂，或者当你开始分享你的代码时，清晰的文档对于保持函数的可读性和组织性变得非常重要。文档字符串是 Julia 中文档函数的主要方式。

文档字符串是紧接在函数定义之前写的一段文本。Julia 的内置帮助系统和其他文档工具可以找到这段文本，并将其显示为该函数的官方帮助信息。这种做法不仅仅是形式；它是编写可维护和可用代码的一个重要组成部分。可以把它看作是直接嵌入 (embedding)到函数中的操作手册，当你（包括你自己）需要理解函数如何工作时，它随时可用。

### 好的文档字符串有哪些特点？

一个有用的文档字符串清楚地传达了关于函数的几个重要信息：

- **目的：** 函数是做什么的？它解决了什么问题？
- **参数 (parameter)：** 它期望什么输入？它们的类型和含义是什么？
- **返回值：** 它产生什么输出？它的类型和含义是什么？
- **行为：** 关于其操作方式或任何副作用（例如打印到控制台或修改全局变量）有哪些重要细节？
- **示例：** 函数通常如何使用？简短、说明性的示例非常有帮助。

### 在 Julia 中编写文档字符串

在 Julia 中，文档字符串最常用三引号（`"""..."""`）编写为多行字符串。这使得你的文档可以跨越多行，便于阅读。

文档字符串必须紧接在 `function` 关键字（或其他定义，如类型或模块，尽管我们这里主要关注函数）之前出现。

这里有一个基本结构：

```julia
"""
函数功能的简要单行总结。

如果需要，可以在此处提供更详细的说明。这部分
可以描述函数的行为、其算法，或其用法中的任何
细节。
"""
function my_simple_function(argument1, argument2)
    # ... 函数实现 ...
end
```

对于不简单的函数，结构更清晰的文档字符串非常有益。虽然 Julia 很灵活，但遵循常见约定会让你的文档字符串更易于理解。让我们看一个更完整的示例：

````julia
"""
    calculate_rectangle_area(length::Number, width::Number)

计算给定长度和宽度的矩形面积。

# 参数
- `length::Number`: 矩形的长度。它应该是一个数值。
- `width::Number`: 矩形的宽度。它也应该是一个数值。

# 返回值
- `Number`: 计算出的矩形面积。

# 示例
下面的 `jldoctest` 块演示了如何使用该函数以及预期的输出。
这些示例也可以自动测试。
```jldoctest
julia> calculate_rectangle_area(10, 5)
50

julia> calculate_rectangle_area(3.5, 2.0)
7.0
```
"""
function calculate_rectangle_area(length::Number, width::Number)
    if length < 0 || width < 0
        error("长度和宽度必须是非负数。")
    end
    return length * width
end
````

让我们分析一下这个 `calculate_rectangle_area` 文档字符串的组成部分：

1. **第一行（函数签名和简洁摘要）：** `calculate_rectangle_area(length::Number, width::Number)`
   通常和好的做法是，文档字符串以函数的签名（其名称和参数 (parameter)名称，通常包含类型注解以提高清晰度）开头。这会立即告诉读者如何调用该函数。有些人喜欢将非常简短的总结放在这一行或紧随其后的下一行。所示的缩进是 Julia 社区中广泛采用的风格。
2. **详细描述：** “计算给定长度和宽度的矩形面积。”
   这提供了函数总体目的的清晰、易于理解的说明。
3. **`# 参数` 部分：**
   本节详细说明每个输入参数：

   - 它通常以 `# Arguments` 这样的标题开头。
   - 每个参数都列出，通常以项目符号（例如，使用 `-`）形式。
   - `length::Number`: 参数名称，可选地后跟其预期类型（`::Number`）。
   - 说明参数代表什么以及任何约束（例如，“它应该是一个数值”）。
4. **`# 返回值` 部分：**
   本节说明函数输出什么：

   - 以 `# Returns` 这样的标题开头。
   - 描述函数返回的值，包括其类型（例如，`Number`）。
5. **`# 示例` 部分：**
   这是文档字符串最有价值的部分之一。

   - 它以 `# Examples` 这样的标题开头。
   - 示例通常放在 `jldoctest` 块中。像 `julia> ...` 这样编写的代码模拟 Julia REPL 的输入，而紧随其后的行（没有 `julia>`）显示预期输出。
   - `jldoctest` 块很特殊：文档生成工具可以自动运行这些示例，以验证它们的正确性并与函数的当前行为匹配。这对于保持文档与代码更改同步非常有帮助。

你并非严格要求包含所有这些部分，特别是对于非常简单、内部的辅助函数。然而，对于任何旨在广泛使用或具有非显而易见行为的函数，这种详细程度强烈推荐，并且将节省时间并避免后续的困惑。

### 访问文档字符串

Julia 使得访问这些文档字符串非常容易。最常见的方法是使用 Julia REPL 中的帮助模式。为此，只需输入 `?`，后跟函数名，然后按回车键：

```
julia> ?calculate_rectangle_area
  calculate_rectangle_area(length::Number, width::Number)

  计算给定长度和宽度的矩形面积。

  参数
  ≡≡≡≡≡≡≡≡≡≡≡

    •  length::Number: 矩形的长度。它应该是一个数值。
    •  width::Number: 矩形的宽度。它也应该是一个数值。

  返回值
  ≡≡≡≡≡≡≡

    •  Number: 计算出的矩形面积。

  示例
  ≡≡≡≡≡≡≡≡≡≡≡

  下面的 jldoctest 块演示了如何使用该函数以及预期的输出。这些示例也可以自动测试。

  julia> calculate_rectangle_area(10, 5)
  50

  julia> calculate_rectangle_area(3.5, 2.0)
  7.0
```

(你在 REPL 中的确切格式，例如像 `≡≡≡≡≡≡≡≡≡≡` 这样的水平线，可能会因你的 Julia 版本、终端设置以及任何已安装的包而略有不同。但是，文档字符串中的核心信息将显示出来。)

许多支持 Julia 的集成开发环境（IDE）和代码编辑器也使用文档字符串。当你将鼠标悬停在函数调用上时，它们可能会自动显示此信息，或者提供包含文档字符串部分内容的自动完成建议。

### 为什么要在文档字符串上投入时间？

投入一些额外时间编写好的文档字符串是一项前期投入，但从长远来看会带来丰厚的回报。它使你的代码更易于理解、更易于维护，并且让每个人在使用时都更愉快。从你开始 Julia 编程经验时就培养这个习惯将使你受益匪浅。

- **为你未来的自己提供清晰度：** 你今天编写的代码可能看起来很显而易见，但当你几周、几个月甚至几年后重新查看它时，好的文档字符串对于快速回想函数如何工作以及为何这样编写将非常有价值。
- **改进协作：** 如果你与他人合作项目，文档字符串是必不可少的。它们使团队成员能够正确理解和使用彼此的代码，而无需阅读实现的每一行。
- **增强可用性：** 文档齐全的函数对于任何人（包括你自己）来说，都更容易掌握并集成到更大的程序中或用于新任务。
- **自动化文档的重要部分：** 像 `Documenter.jl` 这样的工具可以解析你的文档字符串，并为你的 Julia 项目自动生成具有专业外观的 HTML 文档网站。这是大多数 Julia 包的标准做法。

## 参考资料

- [Functions and Methods](https://docs.julialang.org/en/v1/manual/functions/#Docstrings) — The Julia Language Developers (2024)
  Julia的官方文档，解释了文档字符串的基本概念、语法以及它们如何与Julia内置帮助系统集成。
- [Style Guide](https://docs.julialang.org/en/v1/manual/style-guide/) — The Julia Language Developers (2024)
  官方Julia风格指南，提供了编写一致且可读代码的社区推荐，包括有效文档字符串内容和格式的指导。
- [Writing Documentation](https://documenter.julia.dev/stable/man/guide/) — The Documenter.jl Developers (2025)
  Documenter.jl的指南，这是一个用于生成文档的标准Julia包。此部分详细介绍了如何使用文档字符串（包括`jldoctest`块）来构建专业的项目文档。
- [Mastering Julia](https://www.packtpub.com/product/mastering-julia/9781783553327) — Malcolm Sherrington (2015)
  Publisher: Packt Publishing
  一本关于Julia编程的综合性书籍，提供了实用建议，包括代码组织、风格和在Julia项目中创建有效文档的最佳实践。
