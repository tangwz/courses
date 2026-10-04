---
course: "getting-started-julia-programming"
chapter: "variables-data-types-operations"
lesson: "tour-julia-built-in-data-types"
sourceId: 6768
sourceUrl: "https://apxml.com/zh/courses/getting-started-julia-programming/chapter-2-variables-data-types-operations/tour-julia-built-in-data-types"
title: "Julia 内置数据类型概览"
description: "Julia 中可用的基本内置数据类型概述，例如数值、布尔值和字符。"
order: 2
plots: []
sourceHash: "c696d6870d4f508edb12d93ef4c9f779dd568f15196512921a137d2877e1e7e8"
sourceCorrections: []
---

你在 Julia 中处理的每一份数据，无论是简单的数字还是一行文本，都有一个特定的“类型”。可以将数据类型视为一个标签，它告诉 Julia 一个值是什么种类的数据，以及更重要的是，你可以用它做什么。例如，你可以对数字进行算术运算，但尝试将一段文本除以另一段文本就没有意义了。Julia 拥有一套丰富的内置数据类型，理解它们对于编写高效代码非常重要。这些类型还有助于 Julia 高效地管理内存，并通常有助于实现其高性能。

### 数值类型

数字是许多计算的根本，Julia 为各种需求提供了全面的类型集。

**整型**
整型是整数，例如 -5、0 或 42。Julia 有多种整型，它们在可存储值的范围上有所不同，这取决于它们在内存中使用的位数。常见类型包括 `Int8`、`Int16`、`Int32`、`Int64` 和 `Int128`。默认情况下，如果你输入一个像 `100` 这样的整数，Julia 在 64 位系统上通常会将其推断为 `Int64`。这种类型提供了很大的范围，适用于大多数通用任务。

```julia
age = 30
println(typeof(age)) # 输出很可能是 Int64

small_number = Int8(10) # 明确的 8 位整型
println(typeof(small_number)) # 输出：Int8
```

Julia 还提供无符号整型（例如，`UInt8`、`UInt64`），它们只表示非负值。与有符号对应类型相比，这实际上使它们可存储的正值范围增加了一倍。例如，`UInt8` 可以存储从 0 到 255 的值，而 `Int8` 存储从 -128 到 127 的值。

**浮点数**
浮点数表示实数，包括带小数点的数，如 3.14159 或 -0.001。它们对于科学计算以及任何需要小数值的场景都非常重要。最常见类型是 `Float16`（半精度）、`Float32`（单精度）和 `Float64`（双精度）。Julia 通常会将带小数点的数字默认设为 `Float64`，这为大多数应用提供了精度和范围的良好平衡。

```julia
pi_approx = 3.14159
println(typeof(pi_approx)) # 输出：Float64

price = 19.99
println(typeof(price)) # 输出：Float64

scientific_notation = 6.022e23 # 阿伏伽德罗常数
println(scientific_notation)    # 输出：6.022e23
println(typeof(scientific_notation)) # 输出：Float64
```

请记住，浮点运算有时会存在微小的精度限制，这是由于计算机以二进制形式存储这些数字的方式造成的。这是计算机算术的一个普遍方面，并非 Julia 特有。

**其他数值类型**
Julia 还支持更专业的数值类型，例如用于复数的 `Complex`（例如，`3 + 4im`，其中 `im` 是虚数单位）和用于精确分数的 `Rational`（例如，`2//3`）。虽然我们在本入门课程中不会大量使用它们，但了解它们的存在对于更高级的数学工作或需要精确分数运算时很有帮助。

### 布尔值

布尔值表示真假。在 Julia 中，`Bool` 类型只有两个可能的值：`true` 或 `false`。它们对于程序中的决策非常重要，常用于条件语句（我们将在下一章介绍）来控制执行流程。

```julia
is_learning = true
is_active_user = false

println(typeof(is_learning)) # 输出：Bool
println(is_active_user)      # 输出：false
```

可以将布尔值视为代码中是/否问题的答案，指导其逻辑。

### 字符

在 Julia 中，由 `Char` 类型表示的字符是单个文本单位。这可以是一个字母、一个数字、一个符号，甚至是空格。`Char` 值始终用单引号括起来。

```julia
first_letter = 'A'
punctuation_mark = '!'
digit_char = '7'

println(typeof(first_letter)) # 输出：Char
println(punctuation_mark)     # 输出：!
```

Julia 的 `Char` 类型完全支持 Unicode。这意味着它可以表示几乎所有书写系统中的字符，包括带重音的字母、像 `'π'` (pi) 这样的符号或表情符号。

### 字符串

字符串是用于表示文本的字符序列。在 Julia 中，`String` 类型用于此目的。字符串字面量（代码中的实际字符串值）用双引号括起来。

```julia
greeting = "Hello, Julia!"
user_name = "Alex"
empty_string = ""

println(typeof(greeting)) # 输出：String
println(user_name)        # 输出：Alex
```

可以将字符串视为 `Char` 值的有序集合。字符串功能非常多样，你会使用它们来处理从向用户显示消息、读写文本文件到处理文本数据的所有事情。我们将在后面的章节中介绍更多字符串操作。

### `Any` 类型

Julia 有一个特殊的抽象类型，名为 `Any`。它是所有其他类型的“超类型”，这意味着一个明确声明为 `Any` 的变量，或一个在特定点无法更具体地推断其类型的变量，可以保存任何数据类型的值。

```julia
mystery_value::Any = 100
println(mystery_value)         # 输出：100
println(typeof(mystery_value)) # 输出：Int64 (Julia 知道值的具体类型)

mystery_value = "Now I am a string"
println(mystery_value)         # 输出：Now I am a string
println(typeof(mystery_value)) # 输出：String
```

尽管 `Any` 提供了灵活性，但通常建议在你知道正在处理哪种数据时使用更具体的类型。特定类型允许 Julia 更好地优化代码，并使你的程序更易于理解且不易出错。

### 类型层级结构一瞥

Julia 中的数据类型形成一个层级结构。最顶端是 `Any`。许多常见类型，如数字、字符串和布尔值，都是 `Any` 的子类型。这个层级结构对于理解 Julia 如何处理不同类型非常重要，特别是与函数相关的（我们稍后会提及的一个主题叫做多重派发）。

> Julia 类型层级结构的简化视图。`Any` 是最终的超类型。`Number`、`AbstractString`、`Bool` 和 `Char` 是它的一些直接或间接子类型。像 `Integer`（它本身是抽象的，有 `Int64` 这样的具体版本）、`AbstractFloat`（同样是抽象的）和 `String` 这样的具体类型都符合这个结构。

### 检查类型

Julia 提供了内置工具来检查变量或值的类型。这对于理解数据非常有用，尤其是在学习或调试时。

**`typeof()` 函数**
`typeof()` 函数返回值的具体、实际类型。

```julia
x = 10
println(typeof(x)) # 输出：Int64（或根据系统不同为 Int32）

y = 3.14
println(typeof(y)) # 输出：Float64

z = "hello"
println(typeof(z)) # 输出：String

b = true
println(typeof(b)) # 输出：Bool
```

**`isa()` 运算符/函数**
`isa()` 函数（也可以作为运算符与 `value isa Type` 一起使用）检查一个值是否为给定类型或其子类型之一。它返回 `true` 或 `false`。

```julia
println(isa(10, Int))         # 输出：true（如果 10 默认是 Int64 或 Int32，它们都是 Int 的子类型）
println(isa(10, Integer))     # 输出：true（Int64/Int32 是抽象类型 Integer 的子类型）
println(isa(10, Float64))     # 输出：false

println(isa("Julia", String)) # 输出：true
println(isa("Julia", Number)) # 输出：false
println(isa('J', Char))       # 输出：true
println(isa(true, Bool))      # 输出：true
```

当你想知道一个值是否属于更广泛的类型类别时，使用 `isa()` 特别有帮助（例如 `Integer` 包含 `Int8`、`Int16`、`Int32`、`Int64` 等，或者 `Number` 包含所有整型和浮点类型）。

### 常见内置数据类型总结

| 类型 | 描述 | 字面量示例 | `typeof` 示例 |
| --- | --- | --- | --- |
| `Int64` | 64 位有符号整型（常见默认值） | `123`, `-45` | `typeof(123)` 是 `Int64` |
| `Float64` | 64 位浮点数（常见默认值） | `3.14`, `-0.5`, `2e3` | `typeof(3.14)` 是 `Float64` |
| `Bool` | 布尔值（真假值） | `true`, `false` | `typeof(true)` 是 `Bool` |
| `Char` | 单个 Unicode 字符 | `'A'`, `'%'`, `'π'` | `typeof('A')` 是 `Char` |
| `String` | 字符序列（文本） | `"Hello"`, `""` | `typeof("Hello")` 是 `String` |

本篇介绍了 Julia 数据处理的基本组成部分。识别这些类型不仅仅是学术上的。数据的类型决定了：

- **哪些操作有效**: 你可以加两个 `Int` 值，但不能在没有明确转换的情况下直接加一个 `Int` 和一个 `String`。
- **占用多少内存**: `Int8` 比 `Int64` 使用更少的内存。
- **Julia 如何优化你的代码**: 了解类型使 Julia 的编译器能够生成高效的机器码。

随着你的学习，你将看到这些类型如何互相影响，如何定义自己的自定义类型（使用结构体），以及 Julia 的类型系统如何支持多重派发等强大的功能。目前，熟悉这些内置类型将为我们接下来学习的操作和控制流结构奠定坚实根基。

## 参考资料

- [Types](https://docs.julialang.org/en/v1/manual/types/) — The Julia Language (2024)
  Julia类型系统的官方指南，涵盖内置类型、类型层次结构和类型相关函数。理解Julia核心的基础。
- [Think Julia: How to Think Like a Computer Scientist](https://benlauwens.github.io/ThinkJulia.jl/latest/book.html) — Ben Lauwens, Allen B. Downey (2019)
  Publisher: O'Reilly Media
  一本介绍性的教科书，涵盖Julia中的基本编程概念，包括对内置数据类型及其用法的详细解释。
- [Julia Data Science](https://juliadatascience.io) — Jose Storopoli, Rik Huijzer, and Lazaro Alonso (2021)
  Publisher: JuliaDataScience.io
  一本使用Julia进行数据科学的综合指南，其中基础章节提供了对Julia数据类型及其应用的实用见解。
