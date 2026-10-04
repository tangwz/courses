---
course: "getting-started-julia-programming"
chapter: "variables-data-types-operations"
lesson: "converting-between-data-types"
sourceId: 6773
sourceUrl: "https://apxml.com/zh/courses/getting-started-julia-programming/chapter-2-variables-data-types-operations/converting-between-data-types"
title: "数据类型转换"
description: "了解如何将值从一种数据类型转换为另一种，并了解 Julia 的类型提升机制。"
order: 7
plots: []
sourceHash: "1b60abc2dce7f7cd393808590c7c4f6cd986e9777e749121bf1fc01b9eed868a"
sourceCorrections: []
---

你遇到或生成的数据常常并非特定任务所需的精确形式。例如，来自控制台的用户输入通常是文本（字符串），但你可能需要对其进行数学计算，这就要求它是一个数字。Julia 与许多编程语言一样，提供了将值从一种数据类型转换为另一种数据类型的方法。这个过程对于编写灵活且正确的程序非常必要。

### 主动类型转换

显式类型转换是指你作为程序员，直接指示 Julia 改变一个值的类型。这使你对数据处理方式有精确的掌控。其通用语法非常直接：你使用目标数据类型的名称，就像它是一个函数一样，将要转换的值作为参数 (parameter)传递给它。

我们来看一些常见情况：

**数字类型之间的转换**

你经常需要转换不同类型的数字，比如整数和浮点数。

- **整数到浮点数：** 要将整数转换为浮点数，你可以使用 `Float64()`（如果需要特定精度，也可以使用其他浮点类型，例如 `Float32`）。

  ```julia
  julia> int_value = 10
  10

  julia> float_value = Float64(int_value)
  10.0

  julia> typeof(float_value)
  Float64
  ```

  将整数转换为浮点数通常是安全的，因为不会丢失信息。
- **浮点数到整数：** 将浮点数转换为整数使用 `Int()`。当你这样做时，Julia 会截断小数部分，这意味着它会简单地切掉小数点后的所有内容。它不会四舍五入到最接近的整数。

  ```julia
  julia> pi_approx = 3.14159
  3.14159

  julia> integer_part = Int(pi_approx)
  3

  julia> another_float = 3.99
  3.99

  julia> Int(another_float) # 仍然截断
  3
  ```

  请注意，从浮点数转换为整数可能导致信息丢失（即小数部分）。

**字符串到数字的转换**

从文件或用户输入读取的数据通常是字符串格式。要将它们用作数字，你需要“解析”它们。Julia 为此提供了 `parse()` 函数。

```julia
julia> numeric_string = "123"
"123"

julia> my_integer = parse(Int, numeric_string)
123

julia> typeof(my_integer)
Int64

julia> float_string = "2.718"
"2.718"

julia> my_float = parse(Float64, float_string)
2.718

julia> typeof(my_float)
Float64
```

如果你尝试解析一个不代表目标类型有效数字的字符串，Julia 将会报错。例如，`parse(Int, "hello")` 会失败。如何处理此类潜在错误是我们在后面章节会讲到的一个内容。

**数字到字符串的转换**

相反的操作，即将数字转换为其字符串表示形式，也很常见，或许是为了显示目的或写入文件时。`string()` 函数处理此转换：

```julia
julia> lucky_number = 7
7

julia> string_representation = string(lucky_number)
"7"

julia> typeof(string_representation)
String

julia> eulers_number = 2.71828
2.71828

julia> string(eulers_number)
"2.71828"
```

**字符串到布尔值的转换**

对于布尔值，你可以解析诸如“true”或“false”的字符串：

```julia
julia> true_string = "true"
"true"

julia> bool_val_true = parse(Bool, true_string)
true

julia> typeof(bool_val_true)
Bool

julia> parse(Bool, "false")
false
```

对于布尔值，`parse(Bool, ...)` 也支持解析“1”或“0”的字符串。

### Julia 的辅助：自动类型提升

有时，Julia 可以自动为你转换类型，尤其是在涉及不同数字类型的算术运算中。这被称为类型提升。类型提升的目标通常是通过转换为“更宽”或更通用的类型来保持精度，这种类型可以容纳所有相关值而不会丢失信息。

考虑添加一个整数和一个浮点数：

```julia
julia> result = 5 + 2.5
7.5

julia> typeof(result)
Float64
```

这里，整数 `5` 在加法发生之前被自动提升为浮点数 `5.0`，因此结果 `7.5` 也是 `Float64`。这样做是合理的，因为如果 Julia 将 `2.5` 转换为整数（例如 `2`），结果将是 `7`，从而丢失小数部分。Julia 的提升规则有助于在常见运算中避免此类意外的精度损失。

> 一个整数和一个浮点数在运算中结合，整数被提升为浮点数，结果为一个浮点数。

类型提升使编写混合类型算术代码更简洁、更不容易出错，因为你通常不必为基本数学运算而让代码中充满显式转换。

### 类型转换的实际情况

理解类型转换非常重要，因为：

- **用户输入：** `readline()` 的输入总是字符串。如果你期望一个数字，就必须解析它。
- **文件数据：** 从文本文件读取的数据通常需要从字符串解析成适当的数字或布尔类型。
- **函数参数 (parameter)：** 你编写的或从包中使用的函数可能期望特定类型的参数。可能需要进行转换以满足这些要求。
- **数据表示：** 你可能将数据转换为字符串用于记录日志或显示，或者在关注内存时转换为更紧凑的数字类型（尽管这属于更高级别的考虑）。

### 需要记住的要点

- **潜在的数据丢失：** 请注意，一些转换，例如从 `Float64` 到 `Int`，可能导致数据丢失（小数部分被截断）。
- **无效转换：** 尝试进行不可能的转换（例如，`parse(Int, "not a number")`）将会导致错误。如果未能处理此错误，你的程序可能会停止运行。我们将在后面探讨错误处理。
- **清晰性与简洁性：** 尽管类型提升很有帮助，但在复杂情况下，显式转换有时能使你的代码意图更明确。

掌握数据类型转换使你能够流畅地处理来自各种来源的数据，并为任何操作做好准备，这在你的 Julia 编程技能集中占据了重要一环。随着你的学习，你会发现这些转换技巧对于构建有效应用程序非常有用。

## 参考资料

- [The Julia Language Manual: Conversion and Promotion](https://docs.julialang.org/en/v1/manual/conversion-and-promotion/) — The Julia Language (2024)
  Publisher: The Julia Language
  这是理解 Julia 类型转换函数、显式转换和自动类型提升规则的官方来源。
- [Getting Started with Julia](https://www.packtpub.com/product/getting-started-with-julia/9781789617228) — Len Green, Ehi Aiguokhian (2019)
  Publisher: Packt Publishing
  为新的 Julia 程序员提供实用指导，包括变量类型、数据处理和显式类型转换等主题。
