---
course: "getting-started-julia-programming"
chapter: "functions-creating-reusable-code"
lesson: "defining-calling-functions-julia"
sourceId: 6793
sourceUrl: "https://apxml.com/zh/courses/getting-started-julia-programming/chapter-5-functions-creating-reusable-code/defining-calling-functions-julia"
title: "在 Julia 中定义和调用函数"
description: "学习在 Julia 中定义自己的函数以及如何使用实参调用它们的语法。"
order: 1
plots: []
sourceHash: "d0038ef2a20ab25f6263c7d60366452d87a74527c6a6f6372b5ab61704eda9d5"
sourceCorrections: []
---

随着程序变得更复杂，你常常会发现在多个地方编写相同或非常相似的代码块。这种重复会使代码更长，更难阅读，更容易出错。如果你需要修改那部分逻辑，你将不得不找到并更新每个实例。函数提供了一种简洁的解决方案，它允许你将一系列操作打包成一个独立的、命名的单元。你只需定义这个单元一次，然后就可以根据需要多次执行或“调用”它，每次可能使用不同的输入。这种方法对于编写结构良好、易于维护和高效的 Julia 代码非常重要。

### Julia 函数的结构

在 Julia 中，你使用 `function` 关键字来定义一个函数，其后是函数名、一对括号中包含的参数 (parameter)列表，然后是函数体。函数的定义以 `end` 关键字结束。

这是通用结构：

```julia
function function_name(parameter1, parameter2, ...)
    # 代码块：函数执行的语句
    # ...
    return result # 可选：将一个值返回给调用者
end
```

我们来分解一下：

- **`function`**: 这个关键字表示函数定义的开始。
- **`function_name`**: 这是你给函数起的名称。你稍后将使用此名称调用函数。好的函数名通常具有描述性，并遵循 Julia 的约定，即使用小写字母，如果需要，单词之间用下划线分隔（例如，`calculate_area`）。
- **`parameters` (例如，`parameter1`, `parameter2`)**: 这些是函数被调用时期望接收的值（输入）的占位符。参数在函数内表现为局部变量。如果函数不需要任何输入，你仍然使用空括号：`function_name()`。
- **函数体**: 这些是 `function` 和 `end` 之间缩进的代码行。你在这里编写函数执行时将执行的指令。
- **`return result`**: `return` 关键字用于指定函数应输出的值。当 `return` 语句执行时，函数会立即停止并将 `result` 返回给调用它的程序部分。如果函数没有显式 `return` 语句，它会自动返回其主体中最后计算的表达式的值。对于初学者来说，使用显式 `return` 通常更清晰。
- **`end`**: 这个关键字标志着函数定义的结束。

### 定义你的第一个函数

让我们从一个非常简单的函数开始，它不接受任何输入，也不显式返回任何值。它的目的只是打印一句问候语。

```julia
function say_hello()
    println("来自 Julia 函数的问候！")
end
```

在这个例子中：

- `say_hello` 是函数名。
- 它不接受参数 (parameter)，因此括号 `()` 是空的。
- 函数体由一个 `println` 语句组成。
- 没有显式 `return` 语句。当被调用时，它将执行 `println`，然后隐式返回一个特殊值 `nothing`。

现在，让我们定义一个接受输入（参数）并返回计算值的函数。

```julia
function add_two_numbers(x, y)
    sum_value = x + y
    return sum_value
end
```

这里：

- `add_two_numbers` 是函数名。
- 它接受两个参数：`x` 和 `y`。当我们调用这个函数时，需要提供两个值，它们将分别赋值给 `x` 和 `y`。
- 在函数内部，`sum_value = x + y` 计算两个参数的和并将其存储在局部变量 `sum_value` 中。
- `return sum_value` 将计算出的和返回给调用者。

### 调用函数

定义函数并不会执行它。它只是告诉 Julia 函数的作用。要运行函数内部的代码，你必须“调用”它。你通过写函数名，后面跟括号来调用函数。如果函数需要参数 (parameter)（为其形参提供的值），你就在括号内部提供它们。

让我们调用刚刚定义的函数：

```julia
# 调用 say_hello 函数
say_hello()
```

当 Julia 遇到 `say_hello()` 时，它会执行 `say_hello` 函数内部的 `println("来自 Julia 函数的问候！")` 语句。输出将是：

```
来自 Julia 函数的问候！
```

现在，让我们调用 `add_two_numbers`。因为它需要两个实参，我们需要提供它们。函数返回的值可以存储在变量中，也可以直接使用。

```julia
# 使用实参 5 和 10 调用 add_two_numbers
first_sum = add_two_numbers(5, 10)
println("第一个和是：", first_sum)

# 再次使用不同的实参调用它
second_sum = add_two_numbers(3.14, 2.71)
println("第二个和是：", second_sum)
```

输出：

```
第一个和是：15
第二个和是：5.85
```

请注意 `add_two_numbers(5, 10)` 如何将 `5` 传递给函数内的形参 `x`，将 `10` 传递给形参 `y`。函数计算 `5 + 10`，返回 `15`，然后将其存储在 `first_sum` 变量中。可重用性很明显：我们用不同的输入调用了相同的 `add_two_numbers` 函数，得到了不同的结果，而无需重写加法逻辑。

以下图表说明了调用 `first_sum = add_two_numbers(5, 10)` 时的执行流程：

> 函数调用期间控制流和数据的过程。程序将实参传递给函数，函数执行并返回一个值，然后程序从中断处继续执行。

区分形参和实参很重要：

- **形参** 是函数定义中列出的变量名（例如，`add_two_numbers(x, y)` 中的 `x` 和 `y`）。它们就像占位符。
- **实参** 是你调用函数时传递给函数的实际值（例如，`add_two_numbers(5, 10)` 中的 `5` 和 `10`）。

### 短函数语法

对于可以在单行代码中表达的非常简单的函数，Julia 提供了一种更紧凑的语法。这通常用于数学运算或简单的转换。

语法是：
`function_name(parameter1, parameter2, ...) = expression`

例如，我们的 `add_two_numbers` 函数可以简洁地写成：

```julia
add_two_numbers_short(x, y) = x + y
```

以及一个将数字求平方的函数：

```julia
square(n) = n * n
```

你调用这些短形式函数的方式与使用 `function...end` 块定义的函数完全相同：

```julia
another_sum = add_two_numbers_short(7, 8)
println("短函数求和：", another_sum) # 输出：短函数求和：15

squared_value = square(9)
println("9 的平方是：", squared_value)       # 输出：9 的平方是：81
```

这种紧凑形式仅仅是标准 `function...end` 定义的“语法糖”，当函数体只是一个提供返回值的表达式时。它可以使你的简单操作代码更易读。

通过定义和调用函数，你开始构建更结构化、更易于管理的程序。你将逻辑封装成可重用部分，使你的代码更容易理解、测试和修改。随着我们继续，你将学习更多关于函数实参、返回值以及其他使函数成为 Julia 中功能强大的工具的特性。

## 参考资料

- [Functions](https://docs.julialang.org/en/v1/manual/functions/) — The Julia Language (2024)
  Julia官方手册中关于函数定义、参数、返回值及不同语法形式的全面概述。
- [Think Julia: How to Think Like a Computer Scientist](https://benlauwens.github.io/ThinkJulia.jl/latest/book.html) — Ben Lauwens and Allen B. Downey (2019)
  Publisher: O'Reilly Media; Pages: Chapter 3
  一本优秀的入门教材，使用Julia教授编程基础知识，包括函数定义、参数和调用约定的详细解释。
