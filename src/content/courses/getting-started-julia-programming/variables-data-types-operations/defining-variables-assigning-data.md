---
course: "getting-started-julia-programming"
chapter: "variables-data-types-operations"
lesson: "defining-variables-assigning-data"
sourceId: 6767
sourceUrl: "https://apxml.com/zh/courses/getting-started-julia-programming/chapter-2-variables-data-types-operations/defining-variables-assigning-data"
title: "定义变量与赋值数据"
description: "了解如何在Julia中声明变量、按照约定命名它们并赋值。"
order: 1
plots: []
sourceHash: "946cf4a99591477566229ca078c0ee200bd0031b8ce5a2fa4f51c43ec9283828"
sourceCorrections: []
---

编程的核心在于处理信息。变量是我们在Julia程序中为信息命名和管理信息的基本方式。你可以将变量想象成计算机内存中带标签的容器或命名的占位符，你可以在其中存储数据。这些数据可以是数字、文本或程序需要记住和使用的其他类型的信息。

### 为变量赋值

在Julia中，你使用赋值运算符，即等号 (`=`) 来创建变量并为其赋值。基本结构是：

`variable_name = value`

这个语句告诉Julia：“将右侧的`value`存储到与左侧`variable_name`关联的内存位置。” 重要的是要理解，在编程中，`=`是赋值操作，而不是数学上的相等。它表示“被设定为”。

例如，要存储一个人的年龄：

```julia
age = 25
```

这里，`age`是变量名，`25`是赋给它的值。现在，无论何时你的程序需要使用这个年龄，它都可以通过名称`age`来引用它。

你也可以将表达式的结果赋给变量：

```julia
width = 10
height = 5
area = width * height
```

在这种情况下，Julia首先计算`width * height`（即`10 * 5 = 50`），然后将结果`50`赋给变量`area`。

### 变量命名

为变量选择好的名称会使你的代码更容易阅读和理解。Julia在命名方面提供了很大的灵活性，但有一些规则和约定需要遵循：

**规则：**

1. 变量名必须以字母 (A-Z, a-z) 或下划线 (`_`) 开头。
2. 第一个字符之后，名称可以包含字母、数字 (0-9)、下划线、感叹号 (`!`) 以及许多其他Unicode字符（是的，你可以使用像 `π` 或 `α` 这样的符号作为变量名！）。
3. 变量名区分大小写。这意味着`myValue`、`myvalue`和`MYVALUE`将是三个不同的变量。

**约定（良好实践）：**

- **具有描述性：** 选择能清楚表明变量所代表内容的名称。例如，`customer_name`优于`cn`或`x`。
- **风格：** 尽管Julia允许各种风格，但`snake_case`（例如，`user_id`、`total_amount`）或小写单词在Julia社区中很常见。`camelCase`（例如，`userId`、`totalAmount`）也是可以接受的。最重要的是在你的项目中保持一致。
- **避免使用关键字：** 不要使用Julia的保留关键字（例如`if`、`else`、`function`、`true`、`false`）作为变量名。如果你尝试这样做，Julia会报错。
- **Unicode名称：** 尽管Julia支持Unicode字符（例如，`δ = 0.1`），但要确保它们易于输入并且对可能使用你代码的任何人来说都易于阅读。你可以在Julia REPL中使用类似LaTeX的缩写，然后按Tab键来输入许多数学符号（例如，`\delta`+Tab 会变成 `δ`）。

以下是一些有效变量名的示例：

- `name`
- `user_age`
- `temperature_celsius`
- `_temporary_value`
- `isUserActive`
- `π_value` (使用希腊字母π)

以及一些无效或不佳的示例：

- `1st_value` (不能以数字开头)
- `my-value` (不允许使用连字符；请改用`my_value`)
- `for` (这是一个关键字)

### 存储不同类型的数据

变量可以保存各种类型的数据。目前，让我们看一些你已经简要了解过的常见类型：

- **整数（整型数字）：**

  ```julia
  count = 10
  year = 2024
  ```
- **浮点数（带小数点的数字）：**

  ```julia
  price = 19.99
  pi_approx = 3.14159
  ```
- **字符串（文本字符序列）：** 字符串用双引号 (`"`) 括起来。

  ```julia
  greeting = "Hello, Julia learners!"
  user_name = "Alice"
  ```
- **布尔值（逻辑值`true`或`false`）：**

  ```julia
  is_ready = true
  has_permission = false
  ```

下图说明了变量作为一个命名标签指向所存储数值的理念：

> 名为`user_name`的变量被赋以字符串值“Alice”。

### 使用变量

一旦变量被赋值，你就可以在代码中使用它。例如，你可以打印它的值，在计算中使用它，或将其传递给函数。

让我们在Julia REPL（读取-求值-打印循环）中尝试一下：

```julia
julia> item_name = "Laptop"
"Laptop"

julia> quantity = 2
2

julia> unit_price = 750.00
750.0

julia> total_cost = quantity * unit_price
1500.0

julia> println(item_name)
Laptop

julia> println("Total cost for items:", total_cost)
Total cost for items: 1500.0
```

在这个交互式会话中：

1. 我们将“Laptop”赋给了`item_name`。
2. 我们将`2`赋给了`quantity`，并将`750.00`赋给了`unit_price`。
3. 我们计算了`quantity * unit_price`并将结果赋给了`total_cost`。
4. 然后我们使用`println`显示`item_name`的值以及包含`total_cost`的消息。

### 变量可以更改：重新赋值

“变量”一词意味着其值可以变化或更改。你可以在程序的任何点将新值赋给现有变量。

```julia
julia> score = 100
100

julia> println("Initial score:", score)
Initial score: 100

julia> score = score + 50 # 增加分数
150

julia> println("Updated score:", score)
Updated score: 150

julia> score = 0 # 重置分数
0

julia> println("Final score:", score)
Final score: 0
```

每次`score`被赋新值时，旧值都会被替换。这种更新变量的能力是程序如何管理不断变化的“状态”和信息的基本方式。

### 多重赋值

Julia提供了一种简洁的方式，可以一次性为多个变量赋值，或者将相同的值赋给多个变量。

在一行中为不同变量赋不同值：

```julia
julia> x, y, z = 10, 20, "hello"
("hello", 20, 10)

julia> println(x)
10

julia> println(y)
20

julia> println(z)
hello
```

这等同于：

```julia
x = 10
y = 20
z = "hello"
```

REPL显示`("hello", 20, 10)`作为赋值表达式本身的结果；这个元组反映了赋值的值，但在所有Julia版本或上下文 (context)中，这些值不一定按变量`x, y, z`的顺序排列，因此请关注各个`println`输出，以清楚了解每个变量所持有的内容。

将相同的值赋给多个变量：

```julia
julia> a = b = c = 100
100

julia> println(a)
100

julia> println(b)
100

julia> println(c)
100
```

这里，`c`首先被赋为`100`。然后，`b`被赋为`c`的值（即`100`），最后`a`被赋为`b`的值（也是`100`）。

理解变量以及如何为其赋值是你编写能够处理信息的程序的第一步。随着我们继续学习，你将看到这些命名的“数据块”是如何成为Julia中更复杂操作和结构的基础组成部分。接下来，我们将更详细地了解这些变量可以保存的不同数据类型。

## 参考资料

- [Variables](https://docs.julialang.org/en/v1/manual/variables/) — The Julia Language Developers (2024)
  Julia中定义、命名和使用变量的官方指南，包含规则和约定。
- [Think Julia: How to Think Like a Computer Scientist](https://benlauwens.github.io/ThinkJulia.jl/latest/book.html) — Ben Lauwens and Allen B. Downey (2019)
  Publisher: O'Reilly Media; Pages: Chapter 2: Variables, expressions and statements
  一本介绍性教材，使用Julia解释变量、赋值和基本数据类型等编程基础概念。
- [Julia Programming for Engineers and Scientists](https://mitpress.mit.edu/books/julia-programming-engineers-and-scientists) — Alan Edelman and David P. Sanders and Charles E. Leiserson and Jeremy Kepner and Peter K. Sogin and Albert S. Reuther (2023)
  Publisher: MIT Press; DOI: [10.7551/mitpress/14769.001.0001](https://doi.org/10.7551/mitpress/14769.001.0001)
  对Julia编程的介绍，涵盖变量和数据处理等核心概念。
