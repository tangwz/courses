---
course: "getting-started-julia-programming"
chapter: "expressions-control-flow-structures"
lesson: "repeating-tasks-for-loops"
sourceId: 6780
sourceUrl: "https://apxml.com/zh/courses/getting-started-julia-programming/chapter-3-expressions-control-flow-structures/repeating-tasks-for-loops"
title: "使用 for 循环重复任务"
description: "学习如何在 Julia 中使用 for 循环，用于遍历范围、集合和执行重复任务。"
order: 5
plots: []
sourceHash: "3481990f10d876985966728af24464a3cc86bb638188af5b65369ac8e4bd9fb4"
sourceCorrections: []
---

许多编程任务需要重复执行相同或类似的操作。例如，你可能需要处理班级中每个学生的记录，计算多个银行账户的利息，或者仅仅是打印一系列数字。为每次重复编写单独的指令效率低下且容易出错。循环在此提供了一个强效的解决办法。Julia 中的 `for` 循环旨在遍历一系列项，对该序列中的每个项执行一段代码。当你预先知道重复次数或需要处理一个集合中的所有项时，这使其成为自动化重复过程的绝佳工具。

### Julia `for` 循环的结构

核心上，Julia 的 `for` 循环具有直观的结构。让我们分析一下它的语法：

```julia
for variable in iterable
    # Code to be executed in each iteration
    # This is the loop body
end
```

各部分含义如下：

- **`for`**：这个关键字标志着 `for` 循环的开始。
- **`variable`**：这是一个临时变量，在循环的每次通过（或迭代）中，它都持有来自 `iterable` 的当前项。你可以选择此变量的名称（例如，`i`、`item`、`number`、`char`）。在每次迭代中，Julia 会将 `iterable` 中的下一个值分配给此变量。
- **`in`**：这个关键字将循环 `variable` 与 `iterable` 集合分开。
- **`iterable`**：这是循环将遍历的项序列或集合。常见的可迭代对象包括数字范围、数组、元组和字符串。
- **循环体**：`for` 行和 `end` 关键字之间缩进的代码行构成循环体。这是会重复执行的代码。
- **`end`**：这个关键字标志着 `for` 循环的结束。

### 遍历常见的可迭代对象

当处理数据集合时，`for` 循环的优点就十分明显。让我们看看它们如何在 Julia 中与一些最常见的可迭代类型配合使用。

#### 范围

范围是指定数字序列的常用方式。`$1:5$` 这样的范围表示数字 1、2、3、4 和 5。

```julia
println("Counting from 1 to 5:")
for i in 1:5
    println(i)
end
```

当这段代码运行时，Julia 会执行 `println(i)` 行五次。
第一次迭代时，`$i$` 为 1。
第二次迭代时，`$i$` 为 2，依此类推，直到 `$i$` 为 5。
输出将是：

```
Counting from 1 to 5:
1
2
3
4
5
```

你还可以创建带有“步长”的范围。例如，`$2:2:10$` 创建一个从 2 开始，每次增加 2，直到 10 的序列（即 2、4、6、8、10）。

```julia
println("Even numbers from 2 to 10:")
for num in 2:2:10
    println(num)
end
```

Output:

```
Even numbers from 2 to 10:
2
4
6
8
10
```

#### 数组

数组是项的有序集合。你可以使用 `for` 循环来处理数组中的每个元素。

```julia
colors = ["red", "green", "blue"]
println("My favorite colors are:")
for color in colors
    println(color)
end
```

Output:

```
My favorite colors are:
red
green
blue
```

在每次迭代中，`color` 变量都取 `colors` 数组中下一个元素的值。

#### 元组

元组与数组类似，但它们是不可变的（即创建后不能更改）。它们也可以通过 `for` 循环进行迭代。

```julia
point = (10, 20, 30) # A tuple representing coordinates
println("Coordinates:")
for coordinate in point
    println(coordinate)
end
```

Output:

```
Coordinates:
10
20
30
```

#### 字符串

在 Julia 中，字符串可以被视为字符集合。`for` 循环可以遍历字符串中的每个字符。

```julia
message = "Hello"
println("Characters in 'Hello':")
for char in message
    println(char)
end
```

Output:

```
Characters in 'Hello':
H
e
l
l
o
```

### 理解循环变量

循环变量（如我们示例中的 `i`、`color`、`char`）是一个临时占位符。

1. **赋值**：在循环体每次迭代之前，Julia 会将 `iterable` 中的下一个值分配给此变量。
2. **作用域**：循环变量通常是 `for` 循环的局部变量。这意味着它只存在于循环体内，并且在循环执行完毕后无法从循环外部访问（除非它是在循环外部声明的，这是一种更高级的情况）。对于常见的 `for item in collection` 模式，可以将 `item` 视为在每次迭代中都可用的新变量，并且在循环结束后即消失。

### 在循环内部执行操作

循环体可以包含任何有效的 Julia 代码。你可以在这里执行想要重复的操作。你可以进行计算、调用函数，甚至使用像 `if` 这样的条件语句。

一种常见模式是使用循环来累加结果。例如，计算一个范围内所有数字的和：

```julia
total_sum = 0 # Initialize an accumulator variable
for number in 1:10
    total_sum = total_sum + number # Add current number to total_sum
end
println("The sum of numbers from 1 to 10 is: ", total_sum)
```

Output:

```
The sum of numbers from 1 to 10 is: 55
```

在每次迭代中，`total_sum` 都会更新。这说明了表达式（如 `total_sum + number`）如何在循环内求值。

你还可以在循环中包含条件逻辑。例如，我们只对一个范围内的偶数求和：

```julia
sum_of_evens = 0
for num in 1:10
    if num % 2 == 0  # Check if num is even
        sum_of_evens = sum_of_evens + num
    end
end
println("Sum of even numbers from 1 to 10: ", sum_of_evens)
```

Output:

```
Sum of even numbers from 1 to 10: 30
```

在这里，`for` 循环内部的 `if` 语句控制着每次迭代中是否执行加法操作。

### 嵌套 `for` 循环

你可以将一个 `for` 循环放置在另一个 `for` 循环内部。这被称为嵌套循环。当你需要遍历项的组合或处理多维数据结构（如网格或表格）时，嵌套循环很有用。

考虑打印数字和字母的配对：

```julia
for i in 1:2
    println("Outer loop: i = ", i)
    for char_val in ['a', 'b']
        println("  Inner loop: char_val = ", char_val, ", Pair: (", i, ", ", char_val, ")")
    end
end
```

输出说明了内层循环如何在外层循环的每次迭代中完成其所有迭代：

```
Outer loop: i = 1
  Inner loop: char_val = a, Pair: (1, a)
  Inner loop: char_val = b, Pair: (1, b)
Outer loop: i = 2
  Inner loop: char_val = a, Pair: (2, a)
  Inner loop: char_val = b, Pair: (2, b)
```

The outer loop runs twice. For `i = 1`, the inner loop runs twice (for `'a'` and `'b'`). Then, for `i = 2`, the inner loop runs twice again.

### 循环执行的图示

`for` 循环的普遍过程可以如下图所示：

> 此图说明了流程：循环开始，检查是否有项，将项赋值给变量，执行主体，然后通过检查是否有更多项来重复。如果没有剩余项，循环结束。

### 何时使用 `for` 循环

`for` 循环在以下情况是理想的：

1. 你有一个确定的项集合（如数组、元组或字符串），并且想对每个项进行操作。
2. 你预先知道要重复一个操作多少次（例如，使用 `1:100` 这样的范围）。

本章后面，你将了解 `while` 循环，它更适合迭代次数事先未知但取决于某个条件满足的情况。目前，熟练掌握 `for` 循环将为你自动化许多常见编程任务打下坚实根基。正如你将看到的，通过循环控制程序流程是编写更强大、更高效的 Julia 代码的重要一步。

## 参考资料

- [Control Flow (within the Julia Manual)](https://docs.julialang.org/en/v1/manual/control-flow/) — The Julia Language Developers (2014)
  提供关于Julia `for` 循环及其他控制流结构的详细且权威的官方文档。
- [Think Julia: How to Think Like a Computer Scientist](https://www.oreilly.com/library/view/think-julia/9781492045038/) — Ben Lauwens, Allen B. Downey (2019)
  Publisher: O'Reilly Media; Pages: 295
  一本优秀的入门教材，使用Julia语言解释了包括循环和迭代在内的基础编程概念。
- [Hands-On Julia Programming: Learn to build, analyze, and scale applications with Julia 1.x](https://www.amazon.com/Hands-Julia-Programming-applications-9781838555435/dp/1838555432) — Tanmay Kumar (2020)
  Publisher: Packt Publishing
  一本实用的Julia编程指南，涵盖了包括`for`循环在内的核心语言特性，并提供了清晰的示例和解释。
