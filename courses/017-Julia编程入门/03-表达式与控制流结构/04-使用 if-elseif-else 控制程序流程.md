# 使用 if-elseif-else 控制程序流程

来源：[原文](https://apxml.com/zh/courses/getting-started-julia-programming/chapter-3-expressions-control-flow-structures/control-flow-if-elseif-else)

[返回章节目录](README.md) · [返回课程目录](../README.md)

程序常需要根据特定条件做出判断并执行不同的代码段。就像您可能在下雨时决定带伞，或在堵车时选择不同路线一样，Julia程序可以评估条件并相应地改变它们的行为。这种决策能力是编写实用和交互式软件的基本要求。Julia中实现此目的的主要工具是 `if`、`elseif` 和 `else` 语句，它们能够定义代码中的条件执行路径。

### `if` 语句：基本的条件执行

条件执行的最简单形式是 `if` 语句。它允许代码块*仅在*指定条件为真时才执行。

基本结构如下所示：

```julia
if condition
    # 如果条件为真，执行此处的代码
end
```

这里，`condition` 是一个必须评估为布尔值（`true` 或 `false`）的表达式。如果 `condition` 为 `true`，则执行 `if` 和 `end` 之间缩进的代码。如果 `condition` 为 `false`，则跳过此代码块，程序继续执行 `end` 关键字之后的任何语句。`end` 关键字是必要的；它指示 `if` 代码块的结束。

我们来看一个例子。假设我们只想在数字为正时打印一条消息：

```julia
x = 10
if x > 0
    println("The number is positive.")
end

y = -5
if y > 0
    println("This message will not be printed for y.") # 对于 y，这条消息不会被打印
end
```

在第一部分中，条件 `x > 0`（即 $10 > 0$）评估为 `true`，因此语句 `println("The number is positive.")` 被执行。在第二部分中，`y > 0`（即 $-5 > 0$）评估为 `false`。因此，与 `y` 相关的 `println` 语句被跳过。

### 添加备选方案：`else` 语句

通常，您会希望在一个条件为真时执行一个代码块，而在条件为假时执行*另一个*代码块。这就是 `else` 语句的作用。当 `if` 条件不满足时，它提供了一个备选的执行路径。

其结构为：

```julia
if condition
    # 如果条件为真，执行此处的代码
else
    # 如果条件为假，执行此处的代码
end
```

如果 `condition` 评估为 `true`，则执行第一个代码块（`if` 下的缩进代码），并跳过 `else` 块。反之，如果 `condition` 评估为 `false`，则跳过 `if` 块，并执行 `else` 块。

我们来扩展之前的例子，处理正数和非正数：

```julia
temperature = 15 # 摄氏度

if temperature > 20
    println("It's warm, no need for a jacket.")
else
    println("It might be chilly, consider a jacket.")
end
```

在这种情况下，`temperature > 20`（$15 > 20$）为 `false`。因此，`else` 块内的代码被执行，并打印“It might be chilly, consider a jacket.”。如果 `temperature` 例如是 25，则会执行 `if` 块，打印“It's warm, no need for a jacket。”

### 处理多个条件：`elseif`

有时，您需要按顺序检查多个不同条件。如果第一个条件不满足，您可能希望检查第二个，然后是第三个，依此类推，直到找到一个为真的条件或所有条件都已检查完毕。`if-elseif-else` 结构允许这种多路径的决策。

通用语法是：

```julia
if condition1
    # 如果 condition1 为真，执行此处的代码
elseif condition2
    # 如果 condition1 为假 且 condition2 为真，执行此处的代码
elseif condition3
    # 如果 condition1 和 condition2 都为假 且 condition3 为真，执行此处的代码
# ... 可以有更多的 elseif 子句 ...
else
    # 如果之前所有条件都为假，执行此处的代码
end
```

Julia 从上到下逐一评估这些条件：

1. 如果 `condition1` 为 `true`，其关联的代码块将被执行，并且其余的 `elseif` 和 `else` 块将被完全跳过。
2. 如果 `condition1` 为 `false`，Julia 接着检查 `condition2`。如果 `condition2` 为 `true`，其代码块将被执行，并且任何后续的 `elseif` 和 `else` 块将被跳过。
3. 这个过程会持续到任何额外的 `elseif` 子句。
4. 如果 *均未*评估为 `true`，则可选的 `else` 语句下的代码块将被执行。如果省略 `else` 子句且所有条件都为假，则整个 `if-elseif-else` 结构内没有代码会被执行。

考虑一个根据分数分配等级的例子：

```julia
score = 78

if score >= 90
    println("Grade: A")
elseif score >= 80
    println("Grade: B")
elseif score >= 70
    println("Grade: C")
elseif score >= 60
    println("Grade: D")
else
    println("Grade: F")
end
```

给定 `score = 78`：

- 第一个条件 `score >= 90`（$78 \ge 90$）为 `false`。
- Julia 转移到第一个 `elseif`。条件 `score >= 80`（$78 \ge 80$）也为 `false`。
- Julia 转移到第二个 `elseif`。条件 `score >= 70`（$78 \ge 70$）为 `true`。
- 代码 `println("Grade: C")` 被执行。
- 之后所有的 `elseif` 和 `else` 块都被跳过，程序在最后的 `end` 之后继续执行。

`elseif` 条件的顺序非常重要，尤其当条件可能重叠时。在评分示例中，如果我们错误地将检查顺序设为 `elseif score >= 70` 在 `elseif score >= 80` 之前，那么 85 分会错误地得到“Grade: C”，因为 $85 \ge 70$ 为真并且会首先被评估。

> `if-elseif-else` 结构中的执行流程。程序按顺序评估条件，直到其中一个为真或到达 `else` 块。

### 使用布尔表达式构建条件

`if`、`elseif` 和 `else` 语句中的条件是布尔表达式。这些表达式可以是简单的比较（例如，`age >= 18`，`name == "Julia"），也可以是使用 Julia 的逻辑运算符` &&`（逻辑与）、`||`（逻辑或）和` !\`（逻辑非）构建的更复杂的逻辑结构。您在本章前面已经遇到过这些运算符。

- **逻辑与 (`&&`)**：表达式 `condition1 && condition2` 仅在 `condition1` 和 `condition2` *都*为 `true` 时才评估为 `true`。

  ```julia
  age = 25
  has_drivers_license = true
  if age >= 18 && has_drivers_license
      println("Eligible to drive.")
  else
      println("Not eligible to drive.") # 如果任一条件为假，将打印此消息
  end
  ```
- **逻辑或 (`||`)**：表达式 `condition1 || condition2` 在 `condition1` 或 `condition2` *至少一个*为 `true` 时评估为 `true`。

  ```julia
  is_weekend = true
  is_public_holiday = false
  if is_weekend || is_public_holiday
      println("Enjoy your day off!") # 打印此消息，因为 is_weekend 为真
  else
      println("Back to work or school.")
  end
  ```
- **逻辑非 (`!`)**：`!` 运算符反转条件的布尔值。如果 `condition` 为 `true`，则 `!condition` 为 `false`，反之亦然。

  ```julia
  is_raining = false
  if !is_raining  # 这为真，因为 is_raining 为假
      println("It's not raining, good weather for a walk.")
  end
  ```

您可以组合这些运算符并使用括号 `()` 来分组子表达式，并明确控制评估顺序，这很像算术表达式。例如：`(is_student && age < 25) || has_discount_card`。

Julia（以及许多其他语言）中 `&&` 和 `||` 运算符的一个重要特点是**短路评估**：

- 对于 `a && b`：如果 `a` 评估为 `false`，则表达式 `b` *根本不会*被评估，因为无论 `b` 的值如何，整个 `&&` 表达式都必须为 `false`。
- 对于 `a || b`：如果 `a` 评估为 `true`，则表达式 `b` *不会*被评估，因为无论 `b` 的值如何，整个 `||` 表达式都必须为 `true`。

短路不仅仅是一种优化；它还可以用来防止错误。例如，在 `count > 0 && sum / count > 50` 中，如果 `count` 为 `0`，则 `sum / count` 部分永远不会被评估，从而避免了潜在的除零错误。

### 嵌套条件语句

将 `if-elseif-else` 结构放置在其他 `if-elseif-else` 结构内部也是可行的。这被称为**嵌套**。

```julia
x = 10
y = -3

if x > 0
    println("x is positive.")
    if y > 0
        println("y is also positive.")
    else
        println("y is not positive.") # 这将被打印
    end
else
    println("x is not positive.")
end
```

在此示例中，内部的 `if-else` 块（检查 `y` 的符号）仅在外部条件（`x > 0`）为 `true` 时才会被执行到。

嵌套提供了一种方法来模拟更复杂的决策树。但是，请注意，深度嵌套的条件语句（例如，超过两三个级别）会很快使代码难以阅读、理解和调试。如果您发现代码因嵌套而变得过于复杂，这可能是一个重新思考逻辑的迹象。有时，将问题分解为更小的函数或使用不同的控制流方法可以使代码更清晰。

通过掌握 `if`、`elseif` 和 `else`，您获得了一个有力的工具来控制 Julia 程序的流程，使其能够灵活响应不同的输入和状态。

## 参考资料

- [Control Flow](https://docs.julialang.org/en/v1/manual/control-flow/) — The Julia Language (2024)
  官方文档是Julia控制流结构的权威指南，包括if-elseif-else语句的语法和行为。
- [Hands-On Julia Programming: Learn the language and its applications](https://www.packtpub.com/product/hands-on-julia-programming/9781800561821) — Stefan Karpinski, Huda Nassar, and Alan Edelman (2020)
  Publisher: Packt Publishing; Pages: 67-80
  本书由Julia语言的共同创建者之一撰写，深入介绍了Julia的核心特性，其中包含有关条件执行和布尔逻辑的专门章节。
- [Think Julia: How to Think Like a Computer Scientist](https://benlauwens.github.io/ThinkJulia.jl/latest/book.html) — Ben Lauwens and Allen B. Downey (2019)
  Publisher: O'Reilly Media; Pages: Chapter 5
  一本适合初学者的优质书籍，使用Julia介绍编程概念，并对条件语句、逻辑运算符和嵌套提供了清晰的解释。

---

[上一节](03-%E6%AF%94%E8%BE%83%E8%BF%90%E7%AE%97%E7%AC%A6%E5%92%8C%E9%80%BB%E8%BE%91%E8%BF%90%E7%AE%97%E7%AC%A6.md) · [下一节](05-%E4%BD%BF%E7%94%A8%20for%20%E5%BE%AA%E7%8E%AF%E9%87%8D%E5%A4%8D%E4%BB%BB%E5%8A%A1.md)
