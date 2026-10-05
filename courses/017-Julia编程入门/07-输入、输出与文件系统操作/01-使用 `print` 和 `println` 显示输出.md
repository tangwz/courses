# 使用 `print` 和 `println` 显示输出

来源：[原文](https://apxml.com/zh/courses/getting-started-julia-programming/chapter-7-input-output-file-system-interaction/displaying-output-print-println)

[返回章节目录](README.md) · [返回课程目录](../README.md)

程序进行交流的一种基本方式是显示信息。无论是显示计算结果、提供状态更新，还是给出简单的问候，将输出发送到用户屏幕都是一项重要的操作。有两个主要函数负责处理基本的控制台输出任务在 Julia 中：`print` 和 `println`。

## `print` 函数：不带换行的输出

`print` 函数是将数据发送到控制台的工具。当你使用 `print` 时，Julia 会显示你提供的值，但在输出后，光标会停留在同一行。这意味着如果你连续多次调用 `print`，每次调用的输出将并排显示。

你可以给 `print` 一个或多个要显示的项目。这些项目可以是文本（字符串）或数字等直接值，也可以是保存这些值的变量。

让我们看看它是如何运作的：

```julia
print("Hello, ") # 打印 "Hello, "，光标停留在同一行
print("Julia")   # 在前一个输出之后立即打印 "Julia"
# Output: Hello, Julia
```

在这个例子中，第一个 `print` 调用显示 "Hello, "，第二个 `print` 调用紧接着添加 "Julia"，从而产生单行输出：`Hello, Julia`。

你也可以打印不同类型的数据。如果你提供多个参数 (parameter)，`print` 会依次显示它们。如果你需要空格，请确保它们是你的字符串的一部分，或者作为单独的字符串参数提供。

```julia
name = "World"
version = 1.0
print("Hello, ", name, "! We are using version ", version, ".") # 空格是字符串的一部分
# Output: Hello! We are using version 1.0.
```

在这里，`print` 接受用逗号分隔的多个参数，并连续地将它们显示在同一行。

## `println` 函数：带换行的输出

通常，你会希望每段输出都出现在单独的行上。这就是 `println`（“print line”的缩写）派上用场的地方。`println` 函数的行为与 `print` 非常相似，但有一个显著区别：在显示指定值后，它会自动添加一个换行符。这会将光标移到控制台中下一行的开头。

请看这个例子：

```julia
println("First line of output.")
println("This is the second line.")
# Output:
# First line of output.
# This is the second line.
```

每次调用 `println` 都会在新行上产生输出，这在许多情况下使输出更易于阅读。

就像 `print` 一样，`println` 可以处理多个参数 (parameter)。它们会连续打印，后面跟着一个换行。如果需要，请记住在你的字符串中包含空格或将其作为单独的参数。

```julia
count = 3
item = "apples"
println("I have ", count, " ", item, ".") # 这个 " " 确保了 count 和 item 之间有空格
# Output: I have 3 apples.
```

即使有多个参数，`println` 也会确保该调用产生的整个组合输出后面跟着一个新行。

## `print` 与 `println`：快速比较

`print` 和 `println` 之间的选择完全取决于你希望如何格式化你的输出。如果你正在逐块构建一行文本，`print` 是有用的。如果你想要独立的输出行，`println` 通常更受青睐。

让我们并排看看它们：

```julia
print("Status: ")
print("OK")
println() # 通过不带参数调用 println 手动添加换行

println("Report:")
println("All systems nominal.")

# Output:
# Status: OK
# Report:
# All systems nominal.
```

在第一部分，我们使用两次 `print` 调用来构建单行 `Status: OK`。然后我们调用不带任何参数 (parameter)的 `println()`，只是为了移动到下一行。随后的 `println` 调用都在各自的新行上输出。

## 你可以显示哪些数据类型？

`print` 和 `println` 都相当灵活，可以显示你在 Julia 中会遇到的各种数据类型：

- **字符串**：用双引号括起来的文本，例如 `"Hello Julia"`。
- **数字**：整数（例如 `42`）、浮点数（例如 `3.14159`）。
- **布尔值**：逻辑值 `true` 和 `false`。
- **变量**：如果一个变量存储了上述任何类型，你可以打印该变量来显示其当前值。

```julia
message = "Julia is cool!"
year = 2024
is_learning = true

println(message)
println("Current year: ", year)
println("Am I learning Julia? ", is_learning)

# Output:
# Julia is cool!
# Current year: 2024
# Am I learning Julia? true
```

Julia 会自动将这些基本类型转换为可读的字符串表示以供显示。对于数组或字典等更复杂的数据结构，Julia 通常也提供合理的默认输出格式，不过你将在后续的学习中了解更多关于控制它们显示方式的内容。

## 组合数据以进行输出

你已经看到 `print` 和 `println` 可以接受多个参数 (parameter)。这些参数会一个接一个地打印。如果你希望不同项目之间有空格，你需要明确包含这些空格，可以是作为字符串的一部分，也可以是作为单独的字符串参数（例如，`println(val1, " ", val2)`）。

```julia
fruit = "apple"
quantity = 5
println("Fruit: ", fruit, ", Quantity: ", quantity)
# Output: Fruit: apple, Quantity: 5
```

在这里，空格和逗号是传递给 `println` 的字面字符串的一部分。

虽然传递多个参数可行，但在 Julia 中构建包含嵌入 (embedding)变量值的字符串时，一种更常见且通常更清晰的方法是**字符串插值**。这种技术让你能够使用 `$` 符号作为变量前缀，或 `$(...)` 用于表达式，将变量的值（或表达式的结果）直接插入到字符串中。

让我们使用字符串插值再次查看之前的例子：

```julia
fruit = "apple"
quantity = 5
println("Fruit: $fruit, Quantity: $quantity")
# Output: Fruit: apple, Quantity: 5
```

这种语法通常更具可读性且更方便。你也可以通过在 `$` 符号后将更复杂的表达式用括号括起来，从而嵌入它们：

```julia
radius = 6371 # 地球半径（单位：公里）
println("The Earth's approximate circumference is $(2 * pi * radius) km.")
# Output: The Earth's approximate circumference is 40030.17359196554 km.
```

字符串插值是 Julia 中创建格式化输出字符串的非常有效的功能。

## 显示输出的常见用途

显示输出是许多编程任务的核心部分。以下是一些 `print` 和 `println` 不可或缺的常见场景：

1. **用户交互**：向运行你程序的人提供提示、消息和反馈。

   ```julia
   println("To the Area Calculator!")
   print("Enter the radius: ") # 用户将在此提示后输入
   ```
2. **显示结果**：显示计算或数据处理的结果。

   ```julia
   a = 10
   b = 20
   sum_ab = a + b
   println("The sum of $a and $b is $sum_ab.")
   # Output: The sum of 10 and 20 is 30.
   ```
3. **简单调试**：当你尝试了解你的代码在做什么或为什么它没有按预期工作时，插入 `println` 语句以显示不同点的变量值，可以作为诊断问题的快速方法。

   ```julia
   x = 5
   println("Initial value of x: $x")
   x = x * 2
   println("Value of x after doubling: $x")
   # 这有助于追踪 x 的变化。
   ```

尽管 Julia 拥有更复杂的调试工具，但使用 `println` 进行快速检查是常见且通常有效的第一步。

## 总结

有效显示输出是任何交互式或信息性程序的重要组成部分。Julia 为此提供了两个直观的函数：

- `print(...)`：将给定参数 (parameter)输出到控制台，末尾不添加新行。
- `println(...)`：将给定参数输出到控制台，然后将光标移到下一行。

你可以打印各种数据类型，并且使用 `$` 进行字符串插值提供了一种将变量值直接嵌入 (embedding)到输出字符串中的便捷方式。这些函数是你从 Julia 程序向控制台交流的主要工具。

## 参考资料

- [Julia Language Manual: Input and Output](https://docs.julialang.org/en/stable/manual/io-network/) — Jeff Bezanson, Stefan Karpinski, Viral Shah, Alan Edelman, et al. (2016)
  提供Julia输入输出函数（包括`print`和`println`）详细描述和使用示例的官方文档。

---

[上一节](../06-%E6%A8%A1%E5%9D%97%E3%80%81%E5%8C%85%E4%B8%8E%E9%A1%B9%E7%9B%AE%E7%BB%84%E7%BB%87/08-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E7%BB%84%E7%BB%87%E4%BB%A3%E7%A0%81%E4%B8%8E%E7%AE%A1%E7%90%86%E4%BE%9D%E8%B5%96.md) · [下一节](02-%E4%BD%BF%E7%94%A8%60%40printf%60%E6%A0%BC%E5%BC%8F%E5%8C%96%E8%BE%93%E5%87%BA.md)
