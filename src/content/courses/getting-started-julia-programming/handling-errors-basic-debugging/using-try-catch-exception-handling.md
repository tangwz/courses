---
course: "getting-started-julia-programming"
chapter: "handling-errors-basic-debugging"
lesson: "using-try-catch-exception-handling"
sourceId: 6819
sourceUrl: "https://apxml.com/zh/courses/getting-started-julia-programming/chapter-8-handling-errors-basic-debugging/using-try-catch-exception-handling"
title: "使用 `try-catch` 进行异常处理"
description: "学习如何在 Julia 中使用 `try-catch` 块来妥善处理运行时错误并防止程序崩溃。"
order: 2
plots: []
sourceHash: "42f4c9c7befe1b37cbca2244c824ff4be85917c25f40aacc78fc1be1232425c4"
sourceCorrections: []
---

当您的 Julia 程序遇到无法解决的意外情况时，它会发出“错误”信号，或者更正式地说，抛出“异常”。如果不处理，异常通常会停止程序执行，这在大型应用程序中往往不是期望的行为。Julia 提供了 `try-catch` 语句，作为一种有条理的方式来预见和管理这些异常，让您的程序能够妥善处理错误，而不是崩溃。

基本思路是“尝试”运行一段可能引发错误的代码。如果在这个 `try` 块内确实发生了错误，正常执行会立即停止，然后 Julia 会寻找相应的 `catch` 块来处理具体的问题。

### `try-catch` 的基本结构

`try-catch` 块的最简单形式如下：

```julia
try
    # 可能引发错误的代码
    println("尝试执行操作...")
    risky_value = 10 / 0 # 这会引发一个 DivideError
    println("如果上方发生错误，此行将不会被执行。")
catch
    # 如果 try 块中发生错误，将运行的代码
    println("捕获到一个错误！")
end

println("try-catch 块执行完毕后程序继续运行。")
```

在这个结构中：

- `try` 块内的代码首先执行。
- 如果 `try` 块执行期间没有发生错误（异常），`catch` 块将完全跳过，程序在 `end` 语句后继续执行。
- 如果 `try` 块中*确实*发生错误，`try` 块的执行会在错误发生点停止。Julia 随后立即跳到 `catch` 块，并执行 `catch` 块中的代码。`catch` 块执行完毕后，程序在 `end` 语句后继续执行。

运行上面的例子会产生：

```
尝试执行操作...
捕获到一个错误！
try-catch 块执行完毕后程序继续运行。
```

请注意，“如果上方发生错误，此行将不会被执行...”没有被打印，因为 `DivideError` 导致了立即跳到 `catch` 块。

### 获取错误信息

通常，仅仅知道发生了错误是不够的。你会想知道发生了*什么*错误。你可以在 `catch` 语句中将异常对象赋值给一个变量来做到这一点：

```julia
try
    # 可能引发错误的代码
    println("尝试计算平方根...")
    result = sqrt(-1.0) # 这会引发一个 DomainError
    println("Result: ", result) # 未执行
catch ex # 'ex' 将持有异常对象
    println("发生了一个错误。详细信息如下：")
    println("错误类型：", typeof(ex))
    println("错误消息：", ex.msg) # 许多错误类型都有 .msg 字段
end
```

如果你运行这段代码，`ex` 将是一个 `DomainError` 类型的对象，你可以检查它以获取更多信息。输出会类似于：

```
尝试计算平方根...
发生了一个错误。详细信息如下：
错误类型：DomainError
错误消息：如果使用复数参数调用，sqrt 只会返回复数结果。请尝试 sqrt(Complex(x))。
```

变量 `ex`（你可以随意命名，但 `ex` 或 `e` 是常见的约定）持有实际的异常对象。这个对象包含关于错误的信息，例如它的类型（例如，`DomainError`、`ArgumentError`、`BoundsError`），通常还有一个描述性消息。

### 处理特定类型的错误

一个单独的 `catch ex` 块会捕获任何类型的错误。然而，以不同的方式处理不同类型的错误会更有用。你可以在 `catch` 块内使用 `isa()` 函数检查异常对象 `ex` 的类型来实现这一点。

```julia
function process_data(data, index)
    try
        value = data[index] * 2
        println("处理后的值：", value)
        if value < 0
            error("Processed value cannot be negative.") # 我们也可以手动触发一个错误
        end
    catch ex
        if isa(ex, BoundsError)
            println("错误：索引 ", index, " 超出了所提供数据的范围。")
        elseif isa(ex, DomainError) # 尽管在此示例中并非由 data[index] 直接抛出
            println("错误：发生了一个域错误：", ex.msg)
        elseif isa(ex, ErrorException) # 捕获由 error() 抛出的错误
            println("满足了一个自定义错误条件：", ex.msg)
        else
            println("发生了一个意外错误：", typeof(ex))
            # 对于真正意外的错误，你可能想重新抛出它
            # rethrow(ex) # 这会将错误传递出去
        end
    end
end

my_array = [10, 20, 30]
process_data(my_array, 2)    # 正常执行
process_data(my_array, 5)    # BoundsError
process_data([-5, 10], 1)    # 来自我们自定义 error() 调用的 ErrorException
```

输出：

```
处理后的值：40
错误：索引 5 超出了所提供数据的范围。
满足了一个自定义错误条件：处理后的值不能为负数。
```

在这个例子中：

- `isa(ex, BoundsError)` 检查错误 `ex` 是否为 `BoundsError`，当尝试使用无效索引访问数组时会发生这种错误。
- `isa(ex, ErrorException)` 检查使用 `error()` 函数明确抛出的错误。
- 最后的 `else` 作为所有其他类型错误的“包罗万象”处理。

通过检查错误类型，你可以提供更具体的反馈，或尝试针对特定问题量身定制的不同恢复策略。

### `try-catch` 的执行流程

当发生错误时，`try-catch` 结构会改变程序正常的自上而下的流程。以下是其可视化表示：

> `try-catch` 块内的控制流。如果 `try` 部分的代码执行无误，`catch` 部分将被跳过。如果发生错误，执行将跳转到 `catch` 部分。

### 何时使用 `try-catch`

虽然 `try-catch` 是一个强大的工具，但它并非旨在处理所有条件逻辑。

- **将 `try-catch` 用于异常情况：** 真正的错误或意外事件，例如尝试打开不存在的文件、网络连接失败或接收到格式错误的数据。
- **不要将 `try-catch` 用于常规程序流程：** 例如，如果你想在访问字典之前检查键是否存在，最好使用 `haskey(my_dict, key)`，而不是尝试访问它并捕获 `KeyError`。将 `try-catch` 用于正常逻辑会使代码更难阅读，并可能降低速度。
- **保持 `try` 块的专注性：** 只包含可能抛出你打算处理的异常的特定代码行。将过多代码包装在一个 `try` 块中，可能会使确定是哪个操作导致错误变得困难。

### 关于 `rethrow()` 的说明

有时，`catch` 块可能对错误进行部分处理（例如记录它），但随后决定错误仍应传播，也许由外部的 `try-catch` 块处理，或者在错误关键时停止程序。`rethrow()` 函数用于此目的。

```julia
try
    # 可能存在问题的代码
    x = 1 / 0
catch ex
    println("记录错误：", typeof(ex), " - ", ex.msg)
    if isa(ex, DivideError)
        println("这是一个除法错误，我们无法在此处完全恢复。")
        rethrow(ex) # 传递错误
    end
end
println("如果调用 rethrow() 并且没有在其他地方捕获，此行可能不会被执行。")
```

如果调用 `rethrow(ex)`，原始异常 `ex` 会再次抛出，就好像它从未在此层级被捕获一样。如果没有进一步的 `try-catch` 来处理它，程序将终止。

有效使用 `try-catch` 块是编写更具韧性的 Julia 程序的重要一步，这些程序能够预见并妥善处理问题，而不是在意外发生时简单地失败。随着您的学习，您会发现它们对于构建能可靠运行的应用程序是不可或缺的。

## 参考资料

- [Error Handling](https://docs.julialang.org/en/v1/manual/control-flow/) — The Julia Language (2024)
  涵盖 Julia 中 `try-catch` 语句的官方语法、行为、异常类型以及错误管理的实践。
- [Think Julia: How to Think Like a Computer Scientist (Chapter 14: Files and Exceptions)](https://www.oreilly.com/library/view/think-julia/9781492045021/) — Ben Lauwens, Allen B. Downey (2019)
  Publisher: O'Reilly Media; Pages: Chapter 14: Files and Exceptions
  以适合初学者的方式解释 Julia 中的异常处理原则和 `try-catch` 机制，适合新程序员。
