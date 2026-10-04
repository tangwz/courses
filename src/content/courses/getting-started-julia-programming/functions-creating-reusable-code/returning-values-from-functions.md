---
course: "getting-started-julia-programming"
chapter: "functions-creating-reusable-code"
lesson: "returning-values-from-functions"
sourceId: 6796
sourceUrl: "https://apxml.com/zh/courses/getting-started-julia-programming/chapter-5-functions-creating-reusable-code/returning-values-from-functions"
title: "函数返回值"
description: "了解函数如何使用 `return` 语句或通过隐式返回最后求值的表达式来返回数据。"
order: 4
plots: []
sourceHash: "444e22b9151dc1b6d058795d28592d1777dbe84f0eeab5fa1aab5c55967684d4"
sourceCorrections: []
---

函数将结果传回程序中调用它们的部分时，它们变得真正有用。此结果称为返回值。它是函数计算或处理步骤的成果。可以把函数想象成一个专注的助手：您向它提供指令和材料（参数 (parameter)），它执行指定任务，然后交回完成项（返回值）。此机制使您能够将函数串联起来，将一个函数的输出作为另一个函数的输入，或者将函数结果存入变量供后续使用，从而构建更复杂的程序逻辑。

### 使用 `return` 显式返回值

指定函数应返回什么的最直接方法是使用 `return` 关键字。当 Julia 在函数内部遇到 `return` 时，它会立即停止函数的执行，并将 `return` 后面的值传回函数被调用的位置。

下面是一个简单的函数，它将两个数字相加并返回它们的和：

```julia
function add_numbers(x, y)
    calculation_result = x + y
    return calculation_result
end

sum_value = add_numbers(5, 3)
println("The sum is: $sum_value") # 输出: 和为: 8
```

在此代码中，`add_numbers(5, 3)` 被调用。在函数内部，`x` 被赋值为 `5`，`y` 被赋值为 `3`。它们的和 `8` 被计算并存入 `calculation_result`。语句 `return calculation_result` 随后将此值 `8` 传回，并将其赋值给 `sum_value` 变量。

`return` 关键字也可作为立即退出函数的方式。函数中位于已执行 `return` 语句之后的任何代码都不会运行。

考虑此示例：

```julia
function check_number_type(number)
    if number > 0
        return "Positive"
    end
    # 只有当 number 不大于 0 时才会执行到这里
    return "Not positive (zero or negative)"
    println("This line will not be executed.") # 不可达代码
end

message_one = check_number_type(10)
println(message_one) # Output: Positive

message_two = check_number_type(-2)
println(message_two) # Output: Not positive (zero or negative)
```

当 `check_number_type(10)` 被调用时，条件 `number > 0` 为真，因此字符串 `"Positive"` 被返回，函数终止。在函数定义的最后面的 `println` 语句在此调用中永远不会执行到。

### 隐式返回值：Julia 的一个便利特性

Julia 包含一个有用的特性：如果函数对于特定执行路径不包含已执行的显式 `return` 语句，则函数中最后求值的表达式的值会被自动返回。这可以使代码更紧凑，尤其适用于简短的函数。

让我们重写 `add_numbers` 函数以使用隐式返回值：

```julia
function add_numbers_implicit(x, y)
    calculation_result = x + y
    calculation_result # 这是最后求值的表达式
end

implicit_sum_value = add_numbers_implicit(7, 2)
println("The implicit sum is: $implicit_sum_value") # 输出: 隐式和为: 9
```

在这里，`calculation_result` 是最后求值的表达式，因此它的值（本例中为 `9`）会被自动返回。此函数甚至可以精简为一行：

```julia
function add_numbers_oneline(x, y)
    x + y # x + y 的结果是最后求值的表达式
end

sum_from_oneline = add_numbers_oneline(3, 4)
println("One-line sum: $sum_from_oneline") # 输出: 单行和为: 7
```

尽管隐式返回值在 Julia 中很常见并有助于其简洁性，但通常一个好的做法是，尤其对于 Julia 初学者，使用显式 `return`。这能清晰地表达您的意图，并使您的函数更易于阅读和理解，尤其当它们较长或有多个条件导致不同的返回点时。

### 返回多个值

有时函数需要以输出形式提供多条信息。Julia 巧妙地处理这一点，允许函数返回多个值。当您在 `return` 后面（或作为隐式返回的最后一个表达式）列出多个值并用逗号分隔时，Julia 会自动将它们组合成一个名为元组的数据结构。

假设您需要一个函数，可以计算两个数字的和与积：

```julia
function calculate_sum_and_product(a, b)
    current_sum = a + b
    current_product = a * b
    return current_sum, current_product # 返回一个元组 (current_sum, current_product)
end

all_results = calculate_sum_and_product(4, 5)
println("Raw results: $all_results")        # 输出: 原始结果: (9, 20)
println("Type of results: $(typeof(all_results))") # 输出: 结果类型: Tuple{Int64, Int64}
```

`calculate_sum_and_product` 函数返回两个值，`current_sum` 和 `current_product`。它们被打包成元组 `(9, 20)`，然后赋值给 `all_results` 变量。

此特性一个非常方便的地方是，您可以直接将这些返回值赋值给调用端的不同变量。这有时被称为解构赋值：

```julia
my_calculated_sum, my_calculated_product = calculate_sum_and_product(3, 7)
println("Sum: $my_calculated_sum")          # 输出: 和: 10
println("Product: $my_calculated_product")    # 输出: 积: 21
```

这提供了一种清晰的方式，从单个函数调用获取多个结果。如果您只关心部分返回值，您可以使用下划线 `_` 作为您希望忽略的任何值的占位符：

```julia
just_the_sum, _ = calculate_sum_and_product(10, 2) # 这里我们只需要和
println("Just the sum: $just_the_sum") # 输出: 只需要和: 12
```

### 执行操作的函数：返回 `nothing`

并非所有函数都主要是为了计算并返回新值而设计的。有些函数主要为了其副作用而执行，例如将文本打印到控制台、修改作为参数 (parameter)传入的现有数据结构，或写入文件。例如，`println()` 函数显示输出；它的主要作用并非是返回一个用于后续计算的值。

这类函数返回什么呢？在 Julia 中，如果函数对于给定执行路径没有显式 `return` 语句，并且其最后求值的表达式本身不产生特定值（例如 `println()` 本身），它会隐式返回一个名为 `nothing` 的特殊值。`nothing` 对象是其类型 `Nothing` 的唯一实例，表示值的缺失。

考虑此函数：

```julia
function display_greeting(name)
    println("Hello, $name! Pleased to meet you.")
    # 这里没有显式 return 语句。
    # 最后的操作 println 返回 nothing。
end

output_from_greeting = display_greeting("Dana")
println("The display_greeting function returned: $output_from_greeting")
# 输出:
# 你好, Dana! 很高兴认识你。
# display_greeting 函数返回: nothing
```

`display_greeting` 函数执行了打印消息的任务。由于它没有显式返回任何内容，并且 `println` 本身返回 `nothing`，因此 `output_from_greeting` 变量被赋值为 `nothing`。

函数返回 `nothing` 是完全正常的。这通常表明函数的主要目的通过行动（副作用）实现，而不是通过生成一个用于后续计算的新值。

掌握函数如何返回值是编写高效 Julia 程序的基础。无论您是为了清晰度使用显式 `return`，还是为了简洁性在简单函数中使用隐式返回，抑或通过元组管理多个结果，理解这方面能帮助您构建模块化且有效的程序，其中不同组件可以互相通信并利用彼此的输出。

## 参考资料

- [Functions](https://docs.julialang.org/en/v1/manual/functions/) — The Julia Language (2024)
  Julia官方文档的这一节详细说明了函数定义、参数处理和返回值管理，涵盖了显式、隐式和多种返回类型。
