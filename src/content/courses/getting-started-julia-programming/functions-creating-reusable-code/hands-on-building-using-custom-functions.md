---
course: "getting-started-julia-programming"
chapter: "functions-creating-reusable-code"
lesson: "hands-on-building-using-custom-functions"
sourceId: 6801
sourceUrl: "https://apxml.com/zh/courses/getting-started-julia-programming/chapter-5-functions-creating-reusable-code/hands-on-building-using-custom-functions"
title: "动手实践：构建和使用自定义函数"
description: "定义各种类型的函数并在 Julia 中使用它们解决问题的实践练习。"
order: 9
plots: []
sourceHash: "29b3ae271cba4f158911d2d3dd690d0f5b0a1afce7ebef58350ffa518464df47"
sourceCorrections: []
---

通过一系列动手实践的示例来练习定义函数、处理参数 (parameter)和理解作用域。我们建议您将这些示例输入到Julia REPL或脚本文件中，并试用它们。修改这些示例是巩固您理解的有效方法。

### 1. 您的第一个自定义函数：一个简单的问候

让我们从最基本的函数类型开始：它执行一个固定操作，不接受任何输入，也不返回任何特定值（尽管如您所知，所有Julia函数在技术上都会返回一些东西，即使是`nothing`）。

**目标：** 定义一个打印消息的函数。

```julia
function say_hello()
    println("致Julia函数！")
end

# 让我们调用我们的函数
say_hello()
```

**解释：**

- `function say_hello()` 开始定义一个名为 `say_hello` 的函数。括号 `()` 表示此函数在此版本中不接受任何参数 (parameter)。
- 函数内部的代码（即*函数体*）是 `println("你好，Julia函数！")`，它会向控制台打印一个字符串。
- `end` 标记 (token)函数定义的结束。
- `say_hello()` 是我们*调用*或*执行*该函数的方式。这会执行其函数体内的代码。

**预期输出：**

```
欢迎来到Julia函数！
```

### 2. 带位置参数 (parameter)的函数：个性化问候

函数在能够接受输入时会变得强大得多。位置参数是将数据传递给函数最常用的方式。调用函数时提供值的顺序与函数定义中参数的顺序相对应。

**目标：** 创建一个接受姓名作为输入并打印个性化问候的函数。

```julia
function greet_person(name)
    println("你好，", name, "！你今天怎么样？")
end

# 用不同的姓名调用函数
greet_person("Alice")
greet_person("Bob")
```

**解释：**

- `function greet_person(name)` 定义了一个函数 `greet_person`，它接受一个位置参数，该参数在函数内部将被称为 `name`。
- 当我们调用 `greet_person("Alice")` 时，字符串 `"Alice"` 会在该特定调用中被赋值给函数作用域内的 `name` 参数。
- `println` 函数随后使用这个 `name` 变量来构建个性化消息。

**预期输出：**

```
你好，Alice！你今天怎么样？
你好，Bob！你今天怎么样？
```

### 3. 返回值：一个简单的计算器

通常，函数会计算一个您希望在程序其他地方使用的值。`return` 关键词明确指定函数应该输出什么值。如果省略 `return`，Julia函数会隐式返回最后一个求值表达式的值。

**目标：** 编写一个将两个数字相加并返回它们之和的函数。

```julia
function add_numbers(x, y)
    result = x + y
    return result
end

# 调用函数并存储其返回值
sum_val = add_numbers(5, 3)
println("和为：", sum_val)

another_sum = add_numbers(10.5, 2.7)
println("另一个和为：", another_sum)
```

**解释：**

- `function add_numbers(x, y)` 接受两个位置参数 (parameter)，`x` 和 `y`。
- `result = x + y` 计算它们的和并将其存储在一个局部变量 `result` 中。
- `return result` 明确返回存储在 `result` 中的值。
- `sum_val = add_numbers(5, 3)` 调用函数。返回的值（本例中为 8）随后被赋值给变量 `sum_val`。

**返回多个值**
Julia 让从函数返回多个值变得容易，只需返回一个元组即可。

```julia
function calculate_sum_and_product(a, b)
    s = a + b
    p = a * b
    return s, p # 返回一个元组 (s, p)
end

sum_result, product_result = calculate_sum_and_product(4, 6)
println("和：", sum_result)
println("积：", product_result)

# 您也可以直接获取元组
results_tuple = calculate_sum_and_product(2, 3)
println("元组：", results_tuple)
println("元组的第一个元素：", results_tuple[1])
```

**解释：**

- `return s, p` 是 `return (s, p)` 的一种简洁写法。它创建并返回一个包含和与积的元组。
- `sum_result, product_result = calculate_sum_and_product(4, 6)` 使用*解构赋值*将返回元组的元素直接解包赋值给 `sum_result` 和 `product_result`。

**`add_numbers` 的预期输出：**

```
和为：8
另一个和为：13.2
```

**`calculate_sum_and_product` 的预期输出：**

```
和：10
积：24
元组：(5, 6)
元组的第一个元素：5
```

### 4. 关键词参数 (parameter)：灵活的消息格式化

关键词参数在调用函数时通过名称指定。它们可以提高代码可读性，特别是对于参数较多的函数，并允许在位置参数之后以任何顺序传递参数。

**目标：** 创建一个使用关键词参数显示带可选前缀和后缀消息的函数。

```julia
function format_message(message; prefix="", suffix="!")
    println(prefix, message, suffix)
end

# 使用关键词参数调用
format_message("Hello Julia") # 使用默认前缀和后缀
format_message("Important News"; prefix="INFO: ")
format_message("Task Complete"; suffix="!!")
format_message("Error Occurred"; prefix="ERROR: ", suffix=" Please check.")
format_message("Order matters not"; suffix=" :)", prefix="NOTE: ")
```

**解释：**

- `function format_message(message; prefix="", suffix="!")`：
  - `message` 是一个位置参数。
  - 分号 `;` 将位置参数与关键词参数分开。
  - `prefix=""` 和 `suffix="!"` 定义了带默认值的关键词参数 `prefix` 和 `suffix`。如果在函数调用中未提供这些参数，将使用这些默认值。
- 调用时，您使用关键词名称：`prefix="INFO: "`。
- 如果在调用中未提供关键词参数，将使用其默认值。

**预期输出：**

```
你好 Julia！
信息：重要新闻！
任务完成！！
错误：发生错误 请检查。
注意：顺序不重要 :)
```

### 5. 位置参数 (parameter)的默认值

您也可以为位置参数提供默认值。这些参数在函数定义中必须位于任何没有默认值的位置参数之后。

**目标：** 编写一个计算 `x` 的 `p` 次方（其中 `p` 默认为 2）的函数。

```julia
function power(x, p=2) # p 有一个默认值
    return x^p
end

println("3 的平方：", power(3))       # p 默认为 2
println("3 的立方：", power(3, 3))    # p 被指定为 3
println("2 的 4 次方：", power(2, 4))
```

**解释：**

- `function power(x, p=2)` 将 `p` 定义为一个带默认值 2 的位置参数。
- 如果 `power` 只用一个参数调用（例如 `power(3)`），该参数会赋值给 `x`，而 `p` 则取其默认值 2。
- 如果用两个参数调用（例如 `power(3, 3)`），第一个参数赋值给 `x`，第二个参数赋值给 `p`，从而覆盖默认值。

**预期输出：**

```
3 的平方：9
3 的立方：27
2 的 4 次方：16
```

### 6. 理解变量作用域

在函数内部定义的变量是该函数的局部变量。它们无法从外部访问，并且不会干扰函数外部同名的变量，除非您明确使用全局变量。

**目标：** 演示函数内部的局部变量作用域。

```julia
global_var = 100 # 这是一个全局变量

function scope_test()
    local_var = 10 # 这个变量是 scope_test 函数的局部变量
    println("函数内部：local_var = ", local_var)
    println("函数内部：读取 global_var = ", global_var) # 函数可以读取全局变量

    # 此赋值创建了一个名为 'global_var' 的新局部变量
    # 它是此函数的局部变量，并遮蔽了同名的全局变量。
    # 它不修改函数外部的原始 global_var。
    global_var = 50 
    println("函数内部，局部赋值后：新的局部 global_var = ", global_var)
end

scope_test()
println("函数外部：global_var = ", global_var) # 全局变量保持不变
# println(local_var) # 这会引起错误：local_var 在全局作用域中未定义
```

**解释：**

- `global_var = 100` 在全局作用域中定义。
- 在 `scope_test` 内部，`local_var = 10` 创建了一个只存在于 `scope_test` 内部的变量。
- 函数可以读取全局 `global_var` 的值。
- 当我们在函数*内部*写入 `global_var = 50` 时，Julia 默认会创建一个也名为 `global_var` 的*新局部变量*。这个局部变量在*函数的作用域内*会*遮蔽*（隐藏）同名的全局变量。函数外部的全局 `global_var` 不会因本次赋值而改变。要想在函数内部修改全局变量，您需要在赋值之前在函数内部使用 `global global_var` 声明它为全局变量。

**预期输出：**

```
函数内部：local_var = 10
函数内部：读取 global_var = 100
函数内部，局部赋值后：新的局部 global_var = 50
函数外部：global_var = 100
```

如果您取消最后一行 `println(local_var)` 的注释，您会收到一个 `UndefVarError` 错误，因为 `local_var` 未在全局作用域中定义。

### 7. 匿名函数：快速操作

匿名函数（也称为 lambda 函数）对于小型的一次性操作很有用。它们经常作为参数 (parameter)传递给像 `map` 或 `filter` 这样的其他函数。

**目标：** 使用匿名函数对数字列表进行平方。

```julia
numbers = [1, 2, 3, 4, 5]

# 将匿名函数与 map 结合使用
squared_numbers = map(x -> x^2, numbers)
println("平方后的数字：", squared_numbers)

# 另一个示例：一个用于快速计算的匿名函数，赋值给一个变量
add_one_and_double = x -> (x + 1) * 2
println("将 (x+1)*2 应用于 3：", add_one_and_double(3))
```

**解释：**

- `x -> x^2` 是一个匿名函数。它接受一个参数 `x` 并返回 `x^2`。
- `map(function, collection)` 将给定函数应用于集合的每个元素。这里，它将我们的匿名平方函数应用于 `numbers` 中的每个元素。
- `add_one_and_double = x -> (x + 1) * 2` 将一个匿名函数赋值给一个变量。这为匿名函数赋予了一个名称，使其可以像常规函数一样被调用，这对于非常简洁的函数定义很有用。

**预期输出：**

```
平方后的数字：[1, 4, 9, 16, 25]
将 (x+1)*2 应用于 3：8
```

### 8. 多重分派：一个概览

多重分派允许函数根据其参数 (parameter)的*类型*拥有不同的行为（方法）。这是 Julia 的一个非常有用的特性，它使得代码富有表现力且高效。下面是一个非常简单的例子。

**目标：** 定义一个对数字和字符串表现不同的函数 `process_input`。

```julia
function process_input(data::Number)
    println("处理数字。双倍值：", data * 2)
end

function process_input(data::String)
    println("处理字符串。大写：'", uppercase(data), "'")
end

process_input(10)
process_input("hello julia")
# process_input(true) # 这会引起 MethodError 错误，因为我们尚未为布尔类型定义方法
```

**解释：**

- 我们为同一个函数名 `process_input` 定义了两个*方法*。
- `function process_input(data::Number)`：如果 `data` 是任何一种 `Number`（例如 `Int`、`Float64`），则调用此方法。`::Number` 是一个类型注解，指定此方法应选择用于类型为 `Number` 或其子类型的参数。
- `function process_input(data::String)`：如果 `data` 是 `String`，则调用此方法。
- Julia 会根据调用期间传递的参数类型自动选择要执行的正确方法。此选择在运行时发生。

**预期输出：**

```
处理数字。双倍值：20
处理字符串。大写：'HELLO JULIA'
```

如果您尝试 `process_input(true)`，您会得到一个 `MethodError` 错误，因为没有定义专门匹配 `Bool` 参数的 `process_input` 方法，并且 `Bool` 不是 `Number` 或 `String` 的子类型，无法以匹配这些定义的方式。

### 9. 为您的函数添加文档：添加文档字符串

良好的文档对于可理解和可维护的代码很重要。在 Julia 中，您可以为函数添加文档字符串，以解释它们的作用、参数 (parameter)以及返回值。然后可以从 REPL 轻松访问此文档。

**目标：** 为之前创建的 `add_numbers` 函数添加一个文档字符串，并学习如何访问它。

````julia
"""
    add_numbers(x, y)

将两个数字 `x` 和 `y` 相加并返回它们的和。

# 参数
- `x::Number`: 第一个要相加的数字。
- `y::Number`: 第二个要相加的数字。

# 返回值
- `::Number`: `x` 和 `y` 的和。

# 示例
```jldoctest
julia> add_numbers(5, 3)
8

julia> add_numbers(-1, 1)
0
````

"""
function add\_numbers(x::Number, y::Number)
result = x + y
return result
end

# 在 Julia REPL 中访问文档字符串：

# 输入 ?add\_numbers 然后按 Enter 键。

# 例如：

# julia> ?add\_numbers

```
**解释：**
*   文档字符串是放置在函数定义之前的一个字符串。它通常用三引号 `""" ... """` 括起来。
*   文档字符串内可以使用 Markdown 格式，以提高可读性。常见部分包括简要摘要、`参数`列表（通常包含其类型）、函数的`返回值`以及`示例`。
*   `示例` 中的 `jldoctest` 块很特别。这些块中的代码可以由文档工具自动运行，以确保您的示例是正确的。
*   在 Julia REPL 中，输入 `?` 后跟函数名称（例如 `?add_numbers`）会显示其格式化的文档字符串。

**在 REPL 中输入 `?add_numbers` 后查看输出：**
您将看到格式化的文档字符串显示出来，类似于上面所写的，为您提供有关 `add_numbers` 函数的信息。

这些练习涵盖了在 Julia 中创建和使用函数的核心方面。尝试修改它们，为不同的任务创建您自己的函数，并试用这些功能的组合。例如，尝试编写一个使用关键词参数、某些位置参数具有默认值并返回多个值的函数。您练习得越多，就会越熟练地利用函数编写清晰、模块化和高效的 Julia 代码。
```

## 参考资料

- [Functions](https://docs.julialang.org/en/v1/manual/functions/) — The Julia Language (2024)
  提供Julia中函数定义、调用和不同类型函数（包括参数、作用域和多重分派）的全面信息。
- [Style Guide](https://docs.julialang.org/en/v1/manual/style-guide/) — The Julia Language (2024)
  包含编写清晰且可维护的Julia代码指南，包括文档字符串格式和内容的建议。
- [Think Julia: How to Think Like a Computer Scientist](https://www.oreilly.com/library/view/think-julia/9781492045038/) — Ben Lauwens, Allen B. Downey (2019)
  Publisher: O'Reilly Media; Pages: 295; DOI: [978-1492045038](https://doi.org/978-1492045038)
  一本优秀的入门书籍，使用Julia讲解编程基础知识，包括函数、参数和作用域的详细说明。
