---
course: "getting-started-julia-programming"
chapter: "variables-data-types-operations"
lesson: "practice-variables-types-operations"
sourceId: 6775
sourceUrl: "https://apxml.com/zh/courses/getting-started-julia-programming/chapter-2-variables-data-types-operations/practice-variables-types-operations"
title: "练习：变量、类型和基本操作"
description: "通过实际例子应用您对 Julia 变量、数据类型和简单操作的所学知识。"
order: 9
plots: []
sourceHash: "2f538406125f9be42a88e9bba826fc8b57f79c93dab3bf2bb9ccdcc6e6b8a609"
sourceCorrections: []
---

通过实际例子进行练习，是理解变量、数据类型和操作的最佳方法。实践练习提供了应用数据存储、数据处理和 Julia 类型系统概念的机会。建议您将这些例子输入到 Julia REPL 或脚本文件中，并观察结果。

### 练习 1：声明变量和检查类型

我们从创建一些变量来保存不同类型的信息开始。Julia 是动态类型语言，这意味着通常无需指定变量的类型；Julia 会根据您赋予的值来确定类型。

1. **创建变量**:
   定义变量来存储您的姓名、年龄、平均考试分数以及您是否正在学习 Julia。

   ```julia
   my_name = "Alex"
   my_age = 28
   average_score = 92.5
   is_actively_learning = true
   ```
2. **打印值和类型**:
   使用 `println()` 函数显示这些值，并使用 `typeof()` 函数查看 Julia 为每个变量分配了什么数据类型。

   ```julia
   println("Name: ", my_name, ", Type: ", typeof(my_name))
   println("Age: ", my_age, ", Type: ", typeof(my_age))
   println("Average Score: ", average_score, ", Type: ", typeof(average_score))
   println("Actively Learning: ", is_actively_learning, ", Type: ", typeof(is_actively_learning))
   ```

   当您运行此代码时，您应该看到类似以下输出：

   ```
   Name: Alex, Type: String
   Age: 28, Type: Int64
   Average Score: 92.5, Type: Float64
   Actively Learning: true, Type: Bool
   ```

   (注意：`Int64` 或 `Int32` 取决于您的系统架构，但它们都表示整数。)

   此练习表明了 Julia 如何推断类型：`my_name` 是 `String` 类型，因为它用双引号括起来；`my_age` 是 `Int64` 类型（一个整数）；`average_score` 是 `Float64` 类型（一个带小数点的数字）；`is_actively_learning` 是 `Bool` 类型（`true` 或 `false`）。

### 练习 2：执行数值操作

变量不仅仅用于存储数据；它们对于进行计算不可或缺。我们来处理数字。

1. **基本算术运算**:
   定义两个数字并执行标准算术运算。

   ```julia
   num1 = 15
   num2 = 4

   sum_result = num1 + num2
   diff_result = num1 - num2
   prod_result = num1 * num2
   div_result = num1 / num2 # 浮点除法
   int_div_result = div(num1, num2) # 整数除法（商）
   remainder_result = num1 % num2 # 取模运算符

   println("num1 = ", num1, ", num2 = ", num2)
   println("Sum: ", sum_result)
   println("Difference: ", diff_result)
   println("Product: ", prod_result)
   println("Division (float): ", div_result, ", Type: ", typeof(div_result))
   println("Integer Division (div): ", int_div_result, ", Type: ", typeof(int_div_result))
   println("Remainder: ", remainder_result)
   ```

   仔细观察输出，特别是 `div_result` 的类型。即使 `num1` 和 `num2` 是整数，`/` 运算符默认执行浮点除法，结果是 `Float64`。如果您需要整数商，请使用 `div()` 函数。
2. **温度转换**:
   我们来将温度从摄氏度转换为华氏度。公式为 $F = C \times \frac{9}{5} + 32$。

   ```julia
   celsius = 25.0
   fahrenheit = celsius * (9/5) + 32
   println(celsius, "°C is equal to ", fahrenheit, "°F")
   ```

   这里，`(9/5)` 的结果是 `1.8`，确保使用浮点运算进行计算以获得准确结果。
3. **使用常量**:
   有些值，例如数学常量，不应该改变。Julia 允许您使用 `const` 定义常量。

   ```julia
   const PI_VALUE = 3.14159
   radius = 5.0
   circumference = 2 * PI_VALUE * radius
   println("Circumference of a circle with radius ", radius, ": ", circumference)

   # 尝试重新赋值 PI_VALUE（这将产生警告）：
   # PI_VALUE = 3.14 # 取消注释以查看警告
   ```

   虽然 Julia 在您尝试重新赋值 `const` 变量时会发出警告，但这是向开发者发出的一个信号，表示该值旨在保持不变。

### 练习 3：处理字符串

文本数据，即字符串，是另一种常见的数据类型。

1. **字符串连接和插值**:
   组合字符串以形成新字符串。

   ```julia
   greeting_start = "Hello"
   user = "Julia User"

   # 使用 * 或 string() 进行连接
   full_greeting_concat = greeting_start * ", " * user * "!"
   # 另一种方式，对于多个部分通常更具可读性：
   # full_greeting_concat = string(greeting_start, ", ", user, "!")
   println("Concatenated: ", full_greeting_concat)

   # 使用 $ 进行字符串插值
   full_greeting_interp = "$greeting_start, $user!"
   println("Interpolated: ", full_greeting_interp)
   ```

   在字符串中嵌入 (embedding)变量值时，带有 `$` 的字符串插值因其可读性而常被选用。
2. **基本字符串函数**:
   查找字符串的长度。

   ```julia
   message = "Julia is fun"
   len_message = length(message)
   println("The message '", message, "' has ", len_message, " characters.")
   ```

### 练习 4：理解布尔逻辑

布尔值（`true` 和 `false`）和逻辑运算符（`&&` 表示 AND，`||` 表示 OR，`!` 表示 NOT）是程序中进行决策的根本。

1. **简单比较**:

   ```julia
   x = 10
   y = 5

   is_x_greater = x > y
   is_y_zero = (y == 0) # 括号用于清晰性，此处并非严格需要

   println("Is x greater than y? ", is_x_greater)
   println("Is y equal to zero? ", is_y_zero)
   ```
2. **组合条件**:
   假设您想检查一个数字是否在特定范围之内。

   ```julia
   value = 75
   is_in_range = (value >= 50) && (value <= 100) # 值在 50 到 100（包含）之间
   println("Is ", value, " in the range [50, 100]? ", is_in_range)

   is_outside_range = (value < 0) || (value > 200) # 值小于 0 或大于 200
   println("Is ", value, " outside the range [0, 200]? ", is_outside_range)
   ```

### 练习 5：类型转换

有时，您需要将数据从一种类型转换为另一种类型。

1. **数字转字符串及反之**:

   ```julia
   num_val = 123
   str_val = string(num_val) # 将整数转换为字符串
   println("Number: ", num_val, ", Type: ", typeof(num_val))
   println("As String: '", str_val, "', Type: ", typeof(str_val))

   numeric_string = "456.7"
   # 要将字符串转换为数字，请使用 parse()
   float_from_string = parse(Float64, numeric_string)
   # 如果您尝试从 "456.7" 中解析整数，将会出错。
   # int_from_string = parse(Int64, "456") # 这将成功
   println("Numeric String: '", numeric_string, "', Type: ", typeof(numeric_string))
   println("Parsed to Float64: ", float_from_string, ", Type: ", typeof(float_from_string))
   ```

   `parse()` 函数对于将文本输入（通常读取为字符串）转换为数值类型很重要。
2. **浮点数转整数**:

   ```julia
   float_num = 99.99
   # trunc() 移除小数部分
   int_from_float_trunc = trunc(Int, float_num)
   # round() 四舍五入到最近的整数
   int_from_float_round = round(Int, float_num)
   # floor() 向下取整
   int_from_float_floor = floor(Int, float_num)
   # ceil() 向上取整
   int_from_float_ceil = ceil(Int, float_num)

   println("Original Float: ", float_num)
   println("Truncated to Int: ", int_from_float_trunc) # 结果：99
   println("Rounded to Int: ", int_from_float_round)   # 结果：100
   println("Floored to Int: ", int_from_float_floor)   # 结果：99
   println("Ceiled to Int: ", int_from_float_ceil)     # 结果：100
   ```

   选择正确的转换函数（`trunc`、`round`、`floor`、`ceil`）取决于您将浮点数转换为整数时需要如何处理小数部分。

这些练习为您实践性地介绍了 Julia 中的变量、常见数据类型和基本操作。尝试修改值，以不同方式组合操作，并观察结果。这种实践是建立对 Julia 如何处理数据的直观认识的好方法，当您开始编写更复杂的程序时，这将非常有价值。

## 参考资料

- [The Julia Language Manual: Variables and Basic Types](https://docs.julialang.org/en/v1/manual/) — The Julia Language (2024)
  这份官方 Julia 手册介绍了变量的概念、Julia 的动态类型推断以及整型、浮点型、字符串和布尔型等基本数据类型。读者应查阅“变量”和“基本类型”章节。
- [The Julia Language Manual: Operations and Type Conversion](https://docs.julialang.org/en/stable/manual/) — Jeff Bezanson, Stefan Karpinski, Viral Shah, Alan Edelman, et al. [6] (2024)
  Publisher: The Julia Language Project
  这份官方 Julia 手册解释了基本算术、字符串操作、比较和逻辑运算符，以及如何使用 `parse()`、`string()`、`trunc()` 和 `round()` 等函数执行类型转换。读者应查阅“数学操作”、“字符串操作”、“比较运算符”和“转换与提升”章节。
- [Think Julia: How to Think Like a Computer Scientist](https://www.oreilly.com/library/view/think-julia/9781492045021/) — Ben Lauwens, Allen B. Downey (2019)
  Publisher: O'Reilly Media; Pages: 295
  这是一本出色的入门教材，使用 Julia 作为教学语言，涵盖了包括变量、数据类型和操作在内的编程基础概念。
