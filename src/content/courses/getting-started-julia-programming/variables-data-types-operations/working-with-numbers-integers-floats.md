---
course: "getting-started-julia-programming"
chapter: "variables-data-types-operations"
lesson: "working-with-numbers-integers-floats"
sourceId: 6769
sourceUrl: "https://apxml.com/zh/courses/getting-started-julia-programming/chapter-2-variables-data-types-operations/working-with-numbers-integers-floats"
title: "使用数值：整数和浮点数"
description: "了解 Julia 的数值类型，包括整数、浮点数、有理数和复数。"
order: 3
plots: []
sourceHash: "2948d258d04e92a96d1cd347efb484b1654b015136db4217e929fba754068073"
sourceCorrections: []
---

数值是任何编程语言中计算的根本。Julia 提供了丰富的数值类型集，可以应对各种数值计算任务。Julia 中不同类型数值的表示和处理方式被考察，从整数和浮点数值开始。

### 整数：表示整数

整数是完整的数值，例如 -5、0、42 或 1000。它们不包含任何小数部分。在 Julia 中，当您输入一个完整的数值时，它通常会被解释为整数。

```julia
x = 10
y = -200
typeof(x) # 输出：Int64（或在 32 位系统上为 Int32）
```

`typeof()` 函数，如您之前所见，会告诉我们变量的数据类型。默认情况下，Julia 在 64 位计算机系统上通常将 `Int64` 用于整数，这意味着它使用 64 位内存来存储该数值。这允许存储非常大范围的数值。在较旧的 32 位系统上，它可能默认为 `Int32`。Julia 中的通用 `Int` 类型是系统“原生”整数大小的别名，如今通常是 `Int64`。

如果您需要更多控制，或者处理来自指定特定大小的外部数据源的数据，Julia 还提供了固定大小的整数类型。它们包括：

- `Int8`、`Int16`、`Int32`、`Int64`、`Int128`：有符号整数，可以保存正值和负值。
- `UInt8`、`UInt16`、`UInt32`、`UInt64`、`UInt128`：无符号整数，只能保存非负值（0 和正值）。

类型名称中的数字（例如 `Int8` 中的 8）表示用于存储该数值的位数。位数越多意味着可表示数值的范围越大，但也会占用更多内存。例如，一个 `Int8` 可以存储从 -128 到 127 的数值。

```julia
small_num = Int8(100)
large_num = Int64(9000000000000000000) # 一个非常大的整数
positive_only = UInt8(250)

println(typeof(small_num))    # 输出：Int8
println(typemin(Int8))      # 输出：-128
println(typemax(Int8))      # 输出：127
println(typemax(UInt8))     # 输出：255
```

通常您不需要指定这些大小，除非有特定原因，例如为非常大的小型整数数组优化内存或与硬件交互。对于日常大多数编程任务，`Int`（默认是 `Int64` 或 `Int32`）就足够了。

请注意，如果您尝试存储一个超出其可表示范围的数值，固定大小的整数会“溢出”。例如，赋值 `Int8(300)` 会导致错误。同样，算术运算也可能溢出：`Int8(127) + Int8(1)` 由于“回绕”而导致 `Int8(-128)`。对于默认的 `Int` 类型，Julia 通常被设置为对 `typemax(Int) + 1` 这样的操作抛出 `OverflowError`，这有助于捕获此类问题。

### 浮点数：处理实数和小数

当您需要表示带小数部分的数值，或者非常大或非常小的数值时，您会使用浮点数。例如测量值 3.14159、-0.0025，或科学记数法，例如 $6.022 \times 10^{23}$。

在 Julia 中，写入带小数点的数值会自动使其成为浮点数。您也可以使用 `e` 记号进行科学表示。

```julia
pi_approx = 3.14159
temperature = -15.5
scientific_notation_val = 2.5e-4 # 这表示 2.5 * 10^-4，即 0.00025

println(typeof(pi_approx)) # 输出：Float64
println(scientific_notation_val) # 输出：0.00025
```

Julia 中浮点数的默认类型是 `Float64`，它代表 64 位双精度浮点数。这在大多数科学和通用计算中，在数值范围和精度之间提供了良好的平衡。

与整数类似，如果需要，Julia 也支持其他浮点精度：

- `Float16`：半精度（16 位）。占用内存更少，但范围更小，精度更低。在一些特定应用中很有用，例如在机器学习 (machine learning)的某些方面，内存带宽可能是一个瓶颈。
- `Float32`：单精度（32 位）。当内存或速度比最高精度更重要时，它是 `Float64` 的常见替代方案。
- `Float64`：双精度（64 位）。默认且最常用。

```julia
f16 = Float16(0.5)
f32 = Float32(1.23456789)
f64 = 1.2345678901234567 # 自动为 Float64

println(f16) # 输出：Float16(0.5)
println(f32) # 输出：1.2345679（请注意打印精度）
println(f64) # 输出：1.2345678901234567
```

#### 特殊浮点数值

浮点运算包含一些特殊数值，用于表示像除以零或未定义结果等情况：

- `Inf`：表示正无穷大。例如，`1.0 / 0.0` 的结果是 `Inf`。
- `-Inf`：表示负无穷大。例如，`-1.0 / 0.0` 的结果是 `-Inf`。
- `NaN`：代表“不是一个数值”。它表示一个未定义或无法表示的数值，例如 `0.0 / 0.0` 或 `sqrt(-1.0)` 的结果（不使用复数时）。

```julia
a = 1.0 / 0.0
b = -1.0 / 0.0
c = 0.0 / 0.0

println(a) # 输出：Inf
println(b) # 输出：-Inf
println(c) # 输出：NaN

println(isinf(a)) # 输出：true
println(isnan(c)) # 输出：true
```

任何涉及 `NaN` 的算术运算通常结果都是 `NaN`。例如，`5.0 + NaN` 是 `NaN`。

#### 关于精度的一个说明

重要的是要理解，浮点数在大多数情况下是实数的近似值。计算机使用二进制系统，许多十进制小数无法在二进制中完美表示。这可能导致计算中出现微小的差异。

```julia
val1 = 0.1
val2 = 0.2
sum_val = val1 + val2

println(sum_val)          # 输出：0.30000000000000004
println(sum_val == 0.3)   # 输出：false
```

这种行为是浮点数在几乎所有编程语言中处理方式的一个基本方面，不仅仅是 Julia。对于许多应用来说，这些微小的差异可以忽略不计。然而，如果分数的精确性绝对重要，Julia 提供了有理数。在比较浮点数的相等性时，通常最好检查它们是否“足够接近”，而不是严格相等，可以使用 `isapprox(x, y)` 这样的函数。

```julia
println(isapprox(0.1 + 0.2, 0.3)) # 输出：true
```

### 有理数：精确分数算术

对于需要精确表示分数且不带任何浮点近似误差的情况，Julia 提供了有理数。有理数就是两个整数（分子和分母）的比值。

您可以使用 `//` 运算符创建有理数。

```julia
fraction1 = 1//3
fraction2 = 2//5

sum_fractions = fraction1 + fraction2
println(sum_fractions) # 输出：11//15

println(typeof(fraction1))    # 输出：Rational{Int64}
```

有理数在算术中保持精确性。例如，`1//3 + 1//3 + 1//3` 将精确地等于 `1//1`（即 1），而不是像 `0.9999999999999999` 这样与 1 非常接近的数值，这是浮点运算可能得到的结果。它们在数论等领域非常有用，或者当浮点运算中即使微小的误差也无法接受时。

```julia
float_sum = 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 # 10 次
rational_equivalent = 1//10 * 10

println(float_sum)         # 输出：0.9999999999999999
println(rational_equivalent) # 输出：1//1
```

### 复数：具有实部和虚部的数值

Julia 内置支持复数，它们是形如 $a + bi$ 的数值，其中 $a$ 是实部，$b$ 是虚部，$i$ 是虚数单位（定义为 $i^2 = -1$）。在 Julia 中，虚数单位写为 `im`。

```julia
c1 = 3 + 4im
c2 = 1.0 - 2.5im # 分量可以是浮点数

println(typeof(c1))     # 输出：Complex{Int64}
println(typeof(c2))     # 输出：Complex{Float64}

# 复数运算
println(c1 + c2) # 输出：4.0 + 1.5im
println(c1 * c2) # 输出：13.0 - 3.5im
```

您可以使用 `real()` 和 `imag()` 函数访问复数的实部和虚部：

```julia
real_part = real(c1)
imag_part = imag(c1)

println("c1 的实部：$real_part，c1 的虚部：$imag_part")
# 输出：c1 的实部：3，c1 的虚部：4
```

复数可以使用任意实数数值类型（整数或浮点数）组合其实部和虚部来构成。它们在科学和工程的许多方面都很重要，例如电气工程、量子力学和信号处理。例如，`sqrt(-4.0 + 0im)` 将正确地得到 `0.0 + 2.0im`。

### 基本算术运算

既然我们已经了解了这些不同类型的数值，现在来看看如何进行常见计算。Julia 支持您预期的标准算术运算符：

- `+`（加法）
- `-`（减法，或与一个数值使用时表示取反，例如 `-x`）
- `*`（乘法）
- `/`（除法）
- `^`（求幂/乘方）

```julia
a_int = 10
b_float = 3.0

println(a_int + b_float)  # 输出：13.0（由于类型提升为 Float64）
println(a_int - 5)  # 输出：5 (Int64)
println(b_float * 2)  # 输出：6.0 (Float64)
println(10 / 4) # 输出：2.5（Float64，标准除法通常产生浮点数）
println(2^3)  # 输出：8 (Int64)
println(2.0^3) # 输出：8.0 (Float64)
```

#### 整数专用除法和余数

在处理整数时，有时您希望除法结果也是整数，舍弃任何小数部分。为此，Julia 提供了 `div()` 函数（或其运算符形式 `÷`，您可以在 Julia REPL 中输入 `\div` 后按 TAB 键）。要获取此类除法的余数，您可以使用 `%` 运算符（取模）或 `rem()` 函数。

```julia
numerator = 17
denominator = 5

quotient = div(numerator, denominator) # 或 numerator ÷ denominator
remainder = numerator % denominator    # 或 rem(numerator, denominator)

println("对于 17 除以 5：")
println("整数商 (div)：$quotient")     # 输出：整数商 (div)：3
println("余数 (%)：$remainder")   # 输出：余数 (%)：2
```

这与标准除法 `/` 不同，`17 / 5` 会得到 `3.4`。

### 类型提升：Julia 的智能转换

Julia 方便而强大的特性之一是算术运算中的自动类型提升。当您组合不同类型的数值时（例如 `Int` 和 `Float64`），Julia 通常会将结果“提升”到更通用的类型，以准确表示结果，通常不会丢失信息。

例如，如果您将一个 `Int` 和一个 `Float64` 相加：

```julia
integer_val = 5
float_val = 2.5

result = integer_val + float_val
println(result)       # 输出：7.5
println(typeof(result)) # 输出：Float64
```

在这里，整数 `5` 在加法之前有效地转换为 `5.0`，结果是一个 `Float64`。这可以防止在结果被强制为整数时意外丢失小数部分。这种提升系统也适用于有理数和复数，通常会产生能容纳结果的“最丰富”或最通用的类型。

- `Int + Rational` 结果为 `Rational`。（例如，`1 + 1//2` 是 `3//2`）
- `Float64 + Rational` 结果为 `Float64`。（例如，`0.5 + 1//4` 是 `0.75`）
- `Int + Complex{Int}` 结果为 `Complex{Int}`。（例如，`1 + (2+3im)` 是 `3+3im`）

这种行为有助于避免常见的编程错误，并使数值代码更直观、更易于编写。

### 了解数值的类型

当您处理数值时，尤其是在处理函数结果或复杂表达式时，使用 `typeof()` 仍然是一个非常有用的工具。它允许您检查和理解 Julia 为特定数值使用的具体数值类型。

```julia
val_div = 10 / 2 # 标准除法
println(typeof(val_div)) # 输出：Float64（即使结果是整数）

val_int_explicit = Int(5.0) # 从 Float64 到 Int64 的显式转换
println(typeof(val_int_explicit)) # 输出：Int64
```

理解这些数值类型、它们的特性以及它们如何通过运算和类型提升进行交互，是编写任何涉及计算的 Julia 代码的核心要素。这些知识将对您很有帮助，从简单的算术到高级的科学计算和数据分析任务。

## 参考资料

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) — David Goldberg (1991)
  Journal: ACM Computing Surveys; Publisher: Association for Computing Machinery; Volume: 23; Pages: 5-48; DOI: [10.1145/103162.103163](https://doi.org/10.1145/103162.103163)
  一篇经典且权威的论文，详细阐述了计算机中浮点算术的表示、局限性及面临的挑战。
