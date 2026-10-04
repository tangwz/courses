---
course: "getting-started-julia-programming"
chapter: "expressions-control-flow-structures"
lesson: "arithmetic-assignment-operations"
sourceId: 6777
sourceUrl: "https://apxml.com/zh/courses/getting-started-julia-programming/chapter-3-expressions-control-flow-structures/arithmetic-assignment-operations"
title: "算术与赋值运算"
description: "了解用于计算的算术运算符以及用于更新变量值的赋值运算符。"
order: 2
plots: []
sourceHash: "a91d209966101e7d57a8115a69eb8b43e9a358d567ad832f120f3d999e994d5c"
sourceCorrections: []
---

变量作为数据的容器。Julia程序经常需要使用这些值进行计算，并将结果存回变量中。算术和赋值运算是完成此操作的主要方式，它们是几乎所有Julia程序的基础。

### 算术运算：计算的基本组成部分

算术运算是在Julia中进行数学计算的方式。你可能在日常数学中遇到过这些运算。Julia对大多数这些运算使用标准符号。

- **加法 (`+`)**: 将两个数字相加。

  ```julia
  julia> 5 + 3
  8
  ```
- **减法 (`-`)**: 从第一个数字中减去第二个数字。

  ```julia
  julia> 10 - 4
  6
  ```
- **乘法 (`*`)**: 将两个数字相乘。

  ```julia
  julia> 7 * 6
  42
  ```
- **除法 (`/`)**: 将第一个数字除以第二个数字。除法总是得到一个浮点数，即使是精确除法也是如此。

  ```julia
  julia> 10 / 2
  5.0

  julia> 7 / 2
  3.5
  ```

Julia提供了一些更有用的特殊算术运算符：

- **整除 (`÷` 或 `div()`)**: 执行除法并返回商的整数部分，舍弃任何余数。你可以在Julia REPL或许多支持Julia的编辑器中通过输入 `\div` 然后按Tab键来输入 `÷` 符号。

  ```julia
  julia> 10 ÷ 3  # \div<TAB>
  3

  julia> div(10, 3)
  3
  ```
- **取余 (`%` 或 `rem()`)**: 返回整除后的余数。这通常被称为模运算符。

  ```julia
  julia> 10 % 3
  1

  julia> rem(10, 3)
  1
  ```

  取余运算符对于检查数字是偶数还是奇数（例如，`number % 2 == 0` 表示偶数）等任务很有用。
- **幂运算 (`^`)**: 将第一个数字提高到第二个数字的幂。

  ```julia
  julia> 2^3  # 2 to the power of 3
  8

  julia> 5^2
  25
  ```
- **一元减号和加号 (`-`, `+`)**: 减号也可以用来对数字取反。加号可以用来表示正数，尽管它通常被省略。

  ```julia
  julia> x = 5
  5
  julia> -x
  -5
  julia> +x
  5
  ```

#### 运算符优先级

当一个表达式包含多个运算符时，Julia遵循标准的运算顺序，这通常可以通过像PEMDAS/BODMAS（括号/方括号、指数/阶、乘法和除法、加法和减法）这样的缩写词来记住。这意味着 `*` 和 `/` 在 `+` 和 `-` 之前执行。

例如：

```julia
julia> 2 + 3 * 4
14
```

这里，`3 * 4`（即12）首先被计算，然后将2加到结果上，得到14。如果你想改变顺序，可以使用括号 `()`:

```julia
julia> (2 + 3) * 4
20
```

在这种情况下，由于括号的存在，`2 + 3`（即5）首先被计算，然后结果乘以4，得到20。

如果你不确定运算顺序，或者想让表达式更清晰易读，使用括号是一个好习惯。

下面是常见算术运算符及其典型优先级（优先级越高表示越早执行）的总结：

| 运算符 | 描述 | 示例 | 结果 | 优先级 |
| --- | --- | --- | --- | --- |
| `^` | 幂运算 | `2 ^ 3` | `8` | 最高 |
| `-` (一元) | 取反 | `-5` | `-5` | 高 |
| `*`, `/`, `÷`, `%` | 乘法、除法、整除、取余 | `6 * 2`, `6 / 2`, `7 ÷ 2`, `7 % 2` | `12`, `3.0`, `3`, `1` | 中 |
| `+`, `-` (二元) | 加法、减法 | `5 + 2`, `5 - 2` | `7`, `3` | 低 |

> 下图演示了表达式 `value = 10 + 4 * 2` 的计算顺序。乘法在加法之前执行，然后是赋值。

### 赋值运算：存储和更新值

在第2章中，你学习了变量以及如何使用等号 `=` 为它们赋值。这是最基本的赋值运算。

```julia
julia> width = 10    # Assigns the value 10 to the variable width
10

julia> height = 5     # Assigns the value 5 to the variable height
5

julia> area = width * height  # Calculates width * height (50) and assigns it to area
50
```

`=` 右侧的表达式首先被求值，然后其结果存储在左侧的变量中。

#### 复合赋值运算符

通常，你会希望根据变量的当前值来修改其值。例如，你可能想将一个数字加到现有变量上。你可以这样写：

```julia
julia> score = 100
100

julia> score = score + 10
110
```

这是完全有效的。然而，Julia与许多其他编程语言一样，提供了一种使用**复合赋值运算符**（也称为增广赋值运算符）来编写此类操作的更简洁方式。

这些运算符将算术运算与赋值结合起来。以下是常用的：

- **`+=` (加法并赋值)**: `x += y` 是 `x = x + y` 的简写。

  ```julia
  julia> count = 5
  5
  julia> count += 3  # Same as count = count + 3
  8
  ```
- **`-=` (减法并赋值)**: `x -= y` 是 `x = x - y` 的简写。

  ```julia
  julia> balance = 50.0
  50.0
  julia> balance -= 10.5 # Same as balance = balance - 10.5
  39.5
  ```
- **`*=` (乘法并赋值)**: `x *= y` 是 `x = x * y` 的简写。

  ```julia
  julia> quantity = 3
  3
  julia> quantity *= 4   # Same as quantity = quantity * 4
  12
  ```
- **`/=` (除法并赋值)**: `x /= y` 是 `x = x / y` 的简写。

  ```julia
  julia> total_value = 100.0
  100.0
  julia> total_value /= 4 # Same as total_value = total_value / 4
  25.0
  ```
- **`%=` (取余并赋值)**: `x %= y` 是 `x = x % y` 的简写。

  ```julia
  julia> items = 27
  27
  julia> items_per_box = 5
  5
  julia> items %= items_per_box # Same as items = items % items_per_box (remainder)
  2
  ```
- **`^=` (幂运算并赋值)**: `x ^= y` 是 `x = x ^ y` 的简写。

  ```julia
  julia> num = 2
  2
  julia> num ^= 3 # Same as num = num ^ 3
  8
  ```

使用这些复合运算符可以使你的代码更短，有时也更易读，特别是在变量名较长时。

下面是这些赋值运算符的汇总表：

| 运算符 | 示例 | 等同于 | 描述 |
| --- | --- | --- | --- |
| `=` | `x = 5` | `x = 5` | 基本赋值 |
| `+=` | `x += 2` | `x = x + 2` | 加法并赋值 |
| `-=` | `x -= 2` | `x = x - 2` | 减法并赋值 |
| `*=` | `x *= 2` | `x = x * 2` | 乘法并赋值 |
| `/=` | `x /= 2` | `x = x / 2` | 除法并赋值 |
| `%=` | `x %= 2` | `x = x % 2` | 取余（模）并赋值 |
| `÷=` | `x ÷= 2` | `x = x ÷ 2` | 整除并赋值 |
| `^=` | `x ^= 2` | `x = x ^ 2` | 幂运算并赋值 |

### 关于运算中类型的一个说明

当你对不同类型的数字（例如整数和浮点数）执行算术运算时，Julia通常会自动将结果“提升”为可以准确表示这两个值的类型。例如，将 `Int` 类型与 `Float64` 类型相加通常会得到一个 `Float64` 类型：

```julia
julia> 10 + 5.5
15.5

julia> typeof(10)
Int64

julia> typeof(5.5)
Float64

julia> typeof(10 + 5.5)
Float64
```

这通常是你想要的结果，因为它防止了精度损失。我们在第2章中讨论了数据类型，并且当你编写更复杂的程序时，了解它们在运算过程中如何交互是很重要的。目前，请注意Julia会尝试合理地处理以保留你的数据。

算术和赋值运算是Julia中计算的主力。通过结合它们，你可以在程序中执行各种计算并有效地管理数据。当我们转向控制流时，你将看到这些运算如何融入到做出决策和重复任务的更大结构中。

## 参考资料

- [Mathematical Operations](https://docs.julialang.org/en/v1/manual/mathematical-operations/) — The Julia Language (2024)
  官方文档，涵盖标准和专用算术运算符，包括整数除法、求余、幂运算和一元运算。
- [Operators](https://docs.julialang.org/en/v1/manual/) — The Julia Language (2024)
  Publisher: The Julia Language
  官方文档，详细说明运算符优先级和各种赋值运算符，包括复合赋值。
- [Think Julia: How to Think Like a Computer Scientist](https://benlauwens.github.io/ThinkJulia.jl/latest/book.html) — Ben Lauwens, Allen B. Downey (2019)
  Publisher: O'Reilly Media
  一本入门教材，以教学方式解释 Julia 中的基本编程概念，包括算术和赋值操作。
