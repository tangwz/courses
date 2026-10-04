---
course: "getting-started-julia-programming"
chapter: "expressions-control-flow-structures"
lesson: "comparison-logical-operators"
sourceId: 6778
sourceUrl: "https://apxml.com/zh/courses/getting-started-julia-programming/chapter-3-expressions-control-flow-structures/comparison-logical-operators"
title: "比较运算符和逻辑运算符"
description: "了解如何在 Julia 中使用比较运算符（如 `==`、`<`、`>`）和逻辑运算符（与、或、非）。"
order: 3
plots: []
sourceHash: "507b6bae1b3d32c2c420a88407f43ce364662429a4aeaede3eab999650e18e93"
sourceCorrections: []
---

为了开发有效的程序，仅仅进行计算通常是不够的。要创建能够响应不同情况并做出选择的程序，需要方法来比较值和组合逻辑条件。因此，比较运算符和逻辑运算符变得非常重要。它们使 Julia 程序能够评估条件，最终决定程序的行为方式。

### 进行比较

比较运算符，也称为关系运算符，用于比较两个值。比较的结果总是一个布尔值：`true` 或 `false`。这些运算符是代码中做出判断的根本。

以下是 Julia 中可用的标准比较运算符：

- `==` (等于): 如果两边的值相等，则返回 `true`。

  ```julia
  5 == 5    # true
  5 == 3    # false
  "hello" == "hello" # true
  ```
- `!=` (不等于): 如果值不相等，则返回 `true`。这也可以写成 `≠`（在 Julia REPL 或许多支持 Julia 的编辑器中，你可以输入 `\ne` 后按 Tab 键）。

  ```julia
  5 != 3    # true
  5 != 5    # false
  "julia" != "python" # true
  ```
- `<` (小于): 如果左边的值小于右边的值，则返回 `true`。

  ```julia
  3 < 5     # true
  5 < 3     # false
  ```
- `>` (大于): 如果左边的值大于右边的值，则返回 `true`。

  ```julia
  10 > 5    # true
  5 > 10    # false
  ```
- `<=` (小于或等于): 如果左边的值小于或等于右边的值，则返回 `true`。也可以是 `≤`（输入 `\le` 后按 Tab 键）。

  ```julia
  4 <= 5    # true
  5 <= 5    # true
  6 <= 5    # false
  ```
- `>=` (大于或等于): 如果左边的值大于或等于右边的值，则返回 `true`。也可以是 `≥`（输入 `\ge` 后按 Tab 键）。

  ```julia
  7 >= 5    # true
  5 >= 5    # true
  3 >= 5    # false
  ```

你可以将这些运算符与各种数据类型一起使用，例如数字和字符串。当比较不同类型的数字（例如，整数和浮点数）时，Julia 通常会在比较前将它们提升为通用类型。

```julia
x = 10
y = 5.0

println(x > y)      # 输出: true
println(x == 10.0)  # 输出: true
println(y < 3)      # 输出: false
```

### 使用逻辑运算符组合条件

通常，单个比较常常不够。你可能需要检查多个条件是否都为真，或者多个条件中至少有一个为真。逻辑运算符允许你组合或修改布尔值（`true` 或 `false`）。

Julia 提供了三种主要的逻辑运算符：

- `&&` (逻辑与): 当且仅当两个操作数都为 `true` 时，返回 `true`。
- `||` (逻辑或): 如果至少有一个操作数为 `true`，则返回 `true`。
- `!` (逻辑非): 反转其操作数的布尔值。如果操作数为 `true`，它返回 `false`；如果为 `false`，则返回 `true`。

我们来看看它们的行为，通常概括在所谓的真值表中：

**逻辑与 (`&&`)**

| 操作数 A | 操作数 B | `A && B` |
| --- | --- | --- |
| `true` | `true` | `true` |
| `true` | `false` | `false` |
| `false` | `true` | `false` |
| `false` | `false` | `false` |

**逻辑或 (`||`)**

| 操作数 A | 操作数 B | `A \|\| B` |
| :-------- | :-------- | :------- |
| `true` | `true` | `true` |
| `true` | `false` | `true` |
| `false` | `true` | `true` |
| `false` | `false` | `false` |

**逻辑非 (`!`)**

| 操作数 A | `!A` |
| --- | --- |
| `true` | `false` |
| `false` | `true` |

以下是 Julia 中的一些例子：

```julia
is_sunny = true
is_warm = false

# 逻辑与
can_go_to_beach = is_sunny && is_warm
println("Can go to the beach? $can_go_to_beach")  # 输出: 可以去海滩吗？ false

# 逻辑或
is_good_weather_overall = is_sunny || is_warm
println("Good weather overall? $is_good_weather_overall") # 输出: 总体天气好吗？ true

# 逻辑非
is_not_sunny = !is_sunny
println("Is it not sunny? $is_not_sunny")        # 输出: 是不是没有阳光？ false
```

你可以将逻辑运算符组合条件的方式可视化。例如，如果某个操作需要两个条件都为真才能发生，那么这两个条件必须在与运算符组合时都评估为 `true`。

> 两个条件，`is_registered` 和 `has_paid_fee`，通过逻辑与运算符组合。如果两者都为 `true`，结果则为 `true`。

### 短路评估：高效逻辑

Julia 的逻辑 `&&` 和 `||` 运算符表现出一种称为“短路评估”的行为。这是一项重要的效率特性，也可以用于防止错误。

- 对于 `A && B`：如果 `A` 的值为 `false`，则整个表达式必然为 `false`，因此 `B` 不会被评估。
- 对于 `A || B`：如果 `A` 的值为 `true`，则整个表达式必然为 `true`，因此 `B` 不会被评估。

考虑这个例子：

```julia
function is_valid_user(id::Int)
    println("Checking user ID: $id") # 这行代码帮助我们查看函数是否运行
    return id > 0 && id < 1000       # 一个简单的有效性检查
end

user_id = 0
access_granted = false && is_valid_user(user_id)
# “Checking user ID: 0” 将不会被打印，因为 `false && ...` 发生了短路。
println("Access granted (first attempt): $access_granted") # 输出: false

user_name = "admin"
is_admin = true
has_special_privileges = is_admin || is_valid_user(100)
# “Checking user ID: 100” 将不会被打印，因为 `true || ...` 发生了短路。
println("Has special privileges: $has_special_privileges") # 输出: true

# 第二部分被评估的例子
can_login = (user_name == "admin") && is_valid_user(500)
# “Checking user ID: 500” 将被打印，因为第一部分为 true。
println("Can login: $can_login") # 输出: true (假设 is_valid_user(500) 返回 true)
```

短路评估很有用，例如，当仅在对象本身不是 `nothing` 时才检查其属性，从而防止错误发生。

### 组合运算符和使用括号

你可以通过组合多个比较运算符和逻辑运算符来创建复杂的逻辑表达式。

```julia
age = 25
has_license = true
is_student = false

# 如果年龄大于等于 25 岁且有驾照，则可以租车
can_rent_car = (age >= 25) && has_license
println("Can rent car: $can_rent_car") # 输出: true

# 如果是学生或年龄小于 18 岁（并且隐含地不需要驾照才能获得此折扣），则有资格获得折扣
eligible_for_discount = is_student || (age < 18)
println("Eligible for discount: $eligible_for_discount") # 输出: false
```

当表达式变得复杂时，评估顺序很重要。逻辑运算符有明确定义的优先级（`!` 最高，然后是 `&&`，然后是 `||`）。然而，依赖隐式优先级会使代码更难阅读和调试。

**强烈建议使用括号 `()` 来明确地将表达式的各个部分分组。** 这能使你的意图清晰，并确保逻辑按你预期的方式进行评估。

例如，与其写 `a && b || c`，不如根据你期望的逻辑写 `(a && b) || c` 或 `a && (b || c)`。

### 简要说明：相等运算符 (`==`) 与赋值运算符 (`=`)

新手程序员常见的困惑点是相等运算符 `==` 和赋值运算符 `=` 之间的区别。

- `=`（单个等号）用于**赋值**。它将右边的值赋给左边的变量。

  ```julia
  my_variable = 10  # 将值 10 赋给 my_variable
  ```
- `==`（双等号）用于**比较**。它检查两个值是否相等并返回 `true` 或 `false`。

  ```julia
  my_variable == 10 # 检查 my_variable 是否等于 10；返回 true
  my_variable == 5  # 检查 my_variable 是否等于 5；返回 false
  ```

混淆它们可能导致有时难以发现的错误，所以务必仔细检查你是否为预期目的使用了正确的运算符。

比较运算符和逻辑运算符是构建能够做出决策的程序的核心组成部分。它们产生的 `true` 或 `false` 值正是控制流结构（例如你接下来将学习的 `if-elseif-else` 语句）用来决定执行哪些代码块的依据。掌握这些运算符是编写更动态和智能的 Julia 程序的重要一步。

## 参考资料

- [The Julia Language Manual: Mathematical Operations](https://docs.julialang.org/en/v1/manual/mathematical-operations/#Operators-1) — The Julia Language (2024)
  官方文档，详细介绍了所有 Julia 运算符，包括比较运算符、逻辑运算符及其行为（如短路求值）。
- [Introduction to Computation and Programming Using Python (With Application to Understanding Data)](https://mitpress.mit.edu/books/introduction-computation-and-programming-using-python-third-edition) — John V. Guttag (2021)
  Publisher: The MIT Press
  提供了核心编程概念的清晰介绍，包括布尔逻辑、比较运算符和控制流，这些概念在各种编程语言中普遍适用。
