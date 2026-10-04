---
course: "getting-started-julia-programming"
chapter: "functions-creating-reusable-code"
lesson: "multiple-dispatch-distinctive-julia-feature"
sourceId: 6799
sourceUrl: "https://apxml.com/zh/courses/getting-started-julia-programming/chapter-5-functions-creating-reusable-code/multiple-dispatch-distinctive-julia-feature"
title: "多重派发：Julia 的一个独特功能"
description: "对多重派发的一种入门了解，它是 Julia 的一个核心功能，允许函数根据参数类型表现出不同行为。"
order: 7
plots: []
sourceHash: "8a0c62ce878f9ada437e7d1f128a49a3797916c44bf0fdffef354b49afee407c"
sourceCorrections: []
---

多重派发是 Julia 最鲜明的特点之一，它影响着您在该语言中编写代码的方式。函数是创建可复用代码片段的非常好的工具。多重派发将这个理念更进一步，允许单个函数名根据其接收的参数 (parameter)类型拥有不同的行为，或称作*方法*。这不只是一点小便利；它是 Julia 设计的根本所在，能使代码既有表现力又高效。

### 究竟什么是多重派发？

设想您有一个通用任务，例如“组合两样东西”。如何组合它们很大程度上取决于这两样东西*是什么*。组合两个数字通常意味着加法。组合两段文本（字符串）意味着将它们连接起来。将图片与说明文字组合则涉及一套不同的操作。

多重派发让您定义一个函数，例如 `combine()`，然后为每种您关注的参数 (parameter)类型组合提供单独的实现，称作**方法**。当您调用 `combine(a, b)` 时，Julia 在调用时查看 `a` 和 `b` 的类型，并自动选择要执行的正确方法。多重派发中的“多重”指的是这个选择过程考虑函数*所有*参数的类型，而不仅仅是一个参数。

我们用一个简单的例子来说明。假设我们希望有一个函数 `interact`，它根据其两个输入的类型表现出不同的行为：

```julia
# 方法1：针对两个数字
function interact(x::Number, y::Number)
    println("与两个数字交互。它们的和是：", x + y)
    return x + y
end

# 方法2：针对两个字符串
function interact(x::String, y::String)
    println("与两个字符串交互。让我们连接它们：'", string(x, y), "'")
    return string(x, y)
end

# 方法3：针对一个数字和一个字符串
function interact(x::Number, y::String)
    println("与一个数字和一个字符串交互。数字：", x, ", 字符串：'", y, "'")
    return string(x, " the ", y) # 组合示例
end

# 让我们看看实际效果
interact(5, 10)
interact("Hello, ", "Julia user!")
interact(100, "points")

# 如果我们先提供一个字符串，然后提供一个数字呢？
# interact("Chapter", 5) # 这会引起错误，除非我们为此定义一个方法：
# function interact(x::String, y::Number)
#     println("与一个字符串和一个数字交互。字符串：'", x, "', 数字：", y)
#     return string(x, " ", y)
# end
# interact("Chapter", 5) # 现在这样就会工作了
```

在这段代码中：

1. 我们为名为 `interact` 的函数定义了三个不同的方法。
2. 每个 `function interact(...)` 行都开始一个新的方法定义。
3. 参数名后面的 `::Number` 和 `::String` 是类型注解。它们告诉 Julia，当参数与这些类型匹配时，应使用此特定方法。
4. 当我们调用 `interact(5, 10)` 时，Julia 看到两个参数都是 `Number` 类型（具体来说，是 `Int` 类型，它是 `Number` 的子类型），并执行第一个方法。
5. 当我们调用 `interact("Hello, ", "Julia user!")` 时，Julia 选择第二个方法，因为两个参数都是 `String` 类型。
6. 对于 `interact(100, "points")`，选择了第三个方法。

如果您尝试调用 `interact` 时使用没有定义特定方法的类型组合（例如在方法定义之前最初调用 `interact("Chapter", 5)`），Julia 会通过 `MethodError` 告知您，表明它找不到匹配的方法。这很有用，因为它准确地告诉您尚未定义哪种交互。

### 方法选择的可视化

您可以将多重派发视为一个智能路由系统。当函数被调用时，Julia 会检查参数 (parameter)的类型，并将调用导向到可用的最特定匹配方法。

> 当调用 `interact(arg1, arg2)` 时，Julia 会确定 `arg1` 和 `arg2` 的类型，并选择相应的方法实现。

### 为什么多重派发很重要？

多重派发一开始可能看起来是一个细微的特点，但它对您编写和组织 Julia 代码的方式有重要影响：

1. **思想的自然表达**：它让您根据通用操作（例如 `add`、`plot`、`convert`）来命名函数，然后为不同类型定义专门的行为。这通常会使代码反映您思考问题的方式。例如，Julia 中的 `+` 运算符只是一个拥有许多方法的函数：一个用于整数相加，一个用于浮点数相加，一个用于数组连接，等等。

   ```julia
   println(1 + 2)          # 使用整数加法的方法
   println(1.5 + 2.5)      # 使用浮点数加法的方法
   println("a" + "b")    # 错误：默认情况下没有针对字符串+字符串的方法。请使用 string() 或 *。
   # 但是包可以为它们自己的类型定义 +！
   ```

   (注意：Base Julia 使用 `string()` 或 `*` 进行字符串连接，但重点是如果需要，`+` *可以*为字符串定义。)
2. **可扩展性**：新类型和新方法可以随时添加，甚至由不同模块或包中的不同程序员添加，而无需修改现有代码。如果您创建新的自定义数据类型，例如 `MySpecialNumber`，您只需添加新方法，就可以定义它如何与现有函数（如 `+` 或 `interact`）交互：

   ```julia
   # struct MySpecialNumber ... end # 定义您的类型
   # function interact(x::MySpecialNumber, y::Number) ... end
   ```

   这使得 Julia 的生态系统具有高度的可组合性。库可以顺畅地协同工作，因为它们可以用针对其特定类型的方法来扩展通用函数。
3. **代码复用性**：您为在不同类型中具有相似用途的操作复用函数名，而不是发明略有不同的名称（例如 `add_integers`、`add_floats`）。
4. **性能**：虽然它可能看起来复杂，但 Julia 的设计使其编译器在多重派发方面非常智能。它通常可以准确地确定将调用哪个方法，并为该特定情况编译高度优化、专门化的代码。这显著提升了 Julia 的高性能。

许多编程语言都有看起来有些相似的功能，例如函数重载（在 C++ 或 Java 中常见）或面向对象编程中的方法。然而，Julia 的多重派发更全面，因为方法的选择可以依赖于*所有*参数 (parameter)的动态类型，而不仅仅是编译时的静态选择或在单个“对象”参数上的派发。

随着您在 Julia 编程中的进展，以通用函数和专用方法的角度思考将变得自然而然。这是一种构建程序的有效方式，从而产生灵活、有条理且通常出人意料地快的代码。目前，主要的一点是，当您在 Julia 中定义一个函数时，您实际上是为（可能新的）通用函数定义了一个新的*方法*。您可以为同一个通用函数添加更多方法来处理不同类型的输入，Julia 将选择最适合任务的方法。这种方法是解决问题“Julia 方式”的核心。

## 参考资料

- [Methods](https://docs.julialang.org/en/v1/manual/methods/) — The Julia Language (2024)
  Publisher: The Julia Language
  Julia 官方文档中解释方法如何通过多重派发定义和选择的部分。
- [Julia: A Fresh Approach to Technical Computing](https://arxiv.org/abs/1411.1607) — Jeff Bezanson, Alan Edelman, Stefan Karpinski, and Viral B. Shah (2017)
  Journal: SIAM Review; Publisher: Society for Industrial and Applied Mathematics; Volume: 59; Pages: 65-98; DOI: [10.1137/141000671](https://doi.org/10.1137/141000671)
  Julia 语言创始人撰写的奠基性论文，描述了该语言的设计原则，重点介绍了多重派发。
