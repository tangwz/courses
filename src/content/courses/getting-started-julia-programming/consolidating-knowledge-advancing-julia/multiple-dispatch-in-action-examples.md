---
course: "getting-started-julia-programming"
chapter: "consolidating-knowledge-advancing-julia"
lesson: "multiple-dispatch-in-action-examples"
sourceId: 6827
sourceUrl: "https://apxml.com/zh/courses/getting-started-julia-programming/chapter-9-consolidating-knowledge-advancing-julia/multiple-dispatch-in-action-examples"
title: "多重派发的实际运用：简单示例"
description: "回顾多重派发，通过简单、演示性的例子，更好地理解其实际应用。"
order: 2
plots: []
sourceHash: "d5c6ffe9f16dd9b38535a35ab0c2d4c60a0eb451271b6f44dca5cd1b3f1e9e62"
sourceCorrections: []
---

多重派发是 Julia 的一个标志性特性。通过直接的例子可以观察到它的实际益处。多重派发使 Julia 能够根据提供给函数的所有参数 (parameter)（而不仅仅是第一个）的类型来选择执行函数的特定方法。这是一个强大的机制，有助于 Julia 的灵活性和表达力。

### 通过一个 `greet` 函数来理解

假设我们想要一个提供问候的函数。一个通用的问候语可能很简单：

```julia
function greet(name)
    return "Hello, $(name)!"
end

println(greet("User"))
# 输出: Hello, User!
```

这运行良好。但是如果我们知道对方的头衔，想提供一个更具体的问候语怎么办？或者对于一个数字标识符，想使用不同的问候语呢？借助多重派发，我们可以为 `greet` 函数定义额外的方法。

让我们定义一个简单的 `Person` 类型。在 Julia 中，我们使用 `struct` 来定义新的复合类型：

```julia
struct Person
    name::String
    title::String
end

# 为 Person 定义的更具体的 greet 方法
function greet(person::Person)
    return "Greetings, $(person.title) $(person.name)!"
end

# 另一个用于数字 ID 的方法
function greet(id::Int)
    return "Hello, ID #$(id)!"
end

# 让我们看看它的实际运用
user_name = "Julia Programmer"
ada = Person("Ada Lovelace", "Countess")
user_id = 101

println(greet(user_name))
println(greet(ada))
println(greet(user_id))
```

当你执行这段代码时，你会看到：

```
Hello, Julia Programmer!
Greetings, Countess Ada Lovelace!
Hello, ID #101!
```

当你调用 `greet("Julia Programmer")` 时，Julia 识别出参数 (parameter)是一个 `String`。它找到最初的 `greet(name)` 方法。（如果像 `name` 这样的参数没有指定类型，Julia 会将其视为 `Any`，这是所有类型的超类型。`String` 更具体，因此如果有多个方法适用，Julia 会选择最具体的一个）。
当你调用 `greet(ada)` 时，Julia 识别出 `ada` 的类型是 `Person`。它然后选择 `greet(person::Person)` 方法，因为其签名（函数名及其参数的类型）与参数类型完全匹配。
类似地，`greet(101)` 匹配 `greet(id::Int)`。

每一个具有不同参数类型的 `greet` 定义都是通用 `greet` 函数的一个不同的*方法*。Julia 会自动将调用*派发*给适用于给定参数类型的最具体方法。

> 下面的图表展示了 Julia 的派发机制如何根据函数调用中提供的所有参数的类型来选择合适的函数方法。

> Julia 的派发系统根据参数的类型选择最具体的 `greet` 方法。对于 `String` 参数，选择通用的 `greet(name)` 方法。对于 `Person` 对象，选择 `greet(person::Person)`，而对于 `Int`，则使用 `greet(id::Int)`。

### 扩展不同数据组合的行为

考虑一个旨在将两件事物组合在一起的函数 `combine`。“组合”的具体含义会根据这两种事物的不同而有很大差异。

1. **组合数字：** 对于数字，`combine` 可能表示加法。

   ```julia
   function combine(x::Number, y::Number)
       return x + y
   end

   println(combine(5, 10))
   println(combine(3.14, 2.5))
   # 第一次调用的输出: 15
   # 第二次调用的输出: 5.64
   ```

   在这里，`Number` 是一个抽象类型，因此此方法适用于 `Int`、`Float64` 和其他数值类型。
2. **组合字符串：** 对于字符串，`combine` 可能应该表示拼接。

   ```julia
   function combine(s1::String, s2::String)
       return s1 * s2  # 在 Julia 中，* 运算符用于字符串拼接
   end

   println(combine("Julia ", "rocks!"))
   # 输出: Julia rocks!
   ```
3. **组合字符串和整数：** 如果我们想将一个字符串重复一定次数怎么办？

   ```julia
   function combine(s::String, n::Int)
       return s ^ n # ^ 运算符被重载用于字符串重复
   end

   println(combine("Ha", 3))
   # 输出: HaHaHa
   ```

   请注意，这里方法定义 `combine(s::String, n::Int)` 中的顺序很重要。如果我们调用 `combine(3, "Ha")`，它不会匹配此方法。如果想让该特定顺序能够按其自身逻辑运行，我们需要定义另一个方法 `combine(n::Int, s::String)`，或者在存在更通用的备用方法时依赖它。

让我们尝试调用 `combine`，传入我们没有明确定义方法的类型，例如两个数组：

```julia
# println(combine([1, 2], [3, 4])) # 这会引发错误
```

如果你取消注释并运行上面的行，Julia 会提示你类似 `MethodError: no method matching combine(::Vector{Int64}, ::Vector{Int64})` 的信息。这是因为我们还没有告诉 Julia 如何 `组合` 两个数组。

我们可以轻松扩展 `combine` 来处理数组，也许通过拼接它们：

```julia
# 对于向量（一维数组），我们可以通过展开元素到新向量中进行拼接
function combine(arr1::AbstractVector, arr2::AbstractVector)
    return [arr1..., arr2...] # '...' 是展开运算符
end

println(combine([1, 2], [3, 4]))
println(combine(["a", "b"], ["c"]))
# 第一次调用的输出: [1, 2, 3, 4]
# 第二次调用的输出: ["a", "b", "c"]
```

在这个数组示例中，`AbstractVector` 是一个抽象类型。`Vector{Int64}`（一个64位整数向量 (vector)）和 `Vector{String}`（一个字符串向量）都是 `AbstractVector` 的具体子类型。通过使用 `AbstractVector` 定义我们的方法，它适用于整数、字符串或任何其他元素类型的向量。

### 重要性

这些例子虽然简单，但展示了多重派发的几个优点：

- **代码清晰和自然命名：** 你可以使用一个单一、直观的函数名（如 `greet` 或 `combine`）来表示语义相似但适用于不同数据类型的操作。代码读起来更自然。
- **可扩展性：** 你或其他包的作者可以为现有函数添加新方法，以支持新的自定义类型。如果你创建一个新的 `MySpecialData` 类型，你可以定义 `combine(x::MySpecialData, y::MySpecialData)`，而无需修改原始的 `combine` 函数或使用它们的任何代码。这是 Julia 中一个常见模式：为你自己的新类型扩展现有函数。
- **减少复杂命名方案的需求：** 你不需要发明诸如 `add_numbers`、`concatenate_strings`、`repeat_string_n_times` 或 `combine_arrays_vertically` 之类的名称。参数 (parameter)的类型会自然选择正确的行为。
- **性能：** 尽管从这些简单的例子中并非立即显而易见，但 Julia 的编译器非常擅长使用类型信息来确定要调用哪个方法。通常，这可以在编译时解决，这意味着派发过程本身几乎不引入运行时开销，从而使 Julia 代码运行极快。

多重派发是 Julia 的一个基本特点，它使得编写通用但高度专业且高效的代码成为可能。当你查看更复杂的 Julia 包时，例如用于处理表格数据的 `DataFrames.jl` 或用于可视化的 `Plots.jl`（我们接下来会提到），你会随处可见其作用。它使这些包能够处理各种各样的数据类型和结构，常常使它们能够与原始作者可能都不知道的类型一起工作。

这种固有的可扩展性和可组合性是 Julia 生态系统如此高效的原因。你不仅仅是使用函数；你常常是扩展它们以适应你的特定需求和数据类型，从而为科学计算创建一个丰富且相互关联的环境。

## 参考资料

- [Methods](https://docs.julialang.org/en/v1/manual/methods/) — The Julia Language Developers (2024)
  Julia 方法的官方文档，直接解释了多重分派及其工作原理。
- [Think Julia: How to Think Like a Computer Scientist](https://www.oreilly.com/library/view/think-julia/9781492045038/) — Ben Lauwens, Allen B. Downey (2019)
  Publisher: O'Reilly Media
  一本入门教材，将多重分派作为 Julia 编程的基础内容进行讲解。
- [Julia for Data Science](https://www.manning.com/books/julia-for-data-science) — İlker Arslan (2023)
  Publisher: Manning Publications
  一本实用指南，展示了多重分派如何增强 Julia 中的数据科学工作流程。
- [Julia: A Fresh Approach to Numerical Computing](https://arxiv.org/abs/1209.5145) — Jeff Bezanson, Stefan Karpinski, Viral B. Shah, Alan Edelman (2012)
  Journal: arXiv preprint arXiv:1209.5145; DOI: [10.48550/arXiv.1209.5145](https://doi.org/10.48550/arXiv.1209.5145)
  介绍 Julia 语言的原始学术论文，详细说明了多重分派作为核心设计原则。
