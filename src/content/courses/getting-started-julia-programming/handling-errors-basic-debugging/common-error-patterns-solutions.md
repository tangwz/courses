---
course: "getting-started-julia-programming"
chapter: "handling-errors-basic-debugging"
lesson: "common-error-patterns-solutions"
sourceId: 6822
sourceUrl: "https://apxml.com/zh/courses/getting-started-julia-programming/chapter-8-handling-errors-basic-debugging/common-error-patterns-solutions"
title: "常见错误情况和处理方法"
description: "探讨 Julia 编程中常见错误及其处理方法。"
order: 5
plots: []
sourceHash: "3a01dd7591db8c61c884514b285a099327b7d9d92b6a19e4ac427e9667c6774e"
sourceCorrections: []
---

识别反复出现的错误情况是你在编写更多 Julia 代码时会培养的一项必备能力。虽然遇到错误可能令人沮丧，但 Julia 的错误信息旨在帮助你定位问题。常见的错误类型以及应对方法将得到阐述。

### 理解 Julia 的错误信息

当 Julia 在运行时遇到错误时，它通常会停止执行并向控制台打印错误信息。这条信息是你第一个也是最重要的线索。它通常包含：

1. **错误类型**：例如，`UndefVarError`、`MethodError` 或 `BoundsError`。了解类型可以让你立即了解出了什么问题。
2. **描述性信息**：这通常说明了具体问题，例如 "变量 `my_var` 未定义" 或 "没有匹配 `+(::String, ::Int64)` 的方法"。
3. **堆栈跟踪**：这显示了导致错误的函数调用序列，最新的调用在顶部。它可以帮助你追溯错误在代码中的起源位置。通常会提供行号，指向有问题的代码。

学习阅读和理解这些信息是调试中的一项基础能力。不要被它们吓倒；它们旨在指导你。

### 常见运行时错误类型

虽然语法错误通常在程序运行前就会被 Julia 捕获（例如忘记 `end` 关键字），但运行时错误会在执行期间出现。以下是一些常见情况：

#### UndefVarError：变量未定义的状况

当你尝试使用 Julia 在当前作用域中无法识别的变量时，就会出现 `UndefVarError`。

- **常见原因**：

  - 变量名中简单的拼写错误：`my_variable` 对比 `my_varable`。
  - 在变量被赋值之前尝试使用它。
  - 在变量作用域之外访问它（例如，尝试在函数外部使用函数内部定义的变量，或者反之，而没有适当的传递或全局声明）。
- **示例**：

  ```julia
  function calculate_area(radius)
      pi_val = 3.14159 # pi_val 在此函数中是局部变量
      area = pi_val * radius^2
      println("面积是: ", area)
  end

  calculate_area(5)
  # println(pi_val) # 这将导致 UndefVarError：pi_val 未定义
  ```
- **处理方法**：

  - 仔细检查变量拼写。
  - 确保变量在首次使用前已被赋值。
  - 审查变量作用域。如果变量在函数内部定义，它通常只在该函数内部可访问，除非显式返回或声明为全局（尽管全局变量应谨慎使用）。有关变量作用域的详细信息，请参阅第 5 章。

下图说明了如果变量在不可见的地方被访问，变量作用域如何导致 `UndefVarError`。

> 函数内定义的变量（局部作用域）无法从全局作用域访问，反之亦然，除非使用 `global` 关键字或函数参数 (parameter)/返回值等特定机制。

#### MethodError：函数参数类型不匹配

`MethodError` 是 Julia 中最常见的错误之一，特别是对于初学者。它之所以出现，是因为 Julia 的多重派发系统：你尝试调用一个函数，其参数类型组合没有定义该函数的特定版本（方法）。

- **常见原因**：

  - 向函数传递了意料之外的参数类型（例如，预期是数字的地方传递了字符串）。
  - 误解了特定函数旨在处理的类型。
- **示例**：

  ```julia
  # 期望两个整数的函数
  function add_integers(x::Int, y::Int)
      return x + y
  end

  add_integers(3, 4)       # 输出: 7 (正确)
  # add_integers(3.0, 4)   # MethodError: 没有匹配 add_integers(::Float64, ::Int64) 的方法
  # add_integers("hello", 4) # MethodError: 没有匹配 add_integers(::String, ::Int64) 的方法
  ```

  Julia 的 `MethodError` 错误信息通常非常有帮助，有时会列出该函数名可用的方法，以便你查看支持哪些类型。
- **处理方法**：

  - 仔细检查你传递给函数的参数类型。如果你不确定，可以使用 `typeof()`。
  - 查阅函数文档（如果是你的函数，则查阅你的文档字符串），以了解其预期的参数类型。
  - 你可以使用 `methods(function_name)` 列出函数的所有可用方法，例如 `methods(+)`。
  - 有时，你可能需要将参数转换为预期类型（例如 `parse(Int, "123")` 或 `Int(3.0)`）。

#### TypeError：类型不匹配

当对不适合类型的值执行操作，或当类型断言失败时，就会出现 `TypeError`。它与 `MethodError` 密切相关，但有时更多地是关于给定上下文 (context)中类型本身的基本特性，而不仅仅是缺少函数方法。

- **常见原因**：

  - 将一种类型的值赋值给显式标注为不兼容类型的变量。
  - 将运算符与不支持它们的类型一起使用（尽管这通常会导致 `+` 等运算符出现 `MethodError`）。
- **示例**：

  ```julia
  x::Int = 10     # 正常
  # x = "hello"   # TypeError: 在类型断言中，预期 Int64，得到 String

  function process_number(n::Number)
      println(n * 2)
  end
  process_number(5)    # 正常
  # process_number("text") # 这很可能会首先导致 MethodError，但如果 "text" 在某种情况下通过了类型检查
                         # 并且直接尝试了像 `*` 这样的操作，就可能是一个 TypeError。
  ```
- **处理方法**：

  - 确保类型一致性。如果你使用类型注解 (`::Type`)，请确保赋值符合要求。
  - 注意操作中涉及的类型。如果你混合使用类型，请确保操作有效（Julia 的类型提升规则通常会处理这种情况，但并非总是如此）。

#### BoundsError：越界访问

当你尝试使用超出其有效范围的索引来访问数组（或其他可索引集合）的元素时，就会发生 `BoundsError`。请记住，Julia 数组默认从 1 开始索引。

- **常见原因**：

  - 使用 0 作为索引。
  - 使用等于或大于 `length(array) + 1` 的索引。
  - 使用负索引（除非数组类型特别支持，标准 `Array` 不支持）。
  - 循环条件中的差一错误。
- **示例**：

  ```julia
  my_friends = ["Alice", "Bob", "Charlie"]
  # println(my_friends[0])      # BoundsError: 尝试访问索引 [0] 处的 3 元素 Vector{String}
  # println(my_friends[4])      # BoundsError: 尝试访问索引 [4] 处的 3 元素 Vector{String}
  println(my_friends[1])      # 输出: Alice (正确)
  println(my_friends[length(my_friends)]) # 输出: Charlie (正确)
  ```
- **处理方法**：

  - 始终确保你的索引在 `1` 到 `length(collection)` 的范围内。
  - 循环时，使用诸如 `for element in my_friends` 或 `for i in 1:length(my_friends)` 或 `for i in eachindex(my_friends)` 这样的结构。
  - 如果索引是计算得出的，在尝试访问元素之前检查数组长度。

#### ArgumentError：类型有效，值无效

当函数接收到正确类型的参数，但其值本身不适合函数打算执行的操作时，就会抛出 `ArgumentError`。

- **常见原因**：

  - 向期望正数的函数传递负数（例如，求实数平方根的 `sqrt(-1.0)`）。
  - 向要求非空字符串的函数提供空字符串。
  - 在自定义的类除法函数中分母为零，而该函数本身不处理 `DivideError`。
- **示例**：

  ```julia
  # sqrt(-4.0)  # ArgumentError: DomainError with -4.0: `sqrt` 仅在用复数参数调用时返回复数结果。尝试 `sqrt(Complex(x))`。
  # log(-10)    # ArgumentError: DomainError with -10.0: `log` 仅在用复数参数调用时返回复数结果。尝试 `log(Complex(x))`。
  ```

  注意：对于数学函数，`ArgumentError` 通常会被 `DomainError` 封装，`DomainError` 是 `ArgumentError` 的一种具体类型，表明输入值超出了函数的有效范围。
- **处理方法**：

  - 在将输入值传递给对其有特定限制的函数之前，先验证它们。
  - 阅读函数文档，了解参数值的任何限制。
  - 如果你预计某个操作尽管已检查仍可能收到无效参数，请使用 `try-catch` 块来处理。

#### LoadError：代码载入问题

当 Julia 在尝试载入或包含文件（例如通过 `include("myfile.jl")` 或 `using MyPackage`）时遇到问题时，就会发生 `LoadError`。

- **常见原因**：

  - 指定的文件在给定路径处不存在。
  - 正在载入的文件包含语法错误或在其自身执行期间抛出错误。
- **示例**：

  ```julia
  # include("non_existent_script.jl") # LoadError: 无法打开文件 non_existent_script.jl

  # If "buggy_script.jl" contains:  x = y +  (syntax error)
  # include("buggy_script.jl")       # LoadError: 语法错误：不完整：输入过早结束
  ```
- **处理方法**：

  - 验证文件路径是否正确且文件存在。
  - 如果文件存在，打开它并检查其内容是否存在语法错误或其他可能阻止其成功载入的问题。`LoadError` 信息通常包含来自正在载入的文件中的错误。

#### DivideError：除数为零

这个错误直接明了：当你尝试用零除一个数时，就会发生它。

- **常见原因**：

  - 用作除数的变量的值为零。
- **示例**：

  ```julia
  numerator = 10
  denominator = 0
  # result = numerator / denominator # DivideError: 除数为零
  ```
- **处理方法**：

  - 在执行除法之前，检查除数是否为零。
  - 如果除数可能为零，请使用 `try-catch DivideError` 块或 `if` 条件来优雅地处理这种情况，例如赋值默认值或跳过计算。

#### StackOverflowError：过多递归

当函数递归调用次数过多而没有达到停止递归的基本情况时，通常会发生 `StackOverflowError`。每次函数调用都会向调用堆栈添加一个新的“帧”，如果此堆栈增长过大，就会溢出。

- **常见原因**：

  - 递归函数缺少基本情况。
  - 基本情况从未被满足。
  - 非常深（但合法）的递归超出了默认堆栈大小。
- **示例**：

  ```julia
  # function countdown(n)
  #     println(n)
  #     countdown(n - 1) # 没有停止递归的基本情况！
  # end
  # countdown(5) # StackOverflowError
  ```

  一个修正后的版本：

  ```julia
  function countdown_fixed(n)
      if n < 0
          return # 基本情况
      end
      println(n)
      countdown_fixed(n - 1)
  end
  countdown_fixed(5)
  ```
- **处理方法**：

  - 确保每个递归函数都有一个清晰定义的基本情况，并且最终会达到。
  - 首先用小输入测试你的递归函数。
  - 如果递归确实非常深，你可能需要考虑迭代方法，因为它们通常不会以相同方式消耗堆栈空间。

### 逻辑错误：隐匿的问题

逻辑错误通常是最棘手的，因为代码在 Julia 不报告任何错误的情况下运行，但输出或行为并非你所预期。

- **常见原因**：

  - 不正确的公式或算法。
  - 条件语句 (`if`/`else`) 中有缺陷的推理 (inference)。
  - 循环或索引中的差一错误，它们不会导致 `BoundsError` 但会产生错误结果。
  - 误解运算符优先级。
- **示例**：

  ```julia
  # 计算两个数的平均值
  function calculate_average_wrong(a, b)
      return a + b / 2 # 糟糕！除法优先级更高
  end
  println(calculate_average_wrong(10, 20)) # 输出: 20.0 (不正确，应为 15.0)

  function calculate_average_correct(a, b)
      return (a + b) / 2 # 用括号修正
  end
  println(calculate_average_correct(10, 20)) # 输出: 15.0 (正确)
  ```
- **处理方法**：

  - 这些错误需要仔细调试。打印中间值以追踪程序状态（如本章后面“Julia 代码调试策略”中所述）。
  - 将复杂逻辑分解为更小、可测试的部分。
  - 编写单元测试，以验证你的函数对已知输入产生预期输出。
  - 有时，暂时放下并重新思考问题，或者向他人（甚至是橡皮鸭！）解释你的逻辑，都可以帮助你找出缺陷。

### 解决问题的一般策略

除了识别特定错误类型，一些一般策略可以帮助你解决几乎任何错误：

1. **隔离问题**：如果错误发生在一个大型脚本或函数中，尝试创建一个最小可复现示例 (MRE)。注释掉部分代码或单独测试更小的部分，直到你找到导致问题的具体行或代码块。
2. **验证输入**：通常，错误发生是因为正在处理的数据并非你所预期。就在错误发生之前打印或检查变量，以确认它们的值和类型。
3. **阅读文档**：如果错误涉及 Julia 内置函数或包中的函数，请查阅其文档。你可能误用了它或误解了它的要求。
4. **增量开发**：以小增量编写和测试你的代码。在一个大型、未经测试的代码块中查找错误比在几行新代码中查找要容易得多。

学习有效诊断和修复错误与编写代码本身一样，是编程的一部分。通过练习，你将变得更熟练于解读错误信息并迅速解决问题，从而编写出更可靠的 Julia 程序。

## 参考资料

- [Errors and Exceptions · The Julia Language](https://docs.julialang.org/en/stable/manual/control-flow/) — The Julia Language (2024)
  提供 Julia 错误处理机制、常见错误类型以及如何解释错误消息和堆栈跟踪的官方说明。对于理解所讨论的错误模式至关重要。
- [Julia Programming: A Guide for Beginners and Experienced Users](https://link.springer.com/book/10.1007/978-3-030-71173-1) — Bogumił Kamiński and Ole Kröger (2021)
  Publisher: Springer; DOI: [10.1007/978-3-030-71173-1](https://doi.org/10.1007/978-3-030-71173-1)
  一本全面的 Julia 编程书籍，涵盖 Julia 编程的各个方面，包括错误识别和解决用户遇到的典型错误的实用策略。
