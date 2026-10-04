---
course: "getting-started-julia-programming"
chapter: "handling-errors-basic-debugging"
lesson: "defining-throwing-custom-errors"
sourceId: 6821
sourceUrl: "https://apxml.com/zh/courses/getting-started-julia-programming/chapter-8-handling-errors-basic-debugging/defining-throwing-custom-errors"
title: "定义和抛出自定义错误"
description: "学习如何定义自己的自定义错误类型，并使用 `throw` 函数来表示特定的错误情况。"
order: 4
plots: []
sourceHash: "f9d3e2ef3f5c3660cbdba316654cad916012d4000da95c7f63710cf1b964d570"
sourceCorrections: []
---

虽然 Julia 提供了全面的内置错误类型以应对许多常见问题，但通常会出现需要更明确错误提示的情况，这可以大幅提高程序的清晰度和可靠性。定义自己的自定义错误类型，能让你精确地传达问题所在，从而可以更有针对性地处理这些异常情况。这种方法有助于创建更易于他人（以及未来的你自己）正确使用和调试的函数。

### 为什么要创建自定义错误？

想象一下你正在编写一个处理用户输入的函数。一个通用的 `ArgumentError` 可能只会告诉你某个参数 (parameter)有问题，但一个自定义的 `InvalidEmailFormatError` 或 `PasswordTooShortError` 会立刻表明问题的确切性质。这种明确性在多方面都很有用：

1. **更清晰的语义：** 自定义错误使你代码的意图更明确。它们作为一种文档形式，描述了特定的失败模式。
2. **有针对性的处理：** 使用 `try-catch` 代码块，你可以明确捕获自定义错误类型，并实现针对该特定错误的逻辑，而不是试图从通用错误消息中推断问题。
3. **API 设计：** 如果你正在编写库或模块，自定义错误会构成你 API 的一部分，让你的代码用户能够预料并妥善处理特定问题。

### 定义自定义错误类型

在 Julia 中，所有错误都是内置 `Exception` 类型的子类型。要创建你自己的错误，你需要定义一个继承自 `Exception` 的 `struct`。

```julia
struct MyCustomError <: Exception
    message::String  # 用于保存描述性消息的字段
end
```

在这个定义中：

- `struct MyCustomError` 声明了一个名为 `MyCustomError` 的新复合类型。
- `<: Exception` 表明 `MyCustomError` 是 `Exception` 的子类型。这使得它成为一个可以被 `throw` 和 `catch` 的有效错误类型。
- `message::String` 是我们自定义错误中的一个字段。你可以添加任何需要的字段来携带有关错误的相关信息，例如导致错误的值、特定的错误代码或上下文 (context)。

例如，如果你正在验证数据，你可能希望存储引起问题的值：

```julia
struct DataValidationError <: Exception
    details::String      # 错误的通用描述
    invalid_value::Any   # 验证失败的值
end
```

这里，`invalid_value` 使用 `Any` 类型，表示它可以是任何类型。

### 抛出自定义错误

定义好自定义错误类型后，你可以通过创建该类型的实例并使用 `throw` 函数来表示错误情况。当 `throw` 被调用时，程序的正常执行会停止，Julia 会开始寻找一个合适的 `catch` 块来处理抛出的异常。

```julia
function process_data(data_value::Int)
    if data_value < 0
        throw(DataValidationError("数据值不能为负数。", data_value))
    elseif data_value > 100
        throw(DataValidationError("数据值超出最大限制 100。", data_value))
    end
    println("正在处理数据：", data_value)
    # ... 进一步处理 ...
end
```

在此示例中，如果 `process_data` 以 `-5` 调用，它将实例化一个 `DataValidationError`，消息为“数据值不能为负数。”，值为 `-5`，然后 `throw` 此实例。`println("正在处理数据：", data_value)` 这一行以及该函数调用中的任何后续代码都不会被执行。

### 捕获自定义错误

正如你已经学习过 `try-catch` 块，你可以使用它们来处理这些自定义错误。其强大之处在于能够捕获你特定的错误类型。

```julia
function safe_process(input::Int)
    try
        process_data(input)
        println("数据处理成功，输入：", input)
    catch e::DataValidationError # 特意捕获我们的自定义错误
        println("验证错误：", e.details)
        println("引起问题的值为：", e.invalid_value)
        # 为 DataValidationError 执行特定的恢复或日志记录
    catch e # 捕获任何其他错误
        println("发生了一个意外错误：", e)
        # 通用错误处理，如果无法处理则重新抛出
        # rethrow()
    end
end

safe_process(50)
println("---")
safe_process(-10)
println("---")
safe_process(200)
```

当你运行这段代码时，输出会是这样：

```
正在处理数据：50
数据处理成功，输入：50
---
验证错误：数据值不能为负数。
引起问题的值为：-10
---
验证错误：数据值超出最大限制 100。
引起问题的值为：200
```

注意，`catch e::DataValidationError` 块允许我们访问在 `DataValidationError` 结构体中定义的 `details` 和 `invalid_value` 字段。这种有针对性的方法比捕获一个通用的 `Exception` 并试图解析其消息字符串要有效得多。

### 一个更完整的示例：用户注册

让我们考虑一个稍微复杂些的场景，其中自定义错误可以使逻辑更清晰。假设我们正在验证用户注册详情。

```julia
# 定义自定义错误类型
struct UsernameError <: Exception
    username::String
    message::String
end

struct PasswordError <: Exception
    message::String
end

# 注册用户的函数
function register_user(username::String, password::String)
    if length(username) < 3
        throw(UsernameError(username, "用户名必须至少 3 个字符长。"))
    end
    if !occursin(r"^[a-zA-Z0-9_]+$", username) # 允许字符的正则表达式
        throw(UsernameError(username, "用户名只能包含字母、数字和下划线。"))
    end

    if length(password) < 8
        throw(PasswordError("密码必须至少 8 个字符长。"))
    end
    if !occursin(r"[A-Z]", password) || !occursin(r"[a-z]", password) || !occursin(r"[0-9]", password)
        throw(PasswordError("密码必须包含大写、小写和数字字符。"))
    end

    println("用户 '$username' 注册成功。")
    # 在实际应用中，你将在此处保存用户详情
end

# 尝试注册用户并处理特定错误
function attempt_registration(uname::String, pword::String)
    println("\n尝试注册用户：$uname")
    try
        register_user(uname, pword)
    catch e::UsernameError
        println("用户名问题，针对 '$(e.username)'：$(e.message)")
    catch e::PasswordError
        println("密码问题：$(e.message)")
    catch e
        println("发生了一个意外的注册错误：$e")
    end
end

attempt_registration("Al", "ValidPass123")
attempt_registration("ValidUser", "short")
attempt_registration("Invalid-User", "ValidPass123")
attempt_registration("GoodUser", "NoDigitsHere")
attempt_registration("FinalUser", "SecurePass123!")
```

这个示例说明了不同的自定义错误（`UsernameError`、`PasswordError`）如何从同一个函数（`register_user`）中抛出，并被调用者（`attempt_registration`）明确捕获。这使得可以根据遇到的错误类型采用不同的处理逻辑，从而带来更好的用户体验或更精确的日志记录。

### 使用 `Base.showerror` 自定义错误消息

Julia 提供了一种方式，让你能够自定义错误消息在 REPL 或日志中打印时的显示方式。这通过为 `Base.showerror` 定义一个方法来实现。这是 Julia 多重分派系统的一个例子，函数行为可以根据其参数 (parameter)的类型进行特殊化。

```julia
struct FileProcessingError <: Exception
    filepath::String
    reason::String
    line_number::Union{Int, Nothing} # 可选的行号
end

# 我们错误的自定义显示
function Base.showerror(io::IO, err::FileProcessingError)
    print(io, "FileProcessingError：处理失败，文件为 '", err.filepath, "'.")
    print(io, "\n  原因：", err.reason)
    if err.line_number !== nothing
        print(io, "\n  发生在行附近：", err.line_number)
    end
end

# 抛出此错误的示例
function simulate_file_processing(path::String)
    # 模拟一个错误
    throw(FileProcessingError(path, "检测到无效字符编码。", 42))
}

try
    simulate_file_processing("/path/to/my/data.csv")
catch e
    # 当 'e' 被打印时（例如，通过 REPL 或显式地使用 print(e)），
    # 我们的自定义 Base.showerror 方法将被使用。
    println(e) # 现在这将使用我们的自定义格式。
end
```

如果你在 Julia REPL 中运行这段代码（或一个直接打印错误的脚本），`println(e)` 的输出将根据你 `Base.showerror` 的定义进行格式化：

```
FileProcessingError：处理失败，文件为 '/path/to/my/data.csv'。
  原因：检测到无效字符编码。
  发生在行附近：42
```

虽然对于自定义错误的功能而言并非严格必要，但定义 `Base.showerror` 可以使你的错误提供更多信息并更用户友好，尤其是在日志中或直接向用户显示时。

通过定义和抛出你自己的错误，你能够创建更具表现力和可维护的 Julia 代码。这使得你的程序能够精确地传达失败信息，从而可以实施有效的错误处理策略，最终带来更可靠的应用程序。

## 参考资料

- [Error Handling and Exceptions](https://docs.julialang.org/en/v1/manual/control-flow/#Error-Handling) — The Julia Language Developers (2024)
  Julia 官方文档，涵盖 Julia 的异常类型、`throw` 函数、`try-catch` 块以及用户定义错误的 `Base.showerror` 自定义。
