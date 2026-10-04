---
course: "getting-started-julia-programming"
chapter: "handling-errors-basic-debugging"
lesson: "ensuring-code-execution-finally"
sourceId: 6820
sourceUrl: "https://apxml.com/zh/courses/getting-started-julia-programming/chapter-8-handling-errors-basic-debugging/ensuring-code-execution-finally"
title: "使用 `finally` 保证代码执行"
description: "了解 `try-catch` 块中 `finally` 子句的用法，用于无论是否发生错误都执行清理代码。"
order: 3
plots: []
sourceHash: "167333c80edab9ca0014d4dc9e0170d809c508eb8d0ca7afc9b3b6adf0782a26"
sourceCorrections: []
---

虽然 `try-catch` 块在错误发生时处理错误非常有用，但有时有些代码无论是否发生错误都*必须*运行。这通常是清理代码，比如关闭文件或释放资源。Julia 提供了 `finally` 子句正是为了这个目的，保证重要的清理操作总能执行。

设想你正在处理一个文件。你打开它，写入一些数据，然后你需要关闭它。如果在写入数据时发生错误，你的程序可能会跳转到 `catch` 块，甚至突然终止。如果没有 `finally` 子句，文件可能会保持打开状态，这可能导致资源泄漏和数据损坏等问题。

### `try-catch-finally` 结构

`finally` 子句与 `try` 块结合使用，并且可选地与 `catch` 块结合使用。基本结构如下所示：

```julia
try
    # 可能引发错误的代码
    # 和/或需要清理的代码
catch err
    # 处理错误的代码（可选）
finally
    # 总是会执行的代码，
    # 无论是否发生或捕获到错误。
end
```

`finally` 块的特点是它保证执行。以下是它在不同情境下的行为方式：

1. **`try` 块中没有错误发生：**

   - `try` 块中的代码成功完成。
   - `catch` 块（如果存在）被跳过。
   - `finally` 块中的代码被执行。
2. **`try` 块中发生错误，并被 `catch` 块捕获：**

   - `try` 块中的执行在错误发生处停止。
   - 相应的 `catch` 块被执行。
   - `catch` 块完成后，`finally` 块中的代码被执行。
3. **`try` 块中发生错误，但没有 `catch` 块，或者没有 `catch` 块与错误匹配：**

   - `try` 块中的执行在错误发生处停止。
   - `finally` 块中的代码被执行。
   - `finally` 块完成后，错误向上层传播，如果未在其他地方处理，可能导致程序崩溃。
4. **`catch` 块内部发生错误：**

   - `catch` 块中的执行在错误发生处停止。
   - `finally` 块中的代码被执行。
   - `finally` 块完成后，来自 `catch` 块的新错误传播开来。

要点是，即使存在未被捕获的错误，或者 `return`、`break` 或 `continue` 语句导致控制流离开 `try` 或 `catch` 块，`finally` 块也会执行。

### 实际例子：保证文件被关闭

让我们回顾文件操作的情景。使用 `finally` 可以保证文件正确关闭：

```julia
function process_file(filename::String, data::String)
    io = nothing # 初始化 io，以确保它在 finally 块中可见
    try
        println("尝试打开文件: $filename")
        io = open(filename, "w") # 以写入模式打开文件
        println("文件已打开。正在写入数据...")
        write(io, data)
        # 让我们模拟一个错误以便演示
        # throw(ErrorException("写入时出现问题！"))
        println("数据写入成功。")
    catch err
        println("发生了一个错误: $err")
        # 你可以在此处记录错误或采取其他行动
    finally
        println("进入 finally 块。")
        if io !== nothing && isopen(io)
            println("正在关闭文件。")
            close(io)
        else
            println("文件未打开或已关闭。")
        end
    end
    println("文件处理函数完成。")
end

# 情景 1: 没有错误
process_file("my_data.txt", "Hello, Julia learners!")

println("\n--- 现在模拟一个错误 ---")

# 情景 2: 有错误
function process_file_with_error(filename::String, data::String)
    io = nothing
    try
        println("尝试打开文件: $filename")
        io = open(filename, "w")
        println("文件已打开。正在写入数据...")
        write(io, data)
        println("模拟处理过程中发生错误...")
        throw(ErrorException("模拟磁盘满错误！")) # 模拟一个错误
        # 下面这行代码不会被执行
        # println("Data written successfully.")
    catch err
        println("发生了一个错误: $err")
    finally
        println("进入 finally 块。")
        if io !== nothing && isopen(io)
            println("正在关闭文件。")
            close(io)
        else
            println("文件未打开或已关闭。")
        end
    end
    println("文件处理函数（带错误路径）完成。")
end

process_file_with_error("error_example.txt", "这可能不会完全写入。")
```

如果你运行这段代码：

- 在第一次调用 `process_file` 时，`try` 块完成，`catch` 块被跳过，然后 `finally` 块运行以关闭文件。
- 在第二次调用 (`process_file_with_error`) 时，一个错误被故意抛出。`try` 块被中断，`catch` 块处理 `ErrorException`，然后，重要的是，`finally` 块仍然执行以关闭文件。

示例输出：

```
尝试打开文件: my_data.txt
文件已打开。正在写入数据...
数据写入成功。
进入 finally 块。
正在关闭文件。
文件处理函数完成。

--- 现在模拟一个错误 ---
尝试打开文件: error_example.txt
文件已打开。正在写入数据...
模拟处理过程中发生错误...
发生了一个错误: ErrorException("模拟磁盘满错误！")
进入 finally 块。
正在关闭文件。
文件处理函数（带错误路径）完成。
```

注意“正在关闭文件。”在两种情景下都被打印出来，这表明 `finally` 块每次都执行了。

### `finally` 的常见用途

`finally` 子句对于以下情况必不可少：

- **资源释放：** 关闭文件（如示例所示）、网络连接、数据库连接，或释放程序获取的任何其他系统资源。
- **释放锁：** 如果你的程序使用锁来管理对资源的并发访问，`finally` 保证锁总能被释放，防止死锁。
- **状态恢复：** 撤销对全局状态或系统设置的临时更改。例如，如果你临时更改了一个配置值，可以在 `finally` 块中将其恢复。
- **日志记录或最终报告：** 执行最终的日志记录操作或报告状态，无论成功与否。

### 不带 `catch` 的 `try-finally`

使用带 `finally` 子句但不带任何 `catch` 子句的 `try` 块也是有效的。这种模式在你想保证清理操作发生，但又不想在代码中的这个特定点处理错误时很有用。错误仍然会传播，但在 `finally` 中的清理代码运行之后。

```julia
function operation_with_cleanup(resource_id::Int)
    println("正在获取资源: $resource_id")
    # 获取资源的代码...
    acquired = true

    try
        println("正在使用资源: $resource_id")
        # 模拟可能失败的工作
        if resource_id == 2
            throw(ErrorException("未能使用资源 $resource_id！"))
        end
        println("完成使用资源: $resource_id")
    finally
        if acquired
            println("正在释放资源: $resource_id (在 finally 中)")
            # 释放资源的代码...
        end
    end
end

try
    operation_with_cleanup(1) # 这将成功
    operation_with_cleanup(2) # 这将抛出错误
rescue =># 只有当 operation_with_cleanup(2) 中发生错误时才会执行到这里
    println("主 try-catch: 其中一个操作中发生了错误。")
end
```

在这个例子中，如果调用 `operation_with_cleanup(2)`，它将抛出一个错误。`operation_with_cleanup` 内部的 `finally` 块将执行以释放资源，然后 `ErrorException` 将向外传播，可能被外部的 `try-catch` 块捕获（如 `Main try-catch` 所示）。

`finally` 子句是编写程序的基本工具。通过保证清理代码的执行，它帮助你有效地管理资源，并避免因未处理的错误导致程序处于不一致状态的常见问题。当你构建更复杂的应用程序时，规范使用 `finally` 对提高可靠性将越来越重要。

## 参考资料

- [Control Flow: Exception Handling](https://docs.julialang.org/en/v1/manual/control-flow/#Exception-Handling) — The Julia Language (2023)
  官方文档，解释Julia的错误处理机制，包括`try-catch-finally`结构及其确保执行的特性。
