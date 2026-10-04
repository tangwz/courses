---
course: "getting-started-julia-programming"
chapter: "expressions-control-flow-structures"
lesson: "conditional-iteration-while-loops"
sourceId: 6781
sourceUrl: "https://apxml.com/zh/courses/getting-started-julia-programming/chapter-3-expressions-control-flow-structures/conditional-iteration-while-loops"
title: "使用 `while` 循环进行条件迭代"
description: "了解如何在 Julia 中使用 `while` 循环，以便在条件为真时重复执行代码块。"
order: 6
plots: []
sourceHash: "2fff2895f40a1c85d8d70f9266b851722c21399b7edd340c6a3b93a3fe221d73"
sourceCorrections: []
---

尽管 `for` 循环非常适合已知次数的迭代或遍历集合元素，但在许多情况下，你需要一个循环在某个条件为真时持续运行，而无需预先知道会进行多少次迭代。这就是 `while` 循环的用途。只要指定的布尔表达式结果为 `true`，`while` 循环就会重复执行一段代码。

Julia 中 `while` 循环的基本结构是：

```julia
while condition
    # 当条件为真时执行的代码
end
```

### `while` 循环如何工作

1. **条件检查**: 在循环体每次可能的执行之前，Julia 会评估 `condition`。这个 `condition` 必须是一个结果为布尔值（`true` 或 `false`）的表达式。
2. **执行**：
   - 如果 `condition` 为 `true`，则执行 `while...end` 语句内部的代码块。
   - 代码块执行完毕后，控制返回到步骤 1，并再次检查 `condition`。
3. **终止**：
   - 如果 `condition` 结果为 `false`（无论是初始状态还是经过若干次迭代后），循环将终止，程序继续执行 `end` 关键字后面的语句。

重要的是循环体内部的代码最终要使 `condition` 变为 `false`。否则，循环将永远运行，这种情况称为无限循环。

### `while` 循环的一个简单示例

让我们看一个常见的任务：数到某个数字。虽然 `for` 循环可能更适合简单的计数，但 `while` 循环也能实现这个目的，并有助于说明其工作原理。

假设我们想打印从 1 到 3 的数字：

```julia
let
    count = 1
    while count <= 3
        println("当前计数: ", count)
        count = count + 1 # 或者 count += 1
    end
    println("循环结束！")
end
```

输出：

```
Current count: 1
Current count: 2
Current count: 3
Loop finished!
```

让我们来分析一下：

- `count = 1`: 我们将变量 `count` 初始化为 1。这个变量将用于我们的条件并在循环内更新。
- `while count <= 3`: 这是条件。
  - 第一次迭代：`count` 是 1。`1 <= 3` 为 `true`，所以循环体执行。
  - 第二次迭代：`count` 是 2。`2 <= 3` 为 `true`，所以循环体执行。
  - 第三次迭代：`count` 是 3。`3 <= 3` 为 `true`，所以循环体执行。
  - 第四次迭代：`count` 是 4。`4 <= 3` 为 `false`，所以循环终止。
- `println("Current count: ", count)`: 这一行打印 `count` 的当前值。
- `count = count + 1`: 这是非常重要的一行。它使 `count` 增加。没有这一行，`count` 将始终保持为 1，`count <= 3` 将始终为 `true`，循环将永不结束。
- `println("Loop finished!")`: 这一行在循环终止后执行。

下图展示了 `while` 循环的一般流程：

> `while` 循环的逻辑流程。只要指定条件结果为真，循环就会持续执行其主体。

### 无限循环的危险

当 `while` 循环的条件从不变为 `false` 时，就会发生无限循环。这可能在你忘记更新条件中涉及的变量时发生，或者更新逻辑从不允许条件变为假。

例如，如果我们在前一个例子中忘记了 `count = count + 1` 这一行：

```julia
# 警告：这是一个无限循环！
# let
#     count = 1
#     while count <= 3
#         println("这将永远打印：", count)
#         # 缺少 count = count + 1
#     end
# end
```

这个循环将持续打印“This will print forever: 1”，因为 `count` 始终保持为 1，并且 `1 <= 3` 始终为 `true`。

如果你不小心在 Julia REPL 或脚本中运行了一个无限循环，通常可以通过按 `Ctrl+C` 来停止它。这会向 Julia 发送一个中断信号，请求它停止当前操作。了解如何停止程序是学习循环时一项有用的技能！

### 实际用例：输入验证

`while` 循环的一个常见情况是当你需要从用户那里获取有效输入时。你不知道用户在提供可接受的数据之前会输入多少次无效数据。

假设我们需要用户输入一个正数：

```julia
function get_positive_number()
    num_str = ""
    num = -1 # 初始化为一个无效数字

    while num <= 0
        print("请输入一个正数: ")
        num_str = readline()
        try
            num = parse(Float64, num_str)
            if num <= 0
                println("不是正数。请再试一次。")
            end
        catch e
            println("无效输入。请输入一个数字。")
            num = -1 # 重置以确保循环继续
        end
    end
    println("你输入了: ", num)
    return num
end

# 示例调用（取消注释以在脚本或 REPL 中运行）
# user_number = get_positive_number()
```

在这个例子中：

- 循环在 `num <= 0` 时持续运行。
- 在循环内部，我们提示用户，读取他们的输入，并尝试将其解析为数字。
- 如果输入不是数字或不是正数，则会打印一条提示信息，并且 `num` 保持 `num <= 0` 为 `true` 的状态，导致循环重复。
- 只有当输入一个正数时，`num` 才会大于 0，使 `num <= 0` 变为 `false`，循环终止。

### 结合 `break` 的 `while true` 循环

有时，退出循环的条件更自然地在循环体中间或末尾检查，而不是在最开始。在这种情况下，一种常见模式是使用 `while true`，然后结合 `if` 语句和 `break` 关键字来退出循环。正如你在“修改循环行为：break 和 continue”一节中了解到的，`break` 会立即终止其所在的最近的循环。

示例：模拟在特定事件发生时停止的过程。

```julia
let
    attempts = 0
    while true # 无限循环直到中断
        attempts += 1
        println("尝试次数: ", attempts)
        
        # 模拟某个条件或事件
        if attempts >= 5
            println("达到目标或最大尝试次数。正在退出。")
            break # 退出 while 循环
        end
        
        # 模拟一些工作
        # sleep(0.5) # 暂停 0.5 秒，需要引入 Dates 模块
        println("...处理中...")
    end
    println("循环在 ", attempts, " 次尝试后结束。")
end
```

这个循环被设置为永远运行（`while true`），但内部的 `if attempts >= 5` 条件使用 `break` 提供了一个退出点。如果主循环逻辑复杂且退出条件依赖于循环体内部计算的中间结果，这种方式可以使代码更清晰。

### `for` 和 `while` 之间的选择

`for` 循环和 `while` 循环之间的选择通常取决于你是否提前知道迭代次数：

- **使用 `for` 循环**：

  - 当你需要遍历序列时（例如范围 `1:5`、数组或字符串中的字符）。
  - 当迭代次数在循环开始前是固定的或确定的。
- **使用 `while` 循环**：

  - 当迭代次数预先不知道且取决于执行期间变化的条件时。
  - 对于从文件中读取直到文件末尾、等待满足特定条件的用户输入，或者实现重复直到达到某种状态的算法等任务。

了解 `while` 循环为你的 Julia 编程工具集增加了一个强大的工具，允许你根据动态条件控制程序流程。这对于编写能有效响应不断变化的状态或输入的程序来说非常重要。

## 参考资料

- [Control Flow](https://docs.julialang.org/en/v1/manual/control-flow/#while-loops) — The Julia Language (2024)
  提供Julia编程语言中 `while` 循环的权威规范和使用方法。
- [Computer Science: An Overview](https://www.pearson.com/us/higher-education/program/Brookshear-Computer-Science-An-Overview-13th-edition/PGM296309.html) — J. Glenn Brookshear, Dennis Brylow (2021)
  Publisher: Pearson
  解释了各种编程范式中迭代控制结构（包括 `while` 循环）的基本原理。
- [Julia Programming for Engineers](https://link.springer.com/book/10.1007/978-1-4842-9921-2) — Simon Labrecque and Ben Lauwens (2024)
  Publisher: Apress; DOI: [10.1007/978-1-4842-9921-2](https://doi.org/10.1007/978-1-4842-9921-2)
  提供在Julia编程环境中有效实现循环的实用示例和常见模式。
- [Introduction to Computer Science and Programming in Python: Lecture 3 - Strings and Loops](https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/resources/lecture-3-strings-and-loops/) — Ana Bell, Eric Grimson, and John Guttag (2016)
  Publisher: MIT OpenCourseWare
  提供来自知名计算机科学入门课程的结构化教学内容，涵盖迭代控制流，包括 `while` 循环。
