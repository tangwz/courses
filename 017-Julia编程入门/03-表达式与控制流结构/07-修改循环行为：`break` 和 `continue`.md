# 修改循环行为：`break` 和 `continue`

来源：[原文](https://apxml.com/zh/courses/getting-started-julia-programming/chapter-3-expressions-control-flow-structures/modifying-loop-behavior-break-continue)

[返回章节目录](README.md) · [返回课程目录](../README.md)

`for` 和 `while` 循环非常适合重复代码块，但有时你需要调整它们标准的执行流程。Julia 提供了两个语句，`break` 和 `continue`，它们在循环中提供了这种更精细的控制。这些语句允许你根据循环内部判断的条件，分别提前退出循环或跳过特定的迭代。

### 使用 `break` 提前退出循环

`break` 语句会立即终止遇到它的最内层循环（`for` 或 `while`）。一旦 `break` 执行，程序的控制流会跳到已终止循环后的第一个语句。当你寻找某个东西并在找到后立即停止时，或者当某个条件使得后续迭代不必要或可能出错时，这非常有用。

设想一个情况，你正在遍历一个数字列表，以找到满足特定条件的第一个数字。一旦找到，继续检查剩余的数字将是低效的。

```julia
println("寻找第一个大于5的数字：")
numbers = [1, 3, 7, 2, 8, 4]
found_item = nothing # 使用 'nothing' 表示尚未找到

for num in numbers
    println("正在检查: ", num)
    if num > 5
        found_item = num
        println("找到 ", num, "! 正在退出循环。")
        break # 现在退出循环
    end
    println(num, " 不大于5。")
end

if found_item !== nothing
    println("第一个大于5的数字是: ", found_item)
else
    println("没有找到大于5的数字。")
end
```

在这个例子中，当 `num` 变为 `7` 时，条件 `num > 5` 为真。`found_item` 被更新，打印一条消息，然后 `break` 被执行。循环立即终止，`num` 为 `7` 时的 `println(num, " 不大于5。")` 行被跳过，`num` 为 `2`、`8` 和 `4` 的所有后续迭代也都被跳过。程序随后执行循环后的 `if found_item !== nothing` 检查。

`break` 语句通过减少冗余操作，可以使你的循环更高效。

> 带有 `break` 语句的循环的运行流程。如果 `break` 条件达成，循环终止。

### 使用 `continue` 跳过迭代

`continue` 语句跳过当前循环迭代的剩余部分，并立即进入下一个迭代。与 `break` 不同，`continue` 不会完全终止循环。当你在一个迭代中遇到需要忽略或以不同方式处理的特定情况时，它很有用，而且不影响后续迭代。

假设你只想对列表中所有的正数求和，忽略任何负数或零。

```julia
println("只对正数求和：")
data = [10, -2, 5, 0, -8, 12]
sum_of_positives = 0

for x in data
    print("正在处理: ", x)
    if x <= 0
        println("... 跳过（非正数）。")
        continue # 跳过本次迭代的剩余部分，进入下一个 x
    end
    sum_of_positives += x
    println("... 已添加。当前和: ", sum_of_positives)
end

println("正数的总和: ", sum_of_positives)
```

在这段代码中，当 `x` 是 `-2`、`0` 或 `-8` 时，条件 `x <= 0` 为真。`println` 消息表明跳过，并且 `continue` 被执行。这导致程序立即跳到 `data` 中的下一个值（例如，从 `-2` 跳到 `5`）。`sum_of_positives += x` 行以及随后的 `println` 对于这些非正数都不会被执行。循环继续，直到 `data` 中的所有项都被处理完毕。

`continue` 语句通过允许你绕过某些迭代中逻辑的特定部分，有助于简化循环体。

> 带有 `continue` 语句的执行流程。如果 `continue` 条件达成，当前迭代的剩余部分被跳过，循环进入下一个迭代。

`break` 和 `continue` 都是管理循环执行的有效手段。然而，要谨慎使用它们。过度使用，特别是在复杂的嵌套循环中，有时会降低代码可读性。通常，重构循环条件或使用更多的 `if-elseif-else` 块可以使代码更清晰。尽管如此，当使用得当，`break` 和 `continue` 可以通过直接体现所需的控制流，使你的代码更简洁和高效。

## 参考资料

- [Control Flow](https://docs.julialang.org/en/v1/manual/control-flow/) — The Julia Language (2024)
  Julia官方文档直接而全面地解释了控制流语句，包括`break`和`continue`。
- [Think Julia: How to Think Like a Computer Scientist](https://onlinebooks.library.upenn.edu/webbin/book/lookupid?key=olbp81797) — Ben Lauwens, Allen B. Downey (2019)
  Publisher: O'Reilly Media; Pages: Chapter 7: Iteration
  一本介绍性的书籍，涵盖了Julia中的基本编程概念，其中有专门介绍循环和修改其执行流程的部分。

---

[上一节](06-%E4%BD%BF%E7%94%A8%20%60while%60%20%E5%BE%AA%E7%8E%AF%E8%BF%9B%E8%A1%8C%E6%9D%A1%E4%BB%B6%E8%BF%AD%E4%BB%A3.md) · [下一节](08-%E4%B8%89%E5%85%83%E8%BF%90%E7%AE%97%E7%AC%A6%EF%BC%9A%E7%AE%80%E6%B4%81%E7%9A%84%E6%9D%A1%E4%BB%B6%E8%AF%AD%E5%8F%A5.md)
