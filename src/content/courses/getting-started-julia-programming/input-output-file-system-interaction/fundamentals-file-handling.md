---
course: "getting-started-julia-programming"
chapter: "input-output-file-system-interaction"
lesson: "fundamentals-file-handling"
sourceId: 6813
sourceUrl: "https://apxml.com/zh/courses/getting-started-julia-programming/chapter-7-input-output-file-system-interaction/fundamentals-file-handling"
title: "文件处理基础"
description: "介绍 Julia 中的文件输入/输出操作，包括文件的打开、关闭、读取和写入。"
order: 4
plots: []
sourceHash: "ab6299c3b037433e9d7e38f78b52b279ce41e38342df4416dbc8ef8d76c1490d"
sourceCorrections: []
---

许多应用程序需要处理更持久存储的数据，这超出了仅通过控制台显示消息和获取即时输入的范围。文件处理在此发挥作用。文件使您的 Julia 程序能够在运行之间保存信息、读取配置、处理大型数据集以及执行更多任务。与文件交互是大多数涉及持久化数据的编程任务的主要组成部分。

## 理解文件

从程序角度看，文件本质上是存储在磁盘或其他存储介质上的一系列字节。这些字节可以表示任何内容：文本、图像、数值数据或可执行代码。对于我们最初的讨论，我们将主要关注文本文件，它们存储人类可读字符。

当您的 Julia 程序与文件交互时，通常通过一个中间对象进行，该对象常被称为 *文件流* 或 *I/O 流*。此流提供了一种统一的方式，可以从文件读取数据或向文件写入数据，而不论底层存储细节如何。

## 文件交互的基本生命周期

在任何编程语言（包括 Julia）中处理文件，通常遵循三步流程：

1. **打开文件：** 您告知操作系统您打算访问特定文件以及您计划如何操作它（例如，从中读取或向其写入）。
2. **执行操作：** 文件打开后，您可以读取其内容、向其写入新数据或修改现有数据，具体取决于您打开文件的方式。
3. **关闭文件：** 完成操作后，您必须告知操作系统您已完成文件操作。这一步很重要，因为它确保任何待处理的数据被写入磁盘，并腾出与已打开文件相关的系统资源。

下图展示了这一基本生命周期：

> 程序中文件交互的基本生命周期。

## 使用 `open()` 函数打开文件

在 Julia 中，您使用 `open()` 函数打开文件。它最基本的形式接受两个参数 (parameter)：文件名（一个字符串）以及一个模式字符串，用于指定您希望如何与文件交互。

```julia
# 语法: open(filename::AbstractString, mode::AbstractString)
# 返回一个 IOStream 对象
```

`filename` 可以是简单名称（例如，`"data.txt"`），Julia 将在当前工作目录中查找，或者文件的完整路径（例如，`"/path/to/your/data.txt"` 或 `"C:\\Users\\YourName\\Documents\\data.txt"`）。

`mode` 字符串决定了您被允许对文件执行的操作。以下是最常见的模式：

- `"r"`: **读取模式。** 打开一个现有文件进行读取。如果文件不存在，`open()` 将引发错误。如果您不指定模式，这是默认模式。
- `"w"`: **写入模式。** 打开文件进行写入。如果文件存在，其现有内容将被覆盖（删除）。如果文件不存在，将创建一个新的空文件。
- `"a"`: **追加模式。** 打开文件进行写入，但新数据会添加到文件末尾。如果文件不存在，将创建一个新的空文件。
- `"r+"`: **读写模式。** 打开一个现有文件进行读写。如果文件不存在，会发生错误。
- `"w+"`: **读写模式（截断）。** 打开文件进行读写。如果文件存在，其内容将被覆盖。如果不存在，将创建一个新文件。
- `"a+"`: **读追加模式。** 打开文件进行读取和追加。新数据写入文件末尾。如果文件不存在，将创建一个新文件。

例如，要打开一个名为 `sample.txt` 的文件进行读取：

```julia
file_stream = open("sample.txt", "r")
# 现在您可以使用 file_stream 从 sample.txt 读取数据
# ... (操作) ...
close(file_stream) # 不要忘记关闭！
```

要创建一个名为 `output.txt` 的新文件进行写入（如果存在则覆盖）：

```julia
output_stream = open("output.txt", "w")
# 现在您可以使用 output_stream 写入 output.txt
# ... (操作) ...
close(output_stream)
```

如果成功，`open()` 函数会返回一个 `IOStream` 对象。该对象是您对文件的句柄，您将使用它与其他函数一起读取或写入数据。

## 使用 `close()` 函数关闭文件

完成文件操作后，关闭文件非常重要。关闭文件主要有两件事：

1. **刷新缓冲区：** 当您写入数据时，它可能暂时存储在内存缓冲区中。关闭文件可确保所有缓冲数据实际写入磁盘。
2. **腾出资源：** 操作系统限制程序可以同时打开的文件数量。关闭文件可以腾出这些资源。

您使用 `close()` 函数关闭文件，将 `open()` 返回的 `IOStream` 对象传递给它：

```julia
my_file = open("mydata.log", "a") # 以追加模式打开
# ... 执行一些写入操作 ...
println(my_file, "Log entry: Process completed.")
close(my_file) # 关键步骤！
```

忘记关闭文件可能导致数据丢失（如果缓冲区未刷新）或资源耗尽错误，尤其是在处理大量文件的程序中。

### 确保文件关闭：`do` 块模式

由于记住调用 `close()` 非常重要，特别是如果在文件操作期间发生错误，Julia 提供了一种更便捷、更安全的文件管理方式，即使用 `open()` 函数的 `do` 块。

```julia
open("safer_output.txt", "w") do f
    # 'f' 是 IOStream 对象，仅在此块内部可用
    println(f, "此行将被写入 safer_output.txt。")
    println(f, "Julia 会自动处理文件关闭。")
    # 无需显式调用 close(f)
end
# 此时，即使在 'do' 块内部发生错误，safer_output.txt 也保证已关闭。
```

当您将 `open()` 与 `do` 块一起使用时，文件在块执行完毕时会自动关闭，无论是正常结束还是因错误结束。`IOStream` 对象（在上述示例中名为 `f`）作为参数 (parameter)传递给 `do...end` 块定义的匿名函数。这是在 Julia 中处理文件的推荐方式，因为它有助于防止资源泄漏并使您的代码更整洁。

例如，要使用这种模式从文件读取：

```julia
open("input.txt", "r") do file_handle
    content = read(file_handle, String) # 我们将在下一节看到 read()
    println("文件内容: ", content)
end
# input.txt 现已关闭。
```

理解这些基本原理，以正确模式打开文件并确保它们被妥善关闭，构成了所有后续文件操作（如读取内容或写入数据）的依据，我们将在后续章节中介绍。

## 参考资料

- [I/O and Network](https://docs.julialang.org/en/v1/base/io-network/) — The Julia Language (2024)
  Publisher: The Julia Language
  官方文档，提供Julia内置I/O函数和推荐实践的详细信息，包括使用`open`函数和`do`块进行文件处理。
- [Think Julia: How to Think Like a Computer Scientist](https://www.oreilly.com/library/view/think-julia/9781492045038/) — Ben Lauwens, Allen Downey (2019)
  Publisher: O'Reilly Media; Pages: 295
  一本关于Julia编程的入门书籍，包含文件I/O操作的实用示例和清晰解释，适合初学者学习该语言。
