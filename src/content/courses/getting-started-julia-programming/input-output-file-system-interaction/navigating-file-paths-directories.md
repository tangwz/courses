---
course: "getting-started-julia-programming"
chapter: "input-output-file-system-interaction"
lesson: "navigating-file-paths-directories"
sourceId: 6816
sourceUrl: "https://apxml.com/zh/courses/getting-started-julia-programming/chapter-7-input-output-file-system-interaction/navigating-file-paths-directories"
title: "文件路径与目录管理"
description: "了解Julia中文件路径的使用、检查存在性以及基本的目录操作。"
order: 7
plots: []
sourceHash: "18d7084d34b1170d6e3a0c64ac5c64f7a2791b986f1e3373c4c9a05ccbfcee7b"
sourceCorrections: []
---

当您的Julia程序读取或写入文件时，它们需要知道文件在计算机上的确切位置。这个位置是通过*文件路径*指定的。掌握如何使用文件路径和目录是管理数据和组织项目的基本功。Julia提供了一套工具，帮助您高效地构建、检查和操作这些路径，无论您的代码运行在Windows、macOS还是Linux上。

### 理解文件路径

文件路径本质上是一个地址，指向您计算机文件系统中的特定文件或目录。可以把它想象成文件的邮政地址。路径主要有两种类型：

1. **绝对路径**：绝对路径从文件系统的根目录开始指定文件或目录的位置。

   - 在类Unix系统（Linux、macOS）上，绝对路径通常以正斜杠`/`开头，例如`/home/user/documents/report.jl`。
   - 在Windows上，绝对路径通常以驱动器盘符后跟冒号和反斜杠开头，例如`C:\Users\user\Documents\report.jl`。
2. **相对路径**：相对路径指定文件或目录相对于当前工作目录的位置。*当前工作目录*是您的Julia会话或脚本当前所在的文件夹。

   - 如果您的当前工作目录是`/home/user/documents/`，并且您想访问其中的`report.jl`，则相对路径就是`report.jl`。
   - 如果`report.jl`位于`documents`内名为`project_alpha`的子目录中，则相对路径将是`project_alpha/report.jl`。

Julia将路径作为字符串处理。虽然Windows使用反斜杠`\`作为路径分隔符，而类Unix系统使用正斜杠`/`，但即使在Windows上，如果您在路径字符串中使用正斜杠，Julia通常也能正常工作，因为它经常在内部执行必要的转换。然而，为了稳妥地构建路径，最好使用Julia的内置函数。

让我们来看看路径如何指向位置：

> `report.jl`的绝对路径可能是`/Users/your_username/Documents/ProjectA/report.jl`。如果当前工作目录是`ProjectA`，则`report.jl`的相对路径就是`report.jl`。

### 当前工作目录

您的Julia会话总是在一个特定目录中运行，这个目录被称为**当前工作目录（CWD）**。当您使用相对路径时，它们是根据这个CWD解析的。

要查看您的当前工作目录是什么，可以使用`pwd()`函数（`pwd`代表“print working directory”，一个常见的命令行术语）：

```julia
julia> current_dir = pwd()
"/Users/your_username/julia_projects" # 输出会根据您的系统和启动Julia的位置而异

julia> println("My current directory is: ", current_dir)
My current directory is: /Users/your_username/julia_projects
```

如果您需要更改当前工作目录，可以使用`cd()`函数。这在您希望在特定项目文件夹中操作文件，而无需重复输入长路径时非常有用。

```julia
julia> pwd()
"/Users/your_username/julia_projects"

# 假设您有一个子目录 "my_data_project"
julia> cd("my_data_project") # 假设 "my_data_project" 存在于CWD中

julia> pwd()
"/Users/your_username/julia_projects/my_data_project"

# 返回上一级目录（父目录）：
julia> cd("..")

julia> pwd()
"/Users/your_username/julia_projects"
```

更改目录时要小心，因为它会影响后续操作中相对路径的解析方式。

### 使用`joinpath`安全地构建路径

手动拼接字符串来创建文件路径（例如`folder_name * "/" * file_name`）容易出错，并且由于`/`与`\`等路径分隔符的差异，在不同操作系统之间不具备可移植性。

Julia提供了`joinpath()`函数，这是构建路径的推荐方式。它会智能地连接路径组件，使用适用于您代码运行操作系统的正确分隔符。

```julia
julia> base_directory_unix = "/usr/local/data"
"/usr/local/data"
julia> filename = "config.ini"
"config.ini"

# 在类Unix系统（Linux、macOS）上
julia> full_path_unix = joinpath(base_directory_unix, "settings", filename)
"/usr/local/data/settings/config.ini"

# 现在，我们考虑Windows风格的基础路径
julia> base_directory_windows = "D:\\MyApp\\UserData"
"D:\\MyApp\\UserData"

# 如果在Windows上应用相同的逻辑，使用其典型路径结构
julia> full_path_windows = joinpath(base_directory_windows, "preferences", filename)
"D:\\MyApp\\UserData\\preferences\\config.ini" # Windows上的输出
```

使用`joinpath`使您的代码更容易在不同机器上运行。

### 从路径中提取信息

有时您有一个完整路径，需要提取其中的特定部分，例如文件名或其所在目录。

- `basename(path_string)`：返回路径的最后一个组成部分。如果路径指向文件，它返回文件名。如果指向目录，它返回目录名。
- `dirname(path_string)`：返回路径的目录部分，即除了最后一个组成部分之外的所有内容。

```julia
julia> file_path = "/path/to/my/important_file.txt"
"/path/to/my/important_file.txt"

julia> fname = basename(file_path)
"important_file.txt"

julia> dir = dirname(file_path)
"/path/to/my"

julia> basename("/path/to/my_directory/") # 也适用于目录
"my_directory"

julia> dirname("/path/to/my_directory/")
"/path/to"
```

另一个有用的函数是`splitext(path_string)`，它将路径分割成一对：不含扩展名的路径部分，以及扩展名本身（包括点）。

```julia
julia> file_path = "data/archive.tar.gz"
"data/archive.tar.gz"

julia> root, ext = splitext(file_path)
("data/archive.tar", ".gz")

julia> root2, ext2 = splitext(root) # 对于复合扩展名，可以多次调用它
("data/archive", ".tar")

julia> splitext("filename_no_extension")
("filename_no_extension", "")
```

当您需要检查文件类型或以程序方式更改扩展名时，这很方便。

### 检查路径属性

在尝试读取或写入某个路径之前，通常明智的做法是检查它是否存在以及它是哪种类型的文件系统对象。

- `ispath(path_string)`：如果在给定路径处存在文件或目录，返回`true`，否则返回`false`。
- `isfile(path_string)`：如果路径存在且是一个普通文件，返回`true`，否则返回`false`。
- `isdir(path_string)`：如果路径存在且是一个目录，返回`true`，否则返回`false`。

```julia
julia> data_file = "my_data.csv"
"my_data.csv"

# 假设 my_data.csv 尚不存在
julia> ispath(data_file)
false

julia> isfile(data_file)
false

julia> isdir("my_project_folder") # 假设此文件夹存在
true
```

这些函数对于编写能够适应不同情况或避免错误的脚本非常有用。例如，您可以在尝试读取文件之前检查它是否存在：

```julia
output_file = "results.txt"
if isfile(output_file)
    println("警告：输出文件'", output_file, "'已存在。它可能会被覆盖。")
else
    println("输出文件'", output_file, "'将被创建。")
end
```

### 使用目录

除了检查目录是否存在之外，您还经常需要创建新目录或列出其内容。

**创建目录**

- `mkdir(path_string)`：创建一个新目录。如果路径中的任何父目录不存在，它将引发错误。
- `mkpath(path_string)`：创建一个新目录。如果路径中的父目录不存在，它也会一并创建。这通常更方便、更安全。

```julia
julia> mkdir("new_single_folder") # 在CWD中创建 "new_single_folder"

# julia> mkdir("path/to/deep_folder") # 如果 "path" 或 "path/to" 不存在，这将报错

julia> mkpath("path/to/another_deep_folder") # 依次创建 "path"，然后是 "to"，最后是 "another_deep_folder"
"path/to/another_deep_folder"
```

如果您尝试创建已存在的目录，`mkdir`将报错，而`mkpath`将静默成功（不执行任何操作）。

**列出目录内容**
`readdir(path_string)`函数返回一个字符串数组，其中每个字符串是指定目录内文件或子目录的名称。默认情况下，如果没有提供路径，它会列出当前工作目录的内容。

```julia
# 假设CWD包含 "file1.txt"、"file2.md" 和一个文件夹 "sub_dir"
julia> contents = readdir()
3-element Vector{String}:
 "file1.txt"
 "file2.md"
 "sub_dir"

julia> specific_folder_contents = readdir("my_project_folder") # 列出 "my_project_folder" 的内容
# 输出将是一个 Vector{String}，包含 "my_project_folder" 中各项的名称
```

然后您可以遍历此列表以处理每个文件或子目录。

操作和管理文件路径与目录是编写与文件系统交互的Julia程序的必备技能。使用`joinpath`、`isdir`、`mkdir`和`readdir`等函数将帮助您编写更清晰、更可靠且跨平台兼容的代码。随着您构建更复杂的应用程序，这些工具将变得越来越重要，用于组织数据、管理项目结构，并确保您的程序能找到并存储所需信息。

## 参考资料

- [Julia Documentation: File System](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGJPip5kPwpCEJ6kPLOy21wTR7bbxaDtk3NwnRZc5TH8y9izA13jxNkCR4p28gCZyZ2YlY6rX82oOW1TSfSTN0CjOW_eXp4_GrKCTyPhY0HifZIhDlHYN_7n3xUnd6nkPzMYeDswQ==) — The Julia Language (2020)
  Julia文件系统功能的官方参考资料，包括`joinpath`、`pwd`、`cd`、`readdir`、`mkdir`、`mkpath`、`basename`、`dirname`、`splitext`、`ispath`、`isfile`和`isdir`。
