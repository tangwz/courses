---
course: "getting-started-julia-programming"
chapter: "consolidating-knowledge-advancing-julia"
lesson: "applying-your-skills-small-project"
sourceId: 6832
sourceUrl: "https://apxml.com/zh/courses/getting-started-julia-programming/chapter-9-consolidating-knowledge-advancing-julia/applying-your-skills-small-project"
title: "应用所学：一个小项目"
description: "一个小型项目建议，用于实践和巩固在本课程中获得的 Julia 编程技能。"
order: 7
plots: []
sourceHash: "03f1822a3bc96cc716676e71282ff312260b4b8ea5fb68575fa4cadb50cb4b25"
sourceCorrections: []
---

这是一个很好的机会，可以汇集 Julia 编程技能。通过完成一个小型、完整的项目，您可以巩固对 Julia 编程特性如何组合的理解，并增强编写自己的 Julia 程序的信心。我们建议进行一个“简单 CSV 数据分析器”项目。此任务设计为初学者易于管理，同时仍能涵盖编程的许多主要方面。

### 项目：简单 CSV 数据分析器

本项目目标是编写一个 Julia 脚本，从一个简单的逗号分隔值 (CSV) 文件中读取数据，执行一些基本计算，并将一份汇总报告输出到控制台。

设想您有一个名为 `student_scores.csv` 的文本文件，内容如下：

```
Name,Subject,Score
Alice,Math,85
Bob,Math,92
Charlie,Math,78
Alice,Science,90
Bob,Science,88
Charlie,Science,82
David,Math,95
Eve,Science,88
```

您的程序应能实现以下功能：

1. 从 `student_scores.csv` 文件中读取这些数据。
2. 计算并显示：
   - 已处理的记录总数。
   - 所有条目的平均分数。
   - 获得的最高分数，以及是哪位学生在哪个科目中获得的。
   - 特定科目（例如，“数学”）的平均分数。您可以硬编码科目，或提示用户输入。
3. 以清晰、格式化的方式将这些结果输出到控制台。

### 核心应用技能

本项目将让您实践并整合课程中涵盖的几个主题：

- **文件输入/输出（第 7 章）：** 从文本文件（`student_scores.csv`）中读取数据。
- **字符串处理（第 2 和第 4 章）：** 解析 CSV 文件的每一行（例如，按逗号分隔）。
- **数据类型与转换（第 2 章）：** 处理姓名和科目字符串，以及分数的数字。将字符串分数转换为数值类型。
- **集合（第 4 章）：** 存储解析后的数据，可能在字典数组或自定义结构体数组中。例如，每条记录可以是 `Dict("Name" => "Alice", "Subject" => "Math", "Score" => 85)`。
- **控制流（第 3 章）：** 使用循环（例如，`for` 或 `while`）遍历文件中的行和数据集合。使用条件语句（`if-elseif-else`）处理诸如查找最高分数或按科目筛选之类的逻辑。
- **函数（第 5 章）：** 将代码组织成可重用函数，例如 `read_data(filename)`、`calculate_statistics(records)` 和 `print_report(stats)`。
- **错误处理（第 8 章）：** 实现基本的 `try-catch` 块，特别是针对文件操作（例如，处理文件不存在的情况）。
- **控制台输出（第 7 章）：** 使用 `println` 或 `@printf` 整齐地显示汇总报告。

### 建议的开始步骤

1. **准备数据文件：**
   创建包含上面提供的示例数据的 `student_scores.csv` 文件，或者创建您自己的类似 CSV 文件。确保它保存在您的 Julia 脚本可以访问的位置。
2. **规划程序结构：**
   在编写代码之前，思考如何分解问题。您将需要哪些函数？

   - 一个函数，用于读取 CSV 文件并将其内容解析为更可用的数据结构（例如字典数组）。
   - 一个函数，或几个小函数，用于执行计算（记录总数、平均分数、最高分数、特定科目的平均分数）。
   - 一个函数，用于显示结果。
3. **实现文件读取和解析：**

   - 首先编写一个以文件名作为输入的函数。
   - 在该函数内部，打开文件。请记住使用 `try-catch` 块处理文件不存在时可能出现的 `IOError`。
   - 逐行读取文件。
   - 对于每一行，根据逗号分隔符将其分成几部分（姓名、科目、分数）。
   - 您可能需要跳过标题行。
   - 将分数部分转换为整数（例如，使用 `parse(Int, score_string)`）。
   - 存储这些解析后的信息。一个不错的选择是数组，其中每个元素都是一个字典（例如，`Dict("Name"=>name, "Subject"=>subject, "Score"=>score)`）。此函数应返回您的记录集合。
4. **实现数据处理逻辑：**

   - 编写一个（或多个）函数，将您的记录集合作为输入。
   - 遍历记录以计算：
     - 记录总数（即您的集合的长度）。
     - 所有分数的总和，然后是平均值。
     - 最高分数，并追踪与之相关联的学生姓名和科目。
     - 特定科目（例如，“数学”）的分数总和，以及“数学”条目的计数，以计算其平均值。
5. **实现报告生成：**

   - 创建一个函数，以计算出的统计数据作为输入。
   - 使用 `println` 或 `@printf` 以可读格式显示这些统计数据。例如：

     ```
     学生分数分析报告
     -----------------------------
     已处理记录总数：8
     平均分数（所有科目）：87.0
     最高分数：David 在数学中获得 95 分
     数学平均分数：88.33
     ```
6. **编写主脚本逻辑：**

   - 在您的主脚本文件中，调用 `read_data` 函数以获取记录。
   - 将这些记录传递给 `calculate_statistics` 函数。
   - 将计算结果传递给 `print_report` 函数。
7. **测试与改进：**

   - 运行您的脚本。它是否为您的示例 `student_scores.csv` 生成预期输出？
   - 测试几种不同情况：如果文件为空（除了标题），会怎样？如果分数不同，会怎样？
   - 如果需要调试并查看中间值，可以在函数内部暂时使用 `println` 语句。

### 可选功能增强

在基本版本运行正常后，您可以尝试以下扩展功能：

- **更完善地处理带或不带标题的 CSV 文件：** 检测是否存在标题，或允许用户指定。
- **计算更多统计数据：** 例如最低分数，或文件中所有独特科目的列表。
- **文件名用户输入：** 而不是硬编码 `student_scores.csv`，使用 `readline()` 请求用户输入文件名。
- **将报告写入文件：** 除了输出到控制台之外，或取代输出到控制台，将报告写入新的文本文件。
- **按学生分组统计：** 计算每个学生的平均分数。（这将是字典的一个良好应用）。

### 与后续学习的联系

本项目使用 Julia 的主要功能来执行常见的数据处理任务。在您完成它的过程中，您可能会注意到一些地方，如果数据量更大或更复杂，更高级的工具可能会更有用。

例如，使用像 `DataFrames.jl` 这样的包可以大大简化 CSV 数据的读取、解析和处理，本章前面已向您介绍了该包。`DataFrames.jl` 提供功能出色的且效率高的结构（称为 `DataFrame`），这些结构专为表格数据设计，并配有用于筛选、分组、连接和汇总数据的大量函数。

完成本项目将为您提供坚实的实践基础。在此之后，您将能更好地理解 `DataFrames.jl` 等工具是如何在这些主要思想之上构建，从而为数据处理提供更专业的能力。祝您好运，并享受应用您的 Julia 技能！

## 参考资料

- [The Julia Language Documentation](https://docs.julialang.org/en/v1/) — The Julia Language Developers (2024)
  Julia编程语言的官方文档，涵盖核心语法、数据类型、控制流、函数、I/O和错误处理。
- [DataFrames.jl Documentation](https://dataframes.juliadata.org/stable/) — The DataFrames.jl Developers (2024)
  DataFrames.jl的官方文档，提供在Julia中处理表格数据的工具，包括读取CSV文件和执行数据分析。
- [CSV.jl Documentation](https://csv.juliadata.org/stable/) — The CSV.jl Developers (2024)
  CSV.jl的官方文档，一个在Julia中快速、稳定地读写CSV文件的包。
- [Think Julia: How to Think Like a Computer Scientist](https://benlauwens.github.io/ThinkJulia.jl/latest/book.html) — Ben Lauwens and Allen B. Downey (2019)
  Publisher: O'Reilly Media; Pages: 295
  一本介绍性的教科书，使用Julia涵盖了基本的编程概念和问题解决技术，适合初学者。
