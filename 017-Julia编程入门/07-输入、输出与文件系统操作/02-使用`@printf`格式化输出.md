# 使用`@printf`格式化输出

来源：[原文](https://apxml.com/zh/courses/getting-started-julia-programming/chapter-7-input-output-file-system-interaction/formatting-output-printf)

[返回章节目录](README.md) · [返回课程目录](../README.md)

`print` 和 `println` 非常适合快速直接地向控制台输出内容，但在许多情况下，你需要更精细地控制数据的显示方式。你可能希望在列中对齐 (alignment)数字，指定浮点值的小数位数，或者用空格填充字符串。对于这些任务，Julia 提供了 `@printf` 宏，一个用于生成精确格式化文本的强大工具。该宏是 Julia 标准库中 `Printf` 模块的一部分，无需显式导入即可方便地使用。

`@printf` 宏的工作方式与 C 语言及其他编程语言中的 `printf` 函数类似。它接受一个格式字符串作为第一个参数 (parameter)，后面跟着一个或多个你想要格式化并打印的值。

### `@printf` 的基本用法

使用 `@printf` 的一般语法是：

```julia
@printf "format_string" value1 value2 ...
```

`format_string` 包含照原样打印的普通文本，以及称为 *格式说明符* 的特殊序列。每个格式说明符都充当一个占位符，它定义了相应值应如何格式化。这些说明符总是以百分号 (`%`) 开头。

让我们看一个简单例子：

```julia
name = "Julia"
version = 1.9
@printf "To %s version %.1f!\n" name version
# 输出: Welcome to Julia version 1.9!
```

在此示例中：

- `%s` 是字符串的格式说明符。它会被 `name` 变量的值替换。
- `%.1f` 是浮点数的格式说明符。它会被 `version` 变量的值替换，并格式化为一位小数。
- `\n` 是一个转义序列，代表换行符，打印后将光标移动到下一行。与 `println` 不同，`@printf` 不会自动添加换行符。

### 常用格式说明符

以下是一些 `@printf` 最常用的格式说明符：

- `%s`: 将值格式化为字符串。

  ```julia
  language = "Julia"
  @printf "Current language: %s\n" language
  # 输出: Current language: Julia
  ```
- `%d`: 将整数格式化为十进制数字。

  ```julia
  count = 10
  @printf "There are %d items.\n" count
  # 输出: There are 10 items.
  ```
- `%f`: 以标准十进制表示法（例如，`123.45`）格式化浮点数。

  ```julia
  price = 19.99
  @printf "The price is %f.\n" price
  # 输出: The price is 19.990000. (默认精度通常为 6 位小数)
  ```
- `%e` 或 `%E`: 以科学计数法（例如，`1.2345e+02` 或 `1.2345E+02`）格式化浮点数。

  ```julia
  large_number = 123450000.0
  @printf "Scientific: %e\n" large_number
  # 输出: Scientific: 1.234500e+08
  ```
- `%g` 或 `%G`: 使用 `%f` 或 `%e`（或 `%E`）符号格式化浮点数，以给定值和精度最紧凑的方式显示。这通常是浮点数的一个很好的通用选择。

  ```julia
  val1 = 0.0000123
  val2 = 12345.6
  @printf "Compact val1: %g, Compact val2: %g\n" val1 val2
  # 输出: Compact val1: 1.23e-5, Compact val2: 12345.6
  ```
- `%c`: 格式化单个字符。

  ```julia
  initial = 'J'
  @printf "Initial: %c\n" initial
  # 输出: Initial: J
  ```
- `%%`: 打印字面上的百分号。

  ```julia
  percentage = 75
  @printf "Discount: %d%%\n" percentage
  # 输出: Discount: 75%
  ```

### 控制宽度、对齐 (alignment)和精度

`@printf` 的主要优势之一是它能控制输出的布局。

#### 字段宽度

你可以通过在 `%` 后面直接加上数字来指定字段的最小宽度。如果值短于指定宽度，它会用空格填充（默认情况下在左侧填充，使其右对齐）。

```julia
id = 42
name = "Widget"
@printf "ID: %5d, Name: %10s\n" id name
# 输出: ID:    42, Name:     Widget
```

在这里，`%5d` 表示“在至少 5 个字符宽的字段中打印整数”。`%10s` 表示“在至少 10 个字符宽的字段中打印字符串”。

#### 对齐

默认情况下，数字是右对齐的，字符串也通常是右对齐的（尽管这有时取决于系统或特定 Julia 版本的细微差别；如果对字符串的对齐有要求，最好使用显式对齐）。你可以在宽度说明符前放置连字符 (`-`) 来强制左对齐。

```julia
id = 42
name = "Widget"
@printf "ID: %-5d, Name: %-10s END\n" id name
# 输出: ID: 42   , Name: Widget     END
```

注意 `42` 现在在右侧用空格填充以达到 5 个字符的宽度，“Widget”在右侧填充以达到 10 个字符的宽度。

#### 浮点数的精度

对于浮点数（`%f`、`%e`、`%g`），你可以指定小数点后显示的位数。这通过在类型说明符前添加一个句点 (`.`)，后跟所需的小数位数来实现。

```julia
pi_approx = 3.1415926535
@printf "Pi (default %%f): %f\n" pi_approx
@printf "Pi (%.2f): %.2f\n" pi_approx
@printf "Pi (%.4f): %.4f\n" pi_approx
# 输出:
# Pi (default %f): 3.141593
# Pi (%.2f): 3.14
# Pi (%.4f): 3.1416 (注意：会发生四舍五入)
```

你可以结合使用宽度和精度：

```julia
value = 12.345
@printf "Value: |%10.2f|\n" value
# 输出: Value: |     12.35|
```

这将 `value` 格式化为两位小数，在总宽度为 10 个字符的字段中右对齐。

#### 用零填充

如果你想用前导零而不是空格来填充数字，可以在 `%` 之后或对齐说明符（如果有）之后立即放置一个 `0`。

```julia
item_number = 7
@printf "Item: %03d\n" item_number
# 输出: Item: 007

amount = 5.2
@printf "Amount: %08.2f\n" amount # 用零填充到宽度 8，保留 2 位小数
# 输出: Amount: 00005.20
```

### 组合多个说明符

你可以在单个 `format_string` 中使用多个格式说明符，而且通常会这样做。你在字符串后面提供的值将按它们出现的顺序与说明符匹配。

```julia
product = "Laptop"
quantity = 3
unit_price = 1200.50

@printf "Item: %-15s Quantity: %3d Unit Price: $%8.2f\n" product quantity unit_price
# 输出: Item: Laptop          Quantity:   3 Unit Price: $ 1200.50
```

### 记住换行符 (`\n`)

新用户常感到困惑的一点是，与 `println` 不同，`@printf` 不会在输出末尾自动添加换行符。如果你想让后续输出出现在新行上，你必须在格式字符串中包含 `\n`。

```julia
@printf "First line."
@printf "Still on the first line."
@printf "\nNow on the second line.\n"
# 输出:
# First line.Still on the first line.
# Now on the second line.
```

### 不打印，获取格式化字符串：`Printf.sprintf`

有时，你不想直接将格式化文本打印到控制台。相反，你可能希望将其存储在变量中、写入文件或在程序的其他部分使用。为此，`Printf` 模块提供了 `sprintf` 函数（注意：它是一个函数，而不是宏，因此没有 `@` 前缀）。

`sprintf` 接受与 `@printf` 相同的格式字符串和参数 (parameter)，但它不打印，而是返回结果格式化字符串。

```julia
using Printf # 如果 @printf 未引入 Printf 模块，则使用 sprintf 需先导入

item_name = "Sensor"
value = 23.5789

formatted_data = Printf.sprintf("%s reading: %.2f units", item_name, value)
println(formatted_data)
# 输出: Sensor reading: 23.58 units

# 现在你可以使用 formatted_data，例如将其写入文件
```

### 实践：创建整洁的小表格

让我们将其中一些格式化选项组合起来，创建一个整齐对齐 (alignment)的数据表。这是 `@printf` 擅长的常见应用场景。

```julia
# 表格数据
students = [
    ("Alice", 85, 92.5),
    ("Bob", 78, 88.0),
    ("Charlie", 91, 95.75)
]

# 打印表头
@printf "%-10s | %8s | %10s\n" "Name" "Midterm" "Final Grade"
@printf "-----------|----------|------------\n" # 分隔线

# 打印表格行
for (name, midterm_score, final_grade) in students
    @printf "%-10s | %8d | %10.2f\n" name midterm_score final_grade
end
```

这将产生以下输出：

```
Name       |  Midterm | Final Grade
-----------|----------|------------
Alice      |       85 |      92.50
Bob        |       78 |      88.00
Charlie    |       91 |      95.75
```

注意：

- `%-10s` 将姓名在 10 个字符的字段中左对齐。
- `%8d` 将整数期中分数在 8 个字符的字段中右对齐。
- `%10.2f` 将最终成绩在 10 个字符的字段中右对齐，并格式化为两位小数。

`@printf` 宏为以结构化和可读方式呈现文本数据提供了强大的控制力。尝试不同的说明符和选项是熟悉其功能的最佳途径。

## 参考资料

- [Printf Module](https://docs.julialang.org/en/v1/stdlib/Printf/) — The Julia Language (2025)
  Julia官方标准库Printf模块的文档，详细介绍了@printf宏、sprintf函数和所有格式说明符。
- [Think Julia: How to Think Like a Computer Scientist](https://benlauwens.github.io/ThinkJulia.jl/stable/pages/14_Files_and_Input_Output/) — Ben Lauwens, Allen B. Downey (2019)
  Publisher: O'Reilly Media; Pages: Chapter 14: Files and Input/Output
  一本介绍性的教科书，涵盖了Julia编程的基本概念，包括使用@printf进行输出格式化的详细解释。

---

[上一节](01-%E4%BD%BF%E7%94%A8%20%60print%60%20%E5%92%8C%20%60println%60%20%E6%98%BE%E7%A4%BA%E8%BE%93%E5%87%BA.md) · [下一节](03-%E6%8E%A5%E5%8F%97%E7%94%A8%E6%88%B7%E8%BE%93%E5%85%A5.md)
