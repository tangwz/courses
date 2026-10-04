---
course: "intermediate-python-programming-ml"
chapter: "advanced-python-constructs"
lesson: "comprehensions-generators"
sourceId: 2082
sourceUrl: "https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-1-advanced-python-constructs/comprehensions-generators"
title: "列表推导式和生成器表达式"
description: "学习使用Python中的列表推导式和生成器表达式编写简洁高效的代码。"
order: 2
plots: []
sourceHash: "5e8996136271d4ece6c79c991efba9b13e598bf378b81771a865d9a2fc21baf8"
sourceCorrections: []
---

处理数据序列在数据科学中是基本的。很多时候，您需要根据现有列表创建新列表，或处理大型序列而无需占用过多内存。Python 为这些任务提供了优雅高效的工具：列表推导式和生成器表达式。它们为生成序列提供了比传统 `for` 循环更简洁的方法。

### 列表推导式：简洁的列表创建

设想您有一个数字列表，并且想要创建一个包含每个数字平方的新列表。使用标准的 `for` 循环，您可能会这样写：

```python
numbers = [1, 2, 3, 4, 5]
squares = []
for num in numbers:
  squares.append(num * num)

print(squares)
# Output: [1, 4, 9, 16, 25]
```

这当然可以正常工作，但它需要初始化一个空列表，然后在循环中显式添加项。列表推导式让您可以用一行更易读的代码实现相同的结果。

基本语法是：`[表达式 for 项 in 可迭代对象]`

让我们使用列表推导式重写平方的例子：

```python
numbers = [1, 2, 3, 4, 5]
squares = [num * num for num in numbers]

print(squares)
# Output: [1, 4, 9, 16, 25]
```

在这里，`num * num` 是应用于 `可迭代对象` (`numbers`) 中每个 `项` (此处命名为 `num`) 的表达式。结果是自动构建的新列表 `squares`。

#### 添加条件逻辑

列表推导式还可以包含 `if` 子句，用于从原始可迭代对象中筛选项目。

语法变为：`[表达式 for 项 in 可迭代对象 if 条件]`

假设您只想从列表中获取*偶数*的平方：

```python
numbers = [1, 2, 3, 4, 5, 6]
even_squares = [num * num for num in numbers if num % 2 == 0]

print(even_squares)
# Output: [4, 16, 36]
```

`if num % 2 == 0` 条件确保只有当数字为偶数时，表达式 `num * num` 才会被求值并包含在结果列表中。

#### 列表推导式的优点

- **简洁性：** 它们通常将多行代码（循环设置、添加）简化为一行。
- **可读性：** 一旦熟悉语法，对于简单的转换和筛选，它们可以比等效的 `for` 循环更易读。
- **性能：** 在许多情况下，列表推导式可能比带有 `.append()` 的显式 `for` 循环稍快，因为列表分配和元素插入在内部进行了优化。

需要记住的是，列表推导式会立即在内存中创建*整个*新列表。对于中等大小的列表通常没问题，但对于非常大的数据集可能会出现问题。

### 生成器表达式：内存高效的迭代

如果您需要处理包含数百万甚至数十亿项的序列，或者一个理论上无限的序列怎么办？使用列表推导式将所有处理过的项加载到列表中会消耗大量内存，甚至可能无法实现。这就是生成器表达式的优势所在。

生成器表达式的语法与列表推导式非常相似，但使用圆括号 `()` 而非方括号 `[]`：

`(表达式 for 项 in 可迭代对象 if 条件)`

关键在于，生成器表达式*不会*在内存中创建列表。相反，它会创建一个名为*生成器*的特殊对象。这个生成器充当迭代器，按需逐一生成项（通常被称为“惰性求值”）。

让我们修改平方的例子，使用生成器表达式：

```python
numbers = [1, 2, 3, 4, 5]
squares_generator = (num * num for num in numbers)

print(squares_generator)
# Output: <generator object <genexpr> at 0x...> （地址会因系统而异）
```

请注意，直接打印生成器对象本身并不会显示平方后的数字。它只是告诉我们有一个生成器。要获取这些值，您需要对其进行迭代，例如，使用 `for` 循环或 `next()` 函数：

```python
# 使用 for 循环迭代（最常见）
for square in squares_generator:
  print(square, end=' ') # Output: 1 4 9 16 25

# 或者转换为列表（如果您最终需要完整列表）
# 注意：这会消耗生成器。您只能迭代一次。
# numbers = [1, 2, 3, 4, 5]
# squares_generator = (num * num for num in numbers)
# squares_list = list(squares_generator)
# print(squares_list) # Output: [1, 4, 9, 16, 25]
```

每次循环请求下一项时，生成器表达式只执行足以生成该下一项（`1*1`，然后 `2*2`，依此类推）并产出它。它不会计算或存储序列的其余部分，直到需要时才进行。

#### 生成器表达式的优点

- **内存高效：** 它们的主要优势。它们使用最少的内存，因为它们一次只生成一个项。这使得它们非常适合超大型数据集或数据流。
- **惰性求值：** 计算会推迟到实际需要该值时才进行。如果您不需要遍历整个序列，这可以节省处理时间。
- **可组合性：** 生成器可以优雅地串联起来。一个生成器表达式的输出可以作为另一个生成器的输入，从而创建高效的数据处理流程，而无需中间列表。

考虑处理一个大型日志文件中的行。您可能希望从每条符合特定模式的行中提取信息：

```python
# 假设 'large_log_file.txt' 非常大
# 此代码逐行读取和处理，而无需加载整个文件

# with open('large_log_file.txt', 'r') as f:
#     error_lines = (line.strip() for line in f if 'ERROR' in line)
#     # 现在您可以迭代 error_lines
#     for error in error_lines:
#         # 逐一处理每一条错误行
#         print(error)
```

使用生成器表达式 `(line.strip() for line in f if 'ERROR' in line)` 可以防止将整个潜在的超大文件加载到内存中。

### 列表推导式与生成器表达式的选择

选择主要取决于您对内存的要求以及打算如何使用生成的序列：

- 何时使用**列表推导式**：
  - 结果列表相对较小且内存充足时。
  - 需要多次访问列表元素或按索引访问时。
  - 需要立即获得完整列表时（例如，将其传递给需要列表的函数）。
- 何时使用**生成器表达式**：
  - 处理内存受限的超大序列或数据流时。
  - 只需*一次*遍历序列时。
  - 您正在构建一个不需要中间存储的数据处理流程时。
  - 您想要惰性求值（仅在需要时计算值）时。

列表推导式和生成器表达式都是编写简洁、富有表现力且通常高效的 Python 代码的有力工具，在数据处理和机器学习 (machine learning)流程中尤为宝贵，因为性能和内存管理是重要的考量。熟练掌握它们是编写更专业 Python 代码的一步。

## 参考资料

- [The Python Language Reference - 6.2.7 Generator expressions](https://docs.python.org/3/reference/expressions.html#generator-expressions) — Guido van Rossum and the Python development team (2024)
  官方文档，详细说明了Python中生成器表达式的结构和内存高效运行方式。
- [Fluent Python: Clear, Concise, and Effective Programming](https://play.google.com/store/books/details/Fluent_Python_Clear_Concise_and_Effective_Programm?id=uK0dEAAAQBAJ) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media, Inc.
  一本全面的书籍，涵盖了高级Python特性，包括对列表推导式、生成器表达式及其性能特点的深入分析。第2版。
