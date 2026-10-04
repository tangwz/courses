---
course: "getting-started-julia-programming"
chapter: "working-with-collections"
lesson: "tuples-fixed-size-immutable-sequences"
sourceId: 6788
sourceUrl: "https://apxml.com/zh/courses/getting-started-julia-programming/chapter-4-working-with-collections/tuples-fixed-size-immutable-sequences"
title: "元组：固定大小的不可变序列"
description: "理解Julia中的元组、它们的不可变性以及固定大小项集合的用例。"
order: 4
plots: []
sourceHash: "1854c9c757ea841330d2349846ee5924caca20750f1cbb3cf86a13cac0e3e350"
sourceCorrections: []
---

尽管数组提供了一种灵活的方式来管理可增长、收缩或变化的项集合，但Julia也提供了**元组**，用于需要一旦创建便无法更改的有序项序列的情况。可以将元组视为特定、不可更改的信息记录。这种不可变性，即无法更改的特性，是元组的定义特征，并提供多项优势。

### 元组有何不同？

元组是有序集合，这意味着项以特定顺序存储，你可以通过它们的位置访问它们。然而，与数组不同的是：

- **大小固定**：元组一旦创建，就不能添加或删除元素。其长度在创建时确定。
- **不可变**：你不能更改现有元组中元素的值。如果需要修改后的版本，则会创建一个全新的元组。

这听起来可能有局限性，但不可变性是一个有用的属性。它确保元组的数据在其整个生命周期中保持不变，这可以使你的程序更易于理解和推断，并且当你想要保证数据不会被意外更改时，这一点很重要。例如，像 `(10, 20)` 这样的坐标或像 `(255, 0, 0)` 这样的RGB颜色值是元组的良好选择，因为它们的结构和值本质上是固定的。

### 创建元组

在Julia中，通过将逗号分隔的值序列括在括号 `()` 中来创建元组。

```julia
# 整型元组
my_numbers = (10, 20, 30)
println(my_numbers)

# 混合数据类型的元组
person_data = ("Alice", 35, 1.68) # 姓名、年龄、身高
println(person_data)

# 空元组
empty_tup = ()
println(empty_tup)
```

Output:

```
(10, 20, 30)
("Alice", 35, 1.68)
()
```

对于只有一个元素的元组，你必须在该元素后添加一个逗号。如果没有逗号，括号可能只表示运算顺序。

```julia
single_item_tuple = (99,) # 注意逗号
not_a_tuple = (99)      # 这只是数字 99

println(single_item_tuple)
println(typeof(single_item_tuple))
println(not_a_tuple)
println(typeof(not_a_tuple))
```

Output:

```
(99,)
Tuple{Int64}
99
Int64
```

Julia也允许你在许多情况下不使用括号来创建元组，尤其是在赋值时，尽管使用括号通常更清晰：

```julia
implicit_tuple = 1, "hello", 3.14
println(implicit_tuple)
println(typeof(implicit_tuple))
```

Output:

```
(1, "hello", 3.14)
Tuple{Int64, String, Float64}
```

这就是Julia中的函数可以返回多个值的方式；它们实际上是返回一个元组。

### 访问元组中的元素

访问元组中的元素使用基于1的索引，就像数组一样，将索引放在方括号 `[]` 中。

```julia
point = (15, 25, 30) # 表示一个 (x, y, z) 坐标

x_coord = point[1]
y_coord = point[2]
z_coord = point[3]

println("X 坐标: ", x_coord)
println("Y 坐标: ", y_coord)
println("Z 坐标: ", z_coord)

# 你也可以使用切片，这会创建一个新元组
sub_tuple = point[1:2]
println("子元组 (x, y): ", sub_tuple)
```

Output:

```
X 坐标: 15
Y 坐标: 25
Z 坐标: 30
子元组 (x, y): (15, 25)
```

### 不可变性的优势

元组的核心特性是它们的不可变性。元组一旦创建，其元素就不能更改。如果你尝试为特定索引处的元素赋新值，Julia会引发错误。

```julia
my_tuple = (10, 20, 30)
println("原始元组: ", my_tuple)

# 尝试更改第一个元素
# my_tuple[1] = 5 # 这一行将导致错误！
```

如果你取消注释并运行 `my_tuple[1] = 5`，你将看到类似于以下内容的错误消息：
`ERROR: MethodError: no method matching setindex!(::Tuple{Int64, Int64, Int64}, ::Int64, ::Int64)`
此消息表明无法更改整数元组的元素 (setindex!)。

> 元组的元素是固定的。尝试更改元素会导致错误，原始元组保持不变。

为什么这很有用？

- **数据完整性**：不可变性保证元组中的数据在其整个生命周期中保持不变。这对于表示不应更改的数据（如历史记录或特定配置）很有价值。
- **可预测性**：当你将元组传递给函数时，你可以确定函数不会更改其内容。这使得代码更易于理解。
- **性能**：Julia有时可以对不可变数据结构应用优化。
- **字典键**：如果元组的所有元素也是适合作为键的类型（例如，数字、字符串或其他不可变类型），则元组可以作为字典中的键（我们将在下一节介绍）。像数组这样的可变集合不能直接用作字典键。

如果你需要元组的“修改”版本，则会创建一个包含这些更改的新元组：

```julia
original_tuple = (1, 2, 3)
# 要“更改”第一个元素，请创建一个新元组
modified_tuple = (100, original_tuple[2], original_tuple[3])
println("原始: ", original_tuple)
println("修改后: ", modified_tuple)
```

Output:

```
Original: (1, 2, 3)
Modified: (100, 2, 3)
```

### 元组操作

尽管元组是不可变的，但你仍然可以对它们执行多项操作：

- **长度**：使用 `length()` 获取元素数量。

  ```julia
  record = ("CPU", 3.5, "GHz")
  println(length(record)) # 输出: 3
  ```
- **迭代**：使用 `for` 循环遍历元素。

  ```julia
  rgb_color = (255, 128, 0) # 橙色
  for component in rgb_color
      print(component, " ")
  end
  println() # 输出: 255 128 0 
  ```
- **连接（创建新元组）**：你可以使用展开运算符 (`...`) 组合元组以形成新元组。

  ```julia
  tuple_a = (1, 2)
  tuple_b = ("x", "y")
  combined_tuple = (tuple_a..., tuple_b...) # 将元素展开到新元组中
  println(combined_tuple) # 输出: (1, 2, "x", "y")
  ```

  展开运算符 `...` 将 `tuple_a` 和 `tuple_b` 的元素展开到正在构建的新元组中。
- **检查元素是否存在**：使用 `in` 运算符。

  ```julia
  permissions = ("read", "write")
  println("execute" in permissions) # 输出: false
  println("read" in permissions)    # 输出: true
  ```

### 命名元组：提升清晰度

Julia还支持**命名元组**，它类似于常规元组，但除了索引之外，你还可以通过名称访问元素。这在处理每个元素都有独特含义的结构化数据时，可以显著提高代码可读性。

你可以这样创建一个命名元组：

```julia
point_2d = (x=10, y=20)
println(point_2d)
```

Output:

```
(x = 10, y = 20)
```

你可以使用点表示法通过名称或通过索引访问元素：

```julia
println("X 坐标: ", point_2d.x)
println("Y 坐标: ", point_2d.y)
println("第一个元素: ", point_2d[1])
```

Output:

```
X-coordinate: 10
Y-coordinate: 20
First element: 10
```

命名元组也是不可变的。它们提供了一种轻量级的方式，用描述性名称将相关数据片段组合起来，而无需定义完整的 `struct`（你将在后面学习）。

### 何时使用元组

在以下情况下选择元组：

- 当你有一个固定项集合且顺序很重要时。
- 当各项紧密相关并表示一个单一、不可更改的实体时（例如，一对坐标，或像 `(1, 2, 5)` 这样的版本号）。
- 当你希望确保数据集合在创建后不能被修改时。
- 当你需要从函数返回多个值时（Julia 通过隐式返回元组来实现这一点）。
- 当你需要一个不可变序列用作字典中的键时。

元组提供了一种简单而有效的方法来处理固定大小、有序且不可变的数据集合。它们通过提供数据稳定性保证来补充数组，这在许多编程场景中很有价值。随着你继续学习Julia，你会发现元组经常被使用，通常是因为它们提供的清晰性和不可变性带来的安全性。

## 参考资料

- [The Julia Language Manual: Tuples](https://docs.julialang.org/en/v1/manual/types/#Tuple-Types) — The Julia Language Developers (2023)
  Publisher: The Julia Language
  官方语言文档，涵盖Julia中元组的定义、创建、索引、不可变性及基本操作。
- [Think Julia: How to Think Like a Computer Scientist (Chapter 10: Tuples)](https://benlauwens.github.io/ThinkJulia.jl/stable/book.html) — Ben Lauwens, Allen B. Downey (2019)
  Publisher: O'Reilly Media, Inc.
  一本Julia编程入门指南，清晰解释了元组等数据类型及其应用。
- [Julia Data Science](https://juliadatascience.io) — Jose Storopoli, Rik Huijzer, Lazaro Alonso (2021)
  Publisher: JuliaDataScience.io; DOI: [10.1007/978-1-4842-7022-2](https://doi.org/10.1007/978-1-4842-7022-2)
  一本使用Julia进行数据任务的实用指南，包括对元组等核心数据结构及其在确保数据完整性方面作用的介绍。
