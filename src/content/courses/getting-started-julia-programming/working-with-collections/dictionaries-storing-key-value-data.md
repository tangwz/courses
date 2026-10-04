---
course: "getting-started-julia-programming"
chapter: "working-with-collections"
lesson: "dictionaries-storing-key-value-data"
sourceId: 6789
sourceUrl: "https://apxml.com/zh/courses/getting-started-julia-programming/chapter-4-working-with-collections/dictionaries-storing-key-value-data"
title: "字典：存储键值数据"
description: "学习在 Julia 中使用字典来存储和获取键值对数据。"
order: 5
plots: []
sourceHash: "0d1c01cd29a65f4ad3f35770893c48a7cf65a63c2bad691952711d3c90f115a0"
sourceCorrections: []
---

数组和元组在处理有序数据序列方面表现良好，但有时您需要一种通过特定标签或标识符而非数字位置来存储和获取信息的方式。这时，字典就派上用场了。字典在其他编程语言中常被称为映射、关联数组或哈希表，它们是存储键值对的集合。

想象一下一本词典：您查找一个词（独有的键）来找到它的定义（值）。同样，在 Julia 字典中，每个独有的键都与一个特定值关联。这种结构在您知道对应键后需要快速查找值的情况下非常实用。

与数组不同，数组按数字索引（在 Julia 中通常是 1, 2, 3,...）排序，而字典从数据访问的角度来看通常被认为是无序的。您不会要求字典中的“第三项”；您会要求“与此特定键关联的值”。（值得注意的是，Julia 的 `Dict` 类型自 Julia 1.0 版本以来确实保留了插入顺序，但主要交互模式仍然是基于键的查找。）

### 创建字典

您可以使用 `Dict()` 构造函数在 Julia 中创建字典。您可以创建一个空字典，或者使用键值对对其进行初始化。“对”操作符 `=>` 用于将键与其值关联。

让我们创建一个可以存储任何类型键和值的空字典：

```julia
julia> scores = Dict()
Dict{Any, Any}()
```

这会创建一个字典，其中键和值都可以是 `Any` 类型，这意味着它们可以保存各种类型的数据。

通常，您会希望更具体地指定字典将保存的键和值的类型。这可以提高性能，并通过确保类型一致性来帮助及早发现错误。例如，要创建一个键为字符串（代表学生姓名）而值为整数（他们的分数）的字典：

```julia
julia> student_scores = Dict{String, Int}()
Dict{String, Int64}()
```

请注意，在典型的 64 位系统上，Julia 将 `Int` 推断为 `Int64`。`Int64` 是一种特定类型的整数，可以存储 64 位数字。

您也可以在创建字典时直接用值对其进行初始化：

```julia
julia> fruit_colors = Dict("apple" => "red", "banana" => "yellow", "grape" => "purple")
Dict{String, String} with 3 entries:
  "banana" => "yellow"
  "apple"  => "red"
  "grape"  => "purple"
```

在这种情况下，Julia 根据提供的初始数据推断键和值都是 `String` 类型。字典的一个重要属性是键必须是独有的。如果在初始化期间尝试添加重复的键，将使用与该键最后一次出现关联的值。

### 添加和修改元素

字典创建后，您可以添加新的键值对，或修改与现有键关联的值。这通过方括号表示法 `[]` 完成，类似于您访问数组元素的方式，但您使用的是键而不是数字索引。

要向现有字典添加新条目：

```julia
julia> student_scores["Alice"] = 95
95

julia> student_scores["Bob"] = 88
88

julia> student_scores
Dict{String, Int64} with 2 entries:
  "Bob"   => 88
  "Alice" => 95
```

如果您为字典中已存在的键赋值，Julia 会更新与该键关联的值。这对于更改数据很有用。

```julia
julia> student_scores["Alice"] = 97  # 爱丽丝提高了她的分数！
97

julia> student_scores
Dict{String, Int64} with 2 entries:
  "Bob"   => 88
  "Alice" => 97
```

### 访问值

要从字典中获取值，请在其键上使用方括号：

```julia
julia> alice_score = student_scores["Alice"]
97
```

如果您尝试访问字典中不存在的键，Julia 将抛出 `KeyError`。如果未处理这种类型的错误，它可能会停止您的程序。

```julia
julia> student_scores["Charlie"]
ERROR: KeyError: key "Charlie" not found
Stacktrace:
 [1] getindex(h::Dict{String, Int64}, key::String)
   @ Base ./dict.jl:499
 [2] top-level scope
   @ REPL[7]:1
```

为了安全地访问值并避免潜在错误，您可以首先使用 `haskey()` 函数检查键是否存在：

```julia
julia> haskey(student_scores, "Bob")
true

julia> haskey(student_scores, "Charlie")
false
```

一种更直接且安全地获取值的方法是使用 `get()` 函数。它接受三个参数 (parameter)：字典、您要查找的键，以及如果未找到键则返回的默认值。

```julia
julia> charlie_score = get(student_scores, "Charlie", 0) # 如果 Charlie 未找到，则返回 0
0

julia> alice_score_safe = get(student_scores, "Alice", 0) # Alice 存在，因此返回她的分数
97
```

使用 `get()` 通常更受青睐，因为它将检查和获取结合到一个单一、简洁的操作中，使您的代码更简洁。

### 删除元素

您可以使用 `pop!` 或 `delete!` 等函数从字典中删除键值对。

`pop!(dictionary, key)` 函数会移除给定 `key` 的键值对并返回被移除的值。如果未找到键，它将抛出 `KeyError`，类似于直接访问。

```julia
julia> student_scores
Dict{String, Int64} with 2 entries:
  "Bob"   => 88
  "Alice" => 97

julia> bobs_score = pop!(student_scores, "Bob")
88

julia> student_scores
Dict{String, Int64} with 1 entry:
  "Alice" => 97
```

您还可以为 `pop!` 提供一个默认值，如果未找到键，该值将被返回，从而阻止错误：

```julia
julia> pop!(student_scores, "David", "Not Present") # David 不在字典中
"Not Present"
```

`delete!(dictionary, key)` 函数也会移除键值对。但是，它不会返回值，而是返回修改后的字典本身。如果未找到键，它不会导致错误。

```julia
julia> student_scores["Carol"] = 92
92

julia> student_scores
Dict{String, Int64} with 2 entries:
  "Carol" => 92
  "Alice" => 97

julia> delete!(student_scores, "Alice")
Dict{String, Int64} with 1 entry:
  "Carol" => 92

julia> student_scores
Dict{String, Int64} with 1 entry:
  "Carol" => 92
```

### 遍历字典

经常需要处理字典中的所有条目。Julia 提供了几种方便的方式，可以使用 `for` 循环遍历它们。

您可以直接遍历键值对。在循环的每一步中，您都会得到键及其对应的值。

```julia
julia> fruit_colors = Dict("apple" => "red", "banana" => "yellow", "grape" => "purple")
Dict{String, String} with 3 entries:
  "banana" => "yellow"
  "apple"  => "red"
  "grape"  => "purple"

julia> for (fruit, color) in fruit_colors
           println("The color of $fruit is $color.")
       end
The color of banana is yellow.
The color of apple is red.
The color of grape is purple.
```

如果您只需处理键，可以使用 `keys()` 函数：

```julia
julia> for fruit in keys(fruit_colors)
           println("$fruit is a known fruit.")
       end
banana is a known fruit.
apple is a known fruit.
grape is a known fruit.
```

类似地，如果您只需值，可以使用 `values()` 函数：

```julia
julia> for color in values(fruit_colors)
           println("We have a fruit with the color: $color.")
       end
We have a fruit with the color: yellow.
We have a fruit with the color: red.
We have a fruit with the color: purple.
```

`keys()` 和 `values()` 都返回迭代器。迭代器是高效的结构，允许您逐个访问元素，通常不会在内存中创建新的集合。

### 常用字典函数

Julia 提供了其他一些有用的函数用于处理字典：

- `length(my_dict)`: 返回字典中键值对的数量。

  ```julia
  julia> length(fruit_colors)
  3
  ```
- `isempty(my_dict)`: 如果字典没有条目则返回 `true`，否则返回 `false`。

  ```julia
  julia> isempty(fruit_colors)
  false
  julia> isempty(Dict()) # 一个空字典
  true
  ```
- `merge(dict1, dict2)`: 创建一个新字典，包含 `dict1` 和 `dict2` 中的所有键值对。如果一个键同时存在于两个字典中，新字典将使用 `dict2`（第二个参数 (parameter)）中的值。原始字典不会被更改。

  ```julia
  julia> dict_a = Dict("a" => 1, "b" => 2)
  Dict{String, Int64} with 2 entries:
    "b" => 2
    "a" => 1

  julia> dict_b = Dict("b" => 3, "c" => 4) # "b" 也存在于 dict_a 中
  Dict{String, Int64} with 2 entries:
    "b" => 3
    "c" => 4

  julia> merged_dict = merge(dict_a, dict_b)
  Dict{String, Int64} with 3 entries:
    "b" => 3 # dict_b 中键 "b" 的值
    "c" => 4
    "a" => 1
  ```

  还有一个 `merge!(dict1, dict2)` 函数（带感叹号），它通过添加 `dict2` 中的条目来就地修改 `dict1`。

### 何时选择字典

字典是一种基本数据结构，在特定场景中尤其强大：

- **通过标识符快速查找**：当您需要基于独有标识符而非数据位置来快速获取数据时。例如，使用用户的独有用户名查找其个人资料信息。
- **关联数据**：当您想要存储数据对之间的关系或映射时。例子包括将单词映射到它们在文本文档中的频率，或将产品 ID 映射到它们的详细描述。
- **计数**：字典非常适合统计项目。您可以将项目本身用作键，将其计数用作值。
- **灵活的数据表示**：它们可以表示字段带有名称（键）的记录或对象。例如，像 `Dict("name" => "Julia", "version" => 1.9, "is_awesome" => true)` 这样的字典可以存储实体的各种属性。

与通过整数索引访问的有序序列——**数组**相比，字典（出于访问目的）是无序集合，通过各种类型（如字符串、数字或符号）的键进行访问。如果元素的顺序很重要，并且您通常按顺序或按其数字位置访问它们，那么数组通常是更好的选择。然而，如果您需要将独有键与值关联以便基于这些键快速直接获取，那么字典更适合。

让我们举例说明一个常见用例：统计单词列表中单词的频率。
假设我们有一个单词列表（数组）：

```julia
julia> words = ["hello", "hello", "julia", "hello"]
6-element Vector{String}:
 "hello"
 ""
 "hello"
 "julia"
 ""
 "hello"

julia> word_counts = Dict{String, Int}() # 为计数创建一个空字典
Dict{String, Int64}()

julia> for word in words
           # 如果 'word' 已经是键，获取其计数。否则，默认为 0。
           # 然后，将此计数加 1 并更新字典。
           word_counts[word] = get(word_counts, word, 0) + 1
       end

julia> word_counts
Dict{String, Int64} with 3 entries:
  "julia" => 1
  "global" => 2
  "hello" => 3
```

在此示例中，`get(word_counts, word, 0)` 部分很重要的。它安全地获取 `word` 的当前计数。如果 `word` 是第一次遇到（因此尚未在 `word_counts` 中），`get` 返回默认值 `0`。然后我们将此计数加 1，并使用 `word` 作为键将其存回字典中。这是使用字典统计项目的常见且高效的模式。

下图显示了字典的简单表示：

> 字典中的每个条目都包含一个独有的键，该键与其对应的值关联。访问值是通过其键完成的，而不是通过其位置。

字典提供了一种灵活高效的方式来管理带标签的数据。随着您继续使用 Julia 编程，您会发现它们是处理广泛任务的不可或缺的工具，从简单的数据存储到构成更复杂算法的一部分。它们通过提供一种不同的、基于键的方法来组织和访问您的信息集合，从而有效地补充了数组和元组。

## 参考资料

- [Introduction to Algorithms](https://mitpress.mit.edu/9780262046305/introduction-to-algorithms/) — Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein (2022)
  Publisher: MIT Press; Pages: 273-305 (Chapter 11: Hash Tables)
  一本关于数据结构和算法的权威教科书，详细解释了哈希表、其操作和性能特征。
