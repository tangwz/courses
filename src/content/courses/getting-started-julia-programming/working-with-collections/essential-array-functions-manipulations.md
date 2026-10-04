---
course: "getting-started-julia-programming"
chapter: "working-with-collections"
lesson: "essential-array-functions-manipulations"
sourceId: 6787
sourceUrl: "https://apxml.com/zh/courses/getting-started-julia-programming/chapter-4-working-with-collections/essential-array-functions-manipulations"
title: "数组的基本函数与操作"
description: "了解用于数组操作的常用函数和方法，例如排序、追加和筛选。"
order: 3
plots: []
sourceHash: "4857f3d1108424e30b178940650bffb55d9a770a0a60f27fe6125b9274dbb621"
sourceCorrections: []
---

操作数组，例如改变其内容、添加或移除元素，或重新组织，是一项常见任务。Julia 提供了全面的内置函数，用于高效地处理数组。这些函数中很多会直接修改数组。修改其参数 (parameter)的函数名称在 Julia 中通常以感叹号 (`!`) 结尾。这是一个有用的约定，表明该函数会改变所给数据，而不是返回一个新的修改后的副本。

### 向数组添加元素

使用新数据扩展数组是一种常见需求。Julia 提供了几种添加元素的方法：

- **`push!(array, item)`**: 将 `item` 添加到 `array` 的末尾。这是添加单个元素最常用的方法之一。

  ```julia
  julia> numbers = [10, 20, 30]
  3-element Vector{Int64}:
   10
   20
   30

  julia> push!(numbers, 40)
  4-element Vector{Int64}:
   10
   20
   30
   40

  julia> numbers
  4-element Vector{Int64}:
   10
   20
   30
   40
  ```

  注意 `numbers` 本身已发生改变。
- **`append!(array, collection)`**: 将 `collection` 中的所有元素添加到 `array` 的末尾。当你想将另一个数组（或任何可迭代集合）合并到现有数组时，这很有用。

  ```julia
  julia> list_a = [1, 2]
  2-element Vector{Int64}:
   1
   2

  julia> list_b = [3, 4, 5]
  3-element Vector{Int64}:
   3
   4
   5

  julia> append!(list_a, list_b)
  5-element Vector{Int64}:
   1
   2
   3
   4
   5

  julia> list_a
  5-element Vector{Int64}:
   1
   2
   3
   4
   5
  ```
- **`prepend!(array, collection)`**: 将 `collection` 中的所有元素添加到 `array` 的开头。

  ```julia
  julia> scores = [85, 92]
  2-element Vector{Int64}:
   85
   92

  julia> prepend!(scores, [70, 78])
  4-element Vector{Int64}:
   70
   78
   85
   92
  ```
- **`insert!(array, index, item)`**: 将 `item` 插入到指定的 `index` 处，将后续元素向右移动。

  ```julia
  julia> letters = ['a', 'c', 'd']
  3-element Vector{Char}:
   'a'
   'c'
   'd'

  julia> insert!(letters, 2, 'b') # 在索引 2 处插入 'b'
  4-element Vector{Char}:
   'a'
   'b'
   'c'
   'd'
  ```

下图说明了 `push!` 如何在数组末尾添加元素，以及 `insert!` 如何在特定位置添加元素。

> 该图显示了一个初始数组 `items`。`push!` 追加一个元素，而 `insert!` 在指定索引处放置新元素，同时移动现有元素。

### 从数组中移除元素

正如你可以添加元素一样，你也可以移除它们：

- **`pop!(array)`**: 移除并返回 `array` 中的最后一个元素。

  ```julia
  julia> stack = ["task1", "task2", "task3"]
  3-element Vector{String}:
   "task1"
   "task2"
   "task3"

  julia> last_task = pop!(stack)
  "task3"

  julia> println(last_task)
  task3

  julia> stack
  2-element Vector{String}:
   "task1"
   "task2"
  ```
- **`popfirst!(array)`**: 移除并返回 `array` 中的第一个元素。

  ```julia
  julia> queue = ["A", "B", "C"]
  3-element Vector{String}:
   "A"
   "B"
   "C"

  julia> first_item = popfirst!(queue)
  "A"

  julia> queue
  2-element Vector{String}:
   "B"
   "C"
  ```
- **`deleteat!(array, index_or_range)`**: 移除指定 `index` 处的元素或 `range` 内的元素。

  ```julia
  julia> data = [10, 20, 30, 40, 50]
  5-element Vector{Int64}:
   10
   20
   30
   40
   50

  julia> deleteat!(data, 3) # 移除索引 3 处的元素（即 30）
  4-element Vector{Int64}:
   10
   20
   40
   50

  julia> deleteat!(data, 2:3) # 移除索引 2 和 3 处的元素（现在是 20 和 40）
  2-element Vector{Int64}:
   10
   50
  ```
- **`empty!(array)`**: 移除 `array` 中的所有元素，使其变为空。

  ```julia
  julia> temp_list = [1, 2, 3]
  3-element Vector{Int64}:
   1
   2
   3

  julia> empty!(temp_list)
  0-element Vector{Int64}

  julia> temp_list
  0-element Vector{Int64}
  ```
- **`splice!(array, index_or_range, [replacement_collection])`**: 这是一个功能多样的函数。它移除 `index_or_range` 处的元素并返回它们。可以选择性地在该位置插入 `replacement_collection` 中的元素。

  ```julia
  julia> letters = ['a', 'b', 'x', 'y', 'c', 'd']
  6-element Vector{Char}:
   'a'
   'b'
   'x'
   'y'
   'c'
   'd'

  julia> removed = splice!(letters, 3:4) # 移除索引 3 和 4 处的元素
  2-element Vector{Char}:
   'x'
   'y'

  julia> letters # 'x' 和 'y' 已被移除
  4-element Vector{Char}:
   'a'
   'b'
   'c'
   'd'

  julia> values = [10, 20, 99, 100, 50]
  5-element Vector{Int64}:
   10
   20
   99
   100
   50

  julia> old_values = splice!(values, 3:4, [30, 40]) # 移除 99、100 并插入 30、40
  2-element Vector{Int64}:
   99
   100

  julia> values
  5-element Vector{Int64}:
   10
   20
   30
   40
   50
  ```

### 查询和检查数组

通常，你需要查找关于数组的信息而不改变它：

- **`length(array)`**: 返回 `array` 中的元素数量。

  ```julia
  julia> mixed_bag = [1, "hello", 3.14]
  3-element Vector{Any}:
   1
    "hello"
   3.14

  julia> length(mixed_bag)
  3
  ```
- **`size(array)`**: 返回一个包含 `array` 维度的元组。对于一维数组（向量 (vector)），它返回一个 1 元组 `(长度,)`。

  ```julia
  julia> vec = [1, 2, 3, 4]
  4-element Vector{Int64}:
   1
   2
   3
   4

  julia> size(vec)
  (4,)

  julia> matrix = [1 2; 3 4] # 一个 2x2 矩阵
  2×2 Matrix{Int64}:
   1  2
   3  4

  julia> size(matrix)
  (2, 2)
  ```
- **`isempty(array)`**: 如果 `array` 没有元素，则返回 `true`，否则返回 `false`。

  ```julia
  julia> empty_arr = []
  0-element Vector{Any}

  julia> isempty(empty_arr)
  true

  julia> non_empty_arr = [1]
  1-element Vector{Int64}:
   1

  julia> isempty(non_empty_arr)
  false
  ```
- **`in(item, collection)` (或 `item ∈ collection`)**: 检查 `item` 是否存在于 `collection` 中。返回 `true` 或 `false`。

  ```julia
  julia> fruits = ["apple", "banana", "cherry"]
  3-element Vector{String}:
   "apple"
   "banana"
   "cherry"

  julia> "banana" in fruits
  true

  julia> "grape" in fruits
  false

  julia> "apple" ∈ fruits # 你可以通过输入 \in 然后按 Tab 键来输入 ∈
  true
  ```
- **`findfirst(predicate, array)`**: 返回 `array` 中第一个使 `predicate` 函数返回 `true` 的元素的索引。如果未找到此类元素，则返回 `nothing`。判断函数只是一个接受一个元素并返回 `true` 或 `false` 的函数。

  ```julia
  julia> numbers = [10, 25, 30, 45, 50]
  5-element Vector{Int64}:
   10
   25
   30
   45
   50

  julia> findfirst(x -> x > 30, numbers) # 查找第一个大于 30 的数字
  4 # 45 的索引

  julia> findfirst(iseven, numbers) # `iseven` 是一个内置判断函数
  1 # 10 的索引

  julia> findfirst(x -> x > 100, numbers) # 不会打印任何内容，返回 `nothing`
  ```
- **`findall(predicate, array)`**: 返回一个包含 `array` 中所有使 `predicate` 函数返回 `true` 的元素索引的向量。

  ```julia
  julia> data = [1, 0, 2, 0, 3, 0, 4]
  7-element Vector{Int64}:
   1
   0
   2
   0
   3
   0
   4

  julia> findall(x -> x == 0, data) # 查找所有零
  3-element Vector{Int64}:
   2
   4
   6
  ```

### 转换和重排数组

你也可以根据现有数组创建新数组或重新排列它们：

- **`sort(array)` 和 `sort!(array)`**:

  - `sort(array)` 返回一个元素已排序的新数组。原始数组保持不变。
  - `sort!(array)` 对数组进行原地排序，修改原始数组。

  ```julia
  julia> unsorted_nums = [3, 1, 4, 1, 5, 9, 2, 6]
  8-element Vector{Int64}:
   3
   1
   4
   1
   5
   9
   2
   6

  julia> sorted_copy = sort(unsorted_nums)
  8-element Vector{Int64}:
   1
   1
   2
   3
   4
   5
   6
   9

  julia> unsorted_nums # 原始数组保持不变
  8-element Vector{Int64}:
   3
   1
   4
   1
   5
   9
   2
   6

  julia> sort!(unsorted_nums) # 原地排序
  8-element Vector{Int64}:
   1
   1
   2
   3
   4
   5
   6
   9

  julia> unsorted_nums # 原始数组现在已排序
  8-element Vector{Int64}:
   1
   1
   2
   3
   4
   5
   6
   9
  ```

  你也可以按倒序排序：`sort(array, rev=true)`。
- **`reverse(array)` 和 `reverse!(array)`**:

  - `reverse(array)` 返回一个元素倒序排列的新数组。
  - `reverse!(array)` 对数组进行原地倒序排列。

  ```julia
  julia> items = [10, 20, 30, 40]
  4-element Vector{Int64}:
   10
   20
   30
   40

  julia> reversed_copy = reverse(items)
  4-element Vector{Int64}:
   40
   30
   20
   10

  julia> items # 原始数组保持不变
  4-element Vector{Int64}:
   10
   20
   30
   40

  julia> reverse!(items)
  4-element Vector{Int64}:
   40
   30
   20
   10

  julia> items # 原始数组现在已倒序
  4-element Vector{Int64}:
   40
   30
   20
   10
  ```
- **`map(function, array)`**: 将 `function` 应用于 `array` 的每个元素，并返回一个包含结果的新数组。

  ```julia
  julia> numbers = [1, 2, 3, 4]
  4-element Vector{Int64}:
   1
   2
   3
   4

  julia> squares = map(x -> x^2, numbers)
  4-element Vector{Int64}:
    1
    4
    9
   16
  ```
- **`filter(predicate, array)`**: 返回一个新数组，其中仅包含 `array` 中使 `predicate` 返回 `true` 的元素。

  ```julia
  julia> values = [10, 15, 20, 25, 30, 35]
  6-element Vector{Int64}:
   10
   15
   20
   25
   30
   35

  julia> even_values = filter(iseven, values)
  3-element Vector{Int64}:
   10
   20
   30
  ```

### 组合数组

Julia 提供了便捷的方法来组合多个数组：

- **`vcat(arrays...)`**: 垂直连接数组。对于向量 (vector)（一维数组），这会将它们一个接一个地堆叠成一个新的、更长的向量。你也可以使用分号语法 `[array1; array2; ...]` 来实现。

  ```julia
  julia> arr1 = [1, 2]
  2-element Vector{Int64}:
   1
   2

  julia> arr2 = [3, 4, 5]
  3-element Vector{Int64}:
   3
   4
   5

  julia> combined_v = vcat(arr1, arr2)
  5-element Vector{Int64}:
   1
   2
   3
   4
   5

  julia> also_combined_v = [arr1; arr2] # 与 vcat 相同
  5-element Vector{Int64}:
   1
   2
   3
   4
   5
  ```
- **`hcat(arrays...)`**: 水平连接数组。对于向量，这会将它们并排放置以形成一个二维数组（矩阵）。语法 `[array1 array2 ...]`（使用空格）也执行水平连接。

  ```julia
  julia> col1 = [10, 20]
  2-element Vector{Int64}:
   10
   20

  julia> col2 = [30, 40]
  2-element Vector{Int64}:
   30
   40

  julia> combined_h = hcat(col1, col2)
  2×2 Matrix{Int64}:
   10  30
   20  40

  julia> also_combined_h = [col1 col2] # 与 hcat 相同
  2×2 Matrix{Int64}:
   10  30
   20  40
  ```

  > 注意：如果将 `hcat` 或空格分隔语法与不同长度的一维数组一起使用，你将会遇到错误。它们必须具有兼容的维度才能水平连接成矩阵。

### 复制数组: `copy` 和 `deepcopy`

当你将一个数组赋值给一个新变量时，你并没有创建一个新数组。相反，两个变量都指向内存中的同一个数组。通过一个变量修改数组会影响另一个变量。

```julia
julia> original_list = [1, 2, 3]
3-element Vector{Int64}:
 1
 2
 3

julia> same_list = original_list # 两个变量都指向同一个数组
3-element Vector{Int64}:
 1
 2
 3

julia> push!(same_list, 4)
4-element Vector{Int64}:
 1
 2
 3
 4

julia> original_list # original_list 也被修改了！
4-element Vector{Int64}:
 1
 2
 3
 4
```

为了创建一个独立的副本，Julia 提供了 `copy` 和 `deepcopy`：

- **`copy(array)`**: 创建一个*浅拷贝*。新数组是一个不同的对象，但如果数组包含其他可变集合（例如数组中的数组），那些内部集合仍然是共享的。对于数字或字符串等简单类型的数组，`copy` 通常足够了。

  ```julia
  julia> a = [1, 2, 3]
  3-element Vector{Int64}:
   1
   2
   3

  julia> b = copy(a)
  3-element Vector{Int64}:
   1
   2
   3

  julia> push!(b, 4)
  4-element Vector{Int64}:
   1
   2
   3
   4

  julia> a # `a` 未被修改
  3-element Vector{Int64}:
   1
   2
   3
  ```
- **`deepcopy(array)`**: 创建一个*深拷贝*。新数组及其包含的所有内容（包括嵌套数组或其他可变对象）都会被递归复制。这确保了完全的独立性。

  ```julia
  julia> nested_original = [[1, 2], [3, 4]]
  2-element Vector{Vector{Int64}}:
   [1, 2]
   [3, 4]

  julia> shallow_copy = copy(nested_original)
  2-element Vector{Vector{Int64}}:
   [1, 2]
   [3, 4]

  julia> deep_copied = deepcopy(nested_original)
  2-element Vector{Vector{Int64}}:
   [1, 2]
   [3, 4]

  # 通过浅拷贝修改内部数组
  julia> push!(shallow_copy[1], 99)
  3-element Vector{Int64}:
    1
    2
   99

  julia> nested_original # 原始数组受到浅拷贝修改的影响！
  2-element Vector{Vector{Int64}}:
   [1, 2, 99]
   [3, 4]

  julia> deep_copied # 深拷贝保持独立
  2-element Vector{Vector{Int64}}:
   [1, 2]
   [3, 4]
  ```

了解浅拷贝和深拷贝之间的区别很重要，可以避免在处理包含其他可变结构的数组时出现意外的副作用。

这些函数为操作数组提供了强大的工具包。随着你更多地使用 Julia，你会发现这些操作在清理、转换和准备数据时变得轻而易举。

## 参考资料

- [Interface for Mutable Collections](https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-mutable-collections) — The Julia Language (2023)
  Publisher: The Julia Language
  详细介绍了原位修改数组的函数，如 `push!`、`append!`、`pop!`、`deleteat!` 和 `splice!`，并解释了 Julia 的修改函数命名约定。
- [Multi-dimensional Arrays](https://docs.julialang.org/en/v1/manual/arrays/) — The Julia Language (2023)
  涵盖了基本的数组操作，包括查询数组维度（`length`、`size`、`isempty`）、搜索元素（`findfirst`、`findall`）、组合数组（`vcat`、`hcat`）以及 `map` 和 `filter` 等通用转换。
- [Think Julia: How to Think Like a Computer Scientist](https://benlauwens.github.io/ThinkJulia.jl/stable/) — Ben Lauwens, Allen B. Downey (2019)
  Publisher: O'Reilly Media
  一本入门级教材，提供了 Julia 编程的基础知识，其中包含专门介绍数组及其操作的章节，适合初学者使用。
