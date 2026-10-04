---
course: "getting-started-julia-programming"
chapter: "working-with-collections"
lesson: "practice-managing-data-julia-collections"
sourceId: 6792
sourceUrl: "https://apxml.com/zh/courses/getting-started-julia-programming/chapter-4-working-with-collections/practice-managing-data-julia-collections"
title: "实践：使用 Julia 集合管理数据"
description: "动手示例，练习在 Julia 中创建、操作和使用数组、元组、字典和集合。"
order: 8
plots: ["plots/6792-0.json"]
sourceHash: "18a973a6d938ee83124d5ff84330a3a2581ce9da6a93fab3b115e214675878f2"
sourceCorrections: []
---

Julia 提供了多功能的集合类型：数组、元组、字典和集合，以及用于创建它们的简洁推导式语法。以下练习提供了创建、操作和使用这些集合进行各种数据管理任务的实践。建议将这些示例输入到 Julia REPL 或脚本文件中，以查看它们的效果。

### 处理数组

数组是有序的、可变动的集合，这使得它们非常适合需要修改或扩展的项列表。

**1. 创建和初始化数组**

Julia 提供多种方式来创建数组。让我们尝试一些：

```julia
# 一个空数组，专门用于整数
empty_int_array = Int[]
println("Empty Integer Array: ", empty_int_array)

# 一个带有初始值的数组；Julia 会推断类型
numbers = [10, 20, 30, 40, 50]
println("Numbers Array: ", numbers)
println("Type of numbers array: ", typeof(numbers)) # 显示 Array{Int64, 1}

# 一个带有初始值的特定类型（Float64）数组
float_numbers = Float64[1.5, 2.5, 3.5]
println("Floating-Point Numbers Array: ", float_numbers)

# 一个可以容纳任何类型值的数组
mixed_array = [1, "hello Julia", 3.14, true]
println("Mixed Type Array: ", mixed_array)
println("Type of mixed_array: ", typeof(mixed_array)) # 显示 Array{Any, 1}

# 使用函数创建预填充数组
zeros_array = zeros(Int8, 3) # 创建一个包含三个 Int8 零的数组：[0, 0, 0]
println("Zeros Array (Int8): ", zeros_array)

ones_array = ones(Float32, 2, 3) # 创建一个 2x3 的 Float32 矩阵，全部为一
println("Ones Array (2x3 Float32):")
display(ones_array) # 对于多维数组，`display` 通常更好
```

**2. 访问和修改元素**

请记住，Julia 对数组使用 1-基于索引。

```julia
primes = [2, 3, 5, 7, 11, 13]

# 访问第一个元素
first_prime = primes[1]
println("First prime: ", first_prime) # 输出：2

# 访问第三个元素
third_prime = primes[3]
println("Third prime: ", third_prime) # 输出：5

# 使用 `end` 访问最后一个元素
last_prime = primes[end]
println("Last prime: ", last_prime) # 输出：13

# 修改元素
println("Original primes: ", primes)
primes[3] = 99 # 让我们修改第三个元素
println("Modified primes (element 3 changed): ", primes)
primes[3] = 5 # 改回来
println("Corrected primes: ", primes)
```

**3. 常用数组操作**

数组附带许多有用的操作函数。约定上，以 `!` 结尾的函数会原地修改数组。

```julia
inventory = ["apples", "bananas"]
println("Initial inventory: ", inventory)

# 在数组末尾添加元素
push!(inventory, "oranges")
println("After push! 'oranges': ", inventory)

# 从末尾移除元素并获取其值
last_item = pop!(inventory)
println("Removed item (pop!): ", last_item)
println("Inventory after pop!: ", inventory)

# 在数组开头添加元素
pushfirst!(inventory, "strawberries")
println("After pushfirst! 'strawberries': ", inventory)

# 从开头移除元素并获取其值
first_item = popfirst!(inventory)
println("Removed item (popfirst!): ", first_item)
println("Inventory after popfirst!: ", inventory)

# 获取数组中的元素数量
println("Number of items in inventory: ", length(inventory))

# 排序数组（原地修改）
unsorted_scores = [88, 75, 92, 60, 95]
println("Unsorted scores: ", unsorted_scores)
sort!(unsorted_scores)
println("Sorted scores: ", unsorted_scores)

# 从另一个数组追加元素（修改第一个数组）
more_fruits = ["grapes", "mangoes"]
append!(inventory, more_fruits) # 'inventory' 被修改
println("Inventory after append!: ", inventory)

# 要组合数组而不修改原数组，请使用 `vcat`
list_a = [10, 20]
list_b = [30, 40]
combined_list = vcat(list_a, list_b)
println("List A: ", list_a) # 仍然是 [10, 20]
println("List B: ", list_b) # 仍然是 [30, 40]
println("Combined list (new array): ", combined_list)
```

### 练习使用元组

元组是有序的、不可变序列。它们的固定性质使其适合表示创建后结构和内容不应更改的数据，例如坐标或固定记录。

**1. 创建元组**

```julia
# 表示二维点的元组
point_2d = (10, 20)
println("2D Point: ", point_2d)

# 用于 RGB 颜色值的元组
red_color = (255, 0, 0)
println("Red color (RGB): ", red_color)

# 包含混合数据类型的元组
record = ("John Doe", 34, "New York")
println("Record: ", record)

# 访问元素（1-基于索引，就像数组一样）
name = record[1]
age = record[2]
city = record[3]
println("$name, aged $age, lives in $city.")

# 元组解包，便于赋值
x, y = point_2d
println("x-coordinate: $x, y-coordinate: $y")
```

**2. 演示不可变性**

尝试更改元组中的元素将导致错误，这表明了它们的不可变属性。

```julia
fixed_settings = (true, "fast_mode", 1024)
println("Fixed settings: ", fixed_settings)

# 尝试更改第一个元素：
# fixed_settings[1] = false # 如果取消注释，此行将导致错误
# ERROR: MethodError: no method matching setindex!(::Tuple{Bool, String, Int64}, ::Bool, ::Int64)
```

这种行为是设计使然。如果你需要一个可以修改的集合，数组是合适的选择。

### 了解字典

字典将数据存储为键值对，如果你知道键，可以高效地检索数据。键必须是唯一的。

**1. 创建和填充字典**

```julia
# 创建一个空字典：方法 1
city_populations = Dict{String, Int}()

# 添加键值对
city_populations["New York"] = 8_400_000 # 数字中的下划线是为了提高可读性
city_populations["Los Angeles"] = 3_900_000
city_populations["Chicago"] = 2_700_000
println("City Populations: ", city_populations)

# 使用初始对创建字典：方法 2
element_symbols = Dict("H" => "Hydrogen", "O" => "Oxygen", "C" => "Carbon")
println("Element Symbols: ", element_symbols)
```

**2. 访问、修改和管理条目**

```julia
# 使用其键访问值
oxygen_name = element_symbols["O"]
println("Symbol 'O' stands for: ", oxygen_name)

# 修改现有值
println("Population of Chicago: ", city_populations["Chicago"])
city_populations["Chicago"] = 2_750_000 # 更新人口
println("Updated population of Chicago: ", city_populations["Chicago"])

# 检查键是否存在
has_helium = haskey(element_symbols, "He")
println("Does 'He' exist as a key? ", has_helium) # 输出：false

has_carbon = haskey(element_symbols, "C")
println("Does 'C' exist as a key? ", has_carbon)   # 输出：true

# 使用 `get` 获取值，如果未找到键则提供默认值
helium_name = get(element_symbols, "He", "Symbol not found")
println("Name for 'He': ", helium_name)

carbon_name_get = get(element_symbols, "C", "Symbol not found")
println("Name for 'C' (using get): ", carbon_name_get)

# 移除键值对
delete!(element_symbols, "H")
println("Element Symbols after deleting 'H': ", element_symbols)
```

**3. 遍历字典**

你可以遍历字典的键、值或键值对。

```julia
fruit_prices = Dict("apple" => 0.50, "banana" => 0.25, "orange" => 0.75)

println("\nAvailable Fruits (Keys):")
for fruit in keys(fruit_prices)
    println(fruit)
end

println("\nFruit Prices (Values):")
for price in values(fruit_prices)
    println(price)
end

println("\nComplete Price List (Key-Value Pairs):")
for (fruit, price) in fruit_prices
    println("$fruit costs \$$price")
end
```

让我们将水果价格可视化。



![水果价格](plots/6792-0.json)



> 一个柱状图，显示了不同水果的每单位价格。

### 理解集合

集合是无序的唯一元素集合。它们对于检查成员资格、从列表中移除重复项以及执行标准集合理论操作（并集、交集、差集）特别有用。

**1. 创建集合和添加元素**

```julia
# 创建一个空集合，指定元素类型
unique_tags = Set{String}()

# 向集合添加元素
push!(unique_tags, "julia")
push!(unique_tags, "programming")
push!(unique_tags, "data science")
push!(unique_tags, "julia") # 再次添加 "julia" 没有效果，因为元素必须是唯一的
println("Unique Tags: ", unique_tags) # 不保证元素的顺序

# 使用初始元素创建集合
numbers_set = Set([1, 2, 2, 3, 4, 4, 4, 5])
println("Numbers Set (duplicates removed): ", numbers_set) # 输出：Set([5, 2, 3, 1, 4]) 或类似结果
```

**2. 集合操作**

让我们看下常用集合操作。

```julia
set_europe = Set(["France", "Germany", "Spain"])
set_g7 = Set(["USA", "Canada", "France", "Germany", "UK", "Italy", "Japan"])

# 并集：两个集合中所有不重复的国家
union_countries = union(set_europe, set_g7)
println("Union (Europe or G7): ", union_countries)

# 交集：既在欧洲又在七国集团中的国家
intersection_countries = intersect(set_europe, set_g7)
println("Intersection (European G7 members): ", intersection_countries)

# 差集：在 set_europe 中但不在 set_g7 中的国家
europe_not_g7 = setdiff(set_europe, set_g7)
println("Difference (European countries not in G7): ", europe_not_g7)

# 差集：在 set_g7 中但不在 set_europe 中的国家
g7_not_europe = setdiff(set_g7, set_europe)
println("Difference (G7 countries not in Europe): ", g7_not_europe)

# 检查成员资格（元素是否在集合中？）
is_spain_in_europe = "Spain" in set_europe
println("Is Spain in the European set? ", is_spain_in_europe) # 输出：true

is_brazil_in_g7 = "Brazil" in set_g7
println("Is Brazil in the G7 set? ", is_brazil_in_g7)       # 输出：false
```

### 使用推导式高效创建集合

推导式提供了一种简洁易读的方式来根据现有可迭代对象或范围创建集合，特别是数组和字典。

**1. 数组推导式**

```julia
# 创建一个包含前 5 个偶数（从 2 开始）的数组
first_five_evens = [2 * i for i in 1:5]
println("First five evens: ", first_five_evens) # 输出：[2, 4, 6, 8, 10]

# 从字符串创建大写字符数组
name_str = "julia"
name_chars_upper = [uppercase(char) for char in name_str]
println("Uppercase characters: ", name_chars_upper) # 输出：['J', 'U', 'L', 'I', 'A']

# 带有条件的数组推导式：1 到 10 中奇数的平方
odd_squares = [i^2 for i in 1:10 if i % 2 != 0]
println("Squares of odd numbers: ", odd_squares) # 输出：[1, 9, 25, 49, 81]
```

**2. 字典推导式**

推导式也可以用来创建字典。

```julia
# 创建一个将数字（1 到 4）映射到其字符串表示的字典
num_to_string_dict = Dict(i => string(i) for i in 1:4)
println("Number to String Dictionary: ", num_to_string_dict)
# 输出：Dict(4 => "4", 2 => "2", 3 => "3", 1 => "1")（顺序可能不同）

# 从两个列表（一个用于键，一个用于值）创建字典
student_names = ["Alice", "Bob", "Charlie"]
scores = [85, 92, 78]
student_score_dict = Dict(student_names[i] => scores[i] for i in 1:length(student_names))
println("Student Score Dictionary: ", student_score_dict)
# 输出：Dict("Bob" => 92, "Charlie" => 78, "Alice" => 85)（顺序可能不同）
```

**练习任务：分析文本数据**

让我们尝试一个结合了这些想法的小任务。假设你有一段文本，并且想要：

1. 找到所有不重复的词（忽略大小写）。
2. 计算每个词的出现频率。

```julia
text_sample = "The quick brown fox jumps over the lazy dog. The dog barks."

# 1. 预处理：转换为小写并分割成单词。
# 为获得更清晰的单词列表而去除标点符号（此处采用简单方法）。
words_raw = split(lowercase(replace(text_sample, "." => "")))
println("Raw words: ", words_raw)

# 2. 使用 Set 找到不重复的单词
unique_words = Set(words_raw)
println("\nUnique words: ", unique_words)
println("Number of unique words: ", length(unique_words))

# 3. 使用 Dictionary 计算单词频率
word_frequencies = Dict{String, Int}()
for word in words_raw
    # 如果单词在字典中，增加计数；否则，以计数 1 添加它
    word_frequencies[word] = get(word_frequencies, word, 0) + 1
end

println("\nWord Frequencies:")
for (word, count) in word_frequencies
    if count > 0 # 只是为了确保如果过滤掉了一些单词，我们仍然打印实际单词
        println("'$word': $count")
    end

end
```

这个示例展示了数组（来自 `split`）、集合（用于唯一性）和字典（用于频率计数）如何共同处理数据。

这些练习涵盖了 Julia 主要集合类型的基本操作。随着你对它们更加熟悉，你会发现它们对于 Julia 程序中数据结构化和管理不可或缺。学习的最佳方式是实践，所以尝试修改这些示例或创建自己的小项目，以进一步查看它们的功能。

## 参考资料

- [Julia Language Manual: Collections](https://docs.julialang.org/en/v1.11/manual/collections/) — The Julia Language Developers (2024)
  Publisher: The Julia Language
  Julia核心集合类型（数组、元组、字典、集合）及其操作（包括推导式）的官方权威指南。
- [Think Julia: How to Think Like a Computer Scientist](https://onlinebooks.library.upenn.edu/webbin/book/lookupid?key=olbp81797) — Ben Lauwens, Allen Downey (2019)
  Publisher: O'Reilly Media; Pages: 295
  一本关于Julia编程的入门书籍，其中包含专门的章节，通过大量实用示例来解释数组、元组、字典和集合。
- [Julia for Data Science (Course)](https://juliaacademy.com/p/julia-for-data-science) — Dr Huda Nassar (2024)
  Publisher: Julia Academy
  Julia官方学院的在线课程，提供关于Julia集合类型及其在数据管理中应用的互动课程。
