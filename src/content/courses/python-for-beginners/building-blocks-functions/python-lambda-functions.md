---
course: "python-for-beginners"
chapter: "building-blocks-functions"
lesson: "python-lambda-functions"
sourceId: 2338
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-5-building-blocks-functions/python-lambda-functions"
title: "Lambda 函数：简洁的匿名函数"
description: "了解 lambda 函数，这是一种无需使用 `def` 关键字即可内联创建小型匿名函数的方法。"
order: 7
plots: []
sourceHash: "12c9b21d5104231a75c0edc42f01e5387f4e79e59337a4bc9ecff8a6bf814df8"
sourceCorrections: []
---

有时，您需要一个简单的函数，用于一个非常具体、生命周期短暂的用途。使用 `def` 定义一个完整的函数可能会显得多余，特别是如果您只打算在一个地方使用它，例如将其作为参数 (parameter)传递给另一个函数时。Python 提供了一种使用 `lambda` 关键字内联创建小型、未命名（匿名）函数的方法。

可以将 lambda 函数视为创建由单个表达式定义的简单函数的捷径。它们在处理那些操作其他函数（例如 `map()`、`filter()` 或 `sorted()`）的函数时，尤其有用。

### Lambda 函数的语法

`lambda` 函数的结构很简单：

```python
lambda arguments: expression
```

- **`lambda`**: 表示您正在创建一个匿名函数的关键字。
- **`arguments`**: 一个或多个参数 (parameter)名，用逗号分隔，类似于 `def` 函数定义中的参数。
- **`:`**: 用于将参数与表达式分隔开的冒号。
- **`expression`**: 一个单独的表达式，在调用 lambda 函数时进行求值。此表达式的结果会自动返回。Lambda 函数不能包含复杂的语句，例如循环、`if`/`else` 块（尽管允许条件*表达式*），或在其主体内进行赋值。

### `lambda` 与 `def` 的比较

让我们来看一个标准函数及其等效的 lambda 函数。假设我们想要一个函数来添加两个数字：

```python
# 使用标准函数定义
def add(x, y):
  return x + y

# 使用 lambda 函数
add_lambda = lambda x, y: x + y

# 两者可以用相同的方式调用
result1 = add(10, 5)
result2 = add_lambda(10, 5)

print(f"Result from def function: {result1}")  # 输出：来自 def 函数的结果：15
print(f"Result from lambda function: {result2}") # 输出：来自 lambda 函数的结果：15
```

在此示例中，`add_lambda` 是一个存储 lambda 函数的变量。虽然您*可以*将 lambda 函数这样赋值给一个变量，但这在一定程度上失去了它们作为匿名函数的意义。它们的主要优势在于直接在需要的地方定义，通常作为其他函数的参数 (parameter)。

### Lambda 函数的常见用途

Lambda 函数在与接受另一个函数作为参数 (parameter)的函数（通常称为高阶函数）一起使用时表现出色。

#### 将 `lambda` 与 `map()` 结合使用

`map()` 函数将给定函数应用于可迭代对象（如列表）的每个元素，并返回一个 `map` 对象（可以转换为列表）。Lambda 函数非常适合简洁地为 `map()` 提供函数参数。

```python
numbers = [1, 2, 3, 4, 5]

# 使用 map 和 lambda 函数计算每个数字的平方
squared_numbers = map(lambda x: x * x, numbers)

# 将 map 对象转换为列表以查看结果
print(list(squared_numbers))  # 输出：[1, 4, 9, 16, 25]
```

此处，`lambda x: x * x` 是传递给 `map()` 的匿名函数。它接受一个参数 `x` 并返回 `x * x`。`map()` 将此 lambda 应用于 `numbers` 列表中的每个元素。

#### 将 `lambda` 与 `filter()` 结合使用

`filter()` 函数从可迭代对象的元素中构建一个迭代器，其中函数返回 `True` 的元素将被保留。Lambda 函数非常适合定义过滤条件。

```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 使用 filter 和 lambda 函数只获取偶数
even_numbers = filter(lambda x: x % 2 == 0, numbers)

# 将 filter 对象转换为列表
print(list(even_numbers)) # 输出：[2, 4, 6, 8, 10]
```

Lambda 表达式 `lambda x: x % 2 == 0` 检查一个数字是否为偶数（如果是则返回 `True`，否则返回 `False`）。`filter()` 使用此条件从 `numbers` 列表中选择元素。

#### 将 `lambda` 与 `sorted()` 结合使用

`sorted()` 函数可以接受一个可选的 `key` 参数，它是一个用于从每个元素中提取比较依据的函数。Lambda 函数经常在这里用于指定简单的排序规则。

```python
# 元组列表（姓名，年龄）
people = [('Alice', 30), ('Bob', 25), ('Charlie', 35)]

# 按年龄（每个元组的第二个元素）对列表进行排序
sorted_by_age = sorted(people, key=lambda person: person[1])

print(sorted_by_age) # 输出：[('Bob', 25), ('Alice', 30), ('Charlie', 35)]
```

Lambda 表达式 `lambda person: person[1]` 接受一个元组 `person` 并返回其第二个元素（`person[1]`，即年龄）。`sorted()` 使用这些年龄作为列表的排序依据。

### 局限性和可读性

尽管 lambda 函数提供了简洁性，但它们也有一些局限：

- **单个表达式：** 它们仅限于单个表达式。您不能包含多个语句、循环、复杂的条件逻辑（除了 `x if condition else y` 这样的简单条件表达式），或 `try`/`except` 块。
- **可读性：** 对于复杂的操作，一个带有描述性名称和可能包含注释的标准 `def` 函数通常比一个复杂晦涩的 lambda 函数更具可读性和可维护性。

当操作简单且简洁性能够提高清晰度时，通常在将函数作为参数 (parameter)传递时，使用 lambda 函数。如果逻辑开始变得稍微复杂，则فضل使用 `def` 定义常规函数。

总之，lambda 函数提供了一种紧凑的语法，用于创建由单个表达式定义的小型匿名函数。它们最有效的使用方式是作为高阶函数（如 `map()`、`filter()` 和 `sorted()`）的参数，让您能够直接在需要的地方定义简单操作，而无需完整的 `def` 语句的繁琐。

## 参考资料

- [Built-in Functions (map, filter, sorted)](https://docs.python.org/3/library/functions.html) — Python Software Foundation (2024)
  Publisher: Python Software Foundation
  Python 官方文档，介绍了 map()、filter() 和 sorted() 等高阶函数，它们是 lambda 函数的常见应用场景。
- [Fluent Python: Clear, Concise, and Effective Programming](https://www.porchlightbooks.com/product/fluent-python-clear-concise-and-effective-programming-second-edition:an-o%27reilly-book) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media
  一本全面指南，涵盖 Python 的函数式编程特性，包含使用 lambda 函数的实际示例和考量。
