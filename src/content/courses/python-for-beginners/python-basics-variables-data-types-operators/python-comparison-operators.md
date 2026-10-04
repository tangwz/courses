---
course: "python-for-beginners"
chapter: "python-basics-variables-data-types-operators"
lesson: "python-comparison-operators"
sourceId: 2274
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-2-python-basics-variables-data-types-operators/python-comparison-operators"
title: "使用运算符：比较"
description: "了解如何使用 `==`、`!=`、`<`、`>`、`<=`、`>=` 等运算符在 Python 中比较值。"
order: 6
plots: []
sourceHash: "687c323c071dc0aeb2c94606de87e3cd58c502f5c5c34b32caeeefc79567fa93"
sourceCorrections: []
---

编程经常需要比较值来做出决策。就像我们使用`+`和`-`等算术运算符进行计算一样，Python 提供了*比较运算符*（也称为*关系运算符*）来检查两个值之间的关系。这些比较对于控制程序流程非常重要，能让脚本根据特定条件做出不同的反应。

任何比较操作的结果总是布尔值：`True` 或 `False`。可以将其看作是向 Python 询问关于两段数据之间关系的‘是’或‘否’问题。下面我们来看看 Python 中可用的比较运算符。

### 等于：`==`

此运算符检查其左侧和右侧的值是否相等。区分比较运算符 `==`（两个等号）和赋值运算符 `=`（一个等号）很重要，后者用于给变量赋值。对于新程序员来说，在需要比较的地方使用单个 `=` 是一个常见的错误来源。

```python
# 比较数字
print(5 == 5)
print(10 == 7)

# 比较字符串
print("hello" == "hello")
print("Python" == "python") # 区分大小写！

# 将结果赋值给变量
are_numbers_equal = (100 == 100)
print(are_numbers_equal)
```

运行此代码将输出：

```
True
False
True
False
True
```

### 不等于：`!=`

此运算符检查其左侧和右侧的值是否*不*相等。它是 `==` 运算符的直接反面。

```python
# 比较数字
print(5 != 5)
print(10 != 7)

# 比较字符串
print("hello" != "")
print("Python" != "Python")

# 赋值结果
are_different = ("apple" != "orange")
print(are_different)
```

输出：

```
False
True
True
False
True
```

### 小于：`<`

此运算符检查其左侧的值是否严格小于其右侧的值。

```python
print(5 < 10)
print(10 < 5)
print(5 < 5) # 不小于，所以为 False

# 字符串按字典顺序比较（如同字典排列）
print("apple" < "banana")
print("cat" < "car") # 't' 在 'r' 之后
```

输出：

```
True
False
False
True
False
```

### 大于：`>`

此运算符检查其左侧的值是否严格大于其右侧的值。

```python
print(10 > 5)
print(5 > 10)
print(5 > 5) # 不大于，所以为 False

print("zebra" > "apple")
```

输出：

```
True
False
False
True
```

### 小于或等于：`<=`

此运算符检查其左侧的值是否小于*或*等于其右侧的值。

```python
print(5 <= 10)
print(10 <= 5)
print(5 <= 5) # 等于，所以为 True
```

输出：

```
True
False
True
```

### 大于或等于：`>=`

此运算符检查其左侧的值是否大于*或*等于其右侧的值。

```python
print(10 >= 5)
print(5 >= 10)
print(5 >= 5) # 等于，所以为 True
```

输出：

```
True
False
True
```

### 比较不同数值类型

Python 在比较不同数值类型（如整数和浮点数）时很灵活。它通常在比较前将它们转换为通用类型。

```python
print(5 == 5.0)
print(10 > 9.99)
print(3 <= 3.0)
```

输出：

```
True
True
True
```

### 比较运算符总结

以下是简要总结表：

| 运算符 | 含义 | 示例 | 结果 |
| --- | --- | --- | --- |
| `==` | 等于 | `5 == 5` | `True` |
| `!=` | 不等于 | `5 != 6` | `True` |
| `<` | 小于 | `5 < 10` | `True` |
| `>` | 大于 | `10 > 5` | `True` |
| `<=` | 小于或等于 | `5 <= 5` | `True` |
| `>=` | 大于或等于 | `10 >= 5` | `True` |

理解这些运算符非常重要，因为它们是程序中决策的根本。在下一章“控制程序流程”中，你将看到这些比较产生的 `True` 或 `False` 结果如何用于 `if` 语句和循环中，以指导程序执行。

## 参考资料

- [Comparisons](https://docs.python.org/3/reference/expressions.html#comparisons) — Python Software Foundation (2023)
  Publisher: Python Software Foundation
  Python 语言官方参考，详细说明了比较运算符的行为和语法。
- [Python Crash Course, 3rd Edition: A Hands-On, Project-Based Introduction to Programming](https://nostarch.com/python-crash-course-3rd-edition/) — Eric Matthes (2022)
  Publisher: No Starch Press
  一本广泛推荐的 Python 初学者书籍，清晰地解释了运算符（包括比较运算符）及其在编程逻辑中的应用。
