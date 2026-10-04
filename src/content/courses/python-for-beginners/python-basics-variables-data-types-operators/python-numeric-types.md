---
course: "python-for-beginners"
chapter: "python-basics-variables-data-types-operators"
lesson: "python-numeric-types"
sourceId: 2263
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-2-python-basics-variables-data-types-operators/python-numeric-types"
title: "基本数据类型：数字（整数，浮点数）"
description: "理解 Python 的内置数字类型，包括整数（整型数字）和浮点数（小数）。"
order: 2
plots: []
sourceHash: "d970065953443f570007935db06af7c6d95071d66b31992978deba2f63d2d49f"
sourceCorrections: []
---

计算的核心在于处理数字的能力。无论是计数、计算距离还是测量物理属性，数字都不可或缺。Python 内置支持多种数字类型，但您会经常遇到的两种最基本类型是整数和浮点数。

### 整数 (int)

整数表示整数。它们可以是正数、负数或零，但不能有小数部分。可以把它们看作是用来计数离散项的数字：例如用户数量、循环步数或游戏分数。

在 Python 中，您只需写下不带小数点的数字即可创建一个整数：

```python
# 整数示例
user_count = 150
temperature_celsius = -5
year = 2024
zero_value = 0

print(user_count)
print(temperature_celsius)
```

Python 整数可以任意大，只受限于系统可用内存。您不必像在其他编程语言中那样担心 `short`、`int` 或 `long` 等不同的整数大小。

您可以使用内置的 `type()` 函数确认数字的类型：

```python
age = 30
print(type(age)) # 输出：<class 'int'>
```

### 浮点数 (float)

浮点数表示实数，包括带有小数部分的数字。在处理测量、百分比或任何需要非整数精度的值时，它们非常必要。

您可以通过包含小数点或使用科学计数法（e 符号）来创建一个浮点数。

```python
# 浮点数示例
pi_approx = 3.14159
price = 49.99
temperature_fahrenheit = -20.5
account_balance = 0.0 # 使用小数点使其成为浮点数
distance_meters = 1.5e3 # 科学计数法表示 1.5 * 10^3，即 1500.0

print(pi_approx)
print(distance_meters)
```

即使数字表示一个整数值，包含小数点也会使 Python 将其视为浮点数：

```python
whole_number_float = 10.0
print(whole_number_float)      # 输出：10.0
print(type(whole_number_float)) # 输出：<class 'float'>
```

### 整数与浮点数的选择

- 当您需要精确的整数时，请使用 `int`：例如计数、索引、标识符。
- 在处理测量、分数、概率或计算可能产生非整数结果时，请使用 `float`。

了解计算机上的浮点运算并非总是完全精确很重要。由于这些数字在内部的存储方式（通常使用 IEEE 754 等二进制格式），涉及浮点数的计算有时会产生与您数学预期略有偏差的结果。例如，$0.1 + 0.2$ 的结果可能是 $0.30000000000000004$，而不是精确的 $0.3$。对于大多数应用而言，这种微小的差异可以忽略不计，但需要注意这一点，特别是在需要高精度的财务计算或科学模拟中（Python 为此类情况提供了其他工具，例如 `Decimal` 类型，但这超出了我们当前的讨论范围）。

了解整数和浮点数是使用 Python 进行几乎任何计算或定量分析的基础。随着学习的进行，您将看到这些数字类型如何与运算符进行交互以执行算术运算。

## 参考资料

- [Numeric Types - int, float, complex](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex) — Python Software Foundation (2024)
  Python 内置数值类型的官方文档，详细说明了它们的特性、行为以及整数的任意精度。
- [Learning Python](https://www.oreilly.com/library/view/learning-python-5th/9781449355722/) — Mark Lutz (2013)
  Publisher: O'Reilly Media
  一本全面的Python语言指南，详细介绍了整数和浮点数等基本数据类型。
- [What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://dl.acm.org/doi/10.1145/103162.103163) — David Goldberg (1991)
  Journal: ACM Computing Surveys; Publisher: Association for Computing Machinery, Inc.; Volume: 23; Pages: 5-48; DOI: [10.1145/103162.103163](https://doi.org/10.1145/103162.103163)
  一篇开创性论文，阐释了浮点算术的原理，包括 IEEE 754 标准及其对数值计算的影响。
