---
course: "python-for-beginners"
chapter: "python-basics-variables-data-types-operators"
lesson: "python-arithmetic-operators"
sourceId: 2272
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-2-python-basics-variables-data-types-operators/python-arithmetic-operators"
title: "使用运算符：算术运算"
description: "学习如何使用 Python 中的算术运算符（如 +、-、*、/）执行数学计算。"
order: 5
plots: []
sourceHash: "d59416459b43b93652379c194928be703af3f9f857c222f82f6a52718b0ce1ec"
sourceCorrections: []
---

Python 编程涉及到数值数据，这些数据可以存储在变量中，例如整数（如 `10`、`-5`、`0`）和浮点数（如 `3.14`、`-0.5`、`2.0`）。为了对这些数字执行计算，Python 提供了一套**算术运算符**。这些运算符就像计算器一样，是您在程序中执行数学任务的基本工具。

### 基本算术运算

Python 支持您熟悉的标准算术运算：

- **加法 (`+`)：** 将两个数字相加。

  ```python
  apples = 5
  oranges = 10
  total_fruit = apples + oranges
  print(total_fruit) # Output: 15

  price = 9.99
  tax = 0.80
  total_cost = price + tax
  print(total_cost) # Output: 10.79
  ```
- **减法 (`-`)：** 从第一个数字中减去第二个数字。

  ```python
  budget = 100
  spent = 35.50
  remaining = budget - spent
  print(remaining) # Output: 64.5
  ```
- **乘法 (`*`)：** 将两个数字相乘。

  ```python
  width = 8
  height = 12
  area = width * height
  print(area) # Output: 96
  ```
- **除法 (`/`)：** 用第一个数字除以第二个数字。需要注意的是，Python 中的标准除法*总是*产生一个浮点数，即使除法结果是整数。

  ```python
  total_distance = 100 # kilometers
  time_hours = 2 # hours
  speed = total_distance / time_hours
  print(speed) # Output: 50.0

  items = 10
  people = 4
  items_per_person = items / people
  print(items_per_person) # Output: 2.5
  ```

  请注意 `100 / 2` 的结果是 `50.0`，而不仅仅是 `50`。这种一致的行为有助于避免当您预期小数结果时可能出现的小问题。

### 整数除法和取余

Python 提供了另外两种专门的除法运算符，它们在处理整数时特别有用：

- **整除 (`//`)：** 与标准 `/` 运算符一样执行除法，但它会丢弃小数部分，*向下*取整到最接近的整数。这也称为整数除法。

  ```python
  cookies = 10
  friends = 3
  cookies_per_friend = cookies // friends
  print(cookies_per_friend) # Output: 3 (Each friend gets 3 whole cookies)

  value = 7
  divisor = 2
  result = value // divisor
  print(result) # Output: 3 (7 divided by 2 is 3.5, rounded down is 3)

  # Compare with standard division
  print(10 / 3)  # Output: 3.3333333333333335
  print(10 // 3) # Output: 3
  ```
- **取余 (`%`)：** 此运算符不给出除法的结果，而是给出除法后剩余的*余数*。它通常被称为“余数运算符”。

  ```python
  total_cookies = 10
  friends = 3
  leftover_cookies = total_cookies % friends
  print(leftover_cookies) # Output: 1 (After giving 3 cookies each, 1 is left)

  number = 17
  divisor = 5
  remainder = number % divisor
  print(remainder) # Output: 2 (17 divided by 5 is 3 with a remainder of 2)
  ```

  取余运算符常用于检查数字是偶数还是奇数。如果 `number % 2` 等于 `0`，则该数字为偶数，否则为奇数。

  ```python
  num = 6
  print(num % 2) # Output: 0 (So, 6 is even)

  num = 7
  print(num % 2) # Output: 1 (So, 7 is odd)
  ```

### 幂运算

要将一个数字提高到另一个数字的幂，您可以使用双星号 (`**`) 运算符。

- **幂运算 (`**`)：** 将第一个数字提高到第二个数字的幂。$x^{y}$ 写为 `x ** y`。

  ```python
  base = 2
  exponent = 3
  power_result = base ** exponent # Equivalent to 2 * 2 * 2
  print(power_result) # Output: 8

  side = 5
  area_square = side ** 2 # Equivalent to 5 * 5 (5 squared)
  print(area_square) # Output: 25
  ```

### 运算顺序

当一个表达式包含多个运算符时，Python 遵循标准的运算顺序，这与数学中的顺序相似（通常通过缩写词 PEMDAS/BODMAS 记忆：括号/方括号、指数/幂、乘法和除法、加法和减法）。

1. **括号 `()`：** 括号内的运算首先执行。
2. **幂运算 `**`：** 接下来执行。
3. **乘法 `*`、除法 `/`、整除 `//`、取余 `%`：** 它们具有相同的优先级，从左到右计算。
4. **加法 `+`、减法 `-`：** 它们具有相同的优先级，最后从左到右计算。

请看这个例子：

```python
result = 2 + 3 * 4
print(result) # Output: 14 (Multiplication is done before addition: 2 + 12)
```

如果您希望加法首先执行，请使用括号：

```python
result = (2 + 3) * 4
print(result) # Output: 20 (Parentheses force addition first: 5 * 4)
```

另一个例子：

```python
calculation = 100 - 20 / 5 * 2 + 3**2
# Step-by-step evaluation:
# 1. Exponentiation: 3**2 becomes 9
#    Expression: 100 - 20 / 5 * 2 + 9
# 2. Division/Multiplication (from left to right):
#    20 / 5 becomes 4.0
#    Expression: 100 - 4.0 * 2 + 9
#    4.0 * 2 becomes 8.0
#    Expression: 100 - 8.0 + 9
# 3. Addition/Subtraction (from left to right):
#    100 - 8.0 becomes 92.0
#    Expression: 92.0 + 9
#    92.0 + 9 becomes 101.0
print(calculation) # Output: 101.0
```

使用括号可以使复杂的表达式更加清晰，即使根据优先级规则它们并非严格必要。当不确定时，使用括号来确保计算按照您预期的顺序进行。

### 简写：增广赋值运算符

在编程中，根据变量的当前值修改变量是非常常见的。例如，增加计数器或累加总和：

```python
count = 0
count = count + 1 # Increment count by 1
print(count) # Output: 1

total_cost = 50
item_price = 15.50
total_cost = total_cost + item_price # Add item_price to total_cost
print(total_cost) # Output: 65.5
```

Python 提供了简写运算符，称为**增广赋值运算符**，使这类操作更简洁：

```python
count = 0
count += 1 # Equivalent to count = count + 1
print(count) # Output: 1

score = 100
score -= 10 # Equivalent to score = score - 10
print(score) # Output: 90

multiplier = 5
multiplier *= 2 # Equivalent to multiplier = multiplier * 2
print(multiplier) # Output: 10

total = 50.0
total /= 4 # Equivalent to total = total / 4
print(total) # Output: 12.5
```

这些简写运算符（`+=`、`-=`、`*=`、`/=`、`//=`、`%=`、`**=`）适用于所有算术运算符，并将操作与赋值结合起来。一旦您熟悉它们，它们很方便，并且通常使代码更容易阅读。

您现在拥有在 Python 中执行各种计算的工具。通过结合变量、数值数据类型和这些算术运算符，您可以开始编写有效处理数字的程序。请记住，在需要时使用括号来阐明或控制运算顺序。

## 参考资料

- [The Python Language Reference: Expressions](https://docs.python.org/3/reference/expressions.html) — Python Software Foundation (2023)
  Python官方文档，详细说明了Python的表达式，包括所有算术运算符、其具体行为以及运算符优先级规则。
- [Python Operators: The Ultimate Guide](https://realpython.com/python-operators-expressions/) — John Sturtz (2019)
  Publisher: Real Python
  来自权威来源的综合在线指南，详细解释并提供了所有Python运算符（包括算术运算符、增广赋值运算符）及其优先级的示例。
