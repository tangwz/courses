---
course: "python-for-beginners"
chapter: "building-blocks-functions"
lesson: "python-function-return-values"
sourceId: 2331
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-5-building-blocks-functions/python-function-return-values"
title: "从函数返回值"
description: "学习如何使用 `return` 语句将函数的结果返回到调用它的程序部分。"
order: 3
plots: []
sourceHash: "c8d74658a455455f2f7b70820cee237ab86b7335736d8d2ec9634f646db68360"
sourceCorrections: []
---

函数有助于组织执行特定操作的代码块。然而，许多函数旨在计算一个值或产生一个结果，供调用该函数的程序部分使用。仅仅在函数内部打印结果通常不够；通常，你需要将计算出的值传回。这通过使用 `return` 语句实现。

### 返回结果：`return` 语句

`return` 语句有两个主要作用：

1. 它立即退出当前函数执行。
2. 它将指定的值（如果没有指定值，则为 `None`）发送回调用者。

基本语法很简单：

```python
def function_name(parameters):
    # 执行计算或操作的代码
    result = some_value_or_calculation
    return result
```

一旦遇到 `return` 语句，函数会立即停止执行其内部的任何后续代码，控制流返回到函数被调用的地方。`return` 后指定的值成为函数调用表达式的结果。

### 一个简单的计算例子

我们来定义一个函数，它将两个数字相加并返回它们的和：

```python
def add_numbers(num1, num2):
    """计算两个数字的和。"""
    total = num1 + num2
    return total

# 调用函数并存储返回值
sum_result = add_numbers(5, 3)
print(f"The sum is: {sum_result}")

# 你也可以直接在其他表达式中使用返回值
another_result = add_numbers(10, 2) * 2
print(f"Another calculated result: {another_result}")
```

Output:

```
The sum is: 8
Another calculated result: 24
```

在这个例子中，`add_numbers(5, 3)` 被执行，值 `8` 被计算并赋值给 `total`，然后 `return total` 将值 `8` 返回。值 `8` 替换了 `add_numbers(5, 3)` 表达式，所以就好像我们写了 `sum_result = 8`。

### 返回不同数据类型

函数不仅限于返回数字。它们可以返回Python中任何类型的数据：字符串、布尔值、列表、字典、元组，甚至是自定义对象（我们稍后会看到）。

```python
def create_greeting(name):
    """创建一个个性化的问候字符串。"""
    return f"Hello, {name}! Greet."

def is_even(number):
    """检查一个数字是否为偶数。"""
    if number % 2 == 0:
        return True
    else:
        return False # 也可以直接写 'return number % 2 == 0'

def get_first_three_evens():
    """返回前三个偶数的列表。"""
    return [2, 4, 6]

# 使用这些函数
greeting = create_greeting("Alice")
print(greeting)

check_num = 10
if is_even(check_num):
    print(f"{check_num} is even.")
else:
    print(f"{check_num} is odd.")

even_list = get_first_three_evens()
print(f"First three evens: {even_list}")
```

Output:

```
Hello, Alice!
10 is even.
First three evens: [2, 4, 6]
```

### 如果没有 `return` 语句会怎样？

如果函数执行完毕但没有遇到显式的 `return` 语句，它会自动返回一个特殊值，称为 `None`。`None` 是 Python 表示没有值的方式。

```python
def print_message(message):
    """打印一条消息，但不显式返回任何内容。"""
    print(message)

result = print_message("This function prints.")
print(f"The return value is: {result}")
print(f"Is the result None? {result is None}")
```

Output:

```
This function prints.
The return value is: None
Is the result None? True
```

这里，`print_message` 执行了一个操作（打印）但没有 `return` 语句。当它完成时，它会隐式返回 `None`，然后该值被赋给 `result` 变量。

### 多个 `return` 语句

一个函数可以包含多个 `return` 语句，通常在条件块（如 `if`/`elif`/`else`）中。但是请记住，一旦执行了*任何一个* `return` 语句，函数就会立即终止。单次调用中只会执行一个 `return` 路径。

```python
def get_number_sign(num):
    """确定一个数字的符号。"""
    if num > 0:
        return "Positive"  # 如果 num > 0，则在此处退出
    elif num < 0:
        return "Negative"  # 如果 num < 0，则在此处退出
    else:
        return "Zero"      # 如果 num == 0，则在此处退出

sign1 = get_number_sign(15)
sign2 = get_number_sign(-3)
sign3 = get_number_sign(0)

print(f"Sign of 15: {sign1}")
print(f"Sign of -3: {sign2}")
print(f"Sign of 0: {sign3}")
```

Output:

```
Sign of 15: Positive
Sign of -3: Negative
Sign of 0: Zero
```

### 返回多个值

有时，一个函数需要返回多个信息片段。Python 使这变得容易。你可以在 `return` 关键字后面列出多个值，用逗号分隔。Python 会自动将这些值打包成一个元组并返回该元组。

```python
def get_coordinates():
    """返回一对坐标。"""
    x = 10
    y = 25
    return x, y # 返回元组 (10, 25)

coords = get_coordinates()
print(f"Coordinates tuple: {coords}")
print(f"X coordinate: {coords[0]}")
print(f"Y coordinate: {coords[1]}")

# 你也可以直接将元组解包到变量中
x_val, y_val = get_coordinates()
print(f"Unpacked X: {x_val}")
print(f"Unpacked Y: {y_val}")
```

Output:

```
Coordinates tuple: (10, 25)
X coordinate: 10
Y coordinate: 25
Unpacked X: 10
Unpacked Y: 25
```

这种将多个值作为元组返回然后解包的机制是 Python 编程中一种常见且便捷的模式。

有效使用 `return` 语句可以让你的函数产生结果，这些结果可以存储在变量中、传递给其他函数或在表达式中使用，使你的代码更具灵活性和强大功能。它是构建需要不同部分进行通信和共享计算信息的程序的基本点。

## 参考资料

- [Python Crash Course, 2nd Edition: A Hands-On, Project-Based Introduction to Programming](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE7caTR0d3oUNoKZaUe2ajCIraldH3jvhoL8sHrsvNmOH8OIitXjyjm5VLMX8ETyqzXrbWhg9VfHqAaINFQaNmQpoNhoNqGeRBCKDffMbKp_l-LHLD6av_invXSjtw67DRI9PxIA5Ilgr4nE6qqW55nD_0FqJRB1TNC8QdVecaCM2kHcG4eKUaZ8IO-l4lMpTsMdR0=) — Eric Matthes (2019)
  Publisher: No Starch Press; Pages: 143-149
  一本受欢迎的Python入门书籍，以适合初学者的方式介绍了函数和`return`语句，包括返回单个和多个值。
- [Learning Python, 5th Edition](https://www.oreilly.com/library/view/learning-python-5th/9781449355573/) — Mark Lutz (2013)
  Publisher: O'Reilly Media; Pages: 450-456
  一本全面且经典的Python编程书籍，深入涵盖了函数，包括`return`语句的机制和各种用法。
- [MIT 6.0001 Introduction to Computer Science and Programming in Python, Lecture 4: Functions](https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/resources/mit6_0001f16_lec4/) — Ana Bell, Eric Grimson, John Guttag (2016)
  Publisher: Massachusetts Institute of Technology
  麻省理工学院Python编程基础课程的讲义，提供了函数定义、参数以及`return`语句用于输出的学术视角。
