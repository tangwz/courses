---
course: "python-for-beginners"
chapter: "python-basics-variables-data-types-operators"
lesson: "practice-calculations-io"
sourceId: 2285
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-2-python-basics-variables-data-types-operators/practice-calculations-io"
title: "练习：简单计算与输入输出"
description: "将你所学的变量、数据类型、运算符和用户输入知识应用于实际编码示例中。"
order: 11
plots: []
sourceHash: "b6e2aea17707793617f8be38a59ef4214326dbda21289556042668f757b3a7b6"
sourceCorrections: []
---

现在你已经学习了变量、数字和字符串等不同数据类型、如何使用运算符、在类型之间进行转换，并从用户那里获取输入，是时候将这些部分结合起来了。本练习部分提供实践示例，以巩固你对这些基本组成部分的理解。我们将创建执行计算并与用户交互的小程序。

### 示例 1：简单面积计算器

让我们创建一个计算矩形面积的脚本。它将要求用户输入长度和宽度，执行计算，并显示结果。

1. **获取输入：** 要求用户输入长度和宽度。请记住，`input()` 函数返回文本（一个字符串）。
2. **转换类型：** 使用 `float()` 将输入字符串转换为数字（我们将使用浮点数以允许小数）。
3. **计算：** 将长度乘以宽度以获得面积。
4. **显示输出：** 以用户友好的格式打印结果。

```python
# 获取矩形长度的输入
length_str = input("请输入矩形的长度：")

# 获取矩形宽度的输入
width_str = input("请输入矩形的宽度：")

# 将输入字符串转换为浮点数
length = float(length_str)
width = float(width_str)

# 使用算术乘法运算符计算面积
area = length * width

# 使用 f-string 显示结果，进行格式化输出
print(f"长度是：{length}")
print(f"宽度是：{width}")
print(f"矩形的面积是：{area}")
```

**尝试运行此代码。** 当提示时，输入长度和宽度的数值。观察程序如何获取你的文本输入，将其转换为适合计算的数字，执行乘法，然后打印一个清晰的消息，其中包含计算出的面积。如果你输入文本而不是数字会发生什么？你应该会遇到一个 `ValueError`，这说明了类型转换的重要性以及预见潜在输入问题的重要性（我们稍后将学习如何更优雅地处理这些问题）。

### 示例 2：个性化问候语

本示例侧重于处理字符串和用户输入，以创建个性化消息。

1. **获取用户名：** 要求用户输入他们的名字。
2. **获取喜爱项：** 要求用户输入他们喜欢的东西（例如，颜色、食物、爱好）。
3. **组合并显示：** 创建并打印一个包含用户输入的问候语。

```python
# 获取用户的名字
user_name = input("你叫什么名字？")

# 获取用户喜欢的颜色
favorite_color = input("你最喜欢什么颜色？")

# 使用 f-string 创建个性化问候语
greeting = f"你好，{user_name}！很高兴认识你。{favorite_color} 是一个很棒的颜色！"

# 打印问候语
print(greeting)

# 简单字符串方法的示例
print(f"你的名字大写是：{user_name.upper()}")
```

**运行此脚本。** 当被问及时，输入你的名字和你喜欢的颜色。请注意 `input()` 函数如何直接将你的文本捕获到字符串变量中。f-string 提供了一种简单的方法，将这些变量嵌入 (embedding)到更大的字符串中，以形成最终的 `greeting`。我们还包含了一个使用字符串方法 `.upper()` 来修改存储名字大小写的示例。

### 示例 3：数字比较

让我们练习使用比较运算符。这个脚本将接收两个数字并显示它们比较的结果。

1. **获取两个数字：** 提示用户输入两个数字，并将它们转换为浮点数。
2. **比较：** 使用比较运算符（`>`, `<`, `==`, `!=`）来比较数字。
3. **显示结果：** 打印每次比较的布尔结果（`True` 或 `False`）。

```python
# 获取第一个数字
num1_str = input("请输入第一个数字：")
num1 = float(num1_str)

# 获取第二个数字
num2_str = input("请输入第二个数字：")
num2 = float(num2_str)

# 执行比较
is_greater = num1 > num2
is_less = num1 < num2
is_equal = num1 == num2
is_not_equal = num1 != num2

# 显示结果
print(f"\n比较 {num1} 和 {num2}：")
print(f" {num1} 大于 {num2} 吗？{is_greater}")
print(f" {num1} 小于 {num2} 吗？{is_less}")
print(f" {num1} 等于 {num2} 吗？{is_equal}")
print(f" {num1} 不等于 {num2} 吗？{is_not_equal}")
```

**执行此代码。** 输入两个不同的数字，然后尝试两次输入相同的数字。观察输出。比较运算符评估值之间的关系并产生布尔结果（`True` 或 `False`），这对于在后续章节中控制程序逻辑非常重要。

这些示例涵盖了本章的主要内容：获取用户输入、将其存储在变量中、执行基本的算术和字符串操作、理解数据类型及其转换，以及使用比较运算符。请尝试使用这些脚本。尝试不同的输入，修改计算，或更改输出消息。你练习得越多，就会对这些重要的 Python 基础知识越熟悉。

## 参考资料

- [The Python Tutorial](https://docs.python.org/3/tutorial/) — Guido van Rossum and the Python development team (2023)
  Python 入门指南，涵盖变量、基本数据类型（数字、字符串）和运算符。
