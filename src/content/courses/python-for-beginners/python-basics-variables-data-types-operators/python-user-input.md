---
course: "python-for-beginners"
chapter: "python-basics-variables-data-types-operators"
lesson: "python-user-input"
sourceId: 2283
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-2-python-basics-variables-data-types-operators/python-user-input"
title: "获取用户输入"
description: "了解如何在程序执行期间使用 input() 函数接受用户输入。"
order: 9
plots: []
sourceHash: "1cf10d41d21692603e471513420ce7f81e559e7769f55b6c76a3522fe8bf3a2f"
sourceCorrections: []
---

程序可以处理直接写入代码的数据，但许多应用也需要在运行时与用户交互，请求信息。Python 通过内置的 `input()` 函数提供了一种直接的方式来实现这一点。

当你的程序调用 `input()` 时，它会暂停执行，等待用户在控制台中输入内容并按下回车键。用户输入的任何内容都将作为字符串由该函数返回。

我们来尝试一个基本例子。运行以下代码：

```python
print("你的名字是什么？")
user_name = input()
print("你好，", user_name)
```

当你运行这段代码时，程序会打印“你的名字是什么？”，然后等待。如果你输入 `Alice` 并按下回车，程序会继续执行并打印 `你好， Alice`。你输入的内容 `Alice` 被 `input()` 捕获并存储在 `user_name` 变量中。

### 提供提示信息

通常，将提示信息直接包含在 `input()` 函数中会更便于用户使用。此消息会显示给用户，指示他们应输入什么信息。

```python
user_name = input("请输入你的名字: ")
print("你好，", user_name)
```

这个版本实现了与上一个相同的结果，但将提示和输入请求合并到一行中，使代码稍微更简洁。字符串 `“请输入你的名字: ”` 会被显示，程序紧随其后等待输入。

### 输入始终是字符串

`input()` 函数一个非常重要的方面是，它*总是*将用户输入的数据作为字符串返回，无论他们输入什么。即使用户输入数字，`input()` 也会将其视为字符，形成一个字符串。

考虑这个例子：

```python
age_input = input("请输入你的年龄: ")
print("age_input 的类型:", type(age_input))
# 尝试进行加法运算可能会导致错误或意外行为
# next_year = age_input + 1 # 这会引发 TypeError 错误
```

如果你输入 `25`，输出将是：

```text
Type of age_input: <class 'str'>
```

请注意，类型是 `<class 'str'>`，而不是 `<class 'int'>`。因为 `age_input` 是一个字符串 (`"25"`)，你不能直接对其执行数学运算，例如加 `1`。尝试计算 `age_input + 1` 将导致 `TypeError`，因为 Python 在这种情况下不知道如何将整数添加到字符串。

### 转换输入类型

如果你需要将用户输入视为数字（整数或浮点数）或其他数据类型，你必须显式地转换 `input()` 返回的字符串。你可以使用我们之前讨论过的类型转换函数，例如 `int()` 或 `float()`。

以下是你如何正确处理数字输入的方法：

```python
age_str = input("请输入你的年龄: ")
age_int = int(age_str) # 将输入字符串转换为整数

next_year_age = age_int + 1
print("明年你将是", next_year_age, "岁")

# 你也可以一步完成输入和转换：
height_str = input("请输入你的身高（米）: ")
height_float = float(height_str) # 将输入字符串转换为浮点数

print("你的身高是:", height_float, "米")
```

在这个修改后的例子中：

1. 我们使用 `input()` 获取年龄作为字符串，并将其存储在 `age_str` 中。
2. 我们使用 `int(age_str)` 将字符串 `"25"`（或用户输入的任何内容）转换为整数 `25`，并将其存储在 `age_int` 中。
3. 现在我们可以执行算术运算：`age_int + 1`。
4. 同样，我们使用 `float()` 将身高输入转换为浮点数，以处理可能的十进制值。

请记住，如果用户输入的文本无法转换为目标类型（例如，当使用 `int()` 时输入 `"hello"`），程序将引发 `ValueError` 错误。如何稳妥地处理此类错误将在我们稍后讨论异常处理时介绍。

获取用户输入对于创建交互式程序是基本的。`input()` 函数提供了这种机制，但请始终记住，它的返回值是一个字符串，在使用它进行计算或比较之前，通常需要显式的类型转换。

## 参考资料

- [Input and Output](https://docs.python.org/3/tutorial/inputoutput.html) — Python Software Foundation (2024)
  Publisher: Python Software Foundation
  一个入门教程，展示了如何与用户交互，涵盖了Python中的基本输入操作和字符串格式化。
