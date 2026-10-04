---
course: "python-for-beginners"
chapter: "python-basics-variables-data-types-operators"
lesson: "python-string-type"
sourceId: 2266
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-2-python-basics-variables-data-types-operators/python-string-type"
title: "基础数据类型：文本（字符串）"
description: "学习如何使用 Python 的字符串类型来创建、操作和处理文本数据。"
order: 3
plots: []
sourceHash: "af79161800e1963c8f614c629448a895093b5c7d8d681692bc90d0de5d0d8894"
sourceCorrections: []
---

处理文本是编程的一个基本方面。在 Python 中，文本数据用**字符串**表示。字符串是一个有序的字符序列。任何用引号括起来的内容——字母、数字、符号、空格——都会变成字符串。

### 创建字符串

Python 在定义字符串方面非常灵活。你可以使用单引号 (`'`) 或双引号 (`"`)。如果你的字符串本身包含引号，这种灵活性会很有用。

```python
# 使用单引号
message_single = 'Hello, Python learner!'
print(message_single)

# 使用双引号
message_double = "This also works just fine."
print(message_double)

# 如果你的字符串包含单引号怎么办？使用双引号来括住它。
quote = "He said, 'Python is fun!'"
print(quote)

# 如果你的字符串包含双引号怎么办？使用单引号。
reply = 'She responded, "Indeed it is."'
print(reply)
```

如果你需要一个跨越多行或同时包含单引号和双引号且没有麻烦的字符串怎么办？Python 提供三引号，可以是 `'''` 或 `"""`。

```python
# 用于多行字符串的三重双引号
multi_line_doc = """This is the first line.
This is the second line.
And "quotes" or 'apostrophes' can be used freely inside."""
print(multi_line_doc)

# 三重单引号工作方式相同
multi_line_alt = '''Another way to write
multi-line text,
very convenient.'''
print(multi_line_alt)
```

### 基本字符串操作

你可以使用 `+` 运算符组合字符串，这个操作称为**连接**。你还可以使用 `*` 运算符重复字符串。

```python
first_name = "Ada"
last_name = "Lovelace"
space = " "

# 连接
full_name = first_name + space + last_name
print(full_name) # 输出: Ada Lovelace

# 重复
separator = "-" * 10 # 重复连字符10次
print(separator)   # 输出: ----------
print(first_name * 3) # 输出: AdaAdaAda
```

请注意，你只能将字符串与其他字符串连接。直接尝试将字符串与数字相加会导致错误。你需要先使用 `str()` 将数字转换为字符串，我们很快会在类型转换部分讨论这个。

### 访问字符：索引

由于字符串是序列，你可以通过字符的位置或**索引**来访问单个字符。Python 使用零起始索引，意味着第一个字符在索引 0 处，第二个在索引 1 处，依此类推。

```python
language = "Python"

# 通过正向索引访问字符
print(language[0]) # 输出: P (第一个字符)
print(language[1]) # 输出: y (第二个字符)
print(language[5]) # 输出: n (第六个字符)

# 使用负向索引访问字符（从末尾开始）
print(language[-1]) # 输出: n (最后一个字符)
print(language[-2]) # 输出: o (倒数第二个字符)
```

尝试访问不存在的索引（例如，上面示例中的 `language[6]`）将导致 `IndexError`。

### 提取子字符串：切片

如果你需要不止一个字符，你可以提取字符串的一部分，称为**子字符串**或**切片**。切片使用 `[start:stop:step]` 语法。

- `start`: 切片开始的索引（包含）。如果省略，默认为 0。
- `stop`: 切片结束的索引（不包含）。如果省略，默认为字符串的末尾。
- `step`: 步进量。如果省略，默认为 1。

```python
language = "Python"

# 获取从索引 1 开始（不包括）到索引 4 的字符
print(language[1:4]) # 输出: yth

# 获取从开头到索引 3（不包括）的字符
print(language[:3])  # 输出: Pyt

# 获取从索引 2 到结尾的字符
print(language[2:])  # 输出: thon

# 获取整个字符串（不常用，但显示了默认值）
print(language[:])   # 输出: Python

# 获取每隔一个字符
print(language[::2]) # 输出: Pto

# 反转字符串
print(language[::-1]) # 输出: nohtyP
```

切片总是生成一个新字符串，并且如果索引超出范围，它绝不会导致 `IndexError`；它只会返回请求范围内字符串中可用的部分。

### 字符串不可变性

Python 字符串的一个重要特点是它们是**不可变的**。这意味着一旦字符串被创建，它就不能被原地修改。看起来会修改字符串的操作，比如连接或我们接下来将看到的方法，实际上是创建并返回*新*字符串。

```python
greeting = "hello"
# greeting[0] = "H" # 这将导致 TypeError：'str' 对象不支持项赋值

# 要“改变”问候语，请创建一个新字符串
new_greeting = "H" + greeting[1:]
print(new_greeting) # 输出: Hello
print(greeting)     # 输出: hello (原始字符串未改变)
```

这种不可变性可能看起来有局限性，但它使字符串变得可预测，并且在不同情境下使用起来安全，例如字典的键（你稍后会学到）。

### 常用字符串方法

字符串带有很多内置的**方法**，用于执行常见操作。方法类似于函数，但它们是使用点符号 (`.`) 在对象（本例中是字符串）*上*调用的。这里有一些有用的方法：

- `len(string)`: 尽管技术上来说它是一个内置函数，而非方法，但 `len()` 对于获取字符串中的字符数量非常重要。
- `.lower()` / `.upper()`: 返回一个新字符串，其中所有字符分别转换为小写或大写。
- `.strip()` / `.lstrip()` / `.rstrip()`: 返回一个新字符串，其中删除了开头和/或结尾的空白字符。
- `.find(substring)`: 返回 `substring` 第一次出现的起始索引。如果未找到，则返回 -1。
- `.replace(old, new)`: 返回一个新字符串，其中所有 `old` 的出现都被 `new` 替换。
- `.startswith(prefix)` / `.endswith(suffix)`: 根据字符串是否以给定子字符串开头或结尾，返回 `True` 或 `False`。
- `.split(separator)`: 返回一个子字符串列表，在 `separator` 出现的地方进行分割。如果没有给出分隔符，则按空白字符分割。
- `separator.join(iterable)`: 将可迭代对象（如列表）的元素连接成一个字符串，`separator` 放在元素之间。

```python
text = "  Learning Python is Fun!  "

# 长度
print(len(text)) # 输出: 27

# 大小写转换
print(text.lower()) # 输出:   learning python is fun!
print(text.upper()) # 输出:   LEARNING PYTHON IS FUN!

# 移除空白字符
print(text.strip())  # 输出: Learning Python is Fun!
print(text.lstrip()) # 输出: Learning Python is Fun!
print(text.rstrip()) # 输出:   Learning Python is Fun!

# 查找子字符串
print(text.find("Python")) # 输出: 11 ('Python' 开始的索引)
print(text.find("Java"))   # 输出: -1 (未找到)

# 替换子字符串
print(text.replace("Fun", "Awesome")) # 输出:   Learning Python is Awesome!

# 检查开头/结尾（此处先使用 strip 以确保准确性）
clean_text = text.strip()
print(clean_text.startswith("Learning")) # 输出: True
print(clean_text.endswith("!"))        # 输出: True

# 将字符串分割成列表
words = clean_text.split(" ")
print(words) # 输出: ['Learning', 'Python', 'is', 'Fun!']

# 将列表连接成字符串
joined_string = "---".join(words)
print(joined_string) # 输出: Learning---Python---is---Fun!

# 原始字符串因不可变性而保持不变
print(text) # 输出:   Learning Python is Fun!
```

### 格式化字符串

通常，你需要创建嵌入 (embedding)变量值的字符串。尽管你可以使用连接 (`+`)，但这很快就会变得繁琐且容易出错，尤其是在处理非字符串类型时。Python 提供了更好的方式来格式化字符串。

最现代且普遍推荐的方法是使用**f-字符串**（格式化字符串字面量），它在 Python 3.6 中被引入。你通过在字符串字面量前加上字母 `f` 或 `F` 来创建 f-字符串。在字符串内部，你可以直接将变量或表达式放在花括号 `{}` 中。

```python
name = "Alice"
age = 30
city = "New York"

# 使用 f-字符串进行清晰格式化
intro_fstring = f"My name is {name}, I am {age} years old, and I live in {city}."
print(intro_fstring)
# 输出: My name is Alice, I am 30 years old, and I live in New York.

# 你也可以在花括号中包含表达式
radius = 5
area = 3.14159 * radius**2
print(f"A circle with radius {radius} has an area of {area:.2f}.")
# 输出: A circle with radius 5 has an area of 31.42.
# 注意: :.2f 将浮点数格式化为两位小数。
```

f-字符串可读性强、简洁，并且通常比旧的格式化方法（如 `.format()` 或 C 风格的 `%` 运算符）更快，你可能会在旧代码中遇到它们。

### 转义序列

有时你需要在字符串中包含难以或无法直接输入的字符，或者具有特殊含义的字符。例如，你如何在用相同引号类型作为边界的字符串中包含换行符、制表符或字面引号字符？这可以通过使用**转义序列**来完成，转义序列以反斜杠 (`\`) 开头。

常用转义序列包括：

- `\n`: 换行
- `\t`: 制表符
- `\\`: 字面反斜杠
- `\'`: 字面单引号
- `\"`: 字面双引号

```python
# 换行
print("First line.\nSecond line.")
# 输出:
# First line.
# Second line.

# 制表符
print("Column1\tColumn2")
# 输出: Column1    Column2

# 字面反斜杠
print("This is a path: C:\\Users\\Name")
# 输出: This is a path: C:\Users\Name

# 在由相同引号定义的字符串中使用字面引号
print('He said, \'Hello!\'') # 输出: He said, 'Hello!'
print("She replied, \"Hi!\"") # 输出: She replied, "Hi!"
```

字符串是处理任何文本数据的基础，从用户输入、文件内容到程序中的消息和标签。掌握它们的创建、操作和格式化是学习 Python 的重要一步。

## 参考资料

- [Built-in Types (Text Sequence Type - str)](https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str) — Python Software Foundation (2023)
  提供关于 Python 字符串类型、其方法和行为的全面细节，是权威的技术参考。
- [An Informal Introduction to Python (3.1.3. Strings)](https://docs.python.org/3/tutorial/introduction.html#strings) — Python Software Foundation (2023)
  Publisher: Python Software Foundation
  在官方 Python 教程中提供了对字符串的易懂介绍，涵盖了基本创建、操作和索引，并附有实用示例。
- [Fluent Python, 2nd Edition: Clear, Concise, and Effective Programming](https://www.vitalsource.com/products/fluent-python-luciano-ramalho-v9781492056300) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media
  对 Python 字符串进行了全面而深入的分析，包括基础用法之外的高级主题和最佳实践，适合寻求深入理解的读者。
