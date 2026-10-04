---
course: "python-for-beginners"
chapter: "handling-errors-exceptions"
lesson: "python-understanding-errors"
sourceId: 2380
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-9-handling-errors-exceptions/python-understanding-errors"
title: "理解 Python 中的错误"
description: "区分在程序执行期间可能发生的语法错误（解析问题）和异常（运行时错误）。"
order: 1
plots: []
sourceHash: "681f5fac9641b27113c9ab499aa2b9130821fc6245bd96e760cb8996766d32af"
sourceCorrections: []
---

编写代码时，即使是经验丰富的程序员也会犯错。这些错误会导致程序无法正常运行。Python 错误通常分为两大类：语法错误和异常（也称为运行时错误）。理解它们之间的区别是编写更可靠代码的第一步。

### 语法错误（解析错误）

语法错误是 Python 代码的结构或语法问题。就像人类语言有关于句子结构和标点符号的规则一样，Python 也有关于代码编写方式的规则。如果您违反了这些规则，Python 解释器就无法理解您的指令。

这些错误在程序开始运行*之前*，在称为解析的阶段就会被检测到。解释器会读取您的代码并检查它是否遵循 Python 的语法规则。如果发现违规，它会立即停止并报告 `SyntaxError`。

常见的语法错误原因包括：

- 关键字拼写错误（例如，将 `while` 写成 `whlie`）。
- 缺少标点符号，例如 `if`、`for` 或 `def` 语句后面的冒号（`:`）。
- 括号 `()`、方括号 `[]` 或花括号 `{}` 不匹配。
- 不正确的缩进。

我们来看一个例子：

```python
# 语法错误的例子：缺少冒号
number = 10
if number > 5
    print("Number is greater than 5")
```

如果您尝试运行此代码，Python 会在执行任何内容之前停止并指出问题：

```plaintext
  File "your_script_name.py", line 3
    if number > 5
                 ^
SyntaxError: expected ':'
```

错误消息通常会告诉您文件名（`your_script_name.py`）、检测到错误的行号（第 3 行），有时还会使用插入符号（`^`）来指示问题的确切位置。它还会提供错误类型（`SyntaxError`）和简要描述（`expected ':'`）。由于语法错误会阻止程序启动，因此您必须在代码运行之前修复它们。

### 异常（运行时错误）

异常，即运行时错误，在程序执行*时*发生。与语法错误不同，代码结构根据 Python 的规则是语法正确的。然而，在执行期间，程序会遇到意料之外的情况或无法完成的操作。

当发生异常时，Python 会创建一个包含错误信息的“异常对象”。如果您的代码没有处理此异常，程序将停止执行并打印“回溯”，回溯显示导致错误的函数调用序列以及异常本身的详细信息。

以下是一些常见的异常类型：

- **`TypeError`**：当对不合适的类型对象执行操作时发生。例如，尝试将数字和字符串相加。

age = 30
message = "My age is: " + age # 尝试将字符串与整数相加
print(message)
\`\`\`

````
运行此代码会导致：
```plaintext
Traceback (most recent call last):
  File "your_script_name.py", line 2, in <module>
    message = "My age is: " + age
                           ^~~~~
TypeError: can only concatenate str (not "int") to str
```
````

- **`NameError`**：当您尝试使用尚未定义的变量或函数名时引发。

print(my\_variable) # my\_variable 从未被赋值
\`\`\`

````
运行此代码会导致：
```plaintext
Traceback (most recent call last):
  File "your_script_name.py", line 1, in <module>
    print(my_variable)
          ^^^^^^^^^^^
NameError: name 'my_variable' is not defined
```
````

- **`ValueError`**：当函数接收到正确类型但值不合适的参数 (parameter)时发生。

number\_string = "abc"
number = int(number\_string) # 无法将 'abc' 转换为整数
print(number)
\`\`\`

````
运行此代码会导致：
```plaintext
Traceback (most recent call last):
  File "your_script_name.py", line 2, in <module>
    number = int(number_string)
             ^^^^^^^^^^^^^^^^^^^
ValueError: invalid literal for int() with base 10: 'abc'
```
````

- **`ZeroDivisionError`**：当您尝试将数字除以零时引发。

result = 10 / 0
print(result)
\`\`\`

````
运行此代码会导致：
```plaintext
Traceback (most recent call last):
  File "your_script_name.py", line 1, in <module>
    result = 10 / 0
             ~~^~~
ZeroDivisionError: division by zero
```
````

这只是一些例子；Python 有许多内置的异常类型，用于处理各种运行时错误（例如文件不存在时的 `FileNotFoundError`，或尝试使用无效索引访问列表元素时的 `IndexError`）。

主要区别在于时间和原因：语法错误是由于不正确的代码结构而在执行*之前*捕获的，而异常是由于不可预见的情况或对数据进行无效操作而在执行*期间*发生的。虽然您必须修复语法错误才能运行代码，但异常通常可以在程序内部被预料到并妥善处理。本章的以下部分将详细说明如何使用 Python 的 `try`、`except`、`else` 和 `finally` 块来管理这些运行时异常。

## 参考资料

- [Errors and Exceptions](https://docs.python.org/3/tutorial/errors.html) — Python Software Foundation (2024)
  Publisher: Python Software Foundation
  这是Python错误和异常处理机制的官方权威指南，适合基础学习。
- [Built-in Exceptions](https://docs.python.org/3/library/exceptions.html) — Python Software Foundation (2024)
  这一官方参考文献提供了Python中所有标准内置异常类型的全面列表和详细描述，有助于理解它们的具体原因。
