---
course: "python-for-beginners"
chapter: "handling-errors-exceptions"
lesson: "practice-error-handling"
sourceId: 2388
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-9-handling-errors-exceptions/practice-error-handling"
title: "实践：实现错误处理"
description: "应用异常处理技术，使Python代码更能应对潜在错误。"
order: 8
plots: []
sourceHash: "91bd906e24e41fdb7b01d31ccad3aa39dd85a62b936f290a86bc593a612b9fa1"
sourceCorrections: []
---

Python提供了`try`、`except`、`else`、`finally`和`raise`作为处理错误的主要机制。编写可靠的代码意味着预见潜在问题并谨慎处理它们。这些练习将帮助您将错误处理技术应用于常见的编程情境。

请记住，目标不仅仅是防止程序崩溃，而且是在事情未按预期进行时提供有益的反馈或采取纠正措施。

## Exercise 1: Safe Numeric Input

通常，您需要从用户那里获取数字输入。然而，用户可能会输入文本、符号或什么都不输入，如果您直接尝试将输入转换为整数或浮点数，这将导致`ValueError`。

**任务：** 编写一个脚本，要求用户输入他们的年龄。使用`try...except`块来处理当输入无法转换为整数时发生的`ValueError`。如果输入无效，打印一条提示消息。如果输入有效，打印一条消息确认他们的年龄。

**步骤：**

1. 使用`input()`函数提示用户输入。
2. 使用`try`块尝试使用`int()`将输入转换为整数。
3. 在`try`块中，转换成功后，打印确认消息。
4. 添加一个`except ValueError`块，以捕获转换失败时发生的特定错误。
5. 在`except`块中，打印一条错误消息，说明需要有效的整数输入。

**Example Interaction:**

```
请输入您的年龄: thirty
错误：请输入一个有效的整数作为您的年龄。
```

```
请输入您的年龄: 25
谢谢。您的年龄是 25。
```

**Solution:**

```python
user_input = input("Please enter your age: ")

try:
    age = int(user_input)
    # 只有当上面的转换成功时，此行代码才会运行
    print(f"Thank you. Your age is {age}.")
except ValueError:
    # 如果int()引发ValueError，此代码块将运行
    print("Error: Please enter a valid whole number for your age.")
```

**说明：** 这种简单的结构是基本的。我们*尝试*可能失败的操作（`int(user_input)`）。如果成功，`try`块的其余部分会执行。如果它因为输入不是有效的整数字符串而特别地以`ValueError`失败，代码会跳到`except ValueError`块。

## Exercise 2: Reading from a File Safely

与文件系统交互是异常的另一个常见来源。文件可能丢失，或者您可能没有读取它们的权限。

**任务：** 编写一个函数`read_file_content(filename)`，它接受一个文件名作为参数 (parameter)。该函数应尝试打开并读取文件的内容。它应该通过打印特定消息来处理`FileNotFoundError`。它还应该包含一个`finally`块，打印一条消息，表明文件读取尝试已完成，无论成功或失败。

**步骤：**

1. 定义一个接受`filename`的函数`read_file_content`。
2. 使用`try`块，并通过`with`语句（它会自动处理关闭）以读取模式（`'r'`）打开文件。
3. 在`try`块中，读取文件的内容并打印它。
4. 添加一个`except FileNotFoundError`块，打印一条类似“错误：文件'{filename}'未找到。”的消息。
5. 添加一个`finally`块，打印“尝试读取文件完成。”

**Example Usage:**

```python
# 假设'my_data.txt'存在并包含“Hello Python!”
read_file_content('my_data.txt')

# 假设'non_existent_file.txt'不存在
read_file_content('non_existent_file.txt')
```

**Expected Output:**

```
Hello Python!
尝试读取文件完成。
错误：文件'non_existent_file.txt'未找到。
尝试读取文件完成。
```

**Solution:**

```python
def read_file_content(filename):
    """
    尝试读取并打印文件的内容。
    处理FileNotFoundError，并确保打印最后的消息。
    """
    try:
        # 使用'with'确保文件自动关闭
        with open(filename, 'r') as file:
            content = file.read()
            print("文件内容：")
            print(content)
    except FileNotFoundError:
        print(f"错误：文件'{filename}'未找到。")
    except Exception as e:
        # 捕获其他潜在的I/O错误（可选但推荐）
        print(f"发生了意外错误：{e}")
    finally:
        # 此代码块总是执行
        print(f"尝试读取'{filename}'完成。")

# Example calls
print("--- 读取现有文件 ---")
# 首先创建一个临时文件用于测试
with open('my_data.txt', 'w') as f:
    f.write("Hello Python!")
read_file_content('my_data.txt')

print("\n--- 读取不存在的文件 ---")
read_file_content('non_existent_file.txt')

# 清理临时文件（可选）
import os
if os.path.exists('my_data.txt'):
    os.remove('my_data.txt')
```

**说明：** `with open(...)`语句是文件处理的首选，因为它会自动关闭文件，即使发生错误也不例外。`try`块包含文件操作。`FileNotFoundError`被专门捕获。`finally`块保证“尝试完成...”的消息出现，这对于记录日志或确认操作是否已尝试很有用。我们还添加了一个通用的`except Exception`来捕获其他潜在问题，尽管`FileNotFoundError`在这里是最常见的。

## Exercise 3: Calculator with Multiple Error Types and `else`

让我们结合处理多个特定错误和`else`块。

**任务：** 创建一个函数`divide_numbers(numerator, denominator)`，它尝试执行除法运算。它应该：

- 如果分母为零，处理`ZeroDivisionError`。
- 如果任一输入不是数字（例如，字符串），处理`TypeError`。
- 仅当除法成功时，才使用`else`块打印结果。
- 如果成功则返回结果，如果发生错误则返回`None`。

**步骤：**

1. 定义函数`divide_numbers(numerator, denominator)`。
2. 使用`try`块计算`result = numerator / denominator`。
3. 添加一个`except ZeroDivisionError`块，打印“错误：不能除以零。”
4. 添加一个`except TypeError`块，打印“错误：两个输入都必须是数字。”
5. 添加一个`else`块，打印结果（例如，`f"结果是 {result}"`）并返回`result`。
6. 确保如果发生任何异常，函数返回`None`（如果`except`块没有`return`语句，这会隐式发生）。

**Example Usage:**

```python
divide_numbers(10, 2)
divide_numbers(10, 0)
divide_numbers(10, 'a')
```

**Expected Output:**

```
结果是 5.0
错误：不能除以零。
错误：两个输入都必须是数字。
```

**Solution:**

```python
def divide_numbers(numerator, denominator):
    """
    将两个数字相除，处理ZeroDivisionError和TypeError。
    仅在成功时使用else块打印结果。
    如果发生错误，则返回结果或None。
    """
    try:
        result = numerator / denominator
    except ZeroDivisionError:
        print("Error: Cannot divide by zero.")
        return None # 发生错误时显式返回 None
    except TypeError:
        print("Error: Both inputs must be numbers.")
        return None # 发生错误时显式返回 None
    else:
        # 此代码块仅在try块没有错误地完成时运行
        print(f"The result is {result}")
        return result

# Example calls
print("--- 有效的除法 ---")
divide_numbers(10, 2)

print("\n--- 除数为零 ---")
divide_numbers(10, 0)

print("\n--- 无效的输入类型 ---")
divide_numbers(10, 'a')
```

**说明：** 这展示了处理多个特定错误。`else`块在这里很重要；它保证只有当`try`块中的除法在不引发任何捕获的异常的情况下完成时，成功消息和`return result`语句才会被执行。

## Exercise 4: Raising an Exception

有时，您需要根据程序的逻辑发出错误情况信号，即使Python不会自动引发异常。

**任务：** 编写一个函数`calculate_area(length, width)`来计算矩形的面积。在开始时添加一个检查：如果`length`或`width`中小于或等于零，则引发`ValueError`，并附带消息“尺寸必须为正数”。否则，计算并返回面积。然后，编写代码在`try...except`块中调用此函数，以处理潜在的`ValueError`。

**步骤：**

1. 定义函数`calculate_area(length, width)`。
2. 在函数内部，使用`if`语句检查`length <= 0`或`width <= 0`。
3. 如果条件为真，使用`raise ValueError("尺寸必须为正数")`语句。
4. 如果条件为假，计算`area = length * width`并返回它。
5. 在函数外部，使用有效输入（例如，5, 10）在`try`块中调用`calculate_area`并打印结果。
6. 使用无效输入（例如，5, -2）在另一个`try`块中调用`calculate_area`。
7. 添加一个`except ValueError as e`块来捕获您的函数引发的异常，并打印`e`中包含的错误消息。

**Example Usage:**

```python
# 使用有效尺寸调用
# 使用无效尺寸调用
```

**Expected Output:**

```
面积：50
计算面积出错：尺寸必须为正数
```

**Solution:**

```python
def calculate_area(length, width):
    """
    计算矩形的面积。
    如果尺寸非正数，则引发ValueError。
    """
    if length <= 0 or width <= 0:
        # 引发异常以指示无效状态
        raise ValueError("Dimensions must be positive")
    # 仅当'if'条件为假时，此代码才运行
    area = length * width
    return area

# --- 调用函数并处理潜在错误 ---

print("--- 使用有效尺寸计算 ---")
try:
    valid_area = calculate_area(5, 10)
    print(f"Area: {valid_area}")
except ValueError as e:
    print(f"Error calculating area: {e}")

print("\n--- 使用无效尺寸计算 ---")
try:
    invalid_area = calculate_area(5, -2)
    # 如果引发异常，此print语句将不会被执行
    print(f"Area: {invalid_area}")
except ValueError as e:
    # 捕获函数引发的特定错误
    print(f"Error calculating area: {e}")
```

**说明：** 在这里，`calculate_area`函数使用`raise`来强制执行一项规则（正数尺寸）。这向*调用*该函数代码发出了一个问题信号。调用代码随后使用`try...except`来捕获这个特定的`ValueError`并妥善处理，从而防止程序崩溃并告知用户问题所在。引发异常是处理函数前置条件违规或其他逻辑错误的整洁方式。

这些练习涵盖了您在Python中使用异常处理的主要方式。通过练习捕获特定错误，使用`else`和`finally`，甚至引发自己的异常，您可以编写更可靠和用户友好的程序。

## 参考资料

- [The try statement](https://docs.python.org/3/reference/compound_stmts.html#the-try-statement) — Guido van Rossum and the Python development team (2023)
  Python异常处理语法，包括`try`、`except`、`else`和`finally`的核心参考。
- [Built-in Exceptions](https://docs.python.org/3/library/exceptions.html#built-in-exceptions) — Guido van Rossum and the Python development team (2024)
  标准异常的完整列表和描述，有助于理解`ValueError`和`FileNotFoundError`等特定错误类型。
- [Chapter 12: Exception Basics](https://www.oreilly.com/library/view/learning-python-5th/9781449355739/) — Mark Lutz (2013)
  Publisher: O'Reilly Media; Pages: 441-470
  详细章节，解释Python异常处理的机制和应用。
