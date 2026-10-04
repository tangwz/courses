---
course: "python-for-beginners"
chapter: "building-blocks-functions"
lesson: "python-variable-scope"
sourceId: 2333
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-5-building-blocks-functions/python-variable-scope"
title: "理解变量作用域（局部与全局）"
description: "明白局部变量（在函数内部定义）和全局变量（在函数外部定义）之间的区别。"
order: 4
plots: []
sourceHash: "63242ecccbb7ef893a91c698a654cc55bfba0bd085ea35635f2d37f4c3cab9c9"
sourceCorrections: []
---

当你开始编写函数时，你会注意到一个重要点：并非所有变量都能在程序的任何地方被访问。你定义变量的位置决定了你可以在哪里使用它。这个理念被称为**变量作用域**。明白作用域对于组织代码和避免意外行为很根本。

把它想象成房子里有不同的房间。有些物品属于特定房间（比如浴室里的牙刷），而另一些则可能在很多地方都能用到（比如主电源开关）。变量的运作方式类似。

### 局部变量：函数内部

你在函数定义*内部*定义的变量被称为**局部变量**。它们只存在于那个特定的函数内部。一旦函数执行完毕，这些变量就会消失，它们的内存也会被释放。尝试从函数外部访问局部变量会导致错误。

考虑这个例子：

```python
def greet_user():
    message = "Hello from inside the function!" # message 变量是 greet_user 函数的局部变量
    print(message)

greet_user()

# 现在，让我们尝试在函数外部访问 'message'：
# print(message) # 这一行将导致 NameError
```

如果你运行这段代码并取消注释最后一行，Python 将会停止并报告一个 `NameError`。出现这种情况是因为 `message` 变量只存在于 `greet_user` 函数的作用域内。在该函数之外，Python 不知道 `message` 指的是什么。

这种局部性实际上是一个很有用的特点。这表示你可以在不同的函数内部使用 `i`、`count` 或 `temp` 等常用变量名，而不用担心它们相互干扰。每个函数都有它自己的私有工作区来存放局部变量。

### 全局变量：(几乎)处处可访问

在任何函数定义*外部*定义的变量被称为**全局变量**。它们存在于脚本的主体部分，通常被称为全局作用域或模块作用域。这些变量通常可以在脚本中的任何地方被访问（读取），包括在它们之后定义的任何函数内部。

让我们看一个例子：

```python
app_version = "1.0" # app_version 是一个全局变量

def display_version():
    print("Inside the function, app version is:", app_version) # 访问全局变量

print("Outside the function, app version is:", app_version)
display_version()
```

运行这段代码会产生：

```text
Outside the function, app version is: 1.0
Inside the function, app version is: 1.0
```

正如你所看到的，`display_version` 函数可以毫无问题地读取全局变量 `app_version` 的值。

### Python 如何查找变量：查找顺序

当你在函数内部使用变量名时，Python 遵循特定的顺序来找出它指的是哪个值：

1. **局部作用域：** Python 首先检查在当前函数*内部*是否定义了同名变量。
2. **全局作用域：** 如果在局部作用域中没有找到该变量，Python 就会检查它是否存在于*全局*作用域（所有函数之外）。

(注意：还有其他作用域，比如闭包函数作用域和内置作用域，但目前，明白局部和全局之间的区别是最重要的一步。)

这个查找顺序解释了为什么函数可以访问全局变量，而外部代码不能访问局部变量。

### 局部变量与全局变量冲突时：遮蔽

如果你在函数内部定义一个局部变量，其名称与全局变量*相同*，会发生什么？

```python
animal = "Global Fox" # 全局变量

def print_local_animal():
    animal = "Local Bear" # 同名的局部变量
    print("Inside function:", animal)

print("Outside function (before call):", animal)
print_local_animal()
print("Outside function (after call):", animal)
```

输出：

```text
Outside function (before call): Global Fox
Inside function: Local Bear
Outside function (after call): Global Fox
```

在 `print_local_animal` 函数内部，赋值语句 `animal = "Local Bear"` 创建了一个*新的*、名为 `animal` 的局部变量。这个局部变量在函数作用域内具有优先权，实际上隐藏或“遮蔽”了同名的全局变量。当 `print_local_animal` 执行完毕时，它的局部 `animal` 变量就会消失。重要的是，全局 `animal` 变量在此过程中始终保持不变。

这种行为可以避免函数仅仅因为在局部使用了相同的名称而意外修改全局变量。

### 修改全局变量：`global` 关键字

通常，函数应该对其输入（参数 (parameter)）进行操作并产生输出（返回值）。这使得它们自成一体且更容易理解。然而，在某些情况下，函数可能需要明确修改一个全局变量。

如果你尝试在函数内部为一个与全局变量同名的变量赋值，Python 会认为你想创建一个*新的局部变量*（如遮蔽示例所示）。要告诉 Python 你实际上想要修改*全局*变量，你必须在函数内部给它赋值*之前*使用 `global` 关键字。

```python
counter = 0 # 全局变量

def increment_counter():
    global counter # 声明要使用全局变量 'counter'
    counter += 1   # 修改全局变量
    print("Counter inside function:", counter)

print("Counter before:", counter)
increment_counter()
increment_counter()
print("Counter after:", counter)
```

输出：

```text
Counter before: 0
Counter inside function: 1
Counter inside function: 2
Counter after: 2
```

在这里，`global counter` 告诉 `increment_counter` 函数，`counter` 指的是全局作用域中的变量，而不是一个新的局部变量。因此，`counter += 1` 操作修改的是原始的全局变量。

**谨慎使用 `global`：** 尽管 `global` 关键字提供了一种修改全局状态的方法，但通常建议谨慎使用它。修改全局变量的函数可能会产生从函数调用处不易察觉的副作用。这会使你的程序更难于理解和调试。通常，更好的做法是让函数`返回`新值，并在需要时由函数外部的代码更新全局变量。

```python
# 替代方法（通常更受推荐）
counter = 0 # 全局变量

def calculate_new_count(current_count):
    # 这个函数不直接修改全局状态
    return current_count + 1

print("Counter before:", counter)
counter = calculate_new_count(counter) # 根据返回值更新全局变量
print("Counter after first update:", counter)
counter = calculate_new_count(counter)
print("Counter after second update:", counter)
```

输出：

```text
Counter before: 0
Counter after first update: 1
Counter after second update: 2
```

这种替代方法达到了同样的效果，但使函数（`calculate_new_count`）保持自包含，并且不直接修改全局变量。

明白局部作用域和全局作用域之间的区别对于编写清晰、正确且易于维护的 Python 函数是很根本的。这有助于你管理程序中数据的位置，并避免对变量进行意外修改。

## 参考资料

- [The Python Tutorial: More Control Flow Tools - Defining Functions](https://docs.python.org/3/tutorial/controlflow.html#defining-functions) — Python Software Foundation (2024)
  Publisher: Python Software Foundation
  Python官方文档的这一部分清晰地介绍了函数，涵盖了局部变量和`global`关键字。
- [Fluent Python, 2nd Edition](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056345/) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media; Pages: 1012
  本书全面指导Python，包含对作用域规则、闭包和LEGB规则的详细讨论，有助于更全面的理解。
