---
course: "python-for-beginners"
chapter: "building-blocks-functions"
lesson: "python-defining-functions"
sourceId: 2328
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-5-building-blocks-functions/python-defining-functions"
title: "定义自己的函数"
description: "学习在 Python 中使用 `def` 关键字定义自定义函数的语法。"
order: 1
plots: []
sourceHash: "39d55455259ddade6b52a3513b7b1a5640112abdbed36d7aee15d388510d3c21"
sourceCorrections: []
---

你可以编写一系列 Python 语句来执行任务。然而，当你发现自己重复地编写相同的步骤序列，或者当某个特定任务变得复杂时，将这些步骤打包成一个命名的代码块是很有益的。这个代码块被称为一个*函数*。定义函数让你可以为特定的计算或操作命名，使你的代码更有条理、更易读、可重用。你无需多次重写相同的逻辑，只需在需要执行该操作时，通过其名称“调用”该函数即可。

### `def` 语句

在 Python 中，你使用 `def` 关键字来定义一个函数，接着是函数名、一对括号 `()` 和一个冒号 `:`。函数要执行的实际代码放在 `def` 行之后的一个缩进块中。这个缩进是强制性的，它表明了函数的主体。

以下是基本语法：

```python
def function_name():
    # 缩进的代码块
    # 这是函数体
    statement1
    statement2
    # ... 更多语句
```

我们来定义一个简单的函数，它会打印一条问候消息：

```python
def greet_user():
    print("你好！")
    print("欢迎使用 Python 函数。")
```

具体来说：

1. `def`：表示函数定义开始的关键字。
2. `greet_user`：这是我们为函数选择的名称。函数名遵循与变量名相同的规则（必须以字母或下划线开头，可以包含字母、数字和下划线）。约定俗成是使用小写字母，并用下划线分隔单词（这种风格称为 `snake_case`）。
3. `()`：括号是必需的。稍后我们将了解它们如何用于接收函数的输入数据（参数 (parameter)）。目前它们是空的，表示此函数不接受任何输入。
4. `:`：冒号标志着函数头行的结束。
5. *缩进块*：`print("你好！")` 和 `print("欢迎使用 Python 函数。")` 这几行构成了函数*体*。它们被缩进（通常是 4 个空格），以表明它们属于 `greet_user` 函数。Python 使用这种缩进来确定函数代码的作用域。

### 定义与调用

理解定义函数并不会立即执行其代码，这一点很重要。`def` 语句只是创建了函数对象并将其命名为 `greet_user`。可以把它想象成写食谱；你已经定义了步骤，但还没有真正烹饪任何东西。

要执行函数体内的代码，你需要*调用*该函数。你通过使用函数名后跟括号来调用函数：

```python
# 定义函数
def greet_user():
    print("你好！")
    print("欢迎使用 Python 函数。")

# 现在，调用函数
print("我们来调用函数：")
greet_user()
print("函数调用结束。")
```

运行此脚本将产生以下输出：

```
我们来调用函数：
你好！
欢迎使用 Python 函数。
函数调用结束。
```

请注意，`greet_user` 内部的行只有在 `greet_user()` 被调用时才会被执行。你可以在程序的多个位置多次调用此函数，从而实现代码复用。

```python
# 定义函数（通常在脚本顶部完成）
def greet_user():
    print("你好！")
    print("欢迎使用 Python 函数。")

# 调用一次
print("第一次问候：")
greet_user()

# 也许在这里做一些其他工作...
x = 10
y = 5
print(f"执行计算: {x} * {y} = {x*y}")

# 再次调用
print("\n第二次问候：")
greet_user()
```

输出：

```
第一次问候：
你好！
欢迎使用 Python 函数。
执行计算: 10 * 5 = 50

第二次问候：
你好！
欢迎使用 Python 函数。
```

定义函数是编写结构化、易于管理的 Python 程序的重要一步。通过将逻辑封装在命名块中，你的代码会更容易理解、测试和修改。在接下来的章节中，我们将了解如何通过向函数传递数据并从中获取结果来使其更具灵活性。

## 参考资料

- [Python for Everybody: Chapter 4 - Functions](https://www.py4e.com/html3/04-functions) — Charles R. Severance (2023)
  一个易于理解的入门章节，阐释了函数定义和调用的动机与基本机制。
