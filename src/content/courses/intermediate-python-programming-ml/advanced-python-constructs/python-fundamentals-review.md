---
course: "intermediate-python-programming-ml"
chapter: "advanced-python-constructs"
lesson: "python-fundamentals-review"
sourceId: 2079
sourceUrl: "https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-1-advanced-python-constructs/python-fundamentals-review"
title: "Python 基础知识回顾"
description: "对中级机器学习编程非常重要的 Python 核心要点进行针对性回顾。"
order: 1
plots: []
sourceHash: "cd63c50c405463965bde2d095db1723e0384a10bbc67a526854a7934c952913d"
sourceCorrections: []
---

在介绍数据科学中常用的一些更专业的 Python 结构之前，Python 核心知识将被阐述。这些知识构成了所有后续主题的基本组成部分。对这些基础知识的透彻理解，对于日常使用 Python 处理数据必不可少，并为高级应用提供了坚实的基础。

### 变量和基本数据类型

编程的核心在于操作存储在变量中的数据。Python 是一种动态类型语言，这意味着您无需显式声明变量的类型。然而，了解主要数据类型对于高效处理数据很重要。

- **整型 (`int`)**: 整数，例如 `10`、`-5`、`0`。
- **浮点型 (`float`)**: 带小数点的数字，例如 `3.14`、`-0.001`、`2.7e5`（科学计数法）。请注意浮点运算中固有的潜在精度问题。
- **字符串 (`str`)**: 字符序列，用单引号 (`'...'`) 或双引号 (`"..."`) 括起来。用于文本数据。操作包括连接 (`+`) 和切片。
- **布尔型 (`bool`)**: 表示真值，即 `True` 或 `False`。对于条件逻辑必不可少。

```python
# 变量赋值
count = 100
temperature = 23.5
city_name = "San Francisco"
is_valid = True

# 检查类型（对调试有用）
# print(type(count))
# print(type(temperature))
# print(type(city_name))
# print(type(is_valid))
```

虽然动态类型提供了灵活性，但明确地使用*类型提示*（例如 `count: int = 100`）来指示预期类型变得越来越普遍，尤其是在大型项目中，因为它能提高代码可读性并允许进行静态分析。

### 主要数据结构

Python 提供了几种内置数据结构，用于组织数据集合。选择合适的数据结构通常与性能和具体任务相关。

- **列表 (`list`)**: 有序、可变（可更改）的项序列。用方括号 `[]` 定义。列表功能多样，常用于存储数据点或观测值的序列。

  ```python
  features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
  measurements = [5.1, 3.5, 1.4, 0.2]
  measurements.append(0.3) # 列表是可变的
  # print(features[0]) # 按索引访问
  # print(measurements)
  ```
- **元组 (`tuple`)**: 有序、不可变（不可更改）的项序列。用括号 `()` 定义。由于它们不可变，元组常用于不应更改的数据，例如坐标或固定的配置设置。它们也可以用作字典的键。

  ```python
  point = (10, 20)
  # point[0] = 15 # 这将引发 TypeError
  # print(point)
  ```
- **字典 (`dict`)**: （在 Python 3.7 之前的版本中）无序的键值对集合。用花括号 `{}` 定义。键必须是唯一的且不可变（字符串、数字或元组是常见的键）。字典在映射信息方面非常有用，例如将特征名称映射到值或存储配置参数 (parameter)。

  ```python
  sample = {'sepal_length': 5.1, 'sepal_width': 3.5, 'species': 'setosa'}
  # print(sample['sepal_length']) # 按键访问
  sample['petal_length'] = 1.4 # 添加新的键值对
  # print(sample)
  ```
- **集合 (`set`)**: 无序的唯一、不可变项集合。用花括号 `{}` 或 `set()` 函数定义。集合对成员测试（`in` 运算符）和从序列中删除重复项进行了高度优化。

  ```python
  unique_species = {'setosa', 'versicolor', 'virginica', 'setosa'}
  # print(unique_species) # 输出: {'setosa', 'versicolor', 'virginica'}
  # print('setosa' in unique_species) # 快速成员测试
  ```

### 控制流

控制流语句指示代码的执行顺序。

- **条件语句 (`if`, `elif`, `else`)**: 根据条件是否评估为 `True` 或 `False` 来执行代码块。

  ```python
  value = 75
  if value > 90:
      grade = 'A'
  elif value > 70:
      grade = 'B'
  else:
      grade = 'C'
  # print(f"等级: {grade}")
  ```
- **循环 (`for`, `while`)**: 重复执行代码块。
  - `for` 循环遍历序列（例如列表、元组、字符串、字典或生成器输出）。

    ```python
    # 遍历列表元素
    total = 0
    numbers = [1, 2, 3, 4, 5]
    for num in numbers:
        total += num
    # print(f"和: {total}")

    # 遍历字典键
    # for key in sample:
    #     print(f"{key}: {sample[key]}")
    ```
  - `while` 循环在条件保持 `True` 时持续执行。请注意确保条件最终变为 `False` 以避免无限循环。

    ```python
    count = 0
    while count < 3:
        # print(f"计数是 {count}")
        count += 1
    ```
- **循环控制 (`break`, `continue`)**: `break` 完全退出当前循环，而 `continue` 则跳过当前迭代的其余部分，并进入下一次迭代。

### 函数

函数是可重用的代码块，执行特定任务。它们对于编写有组织、模块化且可维护的程序必不可少。

- **定义 (`def`)**: 使用 `def` 关键字定义函数，后跟函数名、用于参数 (parameter)的括号 `()` 和冒号 `:`。下方缩进的代码块构成函数体。
- **参数和实参**: 参数是函数定义中括号内列出的变量。实参是在函数被调用时传递给函数的值。
- **返回值 (`return`)**: 函数可以选择使用 `return` 语句返回值。如果省略，函数将返回 `None`。

```python
def calculate_mean(data_list):
    """计算数字列表的算术平均值。"""
    if not data_list: # 处理空列表情况
        return 0.0
    return sum(data_list) / len(data_list)

# 调用函数
scores = [88, 92, 75, 98, 85]
average_score = calculate_mean(scores)
# print(f"平均分数: {average_score}")
```

函数有助于将复杂问题分解为更小、更易于管理的部分。它们还促进代码重用，减少冗余并使更新更简便。

这次简要回顾涵盖了绝对核心内容。随着我们继续学习，我们将在这些知识的基础上进行构建，介绍处理序列更高效的方法（推导式、生成器），编写更灵活函数的技术（高级参数、装饰器），以及组织大型应用程序的方法（面向对象编程、上下文 (context)管理器）。扎实掌握这些基本要点将使后续主题的学习过程更加顺畅。

## 参考资料

- [The Python Tutorial](https://docs.python.org/3/tutorial/) — Python Software Foundation (2023)
  Python核心特性（包括基本数据类型、控制流和函数）的官方全面介绍。
- [Introduction to Computer Science and Programming in Python](https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/) — Ana Bell, Eric Grimson, and John Guttag (2016)
  Publisher: MIT OpenCourseWare
  一门大学入门级课程，提供Python基础知识（包括数据类型、控制流和函数）的系统学习方法。
