---
course: "python-for-beginners"
chapter: "reusing-code-modules-packages"
lesson: "practice-modules"
sourceId: 2365
sourceUrl: "https://apxml.com/zh/courses/python-for-beginners/chapter-7-reusing-code-modules-packages/practice-modules"
title: "实践：使用标准模块和外部模块"
description: "通过导入和运用标准模块及已安装的外部模块中的函数和类，获取实际经验。"
order: 8
plots: []
sourceHash: "730e05df0de6918cb1d9535e076969d2169eec62a7e4fa40ec1794f5f2a836b2"
sourceCorrections: []
---

通过使用 Python 标准库中的模块以及安装和使用一个简单的外部包来提供示例。

### 处理标准库

Python 自带一个内容丰富的标准库，为许多常见任务提供即用型模块。对于这些模块，你无需额外安装任何东西；它们是 Python 安装的一部分。

#### 示例 1：使用 `math` 模块

`math` 模块提供对数学函数的访问。假设你需要计算一个数的平方根，或者找到向上取整的值（大于或等于一个数的最小整数）。

1. **导入模块：** 通过导入 `math` 模块来开始你的脚本。

   ```python
   import math
   ```
2. **使用其函数：** 现在你可以使用点号表示法（`module_name.function_name`）访问 `math` 模块中的函数。

   ```python
   import math

   number = 16
   square_root = math.sqrt(number)
   print(f"The square root of {number} is {square_root}") # 输出: 16的平方根是 4.0

   another_number = 9.3
   ceiling_value = math.ceil(another_number)
   print(f"The ceiling value of {another_number} is {ceiling_value}") # 输出: 9.3的向上取整值是 10
   ```

   请注意我们如何用 `math.` 作为 `sqrt` 和 `ceil` 的前缀，以告诉 Python 这些函数来自哪里。

#### 示例 2：使用 `random` 模块

需要生成随机数或进行随机选择？ `random` 模块是你的工具。

1. **导入模块：**

   ```python
   import random
   ```
2. **使用其函数：** 让我们在一个特定范围内生成一个随机整数，并从列表中随机选取一个项。

   ```python
   import random

   # 生成一个介于1到10（包括1和10）之间的随机整数
   random_integer = random.randint(1, 10)
   print(f"A random integer: {random_integer}")

   # 从列表中选择一个随机元素
   options = ['apple', 'banana', 'cherry', 'date']
   random_choice = random.choice(options)
   print(f"A random fruit: {random_choice}")
   ```

   每次运行此脚本时，你很可能会得到不同的随机整数和水果输出。

#### 示例 3：使用 `from ... import`

有时，你可能只需要模块中的一两个特定项，或者想避免重复输入模块名称。你可以使用 `from ... import` 语法。让我们用这种方法重新进行平方根计算。

```python
from math import sqrt, ceil # 直接导入 sqrt 和 ceil

number = 25
square_root = sqrt(number) # 注意：现在不需要 'math.' 前缀了
print(f"The square root of {number} is {square_root}") # 输出: 25的平方根是 5.0

another_number = 4.1
ceiling_value = ceil(another_number) # 不需要 'math.' 前缀
print(f"The ceiling value of {another_number} is {ceiling_value}") # 输出: 4.1的向上取整值是 5
```

虽然这可以使代码更短，但请注意，直接导入许多名称可能会使脚本的命名空间混乱，并且如果不同模块具有相同名称的函数或变量，可能会导致命名冲突。对于大型程序，使用 `import module_name` 通常更清晰。

### 处理外部包

Python 包索引 (PyPI) 托管了社区创建的数千个外部包。这些包显著扩展了 Python 的能力。要使用它们，你首先需要使用 `pip` 进行安装。

#### 示例 4：安装和使用 `requests` 包

`requests` 包是一个非常流行的库，用于进行 HTTP 请求（例如，获取网页）。

1. **安装包：** 打开你的终端或命令提示符（不是 Python 解释器）。输入以下命令并按回车键：

   ```bash
   pip install requests
   ```

   你应该会看到输出，表示 `requests`（以及可能的一些依赖项）正在下载和安装。
2. **在脚本中使用包：** 现在你可以像使用标准库模块一样导入和使用 `requests`。让我们获取一个简单示例网站的内容。

   ```python
   import requests
   import datetime # 我们也导入 datetime，看看我们何时运行它

   try:
       # 对一个 URL 发送 GET 请求
       response = requests.get('https://httpbin.org/get')
       response.raise_for_status() # 对于错误的HTTP状态码（4xx或5xx）抛出异常

       # 打印关于响应的一些信息
       print(f"Request successful at: {datetime.datetime.now()}")
       print(f"Status Code: {response.status_code}")
       # print(f"内容（前150个字符）：{response.text[:150]}...") # 取消注释以查看内容

   except requests.exceptions.RequestException as e:
       # 处理潜在错误，如网络问题或不良响应
       print(f"An error occurred: {e}")
       print(f"Request failed at: {datetime.datetime.now()}")
   ```

   此脚本尝试从一个测试 URL 获取数据。它使用 `try...except` 块（你之前学过）来优雅地处理潜在的网络错误。它打印服务器返回的状态码（200通常表示成功）以及请求发出时间。

### 小型任务：组合模块

让我们将所学知识结合起来。编写一个简短的脚本，该脚本：

1. 导入 `random` 和 `math` 模块。
2. 使用 `random.uniform()` 生成一个介于 1.0 和 100.0 之间的随机浮点数。
3. 使用 `math.floor()` 计算该数的向下取整值（小于或等于该数的最大整数）。
4. 打印原始随机数及其向下取整值。

**解决方案：**

```python
import random
import math

# 1. 生成一个介于 1.0 和 100.0 之间的随机浮点数
random_float = random.uniform(1.0, 100.0)

# 2. 计算其向下取整值
floor_value = math.floor(random_float)

# 3. 打印结果
print(f"Generated random float: {random_float:.2f}") # 格式化为 2 位小数
print(f"Floor value: {floor_value}")
```

此实践展示了标准模块和外部模块如何让你轻松地将强大的功能集成到程序中，而无需从头开始编写所有内容。随着你构建更复杂的应用程序，有效使用模块和包将是保持代码有条理、可读和可维护的根本。你可以尝试 `math`、`random` 和 `datetime` 模块中的其他函数，或者尝试从 PyPI 安装并使用另一个简单的包。

## 参考资料

- [The Python Standard Library](https://docs.python.org/3/library/index.html) — Python Software Foundation (2024)
  提供所有内置模块的全面文档，包括作为示例使用的`math`和`random`模块。
- [Installing Python Modules](https://docs.python.org/3/installing/index.html) — Python Software Foundation (2024)
  解释如何使用pip安装外部包，这是管理Python依赖的核心方法。
- [Requests: HTTP for Humans™](https://requests.readthedocs.io/en/latest/) — Kenneth Reitz and individual contributors (2024)
  `requests`库的官方文档，该库在外部包使用示例中作为主要案例。
- [Fluent Python: Clear, Concise, and Effective Programming](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056355/) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media; Pages: 1014
  详细介绍Python的模块和包系统，并提供实用的结构化指导。
