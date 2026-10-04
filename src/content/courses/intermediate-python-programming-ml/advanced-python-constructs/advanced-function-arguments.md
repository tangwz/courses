---
course: "intermediate-python-programming-ml"
chapter: "advanced-python-constructs"
lesson: "advanced-function-arguments"
sourceId: 2088
sourceUrl: "https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-1-advanced-python-constructs/advanced-function-arguments"
title: "高级函数参数"
description: "掌握 `*args` 和 `**kwargs` 的使用，以创建灵活的 Python 函数。"
order: 4
plots: []
sourceHash: "43397ae6be59771ab30eb1194c5792ad2dd4fee77ed2c52dcf3eb36d97cc877e"
sourceCorrections: []
---

创建函数，特别是那些为数据分析或机器学习 (machine learning)工作流程设计、需要重复使用的复杂函数时，经常会遇到事先无法确定用户可能需要传递多少参数 (parameter)的情况。Python 提供强大的机制来处理这种灵活性：`*args` 用于可变非关键字参数，`**kwargs` 用于可变关键字参数。

### 接受任意数量的位置参数 (parameter)：`*args`

有时，您希望函数接受任意数量的位置参数。例如，设想一个旨在计算多个测量值平均值的函数，但每次调用时测量值的数量可能会不同。这时 `*args` 就派上用场了。

当您定义一个前面带有星号 (`*`) 的函数参数（例如 `*args`）时，Python 会将函数调用期间传递的任何额外位置参数收集到一个元组中，并将其赋值给该参数。`args` 这个名称是一个惯例，但从技术上讲，您可以使用任何有效的变量名（例如 `*numbers`、`*values`）。

请看这个示例：

```python
def calculate_sum(*numbers):
  """计算任意数量数值参数的总和。"""
  print(f"接收到的参数作为元组: {numbers}")
  total = 0
  for number in numbers:
    total += number
  return total

# 用不同数量的参数调用函数
sum1 = calculate_sum(1, 2, 3)
print(f"总和 1: {sum1}")

sum2 = calculate_sum(10, 20, 30, 40, 50)
print(f"总和 2: {sum2}")

sum3 = calculate_sum(5)
print(f"总和 3: {sum3}")

sum4 = calculate_sum() # 即使没有额外参数也能正常工作
print(f"总和 4: {sum4}")
```

在每次调用中，参数 `1, 2, 3` 或 `10, 20, 30, 40, 50` 等会被收集到函数内的 `numbers` 元组中。这使得 `calculate_sum` 函数无论提供了多少数字都能有效运行。这种模式在数据科学中对于可能从可变数量的输入数组或列中聚合数据的函数很有用。

### 接受任意数量的关键字参数 (parameter)：`**kwargs`

同样，您可能需要一个函数来接受任意数量的关键字参数（以 `key=value` 格式传递的参数）。这在为一个过程设置可选的配置参数或定义元数据时很常见。Python 为此使用双星号 (`**`) 语法，通常是 `**kwargs`（关键字参数）。

当 `**kwargs` 用于函数定义时，Python 会将任何与其它已定义参数名不匹配的关键字参数收集到一个字典中。字典的键是参数名（字符串形式），值是对应的参数值。

让我们看一个模拟记录处理详情的函数的例子：

```python
def log_processing_step(step_name, status, **details):
  """记录数据处理步骤的详情。"""
  print(f"--- 日志条目 ---")
  print(f"步骤: {step_name}")
  print(f"状态: {status}")
  if details:
    print("附加详情：")
    for key, value in details.items():
      print(f"  {key}: {value}")
  print(f"-----------------\n")

# 用不同的关键字参数调用函数
log_processing_step("Data Loading", "Success", source_file="data.csv", rows_loaded=1500)

log_processing_step("Feature Scaling", "Completed", method="StandardScaler", columns_scaled=5)

log_processing_step("Model Training", "Warning", algorithm="LinearRegression") # 没有额外详情
```

在第一次调用中，`source_file="data.csv"` 和 `rows_loaded=1500` 被收集到 `details` 字典中：`{'source_file': 'data.csv', 'rows_loaded': 1500}`。在第二次调用中，`details` 变为 `{'method': 'StandardScaler', 'columns_scaled': 5}`。在第三次调用中，`details` 是一个空字典。这使得 `log_processing_step` 能够灵活处理不同级别的详情，这是一种在配置 Scikit-learn 等库中的步骤或定义绘图函数参数时经常出现的模式。

### 结合使用 `*args` 和 `**kwargs`

您可以在同一个函数定义中同时使用 `*args` 和 `**kwargs` 来接受任意位置参数 (parameter)和关键字参数。函数定义中参数的标准顺序是：

1. 标准位置参数
2. `*args`
3. 标准仅限关键字参数（如果有）
4. `**kwargs`

这是一个包含两者的示例：

```python
def flexible_function(required_param, *args, default_kwarg='default', **kwargs):
  """演示如何使用标准参数、*args、默认关键字参数和**kwargs。"""
  print(f"必选参数: {required_param}")
  print(f"默认关键字参数: {default_kwarg}")

  if args:
    print(f"位置参数 (*args): {args}")

  if kwargs:
    print(f"关键字参数 (**kwargs): {kwargs}")
  print("-" * 20)

flexible_function(10)
flexible_function(20, 100, 200)
flexible_function(30, default_kwarg='custom')
fle_function(40, 101, 102, config_option='A', user_id=99)
fle_function(50, config_option='B', user_id=100, default_kwarg='overridden')
```

注意看不同参数组合是如何正确赋值给 `required_param`、`args`、`default_kwarg` 和 `kwargs` 的。

### 函数调用时的参数 (parameter)解包

星号 (`*`) 和双星号 (`**`) 运算符也可以在 *调用* 函数时使用。这允许您将序列（如列表或元组）“解包”为位置参数，将字典“解包”为关键字参数。

```python
def process_data(id, value, category):
  """一个期望三个特定参数的简单函数。"""
  print(f"处理 ID: {id}, 值: {value}, 类别: {category}")

# 存储在列表/元组中的数据
data_list = [101, 55.5, 'Electronics']
process_data(*data_list) # 将列表解包为位置参数

# 存储在字典中的数据
data_dict = {'id': 202, 'value': 78.9, 'category': 'Clothing'}
process_data(**data_dict) # 将字典解包为关键字参数

# 也可以结合使用：
more_data_dict = {'value': 15.2, 'category': 'Books'}
process_data(303, **more_data_dict) # 位置传递 'id'，其他通过字典传递
```

当函数调用的参数是动态生成或存储在数据结构中时，这种解包机制非常有用。

了解 `*args` 和 `**kwargs` 对于编写灵活的 Python 函数很重要。在数据科学和机器学习 (machine learning)中，这种灵活性使您能够创建更通用的工具，优雅地处理不同的输入，并通过可选参数配置模型训练或数据转换管道等复杂操作，从而编写出更具复用性和可维护性的代码。

## 参考资料

- [Fluent Python: Clear, Concise, and Effective Programming](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056300/) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media
  一本关于Python高级特性的综合指南，清晰地说明了灵活函数参数及其在函数设计中的实际应用。
- [Python Cookbook, 3rd Edition](https://www.vitalsource.com/products/python-cookbook-david-beazley-brian-k-jones-v9781449357368) — David Beazley and Brian K. Jones (2013)
  Publisher: O'Reilly Media; Pages: 706
  提供了Python高级编程的实用技巧和解决方案，包括在函数中处理可变位置参数和关键字参数的具体技术。
