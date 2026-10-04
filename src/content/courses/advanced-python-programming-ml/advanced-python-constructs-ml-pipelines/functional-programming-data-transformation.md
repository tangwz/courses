---
course: "advanced-python-programming-ml"
chapter: "advanced-python-constructs-ml-pipelines"
lesson: "functional-programming-data-transformation"
sourceId: 2242
sourceUrl: "https://apxml.com/zh/courses/advanced-python-programming-ml/chapter-1-advanced-python-constructs-ml-pipelines/functional-programming-data-transformation"
title: "Python中用于数据转换的函数式编程模式"
description: "应用函数式编程思想，如map、filter、reduce和partial函数，实现更清晰的数据转换。"
order: 3
plots: []
sourceHash: "2dd749bd883d39d1f8a95ea1bfa4e1d9543054729523da83b29d9983af44cb06"
sourceCorrections: []
---

函数式编程为数据操作提供了一种声明式方法，可以显著提升机器学习 (machine learning)流程中数据转换步骤的清晰度、可测试性和可组合性。函数式模式不命令式地规定数据如何迭代和修改，而是侧重于描述应该发生什么转换，通常将函数作为一等公民对待并强调不变性。这与构建可靠且易于理解的机器学习系统的目标相符。

### 主要思想：纯函数与不变性

函数式编程的核心在于**纯函数**。一个纯函数有两个主要特性：

1. **确定性：** 给定相同的输入，它总是返回相同的输出。
2. **无副作用：** 它不修改其局部范围之外的任何状态（例如，修改全局变量、就地改变输入参数 (parameter)、执行I/O操作）。

用纯函数构建的数据转换更容易推断和测试。你可以根据输入精确预测它们的输出，而不必担心隐藏的状态变化。

与之密切相关的是**不变性**原则。函数式转换通常不是修改现有数据结构，而是创建并返回应用了更改的*新*数据结构。虽然这乍一看可能效率不高，但它能防止流程的某一部分无意中更改了其他地方使用的数据，从而避免意料之外的副作用。现代库通常会实现优化（如写时复制）以减轻性能问题。

### 使用`map`进行逐元素转换

`map(function, iterable)`内置函数将给定的`函数`应用于`可迭代对象`（如列表、元组或生成器）的每个元素，并返回一个生成结果的迭代器。它是表示逐元素操作的一种直接方式。

考虑一个简单的任务：将数值特征按10的系数进行缩放。

```python
import math

features = [1.2, 0.5, 3.4, -0.8, 2.1]

# 使用map和已定义的函数
def scale_feature(x):
  return x * 10

scaled_features_iterator = map(scale_feature, features)
scaled_features = list(scaled_features_iterator)
print(scaled_features)
# Output: [12.0, 5.0, 34.0, -8.0, 21.0]

# 使用map和lambda函数（更简洁）
log_features_iterator = map(lambda x: math.log(x) if x > 0 else 0, features)
log_features = list(log_features_iterator)
print(log_features)
# Output: [0.1823215567939546, -0.6931471805599453, 1.2237754316221157, 0, 0.741937344729377]
```

尽管`map`明确体现了函数式模式，但Python在许多常见情况下更倾向于使用**列表推导式**或**生成器表达式**以提高可读性：

```python
# 使用列表推导式的等效代码
scaled_features_comp = [x * 10 for x in features]
print(scaled_features_comp)
# Output: [12.0, 5.0, 34.0, -8.0, 21.0]

# 使用生成器表达式的等效代码（内存高效）
log_features_genexp = (math.log(x) if x > 0 else 0 for x in features)
# log_features_genexp 是一个生成器对象
print(list(log_features_genexp))
# Output: [0.1823215567939546, -0.6931471805599453, 1.2237754316221157, 0, 0.741937344729377]
```

即使在使用推导式时，理解`map`模式也很有价值，因为它鼓励将转换视为将函数应用于序列中的每个元素。

### 使用`filter`进行选择性筛选

`filter(function, iterable)`内置函数从`可迭代对象`中构建一个迭代器，其中`函数`对元素返回真值。它允许你根据条件选择性地保留数据。

假设我们只想保留列表中的正特征：

```python
features = [1.2, 0.5, 3.4, -0.8, 2.1]

# 使用filter和已定义的函数
def is_positive(x):
  return x > 0

positive_features_iterator = filter(is_positive, features)
positive_features = list(positive_features_iterator)
print(positive_features)
# Output: [1.2, 0.5, 3.4, 2.1]

# 使用filter和lambda函数
non_negative_iterator = filter(lambda x: x >= 0, features)
print(list(non_negative_iterator))
# Output: [1.2, 0.5, 3.4, 2.1]
```

同样，带有`if`子句的列表推导式在Python中通常提供更直接的语法：

```python
# 使用列表推导式的等效代码
positive_features_comp = [x for x in features if x > 0]
print(positive_features_comp)
# Output: [1.2, 0.5, 3.4, 2.1]

# 在推导式中结合map和filter的逻辑
# 只缩放正特征
scaled_positive_features = [x * 10 for x in features if x > 0]
print(scaled_positive_features)
# Output: [12.0, 5.0, 34.0, 21.0]
```

`filter`模式鼓励将数据选择视为应用一个谓词函数。

### 使用`functools.reduce`进行聚合

为了将序列中的值聚合为单个结果，函数式模式使用`reduce`。在Python中，这个函数位于`functools`模块中。`reduce(function, iterable[, initializer])`将`函数`累积地应用于`可迭代对象`中的元素，从左到右，从而将可迭代对象归约为单个值。该`函数`必须接受两个参数 (parameter)。

```python
from functools import reduce
import operator

numbers = [1, 2, 3, 4, 5]

# 计算和
total = reduce(lambda x, y: x + y, numbers)
print(total)
# Output: 15

# 使用operator模块函数计算乘积
product = reduce(operator.mul, numbers)
print(product)
# Output: 120

# 使用初始值（对空序列或设置起始值很有用）
total_with_init = reduce(operator.add, numbers, 100) # 从100开始求和
print(total_with_init)
# Output: 115
```

尽管`reduce`功能强大，但有时它会使代码比简单的`for`循环更难阅读。诸如`sum()`、`max()`、`min()`和`all()`之类的内置函数通常为常见的归约任务提供了更清晰的替代方案。当标准聚合函数不适用或需要组合复杂的归约逻辑时，请慎重使用`reduce`。

### 使用`functools.partial`创建专用函数

在数据流程中，你通常有一个通用转换函数，但需要多次应用它，并固定一些参数 (parameter)。`functools.partial(func, /, *args, **keywords)`返回一个新的partial对象，该对象在被调用时，其行为将类似于`func`被预置了位置参数`args`和关键字参数`keywords`后被调用。

设想一个通用归一化 (normalization)函数，并且你想为Min-Max缩放和Z-score标准化创建特定版本：

```python
from functools import partial
import numpy as np

def normalize(data, method='minmax', epsilon=1e-8):
  """使用指定方法对数据进行归一化。"""
  data = np.asarray(data)
  if method == 'minmax':
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val + epsilon)
  elif method == 'zscore':
    mean_val = np.mean(data)
    std_val = np.std(data)
    return (data - mean_val) / (std_val + epsilon)
  else:
    raise ValueError(f"Unknown normalization method: {method}")

data_points = np.array([10, 20, 30, 40, 50])

# 使用partial创建特定的Min-Max缩放器函数
minmax_scaler = partial(normalize, method='minmax')

# 创建特定的Z-score缩放器函数
zscore_scaler = partial(normalize, method='zscore')

# 应用它们
scaled_minmax = minmax_scaler(data_points)
scaled_zscore = zscore_scaler(data_points)

print("Original:", data_points)
print("Min-Max Scaled:", scaled_minmax)
print("Z-Score Scaled:", scaled_zscore)
# Output:
# Original: [10 20 30 40 50]
# Min-Max Scaled: [0.   0.25 0.5  0.75 1.  ]
# Z-Score Scaled: [-1.41421356 -0.70710678  0.          0.70710678  1.41421356]
```

使用`partial`有助于为你的流程阶段创建清晰、可重用的转换组件，而无需为简单的参数固定而求助于包装函数或类。

### 可组合性

使用纯函数式转换的一个主要优点是**可组合性**。简单的函数可以被链接或组合起来构建复杂的数据处理流程。由于纯函数没有副作用，应用顺序（对于独立转换）通常无关紧要，或者流程非常明确。

考虑清洗和处理文本数据：

```python
import re
from functools import reduce

def lowercase(text):
  return text.lower()

def remove_punctuation(text):
  return re.sub(r'[^\w\s]', '', text)

def tokenize(text):
  return text.split()

# 定义转换序列
transformations = [lowercase, remove_punctuation, tokenize]

# 手动组合函数
def process_text_manual(text):
  result = text
  for func in transformations:
    result = func(result)
  return result

# 替代方法：使用reduce组合（对于简单链条可读性较差）
def compose(*functions):
    # 注意：reduce对于典型组合是右到左应用函数
    # f(g(h(x))) -> reduce(lambda f, g: lambda x: f(g(x)), functions)
    # 但管道是左到右应用：
    # x -> h -> g -> f
    return reduce(lambda res, f: f(res), functions, None) # 需要初始值

def process_text_composed(text):
    # 这个组合辅助函数需要改进以应用于初始文本
    # 更简单的管道方法通常更好：
    pipeline = reduce(lambda f, g: lambda x: g(f(x)), transformations)
    return pipeline(text)

# 针对此情况的更简单链式调用
raw_text = "Here is Some Text, with punctuation!"
processed = tokenize(remove_punctuation(lowercase(raw_text)))
print(processed)
# Output: ['here', 'is', 'some', 'text', 'with', 'punctuation']

# 使用简单循环（对于线性管道通常最清晰）
result = raw_text
for transform in transformations:
    result = transform(result)
print(result)
# Output: ['here', 'is', 'some', 'text', 'with', 'punctuation']
```

尽管使用`reduce`进行直接的函数组合可能很难正确实现，但链接定义良好、独立的转换步骤的*理念*是核心所在。无论你通过显式顺序调用、循环函数，还是专用管道工具（如我们稍后将看到的Scikit-learn中的工具）来实现这一点，函数式风格都鼓励将复杂转换分解为更小、更易管理且可重用的部分。

应用这些函数式编程模式，即`map`用于转换，`filter`用于选择，`reduce`用于聚合（需谨慎使用），`partial`用于专用化，并使用纯函数和不变性，可以使机器学习 (machine learning)流程中的数据转换代码更具声明性、可测试性和可维护性。这些技术补充了之前讨论的生成器和上下文 (context)管理器模式，有助于高效处理数据。

## 参考资料

- [Functional Programming HOWTO](https://docs.python.org/3/howto/functional.html) — A. M. Kuchling (2024)
  提供了函数式编程概念及其在 Python 中实际应用的全面介绍，详细说明了 map、filter 和 reduce 等内置函数以及 lambda 函数。
- [functools - Higher-order functions and operations on callable objects](https://docs.python.org/3/library/functools.html) — Python Documentation (2024)
  `functools` 模块的官方文档，对理解用于函数专门化的 `partial` 和用于聚合模式的 `reduce` 至关重要。
- [Fluent Python: Clear, Concise, and Effective Programming](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056348/) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media; Pages: 1014
  一本关于地道 Python 的权威指南，涵盖了生成器、列表推导式和与数据转换相关的函数式编程方面等高级特性。第二版。
