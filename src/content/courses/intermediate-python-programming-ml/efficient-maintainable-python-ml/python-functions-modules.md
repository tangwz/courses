---
course: "intermediate-python-programming-ml"
chapter: "efficient-maintainable-python-ml"
lesson: "python-functions-modules"
sourceId: 2216
sourceUrl: "https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-6-efficient-maintainable-python-ml/python-functions-modules"
title: "编写高效的函数和模块"
description: "关于在Python中编写模块化、可复用且文档完善的函数和模块的最佳实践。"
order: 3
plots: []
sourceHash: "22c4ade88ddfe2feccf2044799e567485789e9d40f9788a645fccbbc2eb9098a"
sourceCorrections: []
---

随着机器学习 (machine learning)项目复杂度的增加，仅仅编写“能运行”的代码是不够的。轻松理解、修改和复用代码组件的能力变得非常重要，有助于提升效率和协作。这正是函数和模块的精心设计发挥作用的地方。它们是构建整洁、可维护和可扩展代码的基础要素。

### 设计高效函数

函数允许您封装特定逻辑，为其命名，并在需要时执行。精心设计的函数能显著提升代码的可读性并减少重复。在为机器学习 (machine learning)工作流程编写函数时，请考虑以下原则：

#### 单一职责：做好一件事

每个函数都应有一个单一、明确的用途。一个名为 `load_and_preprocess_data` 的函数可能做得太多了。它违反了单一职责原则（SRP）。如果需要更改缺失数据的处理方式，您将不得不修改一个同时处理文件加载和特征缩放的大型函数。

相反，将其分解：

```python
def load_data(file_path: str) -> pd.DataFrame:
    """从CSV文件加载数据。"""
    # 数据加载实现
    pass

def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """填充或移除缺失值。"""
    # 处理NaN的实现
    pass

def scale_features(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """缩放指定的数值特征。"""
    # 特征缩放实现
    pass
```

这种方法使代码更易于理解、测试和修改。如果数据加载格式发生变化，您只需更新 `load_data` 函数。

#### 使用清晰且有描述性的名称

函数名称应清楚表明其用途。动词应用于表示操作（例如，`calculate_accuracy`、`train_model`、`plot_confusion_matrix`），并遵循一致的命名约定，通常是PEP 8推荐的 `snake_case` 风格。避免使用 `process_data` 或 `run_analysis` 等模糊名称。具体说明正在处理或分析的*内容*。

#### 保持函数简短

过长的函数更难阅读、理解和调试。如果一个函数跨越多个屏幕，通常表明它做得太多了，应该分解成更小、更辅助的函数。对长度没有严格规定，但应致力于让函数只专注于一个逻辑步骤。

#### 定义清晰的输入和输出

函数通过其参数 (parameter)（输入）和返回值（输出）与代码的其余部分进行通信。

- **参数：** 明确函数需要哪些数据。使用描述性的参数名称。
- **类型提示：** Python 的类型提示（使用 `typing` 模块）能显著提升清晰度。它们声明了参数和返回值的预期类型。这可作为文档，并允许静态分析工具发现潜在错误。

```python
from typing import List, Tuple
import numpy as np

def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """计算两个边界框的交并比（IoU）。

    参数：
        box1: 表示第一个边界框的列表 [x_min, y_min, x_max, y_max]。
        box2: 表示第二个边界框的列表 [x_min, y_min, x_max, y_max]。

    返回：
        0.0 到 1.0 之间的浮点型 IoU 分数。
    """
    # 实现...
    iou = 0.0 # 占位符
    return iou
```

- **返回值：** 函数理想情况下应返回结果，而不是直接修改全局状态或输入对象（除非这是其明确目的，例如 Pandas 中的就地操作，应谨慎使用）。如果函数需要返回多个值，请使用元组或自定义对象/字典。

#### 使用文档字符串记录您的函数

每个非简单的函数都应有文档字符串，说明其功能、参数、返回值以及可能引发的任何异常。这对可维护性非常重要，尤其是在团队协作或以后回顾自己的代码时。常用格式包括 Google 风格和 NumPy 风格。

```python
import pandas as pd

def summarize_dataframe(df: pd.DataFrame) -> dict:
  """提供 Pandas DataFrame 的基本概要。

  参数：
    df: 输入的 Pandas DataFrame。

  返回：
    包含概要统计信息的字典：
      'num_rows': 行数。
      'num_cols': 列数。
      'missing_counts': 每列缺失值的计数 Series。

  引发：
    TypeError: 如果输入不是 Pandas DataFrame。
  """
  if not isinstance(df, pd.DataFrame):
    raise TypeError("输入必须是 Pandas DataFrame。")

  summary = {
      'num_rows': len(df),
      'num_cols': len(df.columns),
      'missing_counts': df.isnull().sum()
  }
  return summary
```

一致的文档字符串使您的代码无需阅读实现细节即可理解。

### 使用模块组织代码

随着项目规模的增长，将所有函数放在一个文件中变得难以管理。Python 的模块系统允许您将相关代码组织到单独的文件（`.py` 文件是模块）和目录（包）中。

#### 为何使用模块？

- **组织：** 将相关函数和类分组（例如，所有数据加载函数放在 `data_loader.py` 中，所有预处理步骤放在 `preprocessing.py` 中）。
- **命名空间管理：** 避免项目中不同部分之间的命名冲突。`preprocessing.calculate_mean()` 与 `evaluation.calculate_mean()` 是不同的。
- **复用性：** 定义良好的模块可以在不同项目之间导入和复用。
- **可维护性：** 与特定功能（如数据验证）相关的更改都局限于特定模块内。

#### 创建和使用模块

只需将您的 Python 代码保存在 `.py` 文件中。例如，创建一个名为 `feature_engineering.py` 的文件：

```python
# feature_engineering.py
import pandas as pd
from typing import List

def create_polynomial_features(df: pd.DataFrame, columns: List[str], degree: int = 2) -> pd.DataFrame:
  """为指定列创建多项式特征。"""
  df_poly = df.copy()
  for col in columns:
      for d in range(2, degree + 1):
          df_poly[f'{col}_pow{d}'] = df_poly[col] ** d
  return df_poly

def create_interaction_features(df: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
  """在两列之间创建交互特征。"""
  df_interact = df.copy()
  df_interact[f'{col1}_x_{col2}'] = df_interact[col1] * df_interact[col2]
  return df_interact
```

现在，在另一个脚本中（例如，`main_script.py`），您可以导入并使用这些函数：

```python
# main_script.py
import pandas as pd
import feature_engineering as fe # 导入模块

# 假设 'my_data' 是一个 pandas DataFrame
# my_data = pd.read_csv(...)

# 应用模块中的函数
numerical_cols = ['age', 'income']
my_data_poly = fe.create_polynomial_features(my_data, numerical_cols, degree=3)
my_data_final = fe.create_interaction_features(my_data_poly, 'age', 'income')

print(my_data_final.head())
```

或者，您可以导入特定函数：

```python
# main_script.py (另一种导入方式)
import pandas as pd
from feature_engineering import create_polynomial_features

# ...
my_data_poly = create_polynomial_features(my_data, ['age', 'income'])
```

#### 使用包组织项目结构

对于大型项目，您可以将相关模块分组到目录中。为了让 Python 将目录视为包（从中可以导入模块），请在该目录中包含一个名为 `__init__.py` 的空文件。

一个典型的机器学习 (machine learning)项目结构可能如下所示：

```
my_ml_project/
├── data/                 # 原始数据和处理后的数据
│   ├── raw/
│   └── processed/
├── notebooks/            # 用于分析的 Jupyter notebook
├── src/                  # 源代码
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── training.py
│   ├── evaluation.py
│   └── utils.py          # 常用工具函数
├── tests/                # 单元测试
│   ├── test_preprocessing.py
│   └── ...
├── requirements.txt      # 项目依赖
└── main.py               # 运行流水线的主脚本
```

> 组织机器学习项目的常用目录结构，将数据、notebook、源代码（src）和测试分离。

从 `main.py` 或 notebook 中，您可以这样导入：

```python
from src.preprocessing import handle_missing_values
from src.feature_engineering import create_polynomial_features
from src.models import train_linear_regression
```

`__init__.py` 文件也可以用于控制使用 `from package import *` 时暴露哪些符号（尽管为了清晰起见，通常不推荐这种导入方式）或者运行包的初始化代码。

#### 避免循环导入

一个常见问题是创建循环依赖，即模块 `A` 导入模块 `B`，而模块 `B` 又导入模块 `A`。这通常发生在模块定义不佳或试图做太多事情时。在这种情况下，Python 将引发 `ImportError`。适当的结构设计、遵循模块单一职责原则，有时整合相关性强的函数或使用工具模块，都能有助于防止这种情况发生。

通过应用这些原则来编写函数并将其组织成模块，您可以创建一个更容易管理、测试、调试和扩展的代码库。这是构建可靠高效机器学习系统不可或缺的支撑。

## 参考资料

- [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/) — Guido van Rossum, Barry Warsaw, and Alyssa Coghlan (2001)
  Python官方风格指南，建立了代码一致性规范，包括命名和格式。
- [Clean Code: A Handbook of Agile Software Craftsmanship](https://www.informit.com/store/clean-code-a-handbook-of-agile-software-craftsmanship-9780132350884) — Robert C. Martin (2008)
  Publisher: Prentice Hall
  一本关于软件工艺的重要著作，提供了编写清晰、可读和可维护代码的原则，尤其关注有效函数设计。
- [Effective Python: 90 Specific Ways to Write Better Python](https://www.informit.com/store/effective-python-90-specific-ways-to-write-better-python-9780134853987) — Brett Slatkin (2019)
  Publisher: Addison-Wesley Professional
  提供了编写健壮、高效和符合Python惯用法的实用建议和最佳实践，包括函数、模块和类型提示方面的指导。
- [The Python Tutorial - Modules](https://docs.python.org/3/tutorial/modules.html) — Python Software Foundation (2023)
  Python官方文档，解释了模块和包的概念、创建、导入机制和组织方式。
