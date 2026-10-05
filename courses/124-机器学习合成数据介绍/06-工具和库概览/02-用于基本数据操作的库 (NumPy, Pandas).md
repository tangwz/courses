# 用于基本数据操作的库 (NumPy, Pandas)

来源：[原文](https://apxml.com/zh/courses/intro-synthetic-data-ml/chapter-6-tools-libraries-overview/libraries-basic-manipulation)

[返回章节目录](README.md) · [返回课程目录](../README.md)

即使存在用于生成复杂合成数据的专门工具，许多任务仍从处理数字和表格的基本组件开始。Python生态系统中有两个库在这些基础阶段不可或缺：NumPy和Pandas。它们是启动专门机器之前必不可少的工作台工具。

### NumPy: 数字数据处理的根本

NumPy，是Numerical Python的简称，是Python中进行数值计算的核心库。它最主要的特点是提供了强大的N维数组对象，通常称为`ndarray`。这种对象在数值运算方面比标准的Python列表效率高很多，尤其是在处理大量数据时。

NumPy对合成数据有什么用处？

1. **高效的数值数组：** 它能高效创建和操作数字（整数、浮点数）数组。当合成数据需要包含年龄、身高、传感器读数或像素值等数字特征时，这一点非常必要。
2. **随机数生成：** NumPy有一个模块（`numpy.random`），用于生成遵循各种统计分布（如均匀分布、正态分布、泊瓦松分布等）的随机数。这通常是生成模拟随机性和变异性合成特征的起点，如第二章所述。
3. **数学运算：** 它提供了一系列数学函数，可以对数组进行逐元素操作。你可以轻松进行计算、转换（如缩放或平移数值），或应用数学模型来生成数据。

我们来看一个简单的例子。假设我们需要为5个人生成合成年龄，假设年龄在18到65之间均匀分布。

```python
# 导入NumPy库
import numpy as np

# 生成5个介于18（含）到66（不含）之间的随机整数
synthetic_ages = np.random.randint(low=18, high=66, size=5)

# 打印生成的年龄
print(synthetic_ages)
# 可能的输出：[42 25 61 19 33]
```

这个小片段展示了NumPy如何基于特定规则（在一个范围内的随机整数）轻松创建合成数字数据数组。

### Pandas: 组织合成数据

虽然NumPy在处理数值数组方面表现出色，但数据通常以行和列的表格形式出现，可能混合了不同的数据类型（数字、文本、类别、日期）。这时就需要用到Pandas。Pandas提供了高效、易用的数据结构和数据分析工具。

Pandas的两个主要数据结构是：

1. **Series：** 一维带标签数组，类似于电子表格或表格中的单列。
2. **DataFrame：** 二维带标签数据结构，其列可能具有不同类型，类似于电子表格、SQL表或Series对象的字典。这是表格数据最常用的结构。

Pandas对合成数据生成有什么帮助？

1. **创建表格结构：** Pandas DataFrame非常适合将生成的合成数据组织成常见的行和列格式。你可以轻松组合合成的数值数据列（可能使用NumPy生成）和合成的类别数据（如名称或标签）。
2. **处理不同数据类型：** DataFrames自然支持处理具有不同数据类型的列（整数、浮点数、字符串、布尔值），反映了真实数据集的结构。
3. **与NumPy的配合：** Pandas基于NumPy构建，这意味着它们可以很好地协同工作。你可以很方便地将NumPy数组放入Pandas DataFrame中。
4. **导出数据：** 一旦你的合成数据集在DataFrame中结构化，Pandas可以方便地将其保存为常见的格式，如CSV文件，这些文件随后可供机器学习 (machine learning)模型或其他工具使用。

我们来扩展之前的例子。我们将使用NumPy生成的`synthetic_ages`，并添加一个合成的类别特征，例如'Status'（例如'Active'、'Inactive'），创建一个小型表格数据集。

```python
# 导入Pandas库
import pandas as pd
# 导入NumPy（假设之前已导入）
import numpy as np

# 生成5个随机年龄（同前）
synthetic_ages = np.random.randint(low=18, high=66, size=5)

# 生成5个随机状态
possible_statuses = ['Active', 'Inactive']
synthetic_statuses = np.random.choice(possible_statuses, size=5)

# 创建一个Pandas DataFrame
synthetic_table = pd.DataFrame({
    'Age': synthetic_ages,
    'Status': synthetic_statuses
})

# 打印生成的表格
print(synthetic_table)
# 可能的输出：
#    Age    Status
# 0   42    Active
# 1   25  Inactive
# 2   61    Active
# 3   19    Active
# 4   33  Inactive
```

在此示例中，我们使用NumPy进行数值生成（`synthetic_ages`）和基本的随机选择（`synthetic_statuses`），然后使用Pandas将这些数据构建成一个结构化的DataFrame，呈现一个简单的合成表格数据集。

在Python中以编程方式生成合成数据时，NumPy和Pandas通常是你首先会使用的工具。它们提供创建、处理和组织数值及表格数据所需的核心功能，形成更复杂生成技术和专业库的依据。掌握它们的基本功能对任何处理数据的人都非常有益，无论是合成数据还是其他数据。通常你可以使用pip等包管理器安装它们：`pip install numpy pandas`。

## 参考资料

- [NumPy Documentation](https://numpy.org/doc/stable/) — NumPy community (2023)
  NumPy N维数组、数学函数和随机数生成功能的官方综合指南。
- [Pandas Documentation](https://pandas.pydata.org/docs/) — Pandas development team (2025)
  Pandas数据结构（如Series和DataFrame）的官方指南，对于组织和操作表格数据非常重要。
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本广受认可的书，提供了使用NumPy和Pandas进行数据操作、清洗和准备的实践指导。

---

[上一节](01-%E8%BD%AF%E4%BB%B6%E5%9C%A8%E6%95%B0%E6%8D%AE%E7%94%9F%E6%88%90%E4%B8%AD%E7%9A%84%E4%BD%9C%E7%94%A8.md) · [下一节](03-Faker%20%E5%BA%93%E7%AE%80%E4%BB%8B.md)
