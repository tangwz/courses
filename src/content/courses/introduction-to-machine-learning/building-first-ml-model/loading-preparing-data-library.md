---
course: "introduction-to-machine-learning"
chapter: "building-first-ml-model"
lesson: "loading-preparing-data-library"
sourceId: 1537
sourceUrl: "https://apxml.com/zh/courses/introduction-to-machine-learning/chapter-7-building-first-ml-model/loading-preparing-data-library"
title: "使用库加载和准备数据"
description: "介绍如何使用Scikit-learn等机器学习库来加载和准备数据。"
order: 3
plots: []
sourceHash: "6ed63040ae866e19298272909636817bbd9e22dac3b0852aa975ee0a84a7139b"
sourceCorrections: []
---

既然您已经了解了构建机器学习 (machine learning)模型的整体步骤，接下来我们看看如何实际操作。从头手动实现数据加载、预处理和算法功能是可行的，但通常效率低下且容易出错，尤其当数据集和模型变得更复杂时。幸好有强大的库可以有效处理这些常见任务。

### 使用库提高效率

机器学习 (machine learning)库为许多标准操作提供了预构建、优化且经过充分测试的功能。使用它们可以使您专注于模型构建的更上层内容，例如理解数据、选择合适的算法以及解释结果，而不是陷入低级别的实现细节中。

Python生态系统中机器学习常用的库之一是 **Scikit-learn**（通常导入为`sklearn`）。它提供了数据预处理、模型选择、训练各种算法（回归、分类、聚类等）以及评估的工具。在实践中构建机器学习模型时，Scikit-learn将用于完成这些任务。我们还会经常使用 **Pandas**，这是另一个Python库，它非常适合处理和操作结构化数据，例如您经常在表格或电子表格（如CSV文件）中找到的数据。

### 加载数据集

第一步始终是将数据载入编程环境。数据可以来自各种来源，但常见格式包括CSV（逗号分隔值）文件或数据库。像Pandas这样的库使得加载这些数据变得简单明了。

假设您的数据在一个名为`dataset.csv`的CSV文件中。您可以使用一个简单的命令将其加载到Pandas DataFrame中，Pandas DataFrame本质上是一个表格结构：

```python
import pandas as pd

# 从CSV文件加载数据
data = pd.read_csv('dataset.csv')

# 显示前几行以检查数据
print(data.head())
```

此命令读取文件并将其内容存储在`data`变量中。`head()`方法对于快速检查前几行和列名以确保数据正确加载很有用。

Scikit-learn还包含一些小型标准数据集，它们有助于学习和测试算法，而无需寻找外部文件。您可以直接加载这些数据集：

```python
from sklearn.datasets import load_iris

# 加载内置的Iris数据集
iris_data = load_iris()

# 数据本身（特征）通常在'.data'中
# 目标标签通常在'.target'中
# 特征名称和目标名称也可能可用
X_iris = iris_data.data
y_iris = iris_data.target

print("Iris 特征形状:", X_iris.shape)
print("Iris 目标形状:", y_iris.shape)
```

### 分离特征和目标变量

机器学习 (machine learning)模型学习从输入特征到输出目标的映射。因此，您需要将加载的数据分成两个不同的部分：

1. **特征 (X)：** 模型将用来进行预测的输入变量或属性。这通常是一个2D数组或DataFrame，其中行代表样本，列代表特征。
2. **目标 (y)：** 模型试图预测的输出变量（例如，回归中的价格，分类中的类别标签）。这通常是一个1D数组或Series，每个样本有一个值。

如果您使用Pandas加载了数据，可以选择列来创建您的`X`和`y`。假设目标变量在一个名为“target\_column”的列中：

```python
# 假设 'data' 是您之前加载的Pandas DataFrame
# 选择除目标列以外的所有列作为特征
X = data.drop('target_column', axis=1)

# 仅选择目标列作为目标变量
y = data['target_column']

# 显示形状以验证
print("特征 (X) 形状:", X.shape)
print("目标 (y) 形状:", y.shape)
```

`drop`函数中的`axis=1`参数 (parameter)告诉Pandas删除列，而不是行。

### 使用Scikit-learn应用基本准备步骤

在第6章中，我们讨论了为什么处理缺失值、特征缩放和数据分割等步骤是必需的。Scikit-learn提供了方便的工具来执行这些操作。

**1. 处理缺失值：**
如果您的数据集中有缺失条目（通常表示为`NaN`），您可以使用Scikit-learn的`SimpleImputer`来填充它们，例如使用相应列的平均值。

```python
from sklearn.impute import SimpleImputer
import numpy as np # 常用于表示NaN

# 假设X可能包含缺失值（表示为np.nan）
# 创建一个填充器对象，用平均值替换NaN
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# 将填充器拟合到数据（计算平均值）并转换X
X = imputer.fit_transform(X)
```

*拟合*计算必要的统计量（如平均值），而*转换*则应用填充。`fit_transform`可以方便地完成这两个步骤。

**2. 特征缩放：**
如前所述，将特征缩放到相似的范围通常很重要。`StandardScaler`（用于标准化）和`MinMaxScaler`（用于归一化 (normalization)）是常见的选择。

```python
from sklearn.preprocessing import StandardScaler

# 创建一个缩放器对象
scaler = StandardScaler()

# 将缩放器拟合到数据（计算平均值和标准差）并转换X
X_scaled = scaler.fit_transform(X)
```

现在，`X_scaled`包含特征数据，其中每列的平均值约为0，标准差为1。

**3. 分割数据：**
最后，在训练之前，您需要将数据分成训练集和测试集。Scikit-learn的`train_test_split`函数使得这变得非常容易。

```python
from sklearn.model_selection import train_test_split

# 将X和y分成训练集（例如80%）和测试集（例如20%）
# 'test_size=0.2' 表示20%用于测试，80%用于训练
# 'random_state' 确保每次运行代码时分割结果都相同（为了可复现性）
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 验证结果集的形状
print("X_train 形状:", X_train.shape)
print("X_test 形状:", X_test.shape)
print("y_train 形状:", y_train.shape)
print("y_test 形状:", y_test.shape)
```

这是数据准备流程的简单可视化：

> 显示使用库函数进行数据加载和准备的工作流程，最终得到独立的训练集和测试集。

通过使用这些库函数，您已经高效地加载了数据，将其分离为特征和目标，处理了潜在的缺失值，缩放了特征，并创建了下一阶段所需的训练集和测试集：训练您的机器学习 (machine learning)模型。这种结构化的方法，由Pandas和Scikit-learn等库提供便利，构成了实际机器学习项目的基础。

## 参考资料

- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html) — scikit-learn developers (2023)
  为数据预处理模块（如 SimpleImputer、StandardScaler 和 train_test_split）提供全面的细节和示例。
- [pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html) — pandas development team (2023)
  提供加载和操作结构化数据的信息，包括 read_csv 和用于数据选择的 DataFrame 操作。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本实用的指南，展示了如何使用 Scikit-learn 和其他库进行数据准备、模型训练和评估。
