---
course: "getting-started-with-scikit-learn"
chapter: "intro-setup-scikit-learn"
lesson: "practice-setup-verification"
sourceId: 1835
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-1-intro-setup-scikit-learn/practice-setup-verification"
title: "动手实践：安装验证"
description: "确认您的 Scikit-learn 安装并加载一个示例数据集。"
order: 6
plots: []
sourceHash: "787b46317c466dbe919256cc0833fbacde0048d596a793af2e6aa60a869b1d22"
sourceCorrections: []
---

确保您的 Scikit-learn 环境设置正确是构建机器学习 (machine learning)模型之前的一个主要步骤。本实践练习将引导您验证 Scikit-learn 安装并加载其一个示例数据集。成功运行此代码可确认您的环境已准备就绪。

首先，让我们确认 Python 可以找到您的 Scikit-learn 安装并检查其版本。打开您的 Python 解释器、Jupyter Notebook 或 IDE 并运行以下代码：

```python
import sklearn

# 打印安装的版本
print(f"Scikit-learn version: {sklearn.__version__}")
```

您应该会看到打印出的安装版本号，且没有任何 `ImportError` 消息。具体版本可能不同，但看到版本号（例如 `1.2.2` 或更高）则表示成功。如果您遇到错误，请重新查看“安装与环境设置”部分，以确保 Scikit-learn 及其依赖项（如 NumPy 和 SciPy）已在您的活动 Python 环境中正确安装。

接下来，让我们尝试加载前面讨论过的一个内置数据集。`sklearn.datasets` 中的 `load_iris` 函数提供了经典的 Iris 数据集，该数据集常用于入门级分类示例。

```python
from sklearn.datasets import load_iris
import numpy as np

# 加载 Iris 数据集
iris = load_iris()

# 加载的对象是 'Bunch' 对象，类似于字典
print(f"iris 数据集的键：{iris.keys()}")

# 数据通常存储在 .data 中，目标标签存储在 .target 中
# 这些通常是 NumPy 数组
print(f"数据形状：{iris.data.shape}")
print(f"目标形状：{iris.target.shape}")

# 让我们查看前 5 个样本及其对应的目标
print("\n前 5 个样本（特征）：\n", iris.data[:5])
print("\n前 5 个目标（标签）：\n", iris.target[:5])

# 特征名称和目标名称也经常包含在内
print(f"\n特征名称：{iris.feature_names}")
print(f"\n目标名称：{iris.target_names}")
```

执行此代码应输出：

1. `iris` 对象中可用的键（如 `data`、`target`、`target_names`、`feature_names`、`DESCR`）。
2. `data` 数组的形状（通常为 (150, 4)，表示 150 个样本和 4 个特征）。
3. `target` 数组的形状（通常为 (150,)，表示每个样本的类别标签）。
4. 前五个样本的实际数值数据。
5. 前五个样本的目标标签（0、1 或 2）。
6. 四个特征的名称（例如，“sepal length (cm)”、“sepal width (cm)”等）。
7. 三个目标类别的名称（例如，“setosa”、“versicolor”、“virginica”）。

如果两个代码片段都无错误运行并产生与所述类似的输出，则您的 Scikit-learn 环境已正确配置。您已成功导入该库并以它预期的数据格式加载了数据。您现在可以继续学习后续章节了，在这些章节中，您将开始使用 Scikit-learn 构建和评估机器学习模型。

## 参考资料

- [Scikit-learn: Machine Learning in Python](https://scikit-learn.org/stable/) — The scikit-learn developers (2025)
  Scikit-learn的官方文档，包含安装、API理解和使用方法。本资源对安装指南和数据集加载部分尤为重要。
- [NumPy Documentation](https://numpy.org/doc/stable/) — The NumPy Developers (2023)
  NumPy的官方文档，NumPy是Python数值计算的基础库。Scikit-learn的核心数据结构依赖于NumPy。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本广泛使用的机器学习实践书籍，内容涵盖Scikit-learn环境设置和数据集使用示例。
