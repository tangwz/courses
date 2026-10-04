---
course: "introduction-to-machine-learning"
chapter: "building-first-ml-model"
lesson: "hands-on-end-to-end-model"
sourceId: 1544
sourceUrl: "https://apxml.com/zh/courses/introduction-to-machine-learning/chapter-7-building-first-ml-model/hands-on-end-to-end-model"
title: "动手实践：端到端简单模型构建"
description: "一个涵盖从数据加载到模型评估所有步骤的全面实践章节。"
order: 7
plots: ["plots/1544-0.json"]
sourceHash: "92c83246969c19c8682cc3f47d4ef78ea2dca6759e847ae11ab95686c03cefbe"
sourceCorrections: []
---

这里将完整演示一个机器学习 (machine learning)模型构建的例子。我们将使用Python中广受欢迎的Scikit-learn库，它使得许多步骤变得直接简单。我们将一步步地遵循标准的机器学习工作流程：加载数据、准备数据、选择并训练模型、进行预测以及评估结果。

在这个例子中，我们将使用一个名为鸢尾花数据集的知名数据集来处理一个分类问题。该数据集包含不同种类鸢尾花的测量数据。我们的目标是构建一个能够根据鸢尾花测量数据预测其种类的模型。

### 1. 环境设置

首先，请确保您已安装Scikit-learn。如果您使用的是Anaconda，它很可能已经包含在内。如果不是，您通常可以使用pip安装它：

```bash
pip install scikit-learn numpy pandas matplotlib seaborn
```

我们还将使用NumPy进行数值运算，使用Pandas进行数据处理（尽管Scikit-learn可以直接加载鸢尾花数据），并使用Matplotlib/Seaborn进行后续的可视化，例如我们稍后会用到的混淆矩阵。

### 2. 加载数据

Scikit-learn自带了几个内置数据集，包括鸢尾花数据集，这使得加载它非常简单。

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data  # 特征（萼片长度、萼片宽度、花瓣长度、花瓣宽度）
y = iris.target # 目标变量（种类，编码为0、1、2）

# 为清晰起见，我们查看特征名称和目标名称
print("特征名称：", iris.feature_names)
print("目标名称：", iris.target_names)
print("数据形状（样本数，特征数）：", X.shape)
print("目标形状（样本数）：", y.shape)
print("\n前5个样本：\n", X[:5])
print("前5个目标：", y[:5])
```

输出显示我们有150个样本（花），每个样本有4个特征。目标变量`y`包含代表种类的数字（0、1、2），分别对应'setosa'、'versicolor'、'virginica'。

### 3. 数据准备：划分与缩放

在训练之前，我们需要将数据划分为训练集（供模型学习用）和测试集（用于评估模型学习效果）。常见的划分比例是80%用于训练，20%用于测试。

我们还需要对特征进行缩放。K-近邻（K-Nearest Neighbors，我们将使用的算法）等算法依赖于数据点之间的距离。如果特征具有明显不同的范围（例如，一个范围是0-1，另一个是0-1000），那么范围较大的特征会在距离计算中占据主导地位。缩放能将所有特征调整到相似的范围。我们将使用`StandardScaler`，它将数据转换为均值为零、方差为一。

重要提示：我们*只*在训练数据上拟合缩放器，以防止测试集的信息泄露到训练过程中。然后，我们使用*相同*的拟合后的缩放器来转换训练和测试数据。

```python
# 将数据划分为训练集和测试集（80%训练，20%测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"训练集大小：{X_train.shape[0]} 个样本")
print(f"测试集大小：{X_test.shape[0]} 个样本")

# 初始化StandardScaler
scaler = StandardScaler()

# 仅在训练数据上拟合缩放器
scaler.fit(X_train)

# 使用拟合后的缩放器转换训练和测试数据
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 显示前几行缩放后的数据以查看效果
print("\n前5个缩放后的训练样本：\n", X_train_scaled[:5])
```

请注意`X_train_scaled`中的值现在如何围绕零居中。

### 4. 选择并训练模型

现在，我们选择一个算法。由于这是一个分类问题（预测类别/种类），并且我们之前介绍过K-近邻（KNN），所以我们来使用它。KNN根据特征空间中其“k”个最近邻居的多数类别来对新数据点进行分类。

我们需要为“k”（邻居的数量）选择一个值。常见的起始点是k=3或k=5。我们使用k=5。

```python
# 选择模型：K-近邻分类器
# 实例化模型，设置k=5个邻居
knn_model = KNeighborsClassifier(n_neighbors=5)

# 使用缩放后的训练数据训练模型
# 'fit'方法是学习发生的地方
knn_model.fit(X_train_scaled, y_train)

print("\n模型训练完成。")
```

就是这样！`fit`方法已经存储了训练数据（或者从中学习到了模式，这取决于模型类型）。对于KNN，它基本上记住了训练数据点在缩放后的特征空间中的位置。

### 5. 进行预测

有了我们训练好的模型，现在我们可以预测*测试集*中花的种类。请记住，模型在训练期间没有见过这些测试数据。我们使用`predict`方法。

```python
# 使用训练好的模型对缩放后的测试数据进行预测
y_pred = knn_model.predict(X_test_scaled)

# 显示前10个测试样本的预测种类
print("\n前10个测试样本的预测种类：", y_pred[:10])
# 显示前10个测试样本的实际种类
print("前10个测试样本的实际种类：   ", y_test[:10])
```

模型输出一个数组`y_pred`，该数组包含测试集`X_test_scaled`中每个样本的预测种类（0、1或2）。我们可以将这些预测与实际值`y_test`进行比较，以了解模型的表现如何。

### 6. 评估模型

逐一比较预测结果是繁琐的。我们需要定量指标。对于分类问题，准确度是一个常见的起始指标。它告诉我们正确预测的比例。

```python
# 计算模型的准确度
accuracy = accuracy_score(y_test, y_pred)

print(f"\n模型在测试集上的准确度：{accuracy:.4f}")
```

准确度为1.0表示在测试集上完美预测，而0.0则表示完全不正确。我们的KNN模型在这个数据集上表现相当不错。

为了更详细地查看，我们可以使用混淆矩阵。它显示了每个类别有多少样本被正确分类，以及错误分类发生在哪里。行通常代表实际类别，列代表预测类别。

```python
# 生成混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 为了更好的可视化，创建DataFrame并使用Seaborn热力图
cm_df = pd.DataFrame(cm, index=iris.target_names, columns=iris.target_names)

plt.figure(figsize=(7, 5))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues') # 使用Blues颜色映射
plt.title('混淆矩阵')
plt.ylabel('实际种类')
plt.xlabel('预测种类')
plt.tight_layout() # 调整布局以防止裁剪
plt.show() # 显示图表

# 也打印混淆矩阵的值
print("\n混淆矩阵：\n", cm_df)
```



![混淆矩阵](plots/1544-0.json)



> 混淆矩阵显示了每种鸢尾花正确和不正确预测的数量。在这个例子中，所有测试样本都被正确分类。

对角线元素（从左上到右下）显示了每个类别（setosa、versicolor、virginica）的正确预测数量。非对角线元素显示了错误分类。例如，如果“versicolor”行和“virginica”列的单元格中有一个“1”，那将意味着一朵实际是“versicolor”的鸢尾花被错误地预测为“virginica”。在我们的例子中，完美的准确度得分体现在混淆矩阵中除了主对角线之外所有地方都为零。

### 步骤总结

恭喜！您刚刚完整演示了端到端构建、训练和评估您的第一个机器学习 (machine learning)模型。我们执行了以下步骤：

1. **加载数据：** 使用了Scikit-learn中的`load_iris()`。
2. **准备数据：** 划分为训练/测试集（`train_test_split`）并缩放特征（`StandardScaler`）。
3. **选择模型：** 选择了`KNeighborsClassifier`。
4. **训练模型：** 对缩放后的训练数据使用了`.fit()`方法。
5. **进行预测：** 对缩放后的测试数据使用了`.predict()`方法。
6. **评估模型：** 计算了`accuracy_score`并可视化了`confusion_matrix`。

这个工作流程提供了一个扎实的基底。虽然我们在这里使用了KNN，但基本过程（加载、划分、缩放、拟合、预测、评估）对于Scikit-learn中的许多其他监督学习 (supervised learning)算法来说是相似的，只是涉及不同的模型类别和可能不同的评估指标。

## 参考资料

- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html) — scikit-learn developers (2023)
  官方用户指南提供了scikit-learn库所有方面的详尽文档，包括数据预处理、各类模型和评估指标。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本实用指南，详述了使用Scikit-learn完成整个机器学习流程，包括数据准备、模型训练（包括KNN）和评估。
- [An Introduction to Statistical Learning with Applications in Python](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, Jonathan Taylor (2023)
  Publisher: Springer; DOI: [10.1007/978-3-031-38747-0](https://doi.org/10.1007/978-3-031-38747-0)
  一本教科书，易懂地介绍了统计学习概念，包括监督学习、KNN、模型评估和机器学习工作流程，并提供了Python应用实例。
