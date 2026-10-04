---
course: "getting-started-with-scikit-learn"
chapter: "supervised-learning-classification"
lesson: "calculating-classification-metrics"
sourceId: 1875
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-3-supervised-learning-classification/calculating-classification-metrics"
title: "在 Scikit-learn 中计算指标"
description: "使用 Scikit-learn 函数计算分类评估指标。"
order: 8
plots: ["plots/1875-0.json"]
sourceHash: "301270cd9a849b2a31cd3c65a9d1e15322fab1032c8741173bae54629e96720c"
sourceCorrections: []
---

为了高效计算准确率、精确率、召回率、F1-分数和混淆矩阵等指标，可以使用 Scikit-learn 的 `metrics` 模块。此模块提供了优化函数，用于根据真实标签和模型生成的预测来评估分类模型。

假设您已经训练了一个分类器（例如前面讨论过的 `LogisticRegression`、`KNeighborsClassifier` 或 `SVC`），并在测试集上获得了预测结果，您通常会有两个数组：

1. `y_true`: 测试数据的真实标签。
2. `y_pred`: 您的分类器对测试数据预测的标签。

让我们看看如何使用这些数组来计算标准分类指标。

### 开始：示例数据

首先，确保从 `sklearn.metrics` 导入必要的函数。为了演示目的，我们定义一些示例真实标签和预测标签。在实际场景中，这些将来自于您的模型在测试集上的评估结果。

```python
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, f1_score, classification_report)

# 示例真实标签（二元分类）
y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1])

# 模型预测的示例标签
y_pred = np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 1])

# 多元分类的示例标签
y_true_multi = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
y_pred_multi = np.array([0, 2, 1, 0, 0, 2, 0, 1, 2])
```

### 准确率

准确率是最简单的指标，表示正确预测的比例。它计算方式如下：


$$
\text{准确率} = \frac{\text{正确预测的数量}}{\text{预测总数}}
$$


在 Scikit-learn 中，您使用 `accuracy_score` 函数：

```python
# 计算准确率
acc = accuracy_score(y_true, y_pred)

print(f"Accuracy: {acc:.4f}")
# 预期输出：Accuracy: 0.7000
```

虽然容易理解，但请记住准确率可能会产生误导，特别是对于类别不平衡的数据集。

### 混淆矩阵

混淆矩阵提供了更详细的预测性能分解，显示了真反例 (TN)、假正例 (FP)、假反例 (FN) 和真正例 (TP) 的计数。

使用 `confusion_matrix` 函数：

```python
# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:")
print(cm)
# 预期输出：
# Confusion Matrix:
# [[3 2]
#  [1 4]]
```

输出通常排列为：

```
[[TN, FP],
 [FN, TP]]
```

因此，在我们的示例中：

- TN = 3 (正确预测为类别 0)
- FP = 2 (当真实为 0 时错误预测为类别 1)
- FN = 1 (当真实为 1 时错误预测为类别 0)
- TP = 4 (正确预测为类别 1)

将混淆矩阵可视化通常可以使其更易于解释。以下是您如何使用 Plotly 生成热力图的示例：



![混淆矩阵热力图](plots/1875-0.json)



> 每类正确和不正确预测的细分。

对于多元分类问题，矩阵会相应扩展，显示每个实际类别与预测类别对的计数。

```python
# 多元分类示例的混淆矩阵
cm_multi = confusion_matrix(y_true_multi, y_pred_multi)

print("\nMulti-class Confusion Matrix:")
print(cm_multi)
# 预期输出：
# Multi-class Confusion Matrix:
# [[3 0 0]
#  [1 1 1]
#  [0 1 2]]
```

### 精确率、召回率和 F1-分数

这些指标能对性能的特定方面提供认识，尤其是在类别不平衡是问题时。

- **精确率：** 衡量正向预测的准确性。$Precision = \frac{TP}{TP + FP}$。使用 `precision_score`。
- **召回率（灵敏度）：** 衡量实际正向案例中有多少被正确识别。$Recall = \frac{TP}{TP + FN}$。使用 `recall_score`。
- **F1-分数：** 精确率和召回率的调和平均值，提供一个平衡两者的单一分数。$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$。使用 `f1_score`。

```python
# 计算正类别（标签 1）的精确率、召回率、F1-分数
precision = precision_score(y_true, y_pred) # 默认值：pos_label=1
recall = recall_score(y_true, y_pred)     # 默认值：pos_label=1
f1 = f1_score(y_true, y_pred)             # 默认值：pos_label=1

print(f"\nBinary Classification Metrics (for class 1):")
print(f"Precision: {precision:.4f}") # TP / (TP + FP) = 4 / (4 + 2) = 0.6667
print(f"Recall:    {recall:.4f}")    # TP / (TP + FN) = 4 / (4 + 1) = 0.8000
print(f"F1-Score:  {f1:.4f}")       # 2 * (Prec * Rec) / (Prec + Rec) = 0.7273

# 预期输出：
# Binary Classification Metrics (for class 1):
# Precision: 0.6667
# Recall:    0.8000
# F1-Score:  0.7273
```

**处理多元分类指标：**
对于多元分类问题，您需要使用 `average` 参数 (parameter)指定如何跨类别平均这些指标：

- `average='micro'`：通过计算总 TP、FN、FP 来全局计算指标。
- `average='macro'`：计算每个标签的指标，并找到它们的未加权平均值。不考虑标签不平衡。
- `average='weighted'`：计算每个标签的指标，并找到它们的按支持度（每个标签的真实实例数）加权平均值。考虑标签不平衡。
- `average=None`：单独返回每个类别的分数。

```python
# 使用不同平均方法计算多元分类指标
precision_macro = precision_score(y_true_multi, y_pred_multi, average='macro')
recall_weighted = recall_score(y_true_multi, y_pred_multi, average='weighted')
f1_micro = f1_score(y_true_multi, y_pred_multi, average='micro')

print(f"\nMulti-class Metrics:")
print(f"Macro Precision:  {precision_macro:.4f}")
print(f"Weighted Recall:  {recall_weighted:.4f}")
print(f"Micro F1-Score:   {f1_micro:.4f}")

# 预期输出：
# Multi-class Metrics:
# Macro Precision:  0.6111
# Weighted Recall:  0.6667
# Micro F1-Score:   0.6667
```

### 分类报告

通常，您会希望获得每个类别的精确率、召回率和 F1-分数的摘要，以及支持度（每个类别的真实实例数）。`classification_report` 函数以方便的文本格式提供了这些信息。

```python
# 生成二元情况的分类报告
report_binary = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1'])
print("\nBinary Classification Report:")
print(report_binary)

# 预期输出：
# Binary Classification Report:
#               精确率    召回率  f1-分数   支持度
#
#      类别 0       0.75      0.60      0.67         5
#      类别 1       0.67      0.80      0.73         5
#
#     准确率                           0.70        10
#    宏平均       0.71      0.70      0.70        10
# 加权平均       0.71      0.70      0.70        10

# 生成多元情况的分类报告
report_multi = classification_report(y_true_multi, y_pred_multi, target_names=['Class 0', 'Class 1', 'Class 2'])
print("\nMulti-class Classification Report:")
print(report_multi)

# 预期输出：
# Multi-class Classification Report:
#               精确率    召回率  f1-分数   支持度
#
#      类别 0       0.75      1.00      0.86         3
#      类别 1       1.00      0.33      0.50         3
#      类别 2       0.67      0.67      0.67         3
#
#     准确率                           0.67         9
#    宏平均       0.81      0.67      0.67         9
# 加权平均       0.81      0.67      0.67         9
```

报告包括：

- 每个类别的精确率、召回率、F1-分数。
- 支持度：`y_true` 中每个类别的出现次数。
- 准确率：总体准确率。
- 宏平均：每个类别指标的平均值（未加权）。
- 加权平均：每个类别指标的平均值，按支持度加权。

Scikit-learn 的 `metrics` 模块提供了一种直接的方式来量化 (quantization)分类模型的性能。使用 `accuracy_score`、`confusion_matrix`、`precision_score`、`recall_score`、`f1_score` 等函数以及全面的 `classification_report`，您可以超越简单的准确率，对模型在不同类别上的行为方式获得更全面的理解，这对于构建有效的分类系统非常重要。

## 参考资料

- [Model evaluation: quantifying the quality of predictions](https://scikit-learn.org/stable/modules/model_evaluation.html) — scikit-learn developers (2023)
  Publisher: scikit-learn project
  涵盖Scikit-learn中分类指标API和用法的官方文档。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一份实用指南，详细介绍了如何使用Scikit-learn实现和评估机器学习模型，包括分类指标。
- [An Introduction to Statistical Learning: With Applications in Python](https://link.springer.com/book/10.1007/978-3-031-38747-0) — Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, Jonathan Taylor (2023)
  Publisher: Springer; DOI: [10.1007/978-3-031-38747-0](https://doi.org/10.1007/978-3-031-38747-0)
  为统计学习方法提供理论基础，包括模型评估和分类指标的核心概念。
