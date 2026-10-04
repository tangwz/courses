---
course: "introduction-to-graph-neural-networks"
chapter: "training-gnn-models"
lesson: "evaluation-metrics-node-classification"
sourceId: 7638
sourceUrl: "https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-4-training-gnn-models/evaluation-metrics-node-classification"
title: "节点分类的评估指标"
description: "了解用于评估 GNN 分类任务表现的标准指标，如准确率和 F1 分数。"
order: 5
plots: ["plots/7638-0.json"]
sourceHash: "82549eef83d8f4929000fc0577c4e52987f59594f7174b1315040b0210c0441a"
sourceCorrections: []
---

训练好的图神经网络 (neural network) (GNN) 依靠其损失函数 (loss function)来指导模型权重 (weight)的调整，从而达到有效的配置。然而，在训练集上获得较低的损失值并不总是能直接转化为在未见数据上的良好表现。为了真正了解模型的泛化能力，我们需要在预留的测试集上使用比原始损失值更易读的指标进行评估。对于节点分类，这些指标回答了一个简单的问题：模型正确标记 (token)了多少个节点？

### 从原始输出到具体预测

用于节点分类的 GNN 通常以最后一个线性层结束，该层会为每个类别生成原始的、未归一化 (normalization)的分数，通常称为 `logits`。如果模型有 $C$ 个类别，并且正在评估 $N$ 个节点，则输出将是一个形状为 $[N, C]$ 的张量。为了将这些分数转化为明确的预测，我们为每个节点选择分数最高的类别。这是通过沿类别维度执行 `argmax` 操作来完成的。

例如，如果单个节点的输出为 `[0.1, 2.5, -1.3]`，则 `argmax` 为 1，这意味着模型预测为第二个类别（因为索引是从零开始的）。这个过程将模型的连续输出分数转换为离散的类别标签，以便我们与真实标签进行对比。

### 混淆矩阵：更全面地观察表现

在计算任何指标之前，我们必须首先将模型的预测与真实标签进行比较。组织这种对比最基本的工具是**混淆矩阵**。在二分类任务中，混淆矩阵是一个 2x2 的表格，总结了预测的四种可能结果。

> 二分类混淆矩阵中的四种结果。正确的预测（TP, TN）位于主对角线上。

对于节点分类（通常是多分类问题），混淆矩阵会扩展为 $C \times C$ 的矩阵，其中 $C$ 是类别的数量。第 $i$ 行和第 $j$ 列的条目表示真正属于类别 $i$ 但被预测为类别 $j$ 的节点数量。对角线元素代表所有被正确分类的节点。

### 准确率

准确率是最直观的指标。它衡量了正确预测占总预测的比例。


$$
\text{准确率} = \frac{\text{正确预测的数量}}{\text{总预测数量}} = \frac{TP + TN}{TP + TN + FP + FN}
$$


虽然准确率易于理解，但在**类别不平衡**的数据集上可能会产生误导。想象一个图，其中 95% 的节点属于类别 A，5% 属于类别 B。一个总是预测类别 A 的偷懒模型将获得 95% 的准确率，而没有学到关于类别 B 的任何有用信息。在这种情况下，我们需要更敏锐的指标。

### 精确率、召回率和 F1 分数

为了更好地了解模型在不平衡数据上的表现，我们转向精确率、召回率和 F1 分数。这些指标通常是按类别计算的。对于给定类别，我们将其视为“正”类，而将所有其他类别视为“负”类。

#### 精确率 (Precision)

精确率回答了这样一个问题：“在模型标记 (token)为类别 A 的所有节点中，有多少实际上是类别 A？”它衡量了模型正类预测的可靠性。


$$
\text{精确率} = \frac{TP}{TP + FP}
$$


当假正类的代价很高时，高精确率非常。例如，在自动将学术论文标记为“撤稿”的系统中，你需要高精确率以避免错误地标记合法的论文。

#### 召回率 (Recall)

召回率（也称为灵敏度或真正类率）回答了这样一个问题：“在所有真正属于类别 A 的节点中，模型找出了多少个？”它衡量了模型识别所有相关实例的能力。


$$
\text{召回率} = \frac{TP}{TP + FN}
$$


当假负类的代价很高时，高召回率非常。例如，在预测哪些蛋白质与某种疾病相关的 GNN 中，你需要高召回率以避免错过任何潜在的重要蛋白质。

#### F1 分数

通常，精确率和召回率之间存在权衡。F1 分数提供了一种将两者结合成单个指标的方法。它是精确率和召回率的**调和平均数**，它会给较小的值赋予更大的权重 (weight)。这意味着只有当精确率和召回率都很高时，F1 分数才会很高。


$$
\text{F1 分数} = 2 \cdot \frac{\text{精确率} \cdot \text{召回率}}{\text{精确率} + \text{召回率}}
$$


### 多分类问题的平均化方法

由于精确率、召回率和 F1 是按类别计算的，我们需要一种策略将它们汇总为多分类节点分类任务的单个数值。两种最常用的方法是宏平均和加权平均。

- **宏平均 (Macro Average)：** 独立计算每个类别的指标，然后计算其算术平均值。这种方法将每个类别视为同等重要，而不考虑它包含多少个节点。如果你想知道模型在所有类别（包括稀有类别）上的表现，这是一个很好的衡量标准。
- **加权平均 (Weighted Average)：** 为每个类别计算指标，但在平均时，根据每个类别的**支持数**（该类别的真实实例数量）对每个类别的分数进行加权。这考虑了类别不平衡的情况。高加权平均 F1 分数表明模型在最常见的类别上表现良好。



![不平衡数据上的指标平均化](plots/7638-0.json)



> 在不平衡的数据集上，较高的加权平均分数可能会掩盖在少数类上的糟糕表现。较低的宏平均分数则反映了这一弱点。

### 选择合适的指标

评估指标的选择完全取决于你的应用目标。

- 对于平衡的数据集，或者当你同等看待所有错误时，**准确率**是一个很好的起点。
- 如果你的数据集不平衡，并且你想确保模型即使在稀有类别上也能表现良好，请使用**宏平均 F1 分数**。
- 如果数据集不平衡，但你更关心在较大的、影响力更大的类别上的表现，那么**加权平均 F1 分数**更为合适。

在实践中，查看完整的分类报告通常很有帮助，该报告会单独显示每个类别的精确率、召回率和 F1 分数。像 `scikit-learn` 这样的库为此提供了方便的函数。

```python
from sklearn.metrics import classification_report

# y_true: 真实标签（例如来自 data.test_mask）
# y_pred: 模型在测试集上的预测标签

# 假设 class_names 是标签的字符串列表
print(classification_report(y_true, y_pred, target_names=class_names))

#              精确率    召回率  f1-分数   支持数

#    类别 1       0.91      0.95      0.93       105
#    类别 2       0.75      0.82      0.78        80
#    类别 3       0.98      0.96      0.97       150
#
#   准确率                             0.92       335
#  宏平均       0.88      0.91      0.89       335
# 加权平均     0.92      0.92      0.92       335
```

这份详细的报告让你能全面了解模型的优缺点，从而针对如何改进模型做出明智的决策。

## 参考资料

- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  本书为机器学习算法及其评估提供了严谨的统计和数学基础，涵盖了分类、误差估计和模型选择等概念。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125974/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: Chapter 3
  这本实用指南解释了机器学习概念，包括使用流行的Python库（如scikit-learn）进行分类评估指标的详细讨论和示例。
- [scikit-learn User Guide: 3.3. Metrics and scoring: quantifying the quality of predictions](https://scikit-learn.org/stable/modules/model_evaluation.html) — scikit-learn developers (2024)
  官方文档提供了分类评估指标的全面解释和示例，包括 `classification_report` 和不同的平均策略，与Python代码片段直接相关。
