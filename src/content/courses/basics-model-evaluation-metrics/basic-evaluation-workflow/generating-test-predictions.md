---
course: "basics-model-evaluation-metrics"
chapter: "basic-evaluation-workflow"
lesson: "generating-test-predictions"
sourceId: 4042
sourceUrl: "https://apxml.com/zh/courses/basics-model-evaluation-metrics/chapter-5-basic-evaluation-workflow/generating-test-predictions"
title: "在测试集上生成预测"
description: "解释使用训练好的模型对未见过的测试数据进行预测的过程。"
order: 5
plots: []
sourceHash: "4ec4c3a36d10cc0321d7029e4c89714f8ffe6307117e53f0dd9ad0a01c722ae2"
sourceCorrections: []
---

在成功分割数据并在训练集上训练模型后，下一个重要步骤是查看模型对新的、未见过的数据的泛化能力如何。这时测试集就发挥作用了。我们使用训练好的模型为测试集中预留的数据点生成预测。

主要思路很简单：将测试集的输入特征（$X_{test}$）输入给训练好的模型，并记录模型产生的输出。这些输出就是模型的预测，通常表示为$ŷ_{test}$（读作“y-hat test”）。这个过程模拟了模型在遇到训练阶段未见过的新数据时的表现。在测试集上评估性能可以为您提供对模型潜在有效性更实际的评估。请记住，在用于训练的数据上进行评估可能会导致过于乐观的结果，因为模型可能只是简单记住了训练样本。

### 预测生成如何进行

将您训练好的模型看作一个函数或一个过程，它从训练数据（$X_{train}$，$y_{train}$）中学习了模式。现在，您只向它提供测试集中的输入特征（$X_{test}$）。模型将学到的模式应用于这些新输入，以生成对应的预测（$ŷ_{test}$）。

这些预测的具体形式取决于您正在解决的问题类型：

- **对于分类问题：** 模型为测试集中的每个输入实例预测一个类别或类标签。例如，如果您训练了一个模型将电子邮件分类为“垃圾邮件”或“非垃圾邮件”，将测试电子邮件的特征输入给它，将导致预测结果为“垃圾邮件”或“非垃圾邮件”。有些模型可能会首先输出概率（例如，80% 的概率是垃圾邮件），然后通常根据阈值（通常是 0.5）将其转换为最终的类标签。为了计算大多数标准指标，如准确率或精确率，我们使用这些最终的预测标签。
- **对于回归问题：** 模型为每个输入实例预测一个连续数值。如果您的模型预测房屋价格，提供测试集中房屋的特征（如面积、卧室数量）将导致一个预测价格，例如 \$253,400。

### 使用正确输入

非常重要的一点是，从测试集提供给模型（$X_{test}$）的输入特征必须与训练期间使用的特征（$X_{train}$）具有完全相同的结构和格式。如果您的模型是使用“温度”和“湿度”等特征进行训练的，那么您的测试数据也必须以相同的单位和格式提供“温度”和“湿度”，以便模型理解它们。

非常重要的一点是，在生成预测时，您*只*向模型提供输入特征（$X_{test}$）。在此阶段，您**不要**提供测试集中的实际目标值（$y_{test}$）。这样做的目的是在*不知道*真实答案的情况下，查看模型会预测什么。

### 示例：预测

想象您训练了一个简单的线性回归模型，根据身高预测体重。假设学到的模型由以下方程表示：

$\text{预测体重} = -110 + 3.5 \times \text{身高（英寸）}$

现在，假设您的测试集中包含一个身高为 70 英寸的人。为了生成预测：

1. 从测试实例中获取身高特征：$x_{test} = 70$。
2. 将其输入到训练好的模型方程中：
   $ŷ_{test} = -110 + 3.5 \times 70$
   $ŷ_{test} = -110 + 245$
   $ŷ_{test} = 135$
3. 模型预测该人员体重为 135 磅。

您将对测试集（$X_{test}$）中的每个实例重复此过程，从而得到一个预测列表（$ŷ_{test}$），对应每个测试实例。

### 下一步是什么？

一旦您获得了测试数据的完整预测集合（$ŷ_{test}$），就可以进行重要的比较步骤了。在下一节“计算性能指标”中，我们将把这些预测与您小心保留的真实目标值（$y_{test}$）进行比较。这种比较，使用适合您问题类型（分类或回归）的指标，将最终量化 (quantization)您的模型在未见过的数据上的表现如何。

## 参考资料

- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  提供关于机器学习工作流的实用指南，包括使用流行Python库训练模型和在新数据上生成预测。
- [An Introduction to Statistical Learning: with Applications in Python](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, Jonathan Taylor (2023)
  Publisher: Springer
  一本关于统计学习方法的奠基性教材，涵盖模型拟合、预测和泛化能力评估等原则。
- [scikit-learn User Guide: 1.1. Estimator basics](https://scikit-learn.org/stable/user_guide.html#estimator-basics) — scikit-learn developers (2023)
  scikit-learn官方文档，展示了使用scikit-learn估计器进行训练和预测的基本工作流，这是模型评估中的一个基本步骤。
