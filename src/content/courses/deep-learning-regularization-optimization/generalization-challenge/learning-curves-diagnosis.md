---
course: "deep-learning-regularization-optimization"
chapter: "generalization-challenge"
lesson: "learning-curves-diagnosis"
sourceId: 4877
sourceUrl: "https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-1-generalization-challenge/learning-curves-diagnosis"
title: "诊断模型表现：学习曲线"
description: "学习如何解读学习曲线以诊断模型拟合问题。"
order: 4
plots: ["plots/4877-0.json", "plots/4877-1.json", "plots/4877-2.json"]
sourceHash: "f7597ad05cfaf1a9c4f9d4e407f7df1e65e0fe9cb0f18dd76e30410b8a35771d"
sourceCorrections: []
---

我们已经讨论了泛化能力的重要性以及欠拟合 (underfitting)和过拟合 (overfitting)问题。但是，我们怎样才能真正*看出*模型在训练期间是否受到这些问题影响呢？仅仅查看测试集上的最终表现指标是不够的；它无法告知我们模型表现如此的原因，也无法告知它在学习过程中的行为方式。学习曲线在此处就显得有用了。

学习曲线是可视化工具，它绘制了模型在训练集和单独的验证集上的表现指标（通常是损失或准确率），这些指标随着时间（通常是周期或迭代次数）或随着所用训练数据量而变化。通过观察这两条曲线的趋势和间隙，我们能获得有益的了解模型学习动态，并诊断潜在问题，如高偏差（欠拟合）或高方差（过拟合）。

### 如何绘制学习曲线

该过程通常包含以下步骤：

1. **划分数据：** 将数据集划分为训练集、验证集和测试集。学习曲线是使用训练集和验证集绘制的。测试集则保留用于最终、无偏的评估。
2. **迭代训练：** 逐周期地训练模型。
3. **评估表现：** 在每个周期结束时（有时更频繁），计算所选的表现指标（例如，回归的均方误差，分类的交叉熵损失或准确率），评估对象是：
   - 整个训练集（或该迭代中使用的迷你批次，尽管在整个训练集上评估会得到更平滑的曲线）。
   - 整个验证集。
4. **绘图：** 创建一个图表，其中x轴表示周期数，y轴表示表现指标。绘制两条线：一条用于训练表现，另一条用于验证表现。

### 曲线解读

训练曲线和验证曲线的形状和关系能说明很多关于模型拟合情况。让我们看一下常见模式：

#### 1. 欠拟合 (underfitting)（高偏差）

如果模型过于简单而无法捕捉数据中的潜在模式，它将在训练集和验证集上表现不佳。

- **训练曲线：** 训练损失会很高，并可能会相对较快地趋于平稳，表明模型即使从它所见的数据中也未能学到太多。
- **验证曲线：** 验证损失也会很高，通常非常接近训练损失。
- **诊断：** 两条曲线都收敛于高错误水平。这说明模型能力不足（例如，层数或神经元太少）无法有效学习任务。增加更多训练数据不太可能带来显著帮助，因为模型从根本上无法表示目标函数。



![学习曲线：欠拟合（高偏差）](plots/4877-0.json)



> 训练损失和验证损失都保持高位，表明模型不够复杂，无法捕捉数据的模式。

#### 2. 过拟合 (overfitting)（高方差）

当模型对训练数据学习得过于透彻，包括其噪声和特定怪异之处时，它无法泛化到新数据。

- **训练曲线：** 训练损失稳定下降并达到一个很低的值。模型几乎完美地拟合了训练数据。
- **验证曲线：** 验证损失最初下降，但在某一点后开始趋于平稳甚至上升。
- **诊断：** 在低训练损失和更高（且可能增加）的验证损失之间出现显著的差距。这表明模型正在记忆训练样本，而不是学习可泛化的模式。



![学习曲线：过拟合（高方差）](plots/4877-1.json)



> 训练损失变得非常低，而验证损失停滞或增加，显示出很大的泛化差距。

#### 3. 良好拟合

理想情况是模型学习到相关模式而没有记忆噪声。

- **训练曲线：** 训练损失稳定下降并收敛到一个低值。
- **验证曲线：** 验证损失也稳定下降并收敛到一个低值，并保持与训练损失接近。
- **诊断：** 两条曲线收敛于低错误水平，且两条曲线之间只有很小的差距。这表明泛化能力良好。



![学习曲线：良好拟合](plots/4877-2.json)



> 训练损失和验证损失都下降并收敛到低值，且两条曲线之间有很小的差距。

### 使用学习曲线采取行动

诊断问题是第一步。学习曲线会引导您接下来的行动：

- **如果是欠拟合 (underfitting)：** 尝试增加模型复杂性（更多层、更多神经元），训练更长时间（如果曲线尚未趋于平稳），更改模型架构，或构建更好的特征。
- **如果是过拟合 (overfitting)：** 尝试获取更多训练数据，应用正则化 (regularization)技术（如L1/L2、Dropout、批量归一化 (normalization)——我们很快会讲到！），降低模型复杂性，或使用数据增强。早期停止（当验证表现开始下降时停止训练）也是使用学习曲线来判断的常见策略。

学习曲线是深度学习 (deep learning)实践者的工具包中不可或缺的一部分。它们为了解训练过程提供了一个窗口，帮助您了解模型*如何*学习，并引导您构建能有效泛化到未见数据的模型。每当训练新模型时，请随时使用它们。

## 参考资料

- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12.pdf) — Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009)
  Publisher: Springer; Pages: Chapter 7; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  一本基础统计学习教材，包含模型评估、偏差-方差权衡和交叉验证的详细讨论，这些都是学习曲线使用方法的基础。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 5
  一本关于深度学习的权威教材，在神经网络背景下全面涵盖了机器学习基础、泛化、偏差-方差权衡、欠拟合和过拟合。
- [CS229 Lecture Notes: Bias vs. Variance and Diagnosing Bias vs. Variance](http://cs229.stanford.edu/materials.html) — Andrew Ng (2023)
  Publisher: Stanford University, Computer Science Department
  提供关于如何使用学习曲线识别和解决欠拟合（高偏差）和过拟合（高方差）的实用见解和清晰解释，这是广受认可的机器学习课程内容。
