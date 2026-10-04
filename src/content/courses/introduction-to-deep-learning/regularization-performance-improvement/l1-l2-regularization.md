---
course: "introduction-to-deep-learning"
chapter: "regularization-performance-improvement"
lesson: "l1-l2-regularization"
sourceId: 5064
sourceUrl: "https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-6-regularization-performance-improvement/l1-l2-regularization"
title: "L1 和 L2 正则化"
description: "说明如何通过给损失函数增加权重惩罚项（L1/L2 范数）有助于避免过拟合。"
order: 3
plots: ["plots/5064-0.json"]
sourceHash: "c5056b7899967d08ae2011a8360302977f13febc4181b76f4132938f03be44c6"
sourceCorrections: []
---

当模型过度学习训练数据，捕捉到无法泛化到新数据的噪声和特定模式时，就会发生过拟合 (overfitting)。对抗此问题的一个常用方法是正则化 (regularization)，它通过根据模型本身的复杂程度，在模型的损失函数 (loss function)中加入惩罚项。L1 和 L2 正则化是两种广泛运用的技术，它们根据网络权重 (weight)的量级来惩罚模型复杂度。

核心思想是，权重过大的模型可能对输入特征的微小变化过于敏感，这实质上是在记忆训练数据。通过在损失函数中添加一个随权重大小增加的惩罚项，我们促使优化过程（如梯度下降 (gradient descent)）找到不仅能最小化原始预测误差，还能使权重保持相对较小的解决办法。

### L2 正则化 (regularization)（权重 (weight)衰减或岭回归）

L2 正则化增加一个与权重大小的*平方*成比例的惩罚项。修改后的损失函数 (loss function)如下所示：


$$
\text{新损失} = \text{原始损失} + \lambda \sum_{i} w_i^2
$$


其中：

- $w_i$ 代表网络中的每个权重。
- $\sum_{i} w_i^2$ 是所有权重的平方和（即平方 L2 范数）。
- $\lambda$ (lambda) 是正则化强度超参数 (parameter) (hyperparameter)。它是一个正值，需要在训练前设定。

**工作原理：**
L2 惩罚项 $\lambda \sum w_i^2$ 抑制单个权重过大。在反向传播 (backpropagation)过程中，每个权重 $w_i$ 的梯度计算将包含一个与 $w_i$ 本身成比例的额外项（$2 \lambda w_i$）。这意味着在梯度下降 (gradient descent)的权重更新步骤中（$w_i \leftarrow w_i - \text{学习率} \times \text{梯度}$），会有一个额外的作用将权重拉向零。这种作用就是 L2 正则化常被称为**权重衰减**的原因。

**作用：**

- **更小、更分散的权重：** 它鼓励网络在某种程度上使用所有输入，而不是过度依赖少数可能有噪声的输入。权重倾向于分布更均匀，并取较小的值。
- **更平滑的模型：** 权重较小的模型通常复杂度较低，具有更平滑的决策边界，从而更不容易过拟合 (overfitting)。
- **非稀疏性：** L2 正则化会将权重推*向*零，但很少使它们*精确地*变为零。

**$\lambda$ 的作用：**
超参数 $\lambda$ 控制着最小化原始损失（使数据拟合良好）与最小化权重大小（保持模型简洁）之间的平衡。

- 如果 $\lambda = 0$，则没有正则化。
- 如果 $\lambda$ 非常小，正则化效果会很微弱。
- 如果 $\lambda$ 很大，权重将受到很大惩罚，如果惩罚项主导了原始损失，则可能导致欠拟合 (underfitting)。

选择 $\lambda$ 的合适值通常需要通过实验和交叉验证等方法来确定。



![权重分布](plots/5064-0.json)



> L2 正则化倾向于使权重比未正则化的模型更接近零。

### L1 正则化 (regularization)（Lasso）

L1 正则化增加一个与权重 (weight)*绝对值*成比例的惩罚项：


$$
\text{新损失} = \text{原始损失} + \lambda \sum_{i} |w_i|
$$


其中：

- $|w_i|$ 是权重 $w_i$ 的绝对值。
- $\sum_{i} |w_i|$ 是所有权重的绝对值之和（即 L1 范数）。
- $\lambda$ 同样是正则化强度超参数 (parameter) (hyperparameter)。

**工作原理：**
L1 惩罚项 $\lambda \sum |w_i|$ 同样抑制权重过大。然而，L1 项的梯度贡献与权重的符号成比例（$\lambda \times \text{sign}(w_i)$），假设 $w_i \neq 0$。这意味着在优化过程中，每个权重都会被一个恒定量（由 $\lambda$ 和学习率决定）推向零，而不管其当前的大小（这与 L2 不同，L2 中推动力会随着权重接近零而减小）。

**作用：**

- **稀疏性：** 这种恒定的向零推动使得 L1 正则化在促使某些权重变为*精确地*零方面非常有效。
- **特征选择：** 当连接到输入特征的权重变为零时，表示该特征被模型有效忽略。因此，L1 可以执行自动特征选择，识别并舍弃关联度较低的输入。
- **更简洁的模型：** 通过将权重归零，L1 可以使模型更简洁、更稀疏。

> L1（菱形）和 L2（圆形）约束区域的形状。优化会在损失函数 (loss function)等高线与约束条件相切的点。L1 的尖角使得沿坐标轴（其中一个权重为零）的交点更有可能出现。

**$\lambda$ 的作用：**
与 L2 类似，L1 中的 $\lambda$ 控制着正则化的强度。更大的 $\lambda$ 会导致更多权重趋于零，从而得到一个更稀疏的模型。

### L1 与 L2 的选择

- \*\*L2（权重 (weight)衰减）\*\*通常是更常用且常被推荐的起点。它通过保持权重较小并避免极端值，提供良好的泛化能力提升。其平滑的特性使得优化过程与 L1 相比略微容易。
- \*\*L1（Lasso）\*\*在你怀疑许多输入特征可能无关紧要，并且期望一个能自动选择特征的稀疏模型时很有用。然而，由于绝对值函数在零点不可导的特性，它在训练期间有时会不太稳定（尽管这在实践中会得到处理）。如果某些有用特征与其他特征高度关联，它也可能将其舍弃。

在某些情况下，会使用 **Elastic Net** 正则化 (regularization)，它结合了 L1 和 L2 惩罚项，以获取两种方法的优点。

### 实现说明

大多数深度学习 (deep learning)框架使得添加 L1 或 L2 正则化 (regularization)非常直接。对于 L2（权重 (weight)衰减），它通常是优化器中的一个简单参数 (parameter)：

```python
# 在 PyTorch 中使用 AdamW 优化器进行 L2 正则化（权重衰减）的示例
import torch
import torch.optim as optim

# 假设 'model' 是你定义的神经网络
# learning_rate = 1e-3
# regularization_strength = 1e-4 # 这是 L2 的 lambda 值

# AdamW 优化器直接包含权重衰减
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# --- 训练循环将在这里 ---
# loss.backward()
# optimizer.step()
# ...
```

对于 L1，或者有时为了更精细地控制 L2，你可以在调用 `loss.backward()` 之前将惩罚项直接添加到你计算的损失中。

L1 和 L2 等正则化技术是避免过拟合 (overfitting)的重要工具。通过在损失函数 (loss function)中增加一个基于权重大小的惩罚项，它们促使模型更简洁，这类模型通常能更好地泛化到未见数据。请记住，正则化强度 $\lambda$ 是一个超参数 (hyperparameter)，通常需要调整才能获得最佳性能。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  全面介绍深度学习模型中L1和L2正则化的理论与实践，阐述其在对抗过拟合和提高泛化能力方面的作用。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  经典的统计学习著作，详细阐述L1（Lasso）和L2（Ridge）正则化的数学基础，解释它们的统计特性以及与线性模型的联系。
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) — Ilya Loshchilov, Frank Hutter (2017)
  Journal: International Conference on Learning Representations (ICLR 2019); DOI: [10.48550/arXiv.1711.05101](https://doi.org/10.48550/arXiv.1711.05101)
  引入AdamW，这是Adam优化器的改进版本，它正确地将权重衰减与L2正则化项解耦，这在深度学习训练中非常重要。
- [CS231n: Convolutional Neural Networks for Visual Recognition - Lecture Notes on Regularization](https://cs231n.github.io/neural-networks-2/#reg) — Stanford University (2023)
  提供对正则化技术（包括L1和L2）的易懂解释，针对深度学习背景，并包含直观的图示和实用建议。
