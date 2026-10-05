# Dropout 比率作为超参数

来源：[原文](https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-3-dropout-regularization/dropout-rate-hyperparameter)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Dropout 的效果很大程度上取决于对**丢弃比率**（通常表示为 $p$）的选择。这个比率表示在训练前向传播过程中，任何给定神经元的输出被设为零的概率。$p$ 是一个超参数 (parameter) (hyperparameter)，这意味着它的值不像模型权重 (weight)那样从数据中学习得到；相反，它必须在训练开始前被确定。

$p$ 的值直接控制 Dropout 施加的正则化 (regularization)强度。它的取值范围是 $[0, 1]$：

- 丢弃比率 $p=0$ 意味着没有单元被丢弃。这相当于关闭了 Dropout 功能，网络在训练时表现得像一个标准神经网络 (neural network)。
- 丢弃比率 $p=1$ 意味着所有单元都被丢弃。这将完全阻止网络学习任何东西，因为信息无法通过被丢弃的层。
- 0 到 1 之间的值引入了随机性和正则化。$p$ 值越高，正则化越强，因为在每次训练迭代中，平均有更多单元被归零。

### 选择合适的 Dropout 比率

为 $p$ 选择一个合适的值通常受到经验结果的指导，并取决于具体的网络结构和数据集。不过，有一些通用建议可循：

1. **常见起始点：** 隐藏层的常用默认值是 $p=0.5$。这个值通常能在正则化 (regularization)和保证足够的信息流动以供学习之间取得良好平衡。当你首次在模型中应用 Dropout 时，这通常是一个合理的起点。
2. **常用范围：** 尽管 $p=0.5$ 很常见，但理想值通常在 $p=0.2$ 到 $p=0.5$ 的范围内。远高于 0.5 的值有时会因为丢弃过多信息而阻碍训练，尤其是在较小的网络中。低于 0.2 的值则提供较弱的正则化。
3. **分层比率：** 在网络的不同层使用不同的 Dropout 比率并不少见。
   - **输入层：** 直接将 Dropout 应用于输入层较不常见。如果使用，通常涉及一个远小的 Dropout 比率（例如，$p=0.1$ 或 $p=0.2$）。丢弃原始输入特征可能会造成过大干扰。
   - **隐藏层：** 较高的 Dropout 比率（如 $p=0.2$ 到 $p=0.5$ 的范围）通常应用于此。较大的隐藏层有时可能比小的隐藏层受益于略高的 Dropout 比率，因为它们具有更多冗余。
4. **模型复杂度与数据集大小：** 理想的 Dropout 比率通常与模型的容量和训练数据量相关。
   - 参数 (parameter)较多的大型模型更容易过拟合 (overfitting)，可能受益于更高的 $p$ 值。
   - 如果你的数据集较小，通过提高 $p$ 值来加强正则化可能有助于提升泛化能力。反之，对于非常大的数据集，过拟合可能不是主要问题，这可能允许使用较低的 $p$ 值，甚至不使用 Dropout。

### 调整 Dropout 比率

由于 $p$ 是一个超参数 (parameter) (hyperparameter)，找到其最佳值通常需要通过实验。对待 Dropout 比率，就像对待学习率或 L2 正则化 (regularization)强度 ($\lambda$) 等其他超参数一样。你可以使用以下方法：

- **手动调整：** 从一个常用值（例如，隐藏层使用 0.5）开始，并观察训练和验证性能（使用学习曲线，如第 1 章所述）。如果模型仍然明显过拟合 (overfitting)，尝试增加 $p$。如果模型欠拟合 (underfitting)或难以收敛，尝试减小 $p$。
- **网格搜索/随机搜索：** 系统地尝试 $p$ 的不同值（例如，[0.1, 0.2, 0.3, 0.4, 0.5]）以及其他超参数，在验证集上评估每个组合，以找出表现最佳的配置。

请记住，如果你修改网络结构的其他方面或训练过程（如优化器或学习率），最佳的 Dropout 比率可能会改变。它通常与其他这些元素一起调整。在实践中，设置 $p$ 需要平衡正则化的需求与阻碍网络学习复杂模式的风险。像 $p=0.5$ 这样的值是一个好的起点，但根据验证性能进行微调 (fine-tuning)对于最大限度地发挥 Dropout 的优势很重要。

## 参考资料

- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) — Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov (2014)
  Journal: Journal of Machine Learning Research; Volume: 15; Pages: 1929-1958
  介绍 Dropout 正则化技术及其应用和丢弃率初始建议的原始学术论文。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本详细解释 Dropout 正则化方法和超参数选择通用策略的综合教科书。
- [Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization (Course 2 of Deep Learning Specialization)](https://www.coursera.org/learn/deep-neural-networks-improve-tuner) — Andrew Ng (2021)
  Publisher: Coursera / deeplearning.ai
  一个提供超参数调整实用指导的在线课程模块，包括选择和优化丢弃率的最佳实践。

---

[上一节](04-%E5%8F%8D%E8%BD%ACDropout%E5%AE%9E%E7%8E%B0.md) · [下一节](06-%E5%8D%B7%E7%A7%AF%E5%B1%82%E5%92%8C%E5%BE%AA%E7%8E%AF%E5%B1%82%E7%9A%84%E4%BD%BF%E7%94%A8%E8%80%83%E9%87%8F.md)
