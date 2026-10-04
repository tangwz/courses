---
course: "deep-learning-regularization-optimization"
chapter: "adaptive-optimizers"
lesson: "choosing-optimizers-guidelines"
sourceId: 4995
sourceUrl: "https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-6-adaptive-optimizers/choosing-optimizers-guidelines"
title: "优化器选择指南"
description: "提供针对特定任务选择合适优化器的实用指导。"
order: 8
plots: ["plots/4995-0.json"]
sourceHash: "428f84e08aa7e11fed02bbf0aafea13e18d5c5efdb4e6139810ce830bd35cb97"
sourceCorrections: []
---

对于深度学习 (deep learning)项目，AdaGrad、RMSprop 和 Adam 等优化器的选择可能让你困惑：到底该选哪一个？没有一个放之四海而皆准的答案，但理解不同优化器的特点可以帮助你做出明智的决定。此处将提供选择合适优化器的实用指导。

### 带有动量的 SGD：依然是个重要选项

尽管自适应方法很流行，但带有动量的随机梯度下降 (gradient descent)（SGD），特别是其 Nesterov 变体（NAG），仍然是非常有效且应用广泛的优化器。

- **何时考虑使用 SGD+Momentum/NAG：**

  - **成熟的问题类型：** 对于架构和超参数 (parameter) (hyperparameter)都得到充分理解的任务（例如标准图像分类基准），经过调整的 SGD+Momentum 通常能取得很好的结果，有时在最终的泛化表现上会超过自适应方法。
  - **注重泛化能力：** 有些研究表明，SGD+Momentum 发现的解可能比自适应方法发现的解泛化能力稍好，可能停留在“更平坦”的极小值点。这并非普遍成立，但它是持续讨论和研究的焦点。
  - **微调 (fine-tuning)：** 在微调预训练 (pre-training)模型时，使用小学习率的 SGD 通常是可靠的选择。
- **面临的问题：**

  - **超参数敏感性：** SGD+Momentum 对学习率和动量设置的敏感度通常高于自适应方法。找到最佳组合常常需要仔细调整。
  - **学习率调度：** 使用精心设计的学习率调度策略（我们将在下一章讨论）对于 SGD+Momentum 达到最佳性能几乎总是必要的。

### 自适应方法：易用性及初始阶段快速进展

AdaGrad、RMSprop 和 Adam 的开发是为了解决 SGD 的一些问题，主要通过自动调整每个参数 (parameter)的学习率。

- **AdaGrad：** 尽管它在引入参数特有学习率方面具有历史意义，但 AdaGrad 的学习率衰减过快的倾向常常导致深度学习 (deep learning)训练过早停止。如今它在训练深度神经网络 (neural network)时不太常用，但仍可能适用于梯度非常稀疏的情形。
- **RMSprop：** 通过使用梯度平方的移动平均而非无限累积，RMSprop 解决了 AdaGrad 学习率快速衰减的问题。它仍然是一个可行选择，尤其当你想要一种比 Adam 稍简单的自适应方法时。在循环神经网络（RNN）训练中，它有时受青睐。
- **Adam：** Adam 在 RMSprop 的基础上，融入了动量（梯度本身的移动平均）。这种组合常常在训练早期实现快速收敛。
  - **AdamW：** 作为一种流行改进，AdamW 将权重 (weight)衰减（L2 正则化 (regularization)）与自适应学习率计算解耦，这通常比标准 Adam 能够提高正则化效果和最终模型性能，因为在标准 Adam 中 L2 正则化会与自适应缩放相互作用。

### Adam/AdamW：常用的起点

对于许多深度学习 (deep learning)应用，**Adam 或 AdamW 通常是推荐的默认优化器**。

- **Adam/AdamW 为何流行：**
  - **普遍表现良好：** 与 SGD 相比，它在多种任务和架构上表现良好，且超参数 (parameter) (hyperparameter)调整相对较少。
  - **初始阶段收敛快：** 它通常能快速找到不错的解。
  - **初始学习率敏感度较低：** 尽管调整学习率（$\alpha$）仍然有益，但 Adam 对初始选择的敏感度通常低于 SGD。像 $\alpha = 0.001$ 或 $\alpha = 0.0003$ 这样的标准值是常用的起始值。
  - **框架默认设置：** 它常是 PyTorch 和 TensorFlow/Keras 等库中的默认优化器。

### 实用建议与权衡

1. **从 Adam 或 AdamW 开始：** 使用推荐的默认超参数 (parameter) (hyperparameter)（例如，学习率 $\approx 10^{-3}$，$\beta_1 \approx 0.9$，$\beta_2 \approx 0.999$，$\epsilon \approx 10^{-8}$）。如果使用权重 (weight)衰减，AdamW 通常优于标准 Adam。
2. **调整学习率：** 即使使用自适应方法，调整学习率仍然是影响力最大的超参数调整。可以尝试 $3 \times 10^{-4}$、$10^{-4}$、$3 \times 10^{-5}$ 等值。
3. **在以下情况考虑使用 SGD+Momentum/NAG：**
   - Adam/AdamW 性能停滞或表现不理想时，尤其是在对未见数据的泛化能力方面。
   - 你正在一个发展成熟的方面工作，且 SGD 调整策略已为人熟知。
   - 你正在微调 (fine-tuning)预训练 (pre-training)模型。
4. **预期 SGD 需要更多调整：** 如果你改用 SGD+Momentum，请准备投入更多精力调整学习率、动量参数，尤其是学习率调度策略。
5. **进行试验：** 最终，针对你的特定任务、数据集和模型架构，最佳优化器可能需要通过实证试验来确定。密切关注训练和验证曲线。



![优化器收敛比较](plots/4995-0.json)



> 比较展示了典型收敛模式。Adam 等自适应方法通常在初始阶段收敛更快，而经过充分调整的 SGD+Momentum 在某些情形下可能达到稍好的最终验证损失，尽管这需要更多调整工作。

优化器的选择涉及平衡易用性、收敛速度、计算开销和潜在的泛化表现。尽管 Adam/AdamW 提供了一个强大且通常有效的起点，但理解不同优化器的特点和权衡，能让你做出更好的选择，并可能通过试验和仔细调整，进一步提升模型的性能。请记住，优化与训练的其他方面（如初始化和学习率调度）紧密关联，我们将在接下来探讨。

## 参考资料

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma, Jimmy Ba (2015)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍了Adam优化器，其应用广泛且常作为默认选择。
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) — Ilya Loshchilov, Frank Hutter (2019)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1711.05101](https://doi.org/10.48550/arXiv.1711.05101)
  提出了AdamW，将权重衰减与自适应学习率分离，提升了正则化效果。
- [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) — John Duchi, Elad Hazan, Yoram Singer (2011)
  Journal: Journal of Machine Learning Research (JMLR); Publisher: Journal of Machine Learning Research; Volume: 12; Pages: 2121-2159; DOI: [10.5555/1953048.2021021](https://doi.org/10.5555/1953048.2021021)
  介绍了AdaGrad，这是一种基础的自适应学习率方法。
- [On the Importance of Initialization and Momentum in Deep Learning](http://proceedings.mlr.press/v28/sutskever13.pdf) — Ilya Sutskever, James Martens, George Dahl, and Geoffrey Hinton (2013)
  Journal: Proceedings of the 30th International Conference on Machine Learning (ICML); Pages: 1139-1147
  讨论了基于动量的SGD（包括Nesterov动量）在深度学习中的有效性。
