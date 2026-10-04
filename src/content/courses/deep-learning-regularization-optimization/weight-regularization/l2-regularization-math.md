---
course: "deep-learning-regularization-optimization"
chapter: "weight-regularization"
lesson: "l2-regularization-math"
sourceId: 4899
sourceUrl: "https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-2-weight-regularization/l2-regularization-math"
title: "L2 正则化：数学公式"
description: "介绍L2正则化的数学公式及其梯度。"
order: 3
plots: []
sourceHash: "2fcc1ea709716888e3b9470133caf2f8b793ed3f958999215c67fd5b720c1753"
sourceCorrections: []
---

L2正则化 (regularization)通过惩罚大权重 (weight)来鼓励更简单的模型并提升泛化能力。数学上，L2正则化通过修改模型的客观函数——即训练期间旨在最小化的函数——并添加一个代表大权重惩罚的项来进行形式化。

### 正则化 (regularization)损失函数 (loss function)

通常，在训练神经网络 (neural network)时，我们最小化一个损失函数，通常表示为 $J(\theta)$，它衡量模型预测值与真实目标值之间的差异。常见例子包括回归任务的均方误差 (MSE) 或分类任务的交叉熵损失。这里，$\theta$ 代表网络中所有可学习的参数 (parameter)（权重 (weight)和偏置 (bias)）。

L2 正则化向这个原始损失函数添加一个惩罚项。这个惩罚项与权重的平方大小成比例。对于一个具有权重矩阵 $W^{[1]}, W^{[2]}, ..., W^{[L]}$（对应第 $1$ 到 $L$ 层）的网络，L2 正则化项计算为这些权重矩阵的平方 Frobenius 范数之和：


$$
\text{L2惩罚项} = \frac{\lambda}{2m} \sum_{l=1}^{L} ||W^{[l]}||_F^2
$$


我们来逐一分析：

- $W^{[l]}$ 是第 $l$ 层的权重矩阵。
- $||W^{[l]}||_F^2$ 是权重矩阵 $W^{[l]}$ 的平方 Frobenius 范数。矩阵的 Frobenius 范数是通过对矩阵中的每个元素进行平方并求和来计算的：$||W||_F^2 = \sum_{i} \sum_{j} (w_{ij})^2$。本质上，它是该层所有单个权重值的平方和。
- $\sum_{l=1}^{L}$ 表示我们将这些惩罚项在所有层 $l$ 从 $1$ 到 $L$ 上求和。
- $m$ 是小批量中的训练样本数量（或总训练集大小，取决于约定）。除以 $m$ 将惩罚项平均到每个样本上。
- $\lambda$ (lambda) 是**正则化超参数 (hyperparameter)**。这是一个非负值，您在训练前选择。它控制正则化的强度。更大的 $\lambda$ 会对大权重施加更强的惩罚，从而产生更小的权重和更简单的模型。如果 $\lambda = 0$，我们恢复原始的、未正则化的损失函数。
- 包含因子 $\frac{1}{2}$ 主要为了数学上的便利，因为它简化了导数的计算，我们将在后面看到。

新的正则化损失函数，我们称之为 $J_{reg}(\theta)$，变为：


$$
J_{reg}(\theta) = J(\theta) + \frac{\lambda}{2m} \sum_{l=1}^{L} ||W^{[l]}||_F^2
$$

> **注意：** 通常，偏置项（如 $b^{[l]}$）*不*包含在正则化惩罚中。尽管对偏置项进行正则化是可行的，但与权重正则化相比，它对模型复杂度的影响通常可以忽略不计，在实践中也常被省略。

### 对梯度下降 (gradient descent)的影响

这个附加项如何通过梯度下降影响训练过程？在反向传播 (backpropagation)过程中，我们计算损失函数 (loss function)对每个参数 (parameter)的梯度。添加L2惩罚项后，正则化 (regularization)损失 $J_{reg}$ 对特定权重 (weight) $w_{ij}^{[l]}$（连接第 $l-1$ 层神经元 $j$ 到第 $l$ 层神经元 $i$ 的权重）的梯度为：


$$
\frac{\partial J_{reg}}{\partial w_{ij}^{[l]}} = \frac{\partial J(\theta)}{\partial w_{ij}^{[l]}} + \frac{\partial}{\partial w_{ij}^{[l]}} \left( \frac{\lambda}{2m} \sum_{k=1}^{L} ||W^{[k]}||_F^2 \right)
$$


原始损失项的导数 $\frac{\partial J(\theta)}{\partial w_{ij}^{[l]}}$ 照常通过反向传播计算。L2惩罚项对 $w_{ij}^{[l]}$ 的导数为：


$$
\frac{\partial}{\partial w_{ij}^{[l]}} \left( \frac{\lambda}{2m} \sum_{k=1}^{L} \sum_{p} \sum_{q} (w_{pq}^{[k]})^2 \right) = \frac{\lambda}{2m} \frac{\partial}{\partial w_{ij}^{[l]}} (w_{ij}^{[l]})^2 = \frac{\lambda}{2m} (2 w_{ij}^{[l]}) = \frac{\lambda}{m} w_{ij}^{[l]}
$$


(请注意，因子2是如何巧妙地抵消的)。

因此，权重 $w_{ij}^{[l]}$ 的完整梯度为：


$$
\frac{\partial J_{reg}}{\partial w_{ij}^{[l]}} = \frac{\partial J(\theta)}{\partial w_{ij}^{[l]}} + \frac{\lambda}{m} w_{ij}^{[l]}
$$


### “权重 (weight)衰减”效应

现在，我们来看看这个权重的梯度下降 (gradient descent)更新规则，使用学习率 $\alpha$：


$$
w_{ij}^{[l]} := w_{ij}^{[l]} - \alpha \frac{\partial J_{reg}}{\partial w_{ij}^{[l]}}
$$


代入我们得到的梯度：


$$
w_{ij}^{[l]} := w_{ij}^{[l]} - \alpha \left( \frac{\partial J(\theta)}{\partial w_{ij}^{[l]}} + \frac{\lambda}{m} w_{ij}^{[l]} \right)
$$


我们可以稍微重新排列一下：


$$
w_{ij}^{[l]} := w_{ij}^{[l]} - \alpha \frac{\lambda}{m} w_{ij}^{[l]} - \alpha \frac{\partial J(\theta)}{\partial w_{ij}^{[l]}}
$$


$$
w_{ij}^{[l]} := w_{ij}^{[l]} \left( 1 - \alpha \frac{\lambda}{m} \right) - \alpha \frac{\partial J(\theta)}{\partial w_{ij}^{[l]}}
$$


仔细观察这一项 $\left( 1 - \alpha \frac{\lambda}{m} \right)$。由于学习率 $\alpha$ 和正则化 (regularization)强度 $\lambda$ 为正，且 $m$ 是批量大小（也为正），这个因子略小于1（假设 $\alpha \lambda / m$ 足够小，这对于稳定训练通常是成立的）。

这个方程表明了为什么L2正则化常被称为**权重衰减**。在每个更新步骤中，在应用原始损失的梯度之前，权重 $w_{ij}^{[l]}$ 乘以一个略小于1的因子。这有效地使权重在每一步中缩小或“衰减”到接近零。权重越大，衰减效果越明显（由于梯度中的 $\frac{\lambda}{m} w_{ij}^{[l]}$ 项）。这种机制阻止权重变得过大，实现了我们之前讨论的道理：保持权重较小有助于简化模型并提高泛化能力。

超参数 (parameter) (hyperparameter) $\lambda$ 直接控制这种衰减的速度。更高的 $\lambda$ 会导致更快的衰减和更小的最终权重。找到 $\lambda$ 的合适值是一个超参数调优过程，通常使用验证集完成。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/contents/regularization.html) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press; Pages: 226-228
  一本标准教材，解释了深度学习中L2正则化的数学公式和“权重衰减”效应。
- [Pattern Recognition and Machine Learning](https://link.springer.com/book/9780387310732) — Christopher M. Bishop (2006)
  Publisher: Springer; Pages: 148-154
  提供了L2正则化的全面统计处理，涵盖了其在机器学习中的起源和一般原理。
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap3.html#l2_regularization_weight_decay) — Michael Nielsen (2019)
  一份在线资源，提供了关于L2正则化、其数学细节以及“权重衰减”机制的清晰易懂的解释。
- [CS231n: Convolutional Neural Networks for Visual Recognition - Regularization](https://cs231n.github.io/neural-networks-2/#reg) — Andrej Karpathy, Fei-Fei Li, Justin Johnson, Serena Yeung, et al. (2023)
  知名大学课程的讲义，详细阐述了神经网络中L2正则化和权重衰减的实践与数学方面。
