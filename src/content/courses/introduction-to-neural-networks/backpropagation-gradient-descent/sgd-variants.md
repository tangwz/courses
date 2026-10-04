---
course: "introduction-to-neural-networks"
chapter: "backpropagation-gradient-descent"
lesson: "sgd-variants"
sourceId: 1978
sourceUrl: "https://apxml.com/zh/courses/introduction-to-neural-networks/chapter-4-backpropagation-gradient-descent/sgd-variants"
title: "随机梯度下降 (SGD) 及其变体"
description: "了解不同的梯度下降优化算法，如 SGD、小批量梯度下降和 Adam。"
order: 6
plots: ["plots/1978-0.json"]
sourceHash: "2f53ecd4314adacadb2094f56c709bb3927d7cec32c3e7245fba00b42f4df948"
sourceCorrections: []
---

虽然使用整个数据集的梯度下降 (gradient descent)方法（常被称为批量梯度下降）为最小化损失函数 (loss function)提供了一个直接的途径，但它伴随着巨大的计算开销。计算梯度需要处理每一个训练样本，才能对网络的权重 (weight)和偏差进行一次更新。对于包含数百万样本的数据集，这会变得非常缓慢且占用大量内存。

为了解决这个问题，梯度下降发展出了几种变体。这些方法旨在实现更快的收敛，并更有效地处理大型数据集。

### 随机梯度下降 (gradient descent) (SGD)

随机梯度下降 (SGD) 采取了与批量梯度下降相反的方式。SGD 不再根据*整个*数据集计算梯度，而是在每一步只使用从*一个*随机选择的训练样本计算的梯度来更新参数 (parameter)。

权重 (weight) $W$ 的更新规则与批量梯度下降相似，但损失和梯度 $\nabla Loss$ 仅使用单个样本 $(x^{(i)}, y^{(i)})$ 计算：

$W_{new} = W_{old} - \eta \nabla Loss(W_{old}; x^{(i)}, y^{(i)})$

**优点：**

- **速度：** 每次迭代只处理一个样本，更新速度非常快。这使得迭代能快速进行，尤其是在训练初期。
- **逃离局部最小值：** 更新中固有的噪声（因为每个梯度都只基于一个样本）可以帮助优化器跳出局部浅层最小值或鞍点，这些点可能会困住批量梯度下降。

**缺点：**

- **高方差：** 更新可能噪声很大，导致损失函数 (loss function)大幅波动，而非平稳下降。通往最小值的路径可能显得杂乱无章。
- **收敛速度慢（周期）：** 虽然单个更新速度快，但与批量梯度下降相比，可能需要更多次遍历整个数据集（周期）才能收敛，并且收敛可能会在最小值附近振荡，而无法精确稳定下来。

想象一下在有雾的山上行进。批量梯度下降会仔细地观察周围整个区域，然后才朝着最佳方向迈出一步。随机梯度下降则只是快速朝一个方向看一眼（基于一条信息），然后立即迈出一步，并快速重复这一过程。它最初移动得更快，但可能会频繁地之字形前进。



![梯度下降路径比较](plots/1978-0.json)



> 损失曲面上的优化路径。批量梯度下降走的是一条直接路线。随机梯度下降的路径因单样本更新而噪声大。小批量梯度下降则提供了一种平衡。

### 小批量梯度下降 (gradient descent)

小批量梯度下降在批量梯度下降的稳定性和随机梯度下降的速度之间取得了平衡。它根据训练数据的一小部分随机选择的子集（即“小批量”）来计算梯度并更新参数 (parameter)。常用的小批量大小范围从 32 到 512，但具体大小可能因应用和硬件能力而异。

更新规则使用小批量中 $m$ 个样本的平均梯度：

$W_{new} = W_{old} - \eta \frac{1}{m} \sum_{i=1}^{m} \nabla Loss(W_{old}; x^{(i)}, y^{(i)})$

**优点：**

- **效率：** 它利用了 NumPy、TensorFlow 和 PyTorch 等库中优化的矩阵运算，这使得梯度计算比随机梯度下降每次迭代一个样本要快得多。
- **更平滑的收敛：** 对小批量进行平均降低了更新的方差，与随机梯度下降相比，从而带来更稳定的收敛路径。
- **主流选择：** 这是实践中最常用的变体。

**缺点：**

- **小批量大小超参数 (hyperparameter)：** 它引入了一个额外的超参数（小批量大小 $m$），需要进行调整。
- **仍有噪声（与批量梯度下降相比）：** 虽然噪声比随机梯度下降小，但更新仍然基于数据的一个子集，因此损失可能不会单调下降。

小批量梯度下降就像在山上行进时，在迈出一步前，先从周围一小块区域（小批量）获取信息。它比查看所有信息（批量梯度下降）更快，并且比只看一个点（随机梯度下降）更稳定。

### 改进更新：动量和自适应方法

尽管小批量梯度下降 (gradient descent)很有效，研究人员已经发展出更复杂的优化算法，这些算法通常能带来更快的收敛和更好的效果。这些算法通常解决两个主要问题：在复杂的损失曲面上移动（例如，峡谷、鞍点）以及选择合适的学习率。

#### 动量

想象一个球滚下山。它不会立刻停止；它会积累动量，帮助它越过小障碍并加速下坡。动量更新将类似的想法融入到梯度下降中。它将上一更新向量 (vector)的一部分 $\beta$（动量系数，通常约为 0.9）加到当前的梯度步长中。

设 $v_t$ 为步骤 $t$ 时的更新向量（速度）：

$v_t = \beta v_{t-1} + \eta \nabla Loss(W_{t-1})$
$W_t = W_{t-1} - v_t$

(注意：有时学习率 $\eta$ 的计算方式不同，例如 $v_t = \beta v_{t-1} + (1-\beta) \nabla Loss$，更新为 $W_t = W_{t-1} - \alpha v_t$。核心思想不变。)

动量有助于在相关方向上加速随机梯度下降并抑制振荡，这在具有狭窄峡谷的损失曲面上特别有用，因为标准梯度下降可能会在其中来回震荡。

#### 自适应学习率方法

自适应方法不为所有参数 (parameter)使用单一的固定学习率 $\eta$，而是在训练期间调整学习率，通常是*每个参数独立调整*。它们通常对与频繁出现的特征相关的参数给予较小的更新，而对与不频繁出现的特征相关的参数给予较大的更新。

- **RMSprop（均方根传播）：** 该方法为每个参数保持一个平方梯度的移动平均值，并将学习率除以该平均值的平方根。这有效地降低了具有持续大梯度的参数的学习率，并提高了具有小梯度的参数的学习率。设 $E[g^2]_t$ 为在步骤 $t$ 时平方梯度的移动平均值：
  $E[g^2]_t = \gamma E[g^2]_{t-1} + (1-\gamma) (\nabla Loss(W_{t-1}))^2$
  更新规则变为：
  $W_t = W_{t-1} - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \nabla Loss(W_{t-1})$
  这里，$\gamma$ 是衰减率（例如 0.9），而 $\epsilon$ 是一个很小的常数（例如 $10^{-8}$），用于防止除以零。
- **Adam（自适应矩估计）：** Adam 可能是目前最受欢迎且有效的优化器之一。它结合了动量和 RMSprop 的思想。它同时跟踪过去梯度的指数衰减平均值（一阶矩，$m_t$）和过去平方梯度的指数衰减平均值（二阶矩，$v_t$）。
  $m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla Loss(W_{t-1})$
  $v_t = \beta_2 v_{t-1} + (1-\beta_2) (\nabla Loss(W_{t-1}))^2$
  Adam 还加入了偏差校正步骤，这在训练的初始阶段尤其重要，因为此时矩估计被初始化为零：
  $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$
  $\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$
  最终的参数更新使用这些偏差校正后的估计值：
  $W_t = W_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$
  常用的默认值是 $\beta_1 = 0.9$、$\beta_2 = 0.999$ 和 $\epsilon = 10^{-8}$。Adam 通常在超参数 (hyperparameter)调整很少的情况下表现良好，使其成为许多问题的良好默认选择。

### 选择合适的优化器

- **小批量梯度下降 (gradient descent)（常与动量结合）：** 一种可靠的基线方法，计算高效。
- **Adam：** 因其鲁棒性以及在各种任务中的良好表现，常作为默认选择。它通常比基本的随机梯度下降或动量收敛更快。
- **随机梯度下降：** 有时能找到更好（更平坦）的最小值，这些最小值具有更好的泛化能力，但通常需要更仔细地调整学习率和学习率调度。在某些研究环境或自适应方法计算资源有限时可能会被优先考虑。

最佳优化器可能因问题而异。虽然 Adam 是一个很好的起点，但尝试其他方法，如带动量的随机梯度下降或 RMSprop，可能会为特定的网络架构或数据集带来更好的性能。理解这些变体使您能够在训练神经网络 (neural network)时做出明智的选择，以实现更实用、更强大的训练技术。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  全面介绍深度学习基本概念的教材，包括本节讨论的各种优化算法。
- [On the Importance of Initialization and Momentum in Deep Learning](https://proceedings.mlr.press/v28/sutskever13.html) — Ilya Sutskever, James Martens, George Dahl, and Geoffrey Hinton (2013)
  Journal: Proceedings of the 30th International Conference on Machine Learning (ICML); Publisher: PMLR; Volume: 28; Pages: 1139-1147
  探讨动量在加速深度神经网络训练和改善收敛方面作用的重要论文。
- [RMSprop (Lecture 6e, Coursera: Neural Networks for Machine Learning)](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) — Geoffrey Hinton (2012)
  介绍RMSprop算法，一种在深度学习中广泛采用的自适应学习率方法。
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma and Jimmy Ba (2015)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  引入Adam优化器的开创性论文，它结合了动量和自适应学习率，是许多深度学习任务中常用的默认选择。
