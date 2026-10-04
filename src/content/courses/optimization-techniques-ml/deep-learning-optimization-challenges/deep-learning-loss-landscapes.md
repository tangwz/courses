---
course: "optimization-techniques-ml"
chapter: "deep-learning-optimization-challenges"
lesson: "deep-learning-loss-landscapes"
sourceId: 1300
sourceUrl: "https://apxml.com/zh/courses/optimization-techniques-ml/chapter-6-deep-learning-optimization-challenges/deep-learning-loss-landscapes"
title: "深度学习损失曲面的特点"
description: "分析深度网络中的高维度、鞍点、高原区域和尖锐最小值等特性。"
order: 1
plots: ["plots/1300-0.json"]
sourceHash: "c9dc901aa12865ced05ff0732dfb0749ee17f446a70e8c8f2ae5e393eab540d7"
sourceCorrections: []
---

训练深度神经网络 (neural network)通常与优化线性回归或支持向量 (vector)机等更简单的机器学习 (machine learning)模型感觉不同。这种差异很大程度上源于我们要最小化的函数本质，即其“地形”。与在较简单情境中遇到的相对规整的凸碗状地形不同，深度网络的“地形”是一个复杂、高维且充满意想不到特征的空间。了解这些特点有助于理解为何某些优化策略在深度学习 (deep learning)中表现更优。

### 高维度与非凸性

神经网络 (neural network)的优化空间具有高维度。现代神经网络可以拥有数百万甚至数十亿个参数 (parameter)（权重 (weight)和偏置 (bias)）。每个参数代表优化空间中的一个维度。在数十亿维空间中可视化函数是不可能的，因此我们通常基于2D或3D例子形成的直觉可能会产生误导。

从数学角度看，深度网络的损失函数 (loss function) $L(\theta)$（其中 $\theta$ 代表所有参数的向量 (vector)）几乎总 是*非凸的*。这意味着与凸函数（看起来像一个碗）不同，损失函数可以有多个*局部最小值*：即在其直接邻域内损失值较低的点，但这些点不一定是整个空间中可达到的最低损失（*全局最小值*）。

为何存在非凸性？连续层中使用的非线性激活函数 (activation function)（如ReLU、sigmoid、tanh）以及网络的组合结构（$output = f_L(...f_2(f_1(input; \theta_1); \theta_2)...; \theta_L)$）共同作用，在参数 $\theta$ 与最终损失 $L$ 之间建立了高度复杂、非线性的关系。简单的操作，例如在某一层中交换两个相同的隐藏单元，可以产生完全相同的网络输出，但却对应于参数空间 $\theta$ 中的不同点，这直接意味着存在多个等价的最小值，从而导致非凸性。

### 鞍点的普遍性

尽管最初认为存在众多局部最小值是非凸优化的主要困难，但研究表明，对于深度学习 (deep learning)中常见的高维问题，*鞍点*可能是一个更重要的障碍。

鞍点是梯度 $\nabla L(\theta)$ 为零（或非常接近零）的位置，就像最小值或最大值一样，但它*不是*局部极值点。相反，函数在某些方向上向上弯曲，在另一些方向上向下弯曲，就像马鞍或山隘。从数学角度看，在鞍点处，海森矩阵 $\nabla^2 L(\theta)$（二阶偏导数矩阵）同时具有正负特征值。

为何鞍点会带来问题？

1. **梯度为零：** 梯度下降 (gradient descent)等一阶优化方法仅依靠梯度 $\nabla L(\theta)$ 来确定更新方向。在鞍点附近，梯度变得非常小，导致优化器速度急剧下降，可能需要指数级长的时间才能逃离。
2. **高维度的普遍性：** 在高维度空间中，鞍点的数量远多于局部最小值。要使一个点成为局部最小值，海森矩阵必须是半正定（所有特征值 $\ge 0$）。而要成为鞍点，海森矩阵只需要有*一些*正特征值和*一些*负特征值。随着维度的增加，遇到梯度接近零的鞍点的概率会显著提高。

简单的SGD在有效处理鞍点时可能遇到困难。结合动量或自适应学习率的方法（在第三章中介绍）通常能更好地处理鞍点，通过积累速度或根据梯度历史调整步长，有助于“越过”与鞍点相关的平坦区域。二阶方法利用海森信息，可以明确识别逃逸方向（负曲率方向），但对于大型网络来说，计算海森矩阵通常过于昂贵。



![2D损失特点](plots/1300-0.json)



> 简化的2D投影，展示了潜在的特征：局部最小值、鞍点（梯度平坦，但曲率变化）以及更平坦的高原区域。实际的“地形”存在于更高的维度中。

### 高原区域与平坦区域

另一个常见特点是存在大片相对平坦的区域，称为高原区域。在这些区域，损失函数 (loss function)在参数 (parameter)空间中经过较大距离时变化非常缓慢，这意味着梯度 $\nabla L(\theta)$ 始终很小。在高原区域进行优化可能会变得极其缓慢，类似于鞍点附近的减速。这种现象有时与非常深的神经网络 (neural network)中梯度消失问题相关联，在该问题中，梯度难以通过许多层进行反向传播 (backpropagation)。归一化 (normalization)（本章后面会讨论）和残差连接等技术在一定程度上被提出，以缓解这些平坦区域并改善梯度流动。

### 尖锐最小值与平坦最小值

并非所有最小值在泛化能力方面都是相同的。设想两个最小值，A和B，它们在训练数据上达到大致相同的低损失值。

- 最小值A是*尖锐的*：如果您在参数 (parameter)空间中稍微偏离最小值点，损失会非常迅速地增加。
- 最小值B是*平坦的*：损失在最小值点周围的更宽广区域内保持较低。

越来越多的证据表明，收敛到*更平坦*最小值的优化器通常能使模型对未见数据具有更好的泛化能力。其道理是，如果损失在解决方案附近是平坦的，那么输入数据的微小变化（导致略微不同的最优参数设置）不太可能引起损失的大幅增加或模型预测的显著改变。

有趣的是，优化算法的选择及其参数（如批量大小和学习率）可以影响优化器是倾向于找到尖锐最小值还是平坦最小值。例如，在SGD中使用更大的批量大小通常与收敛到尖锐最小值相关，而较小的批量大小（引入更多噪声）可能会进行更充分的搜寻并找到更平坦的最小值。Adam等自适应方法也在此背景下被研究，尽管它们之间的关系复杂且仍在进行中。

认识这些多种多样的特点——高维度、非凸性、鞍点多于局部最小值、高原区域，以及尖锐最小值和平坦最小值之间的区别——非常重要。这些特点解释了为何优化深度网络充满挑战，以及为何从自适应学习率到归一化 (normalization)层和仔细的初始化等特定技术已成为标准做法。这些特点直接影响我们所用优化算法的行为、收敛速度和泛化性能。

## 参考资料

- [On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima](https://arxiv.org/abs/1609.04836) — Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, and Ping Tak Peter Tang (2016)
  Journal: ICLR 2017; DOI: [10.48550/arXiv.1609.04836](https://doi.org/10.48550/arXiv.1609.04836)
  这篇论文审视了批次大小、优化器找到的最小值锐度与深度学习模型泛化性能之间的关系。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  这本权威教材的第八章全面概述了优化挑战，包括深度学习中损失函数地形的特点。
- [Visualizing the Loss Landscape of Neural Networks](https://arxiv.org/pdf/1712.09913.pdf) — Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, Tom Goldstein (2018)
  Journal: Advances in Neural Information Processing Systems; Publisher: Neural Information Processing Systems Foundation; Volume: 31; Pages: 6391-6401; DOI: [10.48550/arXiv.1712.09913](https://doi.org/10.48550/arXiv.1712.09913)
  这篇论文介绍了有效可视化神经网络高维损失函数地形的方法，有助于阐明尖锐和扁平最小值等概念。
- [Optimization Methods for Large-Scale Machine Learning](https://doi.org/10.1137/16M1080173) — Léon Bottou, Frank E. Curtis, and Jorge Nocedal (2018)
  Journal: SIAM Review; Publisher: Society for Industrial and Applied Mathematics; Volume: 60; Pages: 223-311; DOI: [10.1137/16M1080173](https://doi.org/10.1137/16M1080173)
  这篇综述文章考察了广泛的优化方法，讨论了它们在大规模机器学习中的适用性和挑战，特别关注了深度学习的非凸性和高维度。
