---
course: "deep-learning-fundamentals-keras"
chapter: "training-deep-neural-networks"
lesson: "backpropagation-conceptually"
sourceId: 4914
sourceUrl: "https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-3-training-deep-neural-networks/backpropagation-conceptually"
title: "反向传播"
description: "解释反向传播如何在训练期间更新网络权重。"
order: 4
plots: []
sourceHash: "607117ce5f259a70cea4989e50d84c5eb493e731b7599a9af11a1616f7509a61"
sourceCorrections: []
---

我们已经编译了模型，定义了它如何学习（优化器）、应该最小化什么（损失函数 (loss function)），以及如何衡量进展（评估指标）。但优化器究竟是如何利用损失值来改进网络的预测呢？这就是反向传播 (backpropagation)方法发挥作用的地方。

设想你的网络刚刚为一个训练样本做出了预测。你使用损失函数将此预测与真实目标标签进行比较，损失函数会计算出一个数字，表示预测的误差或“偏差”。损失值高意味着预测偏差大；损失值低则表示预测接近真实值。训练的目标是使所有训练样本的损失值最小化。

如何调整网络以减少损失呢？网络的表现由其权重 (weight)和偏差决定。如果我们能确定每个权重微小变化会对最终损失产生多大影响，我们就可以系统地调整权重，使其朝着损失减少最多的方向移动。

这正是反向传播的作用。它是一种计算网络中每个权重和偏差对应的损失函数*梯度*的算法。回想一下微积分知识，梯度指向函数最陡峭的上升方向。因为我们要*最小化*损失，所以需要朝梯度的*反方向*调整权重。

### 反向传播 (backpropagation)：误差归因

可以把反向传播看作一种将误差责任逐层向后分配给网络各部分的方式。

1. **计算输出层误差：** 这一过程从输出层开始。我们直接将网络的输出与真实目标进行比较，并计算损失。我们还可以确定输出神经元的激活值对最终误差的贡献程度。
2. **将误差传播到前一层：** 基于为输出层计算的梯度，反向传播确定*前一个*隐藏层的激活值对输出层误差的贡献程度。这涉及理解连接这些层的权重 (weight)如何影响了结果。
3. **继续反向传播：** 这个过程重复进行，逐层向后移动。在每一层，我们计算其神经元的激活值对*其后*一层所计算误差的贡献。
4. **计算权重梯度：** 当我们反向传播误差信息时，我们同时计算损失函数 (loss function)相对于每层中每个权重的梯度。这表示损失对该特定权重微小变化的敏感度。$\frac{\partial L}{\partial W_{ij}}$ 本质上在问，“如果我稍微调 (fine-tuning)整这个特定的权重 $W_{ij}$，总损失 $L$ 会改变多少，以及朝哪个方向改变？”

从数学角度看，这种信息反向流动主要依赖于微积分中的链式法则，使得我们能够从最终损失开始，逐层有效地计算这些梯度。

> 前向传播将输入数据送入网络以生成预测。通过比较预测值与目标值来计算损失。然后，反向传播将误差梯度从损失处向后传递，计算权重（W1, W2）和激活值（H1, Output）的变化如何影响损失。

### 利用梯度：更新步骤

一旦反向传播 (backpropagation)计算出所有权重 (weight)和偏差的梯度，优化器（你在`compile`步骤中选择的，例如Adam或SGD）就会介入。它会利用这些梯度以及其特有的更新规则（通常包含学习率和动量等因素），来调整网络的权重和偏差。目标是将权重“微调 (fine-tuning)”到能减少损失的方向。

这个完整的循环，包括前向传播、损失计算、反向传播（backpropagation）和权重更新，构成了训练过程的一个步骤。它在多个时期内对数据批次重复多次，逐步引导网络参数 (parameter)趋向于使总损失函数 (loss function)最小化的值，从而使网络在其任务中表现更好。

尽管当你调用`fit()`方法时，Keras会自动处理反向传播的实现细节，但理解这个流程对于诊断训练问题和就模型架构、损失函数和优化器做出明智选择非常重要。

## 参考资料

- [Learning representations by back-propagating errors](https://doi.org/10.1038/323533a0) — David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams (1986)
  Journal: Nature; Publisher: Springer Nature; Volume: 323; Pages: 533-536; DOI: [10.1038/323533a0](https://doi.org/10.1038/323533a0)
  普及用于训练多层神经网络的反向传播算法的开创性论文。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本关于反向传播及其在深度神经网络中作用的详细且严谨的数学教材。
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap2.html) — Michael Nielsen (2019)
  Publisher: Online
  一本优秀的免费在线书籍，为反向传播提供了直观而数学精确的解释，适合自学。
- [CS231n: Deep Learning for Computer Vision, Lecture Notes on Backpropagation](https://cs231n.github.io/optimization-2/) — Stanford University (2023)
  Publisher: Stanford University
  从计算图的角度对反向传播进行了清晰且通常有视觉支持的解释，这对于深度学习很有用。
