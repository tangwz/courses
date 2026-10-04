---
course: "optimization-techniques-ml"
chapter: "adaptive-learning-rate-algorithms"
lesson: "adagrad-algorithm"
sourceId: 1261
sourceUrl: "https://apxml.com/zh/courses/optimization-techniques-ml/chapter-3-adaptive-learning-rate-algorithms/adagrad-algorithm"
title: "AdaGrad：根据过往梯度调整学习率"
description: "对 AdaGrad 算法及其逐参数学习率调整机制的解释。"
order: 2
plots: []
sourceHash: "6c7847845c644d5f440487e3ddd651aee306f4ea9eb3568afd0cc7deb18979d7"
sourceCorrections: []
---

在整个训练过程中，对所有参数 (parameter)使用单一固定的学习率 $\eta$ 可能效率不高。与常出现特征相关的参数，较小的更新可能更有益，以避免发散，而与稀疏特征相关的参数，则可能需要更大的更新才能取得显著进展。手动调整数百万个参数的单独学习率是不现实的。

AdaGrad（自适应梯度算法）是为解决这一难题而设计的最早算法之一，它引入了*针对每个参数*的学习率，这些学习率会根据每个参数的过往梯度自动调整。

### 核心思想：学习率与过往梯度成反比缩放

AdaGrad 背后的想法很直接：过去频繁遇到大梯度的参数 (parameter)，其学习率应被降低；而见到小梯度或不常出现梯度的参数，其学习率应被提高（或者更准确地说，降低得更少）。这有助于在损失曲面的平坦方向上（通常与稀疏特征相关）更快地进展，并在陡峭方向上（通常与密集特征相关）更谨慎地迈步。

### AdaGrad 更新规则

AdaGrad 修改了标准的梯度下降 (gradient descent)更新。对于时间步 $t$ 的每个参数 (parameter) $w_i$，更新如下：


$$
w_{t+1, i} = w_{t, i} - \frac{\eta}{\sqrt{G_{t, ii} + \epsilon}} g_{t, i}
$$


我们来分析一下各部分：

1. $w_{t, i}$: 在时间步 $t$ 时第 $i$ 个参数的值。
2. $\eta$: 一个全局学习率，类似于标准梯度下降。尽管 AdaGrad 会根据每个参数调整学习率，但这个初始全局学习率仍设定了整体的尺度。
3. $g_{t, i}$: 在时间步 $t$ 时，损失函数 (loss function)对参数 $w_i$ 的梯度，即 $\frac{\partial L}{\partial w_{t, i}}$。
4. $G_{t, ii}$: 主要部分。这是从第一个时间步到当前时间步 $t$，参数 $w_i$ 的梯度平方的累积和。具体来说：
   
   $$
   G_{t, ii} = \sum_{\tau=1}^{t} g_{\tau, i}^2
   $$
   
   在实际应用中，这通常是迭代计算的。如果 $G_{t-1}$ 存储了之前步骤的总和，那么 $G_t$ 可以通过加上当前梯度向量 (vector) $g_t$ 的逐元素平方来计算：$G_t = G_{t-1} + g_t \odot g_t$，其中 $\odot$ 表示逐元素相乘。$G_{t,ii}$ 是这个累积矩阵的第 $i$ 个对角线元素（如果 $G_t$ 以向量形式存储，则就是第 $i$ 个元素）。
5. $\epsilon$: 一个小的平滑项（例如 $10^{-7}$ 或 $10^{-8}$），添加到分母中以避免除以零，尤其是在初始步骤中 $G_{t, ii}$ 可能为零的情况下。

注意更新规则如何融入 $G_{t, ii}$。项 $\sqrt{G_{t, ii} + \epsilon}$ 充当每个参数的缩放因子。如果参数 $w_i$ 一直有大的梯度（$g_{\tau, i}$），其对应的 $G_{t, ii}$ 将变大，使得分母变大，从而降低该特定参数的*有效学习率* $\frac{\eta}{\sqrt{G_{t, ii} + \epsilon}}$。反之，如果参数的梯度很小或为零（稀疏特征常见），$G_{t, ii}$ 将保持较小，导致该参数的有效学习率更大。

### AdaGrad 的优点

- **参数 (parameter)特定调整：** 自动为单个参数调整学习率，与基本 SGD 相比，通常无需进行大量的手动调整。
- **对稀疏数据有效：** 特别适合处理具有稀疏特征的问题（例如使用词嵌入 (embedding)的自然语言处理任务或推荐系统）。不常出现的特征会获得更大的更新，使模型即便在这些特征稀少的情况下也能从中高效学习。

### 缺点：学习率衰减过快

AdaGrad 的主要缺点直接源于其核心机制：即分母中梯度平方的累积 ($G_{t, ii}$)。由于梯度平方总是非负的，这个和在整个训练过程中会单调增长。随着 $G_{t, ii}$ 的增加，有效学习率 $\frac{\eta}{\sqrt{G_{t, ii} + \epsilon}}$ 对每个参数 (parameter)都会持续减小。

在许多情况下，尤其是在训练可能持续许多周期的深度学习 (deep learning)中，这种衰减变得过于激进。学习率会相对快速地减小到接近零的值，导致训练过程急剧放缓，甚至提前停止，远未达到一个好的最小值。

这一限制促使了后续自适应方法（如 RMSprop 和 Adam）的出现，这些方法旨在保留逐参数调整的优点，同时缓解学习率过快衰减的问题。我们接下来将介绍这些方法。

## 参考资料

- [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) — John Duchi, Elad Hazan, and Yoram Singer (2011)
  Journal: Journal of Machine Learning Research; Publisher: Microtome Publishing; Volume: 12; Pages: 2121-2159
  介绍AdaGrad算法及其理论基础的开创性论文。
- [Deep Learning](https://www.deeplearningbook.org/contents/optimization.html) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 8.5.1
  一本基础教材，在深度学习优化技术的更广阔背景下讨论了AdaGrad，包括其机制和局限性。
- [CS231n: Convolutional Neural Networks for Visual Recognition - Optimization](https://cs231n.github.io/optimization-2/) — Stanford University (2023)
  Publisher: Stanford University
  官方课程笔记，为AdaGrad在深度学习优化中的实际应用提供了易懂的解释。
