---
course: "optimization-techniques-ml"
chapter: "adaptive-learning-rate-algorithms"
lesson: "fixed-learning-rate-limitations"
sourceId: 1259
sourceUrl: "https://apxml.com/zh/courses/optimization-techniques-ml/chapter-3-adaptive-learning-rate-algorithms/fixed-learning-rate-limitations"
title: "固定学习率的局限性"
description: "了解为什么固定学习率可能并非最优，以及对自适应方法的需求。"
order: 1
plots: ["plots/1259-0.json"]
sourceHash: "fd229342b111bec1804c6259227a9c3ea0e5aa49608d2a9130b7dfd0e4e034fb"
sourceCorrections: []
---

标准梯度下降 (gradient descent)方法通常依赖于一个单一的超参数 (parameter) (hyperparameter)：学习率，常用 $\eta$ 表示。该值调整梯度 $\nabla J(\theta)$ 的大小，以确定在每次迭代中在参数空间中迈出的步长：$\theta_{t+1} = \theta_t - \eta \nabla J(\theta_t)$。尽管简单，但选择一个有效的固定 $\eta$ 是一个重要的实际难题。

主要问题在于训练过程对这个单一值的敏感性。如果 $\eta$ 设置得太小，优化器会迈出微小的步子。收敛会变得非常缓慢，可能需要不切实际的迭代次数才能达到一个令人满意的最小值。想象一下，你试图下山，但每一步都只有鹅卵石那么大；你最终会到达，但这将花费很长时间，并且你可能会停滞在相对平坦的区域（高原），而这些区域并非真正的谷底。

相反地，将 $\eta$ 设置得太大则会引入不稳定性。优化器可能会完全跳过最小值，在最优区域来回振荡而无法稳定下来。在更糟的情况下，更新步长可能非常大，以至于损失函数 (loss function)反而增加，导致发散。想象一下从山上跳下；你可能会直接跳过最低点，甚至把自己抛下悬崖。



![固定学习率 (η) 对收敛的影响](plots/1259-0.json)



> 凸问题上不同固定学习率的收敛行为。小的学习率收敛缓慢，大的学习率可能振荡或发散。

寻找合适的学习率通常需要大量的手动调整和实验，这可能导致计算成本高昂。此外，最优的固定学习率本身可能根本不存在。损失的特点在训练过程中会发生变化。在早期，当参数远离最优值时，更大的步长可能有助于快速进展。然而，当训练接近最小值时，通常需要更小、更谨慎的步长来避免跳过。单一的 $\eta$ 无法有效地适应这两种情况。

这种“一刀切”的方法也忽略了针对参数进行特定调整的潜在优势。假设一个处理稀疏特征文本数据的模型。与常见特征相关的参数可能会收到频繁的、可能带有噪声的梯度更新。稀有特征的参数则接收不频繁但可能信息量很大的更新。对于频繁更新的参数使用较小的更新步长（以平均噪声并防止不稳定性）可能更有利，而对于稀疏更新的参数使用较大的更新步长（以确保它们能从有限的数据中学到有效信息）。全局的固定 $\eta$ 对所有参数一视同仁，无论它们的更新频率或梯度的大小如何，这会阻碍高效的学习。

最后，深度学习 (deep learning)中损失表面的复杂非凸性质（我们在第1章中讨论过）加剧了这些问题。固定学习率在应对鞍点（梯度接近零但并非最小值）和大面积高原（平坦区域）时会遇到很大困难。小的 $\eta$ 可能导致优化器以不切实际的缓慢速度爬过这些特征。大的 $\eta$ 可能有助于更快地摆脱高原，但增加了振荡或发散的风险，特别是在穿越狭窄的谷地或接近尖锐的最小值时。

选择和应用单一固定学习率的这些固有困难促使人们需要更精巧的方法。以下章节中讨论的AdaGrad、RMSprop、Adam及其相关算法通过在训练过程中动态调整学习率（通常是针对每个参数进行调整），直接解决了这些局限性。这种适应性通常能带来更快的收敛，适用于更广泛的问题和架构，且通常需要更少的手动调整。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  第4章和第8章全面解释了梯度下降优化，包括学习率的作用及其在深度学习中的实际挑战。
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma, Jimmy Ba (2015)
  Journal: International Conference for Learning Representations; DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  本文介绍了Adam，这是一种广泛采用的自适应学习率算法，通过动态调整各个参数的步长来解决固定学习率的许多局限性。
- [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) — John Duchi, Elad Hazan, and Yoram Singer (2011)
  Journal: Journal of Machine Learning Research (JMLR); Volume: 12; Pages: 2121-2159
  介绍了AdaGrad，这是一种早期且有影响力的自适应学习率算法，它根据每个参数的历史梯度平方和来调整学习率。
- [Lecture 6.5 - RMSprop: Divide the gradient by a running average of its recent magnitude](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) — Geoffrey Hinton, Nitish Srivastava, and Kevin Swersky (2012)
  Journal: Coursera Lecture Notes (University of Toronto); Publisher: University of Toronto
  介绍RMSprop的原始讲义，这是一种自适应学习率方法，旨在缓解AdaGrad学习率过度下降的问题。
