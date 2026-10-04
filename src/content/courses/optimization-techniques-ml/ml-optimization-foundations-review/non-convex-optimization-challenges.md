---
course: "optimization-techniques-ml"
chapter: "ml-optimization-foundations-review"
lesson: "non-convex-optimization-challenges"
sourceId: 1235
sourceUrl: "https://apxml.com/zh/courses/optimization-techniques-ml/chapter-1-ml-optimization-foundations-review/non-convex-optimization-challenges"
title: "非凸优化中的挑战"
description: "识别非凸环境中的局部最小值、鞍点和平坦区等问题。"
order: 5
plots: ["plots/1235-0.json"]
sourceHash: "8fd6cee6f7a7a2620bba11043061c4719854761a590dd3f303b85cf8c88cec43"
sourceCorrections: []
---

凸优化问题提供找到单一全局最小值的确定性。然而，大多数现代机器学习 (machine learning)任务，尤其是深度学习 (deep learning)的实际情况，却是非凸的。当条件 $f(\lambda x + (1-\lambda)y) \le \lambda f(x) + (1-\lambda)f(y)$ 并非全局成立时，优化过程会变得显著复杂且难以应对。在考察旨在缓解这些问题的算法之前，理解这些难点是首要条件。

非凸性带来了若干难题，标准梯度下降 (gradient descent)法难以应对：

### 局部最小值

在非凸空间中，可能存在多个梯度为零的点，即 $\nabla f(x) = 0$，这些点在某个邻域内是最小值，但它们不一定是函数值最低的整体最小值点（全局最小值）。梯度下降 (gradient descent)等一阶方法沿着最陡峭的下降方向移动，一旦到达任何梯度为零的点就会停止。如果此点是局部最小值，算法可能会收敛到次优解，而不会有任何迹象表明参数 (parameter)空间的其他位置存在更好的解。因此，使用梯度下降训练模型的最终表现可能严重依赖于初始化点，因为不同的起始点可能使优化器进入对应不同局部最小值的不同吸引盆。

### 鞍点

在高维非凸问题中，鞍点可能比局部最小值更为普遍。鞍点也是一个梯度为零的临界点，即 $\nabla f(x) = 0$，但它不是局部极值点。相反，函数在某些方向上增加，在另一些方向上减少，形似马鞍。



![非凸损失曲面示例](plots/1235-0.json)



> 这是一个非凸曲面的简化示意图，其中包含一个梯度为零但并非最小值的鞍点（红色X）和多个局部最小值（绿色点）。简单的梯度下降 (gradient descent)法在这些点附近可能会减速或停滞。

尽管使用海森矩阵的二阶方法可以通过检查特征值来区分最小值、最大值和鞍点，但一阶方法缺乏这种信息。在鞍点附近，梯度在所有方向上都变得非常小，导致SGD等方法急剧减速，有时甚至看起来已经收敛。有效地摆脱鞍点是一个重大的难题，尤其是在问题维度增加时（这在深度神经网络 (neural network)中很常见），理论研究表明鞍点出现的频率会呈指数级增长，远超局部最小值。

### 平坦区

平坦区是损失曲面上较大且相对平坦的区域，梯度始终接近零，但函数值本身可能仍然很高。当优化器进入平坦区时，由于梯度很小导致更新量微乎其微，无论学习率如何，进展都会变得非常缓慢。这可能导致训练时间过长，难以接受。平坦区常与深度网络中的梯度消失等现象相关联。

### 高维度与复杂几何形态

大型机器学习 (machine learning)模型的损失曲面存在于极高维度空间（数百万或数十亿参数 (parameter)）中。我们基于2D或3D可视化的直觉在此处常常失效。在低维空间中罕见的特性（如具有多个负曲率方向的鞍点）在高维空间 (high-dimensional space)中可能变得普遍。几何形态可能极其复杂，包含尖锐的山谷、平坦的区域以及众多次优临界点。有效地应对这种情况需要优化算法，使其不易被这些特性永久困住或显著减缓速度。

解决这些非凸优化挑战是后续章节中讨论的高级技术发展的主要动力，例如自适应学习率（第三章）、二阶近似（第二章）和深度学习 (deep learning)专用方法（第六章）。尽管对于复杂模型来说，找到可证明的全局最小值通常是不可行的，但目标变为找到足够好的局部最小值或有效地处理鞍点，以便在合理的时间内实现良好的模型性能。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 8
  经典教材，其中一章专门解释深度学习模型中的优化挑战。
- [Identifying and Attacking the Saddle Point Problem in High-Dimensional Non-Convex Optimization](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHpjPStHvNFeICSIT2ZxwBvS-uTq64GhTi8gPqfYFeNjtLdmRfm8MeiA_P4WqmbC3ydwADUPnQNAJEUdFJxnOrF0B919G6vNMwCS083lUHc7FBtN2HNgjiKbXllXXYKzVwAEgoquJef7826bVP-o7W6QX0-gYekjC2XX4ltuCG0D-fFhupiDnEiFruKOyygEyf-Lz_Q8yw1ovsAKyLZ0QnGaSgo2VmIuj0qthP4Dqm3usD-lTlLND_EgWqRh) — Yann N. Dauphin, Razvan Pascanu, Caglar Gulcehre, Kyunghyun Cho, Surya Ganguli, and Yoshua Bengio (2014)
  Journal: Advances in Neural Information Processing Systems; Publisher: Advances in Neural Information Processing Systems; Volume: 27; Pages: 2933-2941
  基础性论文，讨论了高维非凸优化中鞍点比局部最小值更普遍的现象。
- [How to Escape Saddle Points Efficiently](http://proceedings.mlr.press/v70/jin17a/jin17a.pdf) — Chi Jin, Rong Ge, Praneeth Netrapalli, Sham M. Kakade, and Michael I. Jordan (2017)
  Journal: International Conference on Machine Learning (ICML); Publisher: Proceedings of Machine Learning Research (PMLR); Volume: 70; Pages: 1724-1732; DOI: [10.5555/3305890.3306000](https://doi.org/10.5555/3305890.3306000)
  本文提出了在非凸优化中有效逃离鞍点的算法和理论保证。
- [Optimization Algorithms for Deep Learning](https://arxiv.org/pdf/1904.12260) — Xiangxiang Zhang, Anna Choromanska, and Yann LeCun (2019)
  Journal: arXiv preprint arXiv:1904.12260
  对深度学习中的优化技术及相关挑战进行了全面回顾。
