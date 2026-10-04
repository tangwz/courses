---
course: "advanced-diffusion-architectures"
chapter: "consistency-models"
lesson: "consistency-property-idea"
sourceId: 5449
sourceUrl: "https://apxml.com/zh/courses/advanced-diffusion-architectures/chapter-5-consistency-models/consistency-property-idea"
title: "核心思想：一致性"
description: "理解一致性：将轨迹上的点映射到轨迹的起点。"
order: 2
plots: []
sourceHash: "3f9f40db9a7c9603cf5907082932c068f2b8efad45eb4d53cfee71cca7a7bedd"
sourceCorrections: []
---

标准扩散采样器的迭代特性因计算成本高昂而带来实际难题。这种迭代过程通常源于扩散模型与常微分方程（ODE）之间的内在联系。一致性模型提供了一种独特的方法，以避免这种多步生成过程。这种方法的根本在于源于扩散底层连续时间视图的*一致性*。

### 概率流常微分方程

扩散模型可以从连续随机过程及其对应的常微分方程的视角来理解。具体来说，生成过程可以看作是逆转一个由常微分方程描述的前向过程，该常微分方程通常称为概率流（PF）常微分方程。设$x(t)$表示在时间$t$时的状态（例如，一张图像），时间$t$的范围从$t=0$（真实数据）到$t=T$（近似纯噪声）。PF ODE描述了将数据点$x(0)$转换为噪声$x(T)$以及逆向转换的路径或轨迹。标准扩散采样器本质上通过从$x(T)$向后进行许多小的离散步骤来估计$x(0)$，从而近似该常微分方程的解。

### 定义一致性

一致性模型的核心思想围绕着定义在这些常微分方程轨迹上的函数的一个特定属性。设想有一个函数，我们称之为$f$，它接收沿特定PF ODE轨迹在任意时间$t$的含噪声状态$x(t)$，并直接将其映射回该轨迹的*起点*，$x(0)$。

数学上，设$\{x(t)\}_{t \in [t_{\text{min}}, T]}$是从$x(0)$开始的PF ODE的解轨迹，$t_{\text{min}}$是一个接近零的小正值（作为积分的下限），且$T$是最大噪声时间。如果一个函数$f(x, t)$满足**一致性**，即对于沿此特定轨迹在区间$[t_{\text{min}}, T]$内的所有$t$：


$$
f(x(t), t) = x(0)
$$


该等式表明，无论你在轨迹上的哪个点（在任何时间$t > t_{\text{min}}$），一致性函数$f$总是生成该轨迹的初始数据点$x(0)$。

此外，这意味着一种自一致性形式：对于*相同*轨迹上的任意两个点$x(t_1)$和$x(t_2)$（其中$t_1, t_2 \in [t_{\text{min}}, T]$）：


$$
f(x(t_1), t_1) = f(x(t_2), t_2) = x(0)
$$


正是同一轨迹上点在时间上的这种自一致性特征赋予了这些模型其名称。

> 示意图说明了一致性。点$x(t_{min})$，$x(t_1)$，$x(t_2)$和$x(T)$都位于从$x(0)$开始的同一概率流ODE轨迹上。一致性函数$f(x, t)$将这些点中的每一个都映射回原始数据点$x(0)$。

### 对采样的影响

这种一致性函数$f$的存在对生成过程有深远影响。如果我们能够学习或准确近似$f(x, t)$，我们就可能*一步*完成生成。我们将从噪声分布$p_T(x)$中抽取一个样本$x_T$（通常是标准高斯分布），然后简单计算：


$$
\hat{x}_0 = f(x_T, T)
$$


这直接估计了对应于以噪声样本$x_T$结束的轨迹的数据点$x_0$，从而避免了DDPM、DDIM或其他ODE求解器中所需的迭代细化。

### 与基于分数的采样对比

这与标准扩散模型采样根本不同。传统方法训练一个模型（通常是U-Net或Transformer），用于预测得分函数$\nabla_x \log p_t(x)$或在时间$t$添加的噪声$\epsilon$。采样过程涉及在一个数值求解器中借助这些预测（例如用于SDE的Euler-Maruyama或用于ODE的DDIM/DPM-Solver），进行许多小的$\Delta t$时间步长反向操作，逐步将噪声$x_T$转换为样本$\hat{x}_0$。

对比之下，一致性模型旨在学习从时间$T$回溯到$t_{\text{min}} \approx 0$的整个积分过程的*结果*，该结果封装在函数$f$中。后续章节将讨论如何训练神经网络 (neural network)来近似这个一致性函数$f(x, t)$，从而实现极快的生成。

## 参考资料

- [Consistency Models](https://arxiv.org/abs/2303.01469) — Yang Song, Prafulla Dhariwal, Mark Chen, Ilya Sutskever (2023)
  Journal: International Conference on Machine Learning (ICML); DOI: [10.48550/arXiv.2303.01469](https://doi.org/10.48550/arXiv.2303.01469)
  介绍了一致性模型框架、一步生成的核心思想以及一致性属性本身。这是本节主题的基础论文。
- [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) — Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole (2021)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2011.13456](https://doi.org/10.48550/arXiv.2011.13456)
  建立了使用随机微分方程（SDEs）和概率流常微分方程（PF ODEs）的扩散模型的连续时间公式，为本节讨论的ODE轨迹提供了理论基础。
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Jonathan Ho, Ajay Jain, Pieter Abbeel (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 33; Pages: 6840-6851; DOI: [10.48550/arXiv.2006.11239](https://doi.org/10.48550/arXiv.2006.11239)
  一篇普及现代扩散模型的开创性论文，展示了一致性模型旨在避免的迭代采样过程。为文中提及的“标准扩散采样器”提供了背景。
- [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) — Tero Karras, Miika Aittala, Timo Aila, Samuli Laine (2022)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 35; Pages: 24933-24949; DOI: [10.48550/arXiv.2206.00364](https://doi.org/10.48550/arXiv.2206.00364)
  为各种扩散模型提供了一个统一的理论和实践框架，包括对ODE视角和实现快速推理的采样策略的见解。
