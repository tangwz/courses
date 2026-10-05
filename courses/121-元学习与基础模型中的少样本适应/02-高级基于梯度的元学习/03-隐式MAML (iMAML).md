# 隐式MAML (iMAML)

来源：[原文](https://apxml.com/zh/courses/meta-learning-foundation-models/chapter-2-advanced-gradient-based-meta-learning/implicit-maml-imaml)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管模型无关元学习 (MAML) 提供了一个有用的框架，可学习适应性初始化，但它对二阶导数（或通过整个内循环优化路径进行反向传播 (backpropagation)）的依赖带来了很大的计算和内存难题，特别是对于大型基础模型。计算甚至近似海森矩阵，或者存储大量内梯度步骤的计算图，很快就会变得难以承受。

隐式MAML (iMAML) 提供了一种替代方法，它巧妙地避免了这些困难，借助了隐式微分的能力。iMAML不是*通过*内循环优化器的步骤进行微分，而是*通过*内循环旨在满足的最优条件进行微分。

### 核心思想：微分最优条件

回顾一下，MAML 中内循环的目标是从元参数 (parameter) $\theta$ 开始，找到特定于任务的参数 $\theta'_i$。对于具有支持集损失 $L_{task_i}$ 的任务 $i$，这通常通过梯度下降 (gradient descent)来完成：


$$
\theta'_{i, k+1} = \theta'_{i, k} - \alpha \nabla_{\theta'} L_{task_i}(\theta'_{i, k})
$$


其中 $\theta'_{i, 0} = \theta$。经过 $K$ 步后，我们得到 $\theta'_i = \theta'_{i, K}$。MAML 通过展开这个过程并使用链式法则计算元梯度 $\nabla_{\theta} L_{meta}(\theta'_i)$，这涉及海森矩阵 $\nabla^2 L_{task_i}$。

iMAML 采用不同视角。它假设内循环优化收敛（或近似收敛）到某个满足最优条件的点 $\theta'_i$。这个条件的常见选择是调整后参数处的任务损失梯度为零（或接近零）：


$$
\nabla_{\theta'} L_{task_i}(\theta'_i) \approx 0
$$


这个方程隐式地将调整后的参数 $\theta'_i$ 定义为初始参数 $\theta$ 的函数。隐函数定理 (IFT) 提供了一种计算这个隐式定义函数的导数 $\frac{\partial \theta'_i}{\partial \theta}$ 的方法，无需直接对优化步骤进行微分。

### 应用隐函数定理

我们来定义一个基于内循环优化的函数 $G(\theta, \theta'_i)$。一种方法是使用内循环目标的最优条件。为简化起见，我们假设内循环从 $\theta$ 开始最小化 $L_{task_i}(\phi)$，从而得到 $\theta'_i$。最优条件是 $\nabla_{\phi} L_{task_i}(\phi) |_{\phi=\theta'_i} = 0$。我们可以将其看作一个方程 $G(\theta, \theta'_i) = \nabla_{\theta'} L_{task_i}(\theta'_i) = 0$，假设 $\theta'_i$ 是通过优化过程由 $\theta$ 隐式确定的。

或者，在实践中更常见的是，特别是在使用固定数量的梯度步骤时，我们可以根据梯度下降 (gradient descent)更新本身的定点方程定义 $G$。如果 $\theta'_i$ 是从 $\theta$ 开始的 $K$ 步 SGD（学习率为 $\alpha$）的结果，我们可以考虑单步的定点方程（或相关条件）。为了理解核心机制，我们仍使用最优条件 $\nabla_{\theta'} L_{task_i}(\theta'_i) = 0$。

元目标是最小化查询集上的损失 $L_{meta}(\theta'_i)$，并在任务上平均。元梯度包含项 $\nabla_{\theta} L_{meta}(\theta'_i)$。使用链式法则：


$$
\nabla_{\theta} L_{meta}(\theta'_i) = \left( \frac{\partial \theta'_i}{\partial \theta} \right)^T \nabla_{\theta'} L_{meta}(\theta'_i)
$$


挑战在于计算雅可比矩阵 $\frac{\partial \theta'_i}{\partial \theta}$。在 $G(\theta, \theta'_i) = \nabla_{\theta'} L_{task_i}(\theta'_i) = 0$ 上使用 IFT，我们有：


$$
\frac{\partial G}{\partial \theta} + \frac{\partial G}{\partial \theta'_i} \frac{\partial \theta'_i}{\partial \theta} = 0
$$


重新排列得到：


$$
\frac{\partial \theta'_i}{\partial \theta} = - \left( \frac{\partial G}{\partial \theta'_i} \right)^{-1} \frac{\partial G}{\partial \theta}
$$


代入 $G(\theta, \theta'_i) = \nabla_{\theta'} L_{task_i}(\theta'_i)$，我们得到：


$$
\frac{\partial G}{\partial \theta'_i} = \nabla^2_{\theta'} L_{task_i}(\theta'_i) \quad \text{（海森矩阵！）}
$$


$$
\frac{\partial G}{\partial \theta} = 0 \quad \text{（如果 } \theta'_i \text{ 仅通过初始化依赖于 } \theta \text{。需要仔细处理。）}
$$


这种使用精确最优条件的特定表述对于典型的基于梯度下降的内循环并不完全正确，因为 $\theta'_i$ *确实*依赖于 $\theta$。更实际的表述考虑了更新规则的定点，或直接将 IFT 应用于更新序列。

我们来考虑一个在实践中更常用的直接应用。我们想计算向量 (vector)-雅可比积 $v^T \frac{\partial \theta'_i}{\partial \theta}$，其中 $v = \nabla_{\theta'} L_{meta}(\theta'_i)$。iMAML 在不显式形成雅可比或海森矩阵的情况下找到这个积。它使用的事实是，这个积通常可以通过求解涉及海森矩阵 $\nabla^2_{\theta'} L_{task_i}(\theta'_i)$ 的线性系统来找到。设 $H = \nabla^2_{\theta'} L_{task_i}(\theta'_i)$。所需项可以通过求解 $H z = v$ 形式的方程来近似或计算 $z$，并且元梯度与 $z$ 有关。

主要观察点是，我们不需要完整的海森矩阵 $H$。我们只需要计算海森-向量积 ($H v$)，这可以使用有限差分或自动微分有效完成（类似于计算 Pearlmutter 的 $R\{.\}$ 运算符），而无需实例化完整的海森矩阵。这个海森-向量积正是共轭梯度 (CG) 算法等迭代方法求解线性系统 $H z = v$ 所需的。

### iMAML 算法概述

1. **对于每个元任务批次：**
   - 对于每个任务 $i$：
     - 初始化任务参数 (parameter)：$\theta'_{i, 0} = \theta$。
     - **内循环：** 执行 $K$ 步梯度下降 (gradient descent)以计算支持集损失 $L_{task_i}$，以获得调整后的参数 $\theta'_i = \theta'_{i, K}$。
     - 计算查询集梯度：$v_i = \nabla_{\theta'} L_{meta}(\theta'_i)$。
     - **隐式梯度计算：** 使用共轭梯度（近似地）求解线性系统，以找到与 $v_i$ 相关的隐式元梯度贡献。这涉及计算海森-向量 (vector)积，其中 $H_i = \nabla^2_{\theta'} L_{task_i}(\theta'_i)$，但不是 $H_i$ 本身。所求解的精确系统取决于具体的 iMAML 变体和推导（例如，与 $H_i z = v_i$ 或 $(I + \alpha H_i) z = v_i$ 有关）。设结果为 $g_{implicit, i}$。
     - 存储 $g_{implicit, i}$。
   - **元更新：** 汇总隐式梯度并更新元参数 $\theta$：
     
     $$
     \theta \leftarrow \theta - \beta \frac{1}{N} \sum_{i=1}^N g_{implicit, i}
     $$
     
     （其中 $\beta$ 是元学习率）。

> MAML（通过展开的优化步骤进行显式反向传播 (backpropagation)）与 iMAML（通过求解与内循环最优值相关的线性系统进行隐式微分）的梯度计算路径比较。

### 优点与权衡

**优点：**

- **内存效率：** 这是主要优点。iMAML 避免存储内循环优化的计算图，使其内存占用基本不依赖于内循环步骤的数量 $K$。这对于调整基础模型非常有益，因为即使是单一步骤的计算图也可能很大。
- **计算成本：** 虽然使用 CG 求解线性系统会增加计算量，但它可能比计算完整的二阶 MAML 梯度明显更快，特别是对于大的 $K$。它避免显式形成或存储海森矩阵。
- **潜在的稳定性：** 通过关注定点或最优值，iMAML 可能避免与通过内循环中潜在不稳定的优化动态进行微分相关的状况，尤其是在步骤很多的情况下。

**缺点：**

- **近似质量：** iMAML 的准确性取决于定点假设的有效性和线性系统求解器的精度（例如，CG 迭代次数）。如果内循环收敛不佳或 CG 提前终止，则得到的梯度可能不准确。
- **求解器复杂性：** 实现和调整迭代求解器（如 CG）与标准自动微分相比增加了复杂性。确保 CG 的收敛有时可能需要仔细的预处理或参数 (parameter)调整。
- **海森-向量 (vector)积成本：** 尽管比计算完整的海森矩阵便宜，但计算海森-向量积仍需谨慎并带来计算成本（大致相当于两次反向传播 (backpropagation)）。

### 在基础模型中的情境

iMAML 提供的显著内存节省使其成为使用基础模型进行元学习的有吸引力的选择。标准 MAML 常常由于需要通过具有数十亿参数 (parameter)模型的内部更新进行反向传播 (backpropagation)所需的内存而变得不可行。尽管像 FOMAML 这样的零阶方法也节省内存，但 iMAML 尝试隐式保留一些二阶信息，可能带来更好的适应性能。然而，海森-向量 (vector)积的计算成本和 CG 求解器的复杂性在扩展到最大模型时仍是实际考虑因素。将 iMAML 与其他技术（如混合精度训练或模型并行化）结合，对于在超大规模环境中的实际运用可能是必要的。

## 参考资料

- [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://proceedings.mlr.press/v70/finn17a.html) — Chelsea Finn, Pieter Abbeel, Sergey Levine (2017)
  Journal: Proceedings of the 34th International Conference on Machine Learning; Publisher: PMLR; Volume: 70; Pages: 1126-1135; DOI: [10.48550/arXiv.1703.03400](https://doi.org/10.48550/arXiv.1703.03400)
  介绍了模型无关元学习 (MAML) 的基础论文，iMAML 旨在改进其方法。
- [Meta-Learning with Implicit Gradients](https://openreview.net/forum?id=rkgO_hA5tX) — Aravind Rajeswaran, Chelsea Finn, Sham Kakade, Sergey Levine (2019)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Publisher: Neural Information Processing Systems Foundation, Inc. (NeurIPS); Volume: 32; Pages: 13149-13161; DOI: [10.48550/arXiv.1909.04630](https://doi.org/10.48550/arXiv.1909.04630)
  引入了隐式元学习 (iMAML)，作为 MAML 的高效替代方案，利用隐式微分避免了显式 Hessian 计算。
- [Optimizing Millions of Variables by Implicit Differentiation](https://proceedings.mlr.press/v119/donti20a.html) — Prashant Donti, Brandon Amos, J. Zico Kolter (2020)
  Journal: International Conference on Machine Learning (ICML); Publisher: Proceedings of Machine Learning Research; Volume: 119; Pages: 2696-2706; DOI: [10.48550/arXiv.2002.06206](https://doi.org/10.48550/arXiv.2002.06206)
  提供了一个通过隐式微分优化高维问题的扩展框架，与 iMAML 中使用的计算技术高度相关。
- [A Survey of Meta-Learning](https://arxiv.org/abs/1810.03548) — Joaquin Vanschoren (2018)
  Journal: arXiv preprint arXiv:1810.03548; DOI: [10.48550/arXiv.1810.03548](https://doi.org/10.48550/arXiv.1810.03548)
  一篇全面的综述，提供了元学习技术（包括基于梯度的方法及其发展）的概览。

---

[上一节](02-%E4%B8%80%E9%98%B6MAML%20%28FOMAML%29%20%E4%B8%8E%20Reptile.md) · [下一节](04-%E5%A4%84%E7%90%86%E7%A8%B3%E5%AE%9A%E6%80%A7%E5%92%8C%E6%A2%AF%E5%BA%A6%E6%96%B9%E5%B7%AE.md)
