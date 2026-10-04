---
course: "meta-learning-foundation-models"
chapter: "advanced-gradient-based-meta-learning"
lesson: "maml-stability-variance"
sourceId: 4314
sourceUrl: "https://apxml.com/zh/courses/meta-learning-foundation-models/chapter-2-advanced-gradient-based-meta-learning/maml-stability-variance"
title: "处理稳定性和梯度方差"
description: "学习率调度、梯度裁剪和元优化器选择等方法。"
order: 4
plots: ["plots/4314-0.json"]
sourceHash: "724d85e297499a27f4452e33d3eea17acb907c0e724c3ec2134a7ebdb1abfed4"
sourceCorrections: []
---

基于梯度的元学习算法，如 MAML，通过嵌套循环结构优化参数 (parameter) $\theta$。内循环使用支持集 $D_i^{supp}$ 将参数适应特定任务 $\mathcal{T}_i$，而外循环则根据跨多个任务查询集 $D_i^{qry}$ 的表现来更新元参数 $\theta$。这种双层优化过程，特别是在标准 MAML 中涉及高阶导数时，会带来与优化稳定性和元梯度估计方差相关的重大挑战。有效训练这些模型，特别是大型基础模型，需要仔细关注这些方面。

### 优化不稳定

元训练期间的不稳定可能表现为梯度爆炸或梯度消失，导致发散或进展极其缓慢。有几个因素促成此情况：

1. **急剧曲率**：内部任务特定优化和外部元优化的损失曲面都可能展现急剧曲率。在适应大型复杂模型时尤其如此。高曲率可能导致梯度步长越过最优解或剧烈震荡。
2. **内循环与外循环的关联**：内循环更新直接影响元梯度的计算。如果内循环优化不稳定（例如，需要许多步骤或大学习率 $\alpha$），它会放大外循环更新中的噪声和不稳定。
3. **二阶导数 (MAML)**：标准 MAML 需要对内循环优化过程进行微分，涉及 Hessian 矩阵或 Hessian 向量 (vector)积。计算和采用这些二阶项可能计算量大且数值不稳定，尤其是在高维度时。小的误差或近似可能会累积，导致元参数 (parameter)更新不佳。
4. **学习率敏感性**：内循环学习率 $\alpha$ 和外循环学习率 $\beta$ 的选择影响很大。不合适的 $\alpha$ 可能导致任务适应不良，影响元梯度质量。不合适的 $\beta$ 可能导致元优化本身发散。

### 梯度方差高

元梯度通常使用从元训练分布 $p(\mathcal{T})$ 中采样的一小批任务来估计。梯度估计 $\nabla_\theta L_{meta}(\theta)$ 通常存在高方差，这归因于以下几个因素：

1. **任务采样**：任务固有的多样性意味着任何有限的任务批次都只能提供对 $p(\mathcal{T})$ 上真实期望梯度的嘈杂估计。
2. **有限支持数据**：少样本任务，顾名思义，支持集 $D_i^{supp}$ 较小。基于这些有限数据进行的内循环优化得到的任务特定参数 (parameter) $\phi_i$ 本身就是任务 $\mathcal{T}_i$ 最优参数的一些嘈杂估计。这种噪声会传播到基于查询集 $D_i^{qry}$ 的元梯度计算中。
3. **优化中的随机性**：如果内循环使用随机梯度下降 (gradient descent) (SGD) 或涉及其他随机性来源，这会进一步增加适应参数 $\phi_i$ 的方差，进而影响元梯度。

元梯度中的高方差会减缓外循环优化的收敛速度，需要更小的学习率 $\beta$ 或更多的元迭代。它还可能使元训练过程可靠性较低，可能收敛到次优元参数。

### 缓解方法

处理稳定性和方差对成功进行梯度元学习很重要。可以采用几种方法：

#### 1. 仔细管理学习率

- **区分内外循环学习率**：对内循环使用不同的学习率 $\alpha$，对外循环使用 $\beta$。$\alpha$ 控制任务适应的速度，而 $\beta$ 控制元学习的速度。通常需要独立调整这些。
- **学习率调度**：应用学习率衰减计划，特别是对外循环学习率 $\beta$。一种常用策略是开始时使用较大的 $\beta$，并随着训练的进行逐渐减小它。
- **自适应优化器 (外循环)**：对外循环更新（更新 $\theta$）使用 Adam 或 RMSProp 等自适应优化器可能有益。它们对每个参数 (parameter)的自适应学习率有助于处理复杂的元损失曲面，并减轻跨不同参数组的梯度缩放相关问题。但是，仍需要仔细调整超参数 (hyperparameter)，如 $\beta_1$、$\beta_2$ 和 $\epsilon$。在*内*循环中使用 Adam 较不常见，有时会干扰元优化目标，但并非完全排除。

#### 2. 梯度裁剪

独立地对内循环梯度（在任务适应期间）和外循环元梯度（在元更新期间）应用梯度裁剪可以防止梯度爆炸。

- **内循环裁剪**：在内循环更新步骤中，裁剪在 $D_i^{supp}$ 上相对于 $\theta$（或 $\phi_i$）计算的梯度。
- **外循环裁剪**：在对 $\theta$ 应用更新之前，裁剪在任务批次中计算的最终元梯度。

选择合适的裁剪阈值通常需要实验。通常优于值裁剪的是范数裁剪（例如，`clip_grad_norm_`）。

#### 3. 元优化器选择

虽然带动量的 SGD 或 Adam 是外循环的常见选择，但研究人员正在审视专门为双层问题或元学习设计的优化器。然而，经过良好调整的标准优化器在许多情况下仍然普遍且有效。Adam 的动量和自适应缩放带来的稳定性优势通常使其成为外循环的强有力基线。

#### 4. 增大任务批次大小

元梯度估计的方差通常随着元批次中任务数量 ($B$) 的增加而降低。这是大数定律的直接结果。采样更多的任务可以提供对任务分布上期望梯度的更准确估计。



![元梯度方差 vs. 每元批次任务数量](plots/4314-0.json)



> 该图说明了每次元更新采样的任务数量 ($B$) 增加如何降低元梯度估计的方差。

主要缺点是每次元迭代的计算成本增加，因为必须对批次中的每个任务执行适应和梯度计算。在方差减少和计算可行性之间找到合适的平衡很重要。

#### 5. 采用一阶或隐式方法

- **FOMAML/Reptile**：如前所述，FOMAML 和 Reptile 等一阶近似完全避免了二阶导数的计算。这显著提高了计算效率，并且通常通过移除不稳定的 Hessian 项来增强稳定性，尽管可能改变优化目标。
- **隐式 MAML (iMAML)**：iMAML 使用隐式微分计算元梯度，避免了展开内部优化路径的需要。这可以使内存需求相对于内部步骤数量保持恒定，并可能提供更好的稳定性保证，特别是在需要许多内部步骤时。

#### 6. 初始化与正则化 (regularization)

- **元参数初始化**：好的初始化 $\theta_0$ 可以将元参数置于优化从一开始就更稳定的区域。学习初始化的方法在第4章中进一步介绍。
- **正则化**：应用于元参数 $\theta$ 的标准正则化方法（例如，L2 权重 (weight)衰减）有时有助于平滑元损失并提升稳定性，尽管它们与双层目标的关联需要仔细考量。

#### 基础模型的考量

将这些方法应用于大型基础模型时，挑战常常被放大：

- **更高维度**：庞大的参数数量增加了遇到困难优化动态的可能性。
- **计算成本**：计算梯度，特别是二阶梯度，变得极其昂贵。这使得 FOMAML、Reptile 或 iMAML 更具吸引力。
- **内存限制**：存储激活和梯度以进行元梯度计算是内存密集型的。第6章（元学习实现扩展）中讨论的方法变得不可或缺。

因此，将基于梯度的元学习与 FOMAML 等方法、仔细的学习率控制、梯度裁剪、用于外循环的自适应优化器，以及可能增大的任务批次大小（在计算限制内）结合起来，形成了在适应大型模型时稳定训练和管理梯度方差的常用策略。

## 参考资料

- [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400) — Chelsea Finn, Pieter Abbeel, and Sergey Levine (2017)
  Journal: Proceedings of the 34th International Conference on Machine Learning (ICML); DOI: [10.48550/arXiv.1703.03400](https://doi.org/10.48550/arXiv.1703.03400)
  介绍了MAML算法，将元学习构建为双层优化问题，并讨论了其梯度计算。
- [On First-Order Meta-Learning Algorithms](https://arxiv.org/abs/1803.02999) — Alex Nichol, Joshua Achiam, John Schulman (2018)
  Journal: arXiv preprint arXiv:1803.02999; DOI: [10.48550/arXiv.1803.02999](https://doi.org/10.48550/arXiv.1803.02999)
  介绍了Reptile，一种简单的一阶元学习算法，无需二阶导数即可获得有竞争力的性能，提高了效率和稳定性。
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma and Jimmy Ba (2014)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍了Adam，一种自适应优化算法，因其在训练深度学习模型中的效率和有效性而被广泛使用，适用于外循环元优化。
