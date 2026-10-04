---
course: "bayesian-machine-learning"
chapter: "advanced-probabilistic-graphical-models"
lesson: "bayesian-network-parameter-learning"
sourceId: 3612
sourceUrl: "https://apxml.com/zh/courses/bayesian-machine-learning/chapter-5-advanced-probabilistic-graphical-models/bayesian-network-parameter-learning"
title: "贝叶斯网络中的参数学习"
description: "学习贝叶斯方法以估计贝叶斯网络中的条件概率分布。"
order: 2
plots: ["plots/3612-0.json"]
sourceHash: "325ada5723637fa45f23530c1e14242af83598dc71c1a541c2c9ed0acd0f2377"
sourceCorrections: []
---

一旦贝叶斯网络（BN）的结构 $G$ 确定，无论是通过领域知识还是结构学习算法，下一个重要的步骤是量化 (quantization)连接节点间的概率关系。这涉及学习模型的参数 (parameter)，这些参数通常定义图中每个变量给定其父节点时的条件概率分布（CPD）。与仅仅为这些参数找到一个单一的最佳估计（如最大似然估计）不同，贝叶斯方法将参数本身视为随机变量，旨在计算其在观测数据下的后验分布。

### 参数 (parameter)作为随机变量

在贝叶斯网络中，参数（统称为 $\theta$）表示定义模型的条件概率。对于具有变量 $X_1, \dots, X_n$ 的网络，参数 $\theta$ 包含每个变量 $X_i$ 的集合 $\theta_{i | pa(X_i)}$，它们指定了 $X_i$ 在其父节点 $pa(X_i)$ 的每种配置下的概率分布 $P(X_i | pa(X_i), \theta_{i | pa(X_i)})$。

贝叶斯参数学习的核心思想是，首先对这些参数设定一个 *先验分布* $P(\theta | G)$，这反映了我们在观测到任何数据 $D$ 之前的信念。然后，我们**使用**贝叶斯定理，根据数据更新这些信念，从而得到一个 *后验分布* $P(\theta | D, G)$:


$$
P(\theta | D, G) = \frac{P(D | \theta, G) P(\theta | G)}{P(D | G)}
$$


这里，$P(D | \theta, G)$ 是给定一组特定参数 $\theta$ 和图结构 $G$ 时观测到数据 $D$ 的似然。项 $P(D | G)$ 是边际似然或证据，通常难以直接计算，但它作为一个归一化 (normalization)常数。

贝叶斯网络的一个基本属性大大简化了这一过程。给定一个完整数据集（无缺失值），似然函数根据图结构进行分解：


$$
P(D | \theta, G) = \prod_{i=1}^{n} \prod_{j=1}^{q_i} P(\theta_{ij} | D)
$$


其中 $\theta_{ij}$ 表示变量 $X_i$ 在其父节点 $pa(X_i)$ 处于第 $j$ 种配置（共 $q_i$ 种可能配置）时的参数。此外，如果我们假设先验中存在 *参数独立性*，即 $P(\theta|G) = \prod_i \prod_j P(\theta_{ij})$，那么后验也会分解：


$$
P(\theta | D, G) = \prod_{i=1}^{n} \prod_{j=1}^{q_i} P(\theta_{ij} | D)
$$


这种分解表明我们可以独立地学习每个局部 CPD 的参数，将一个可能巨大的学习问题分解为更小、易于处理的部分。

### 离散变量的参数 (parameter)学习

让我们考虑贝叶斯网络中所有变量都是离散变量的常见情况。每个变量 $X_i$ 可以取 $r_i$ 种状态中的一种。参数 $\theta_{ijk}$ 表示变量 $X_i$ 在其父节点 $pa(X_i)$ 处于第 $j$ 种配置时，取其第 $k$ 种状态的概率：


$$
\theta_{ijk} = P(X_i = k | pa(X_i) = j, \theta_{ij})
$$


其中，对于所有 $i, j$，$\sum_{k=1}^{r_i} \theta_{ijk} = 1$。

给定数据 $D$，特定 CPD $P(X_i | pa(X_i)=j)$ 的似然贡献遵循基于计数 $N_{ijk}$ 的多项分布 —— $N_{ijk}$ 是数据集中 $X_i=k$ 且 $pa(X_i)=j$ 的次数。

对于参数 $\theta_{ij} = (\theta_{ij1}, \dots, \theta_{ij r_i})$ 的先验分布，一个便捷且广泛使用的选择是 **狄利克雷分布**。狄利克雷分布 $Dir(\alpha_{ij1}, \dots, \alpha_{ij r_i})$ 定义在概率单纯形上（分量非负且和为1）。其概率密度函数为：


$$
P(\theta_{ij} | \alpha_{ij}) = \frac{\Gamma(\sum_{k=1}^{r_i} \alpha_{ijk})}{\prod_{k=1}^{r_i} \Gamma(\alpha_{ijk})} \prod_{k=1}^{r_i} \theta_{ijk}^{\alpha_{ijk} - 1}
$$


超参数 (hyperparameter) $\alpha_{ijk} > 0$ 可以被理解为 *伪计数* 或 *先验计数*，反映了在给定父节点配置 $j$ 时，对 $X_i$ 每个状态 $k$ 出现频率的先验信念。和 $\alpha_{ij} = \sum_k \alpha_{ijk}$ 表示此先验信念的强度（等效样本量）。

狄利克雷先验与多项式似然是 *共轭* 的。这意味着如果先验是狄利克雷分布，似然是多项式分布，那么后验分布也将是狄利克雷分布。具体来说，$\theta_{ij}$ 的后验为：


$$
P(\theta_{ij} | D, G) = Dir(\alpha_{ij1} + N_{ij1}, \dots, \alpha_{ij r_i} + N_{ij r_i})
$$


这个优雅的结果意味着后验超参数仅仅是先验超参数加上数据中观测到的相应计数。参数学习归结为统计数据中的出现次数并将其添加到先验伪计数中。

### 参数 (parameter)估计：后验分布的使用

尽管贝叶斯方法可以得出完整的后验分布 $P(\theta | D, G)$，但通常我们需要参数的单点估计，例如，为了填充 CPT 以进行推断。常见的选择包括：

1. **后验均值（贝叶斯估计）**：这是参数在后验分布下的期望值。对于狄利克雷后验 $Dir(\alpha'_{ij1}, \dots, \alpha'_{ij r_i})$，**此时** $\alpha'_{ijk} = \alpha_{ijk} + N_{ijk}$，后验均值为：

   
   $$
   \mathbb{E}[\theta_{ijk} | D, G] = \frac{\alpha'_{ijk}}{\sum_{l=1}^{r_i} \alpha'_{ijl}} = \frac{\alpha_{ijk} + N_{ijk}}{\sum_{l=1}^{r_i} (\alpha_{ijl} + N_{ijl})}
   $$
   

   请注意，此估计如何使用先验伪计数 $\alpha_{ijk}$ 来平滑经验频率 $N_{ijk} / \sum_l N_{ijl}$（即最大似然估计，MLE）。这有助于避免数据中未出现事件的零概率，这在较小的数据集中尤为重要。一个常见的无信息先验选择是拉普拉斯平滑先验，**即**对于所有 $i, j, k$，$\alpha_{ijk} = 1$。
2. **最大后验（MAP）估计**：这会找到使后验概率密度最大的参数值。对于狄利克雷后验，MAP 估计为：

   
   $$
   \theta_{ijk}^{MAP} = \frac{\alpha'_{ijk} - 1}{\sum_{l=1}^{r_i} (\alpha'_{ijl} - 1)} = \frac{\alpha_{ijk} + N_{ijk} - 1}{\sum_{l=1}^{r_i} (\alpha_{ijl} + N_{ijl} - 1)}
   $$
   

   前提是所有 $\alpha'_{ijk} > 1$。如果任何 $\alpha'_{ijk} \le 1$，则众数位于单纯形的边界。

后验均值和 MAP 之间的选择取决于应用，但后验均值因其平均特性以及与最小化平方误差损失的关联而常被优先选用。

### 示例：先验到后验的更新

考虑一个简单的参数 (parameter) $\theta$，它表示硬币正面朝上的概率。假设一个 Beta 先验（它是 K=2 状态的狄利克雷分布：正面、反面），例如 $Beta(\alpha_H=2, \alpha_T=2)$。这反映了以 0.5 为中心但存在一定不确定性的先验信念。

假设我们抛硬币 10 次，观测到 $N_H=7$ 次正面和 $N_T=3$ 次反面。后验分布变为：


$$
P(\theta | D) = Beta(\alpha_H + N_H, \alpha_T + N_T) = Beta(2 + 7, 2 + 3) = Beta(9, 5)
$$


$\theta$ 的后验均值估计（正面朝上概率）为：


$$
\mathbb{E}[\theta | D] = \frac{9}{9 + 5} = \frac{9}{14} \approx 0.643
$$


MLE 将是 $7/10 = 0.7$。贝叶斯估计略微偏向先验均值 (0.5)。

我们可以将此更新可视化：



![硬币翻转概率（Theta）的贝叶斯更新](plots/3612-0.json)



> 该图展示了在观测到数据后，概率密度如何移动并变得更窄（更确定）。先验（蓝色）以 0.5 为中心，而后验（粉色）在 0.64 附近达到峰值，反映了观测到的数据（7 次正面，3 次反面）。

这一原理直接扩展到贝叶斯网络 CPT 中的多项式参数。CPT 中对应特定父节点配置的每一行都使用其相关的先验计数和观测数据计数独立更新。

### 挑战与扩展

尽管贝叶斯参数 (parameter)学习对于完整数据和共轭先验来说非常优雅，但它也面临挑战：

- **缺失数据/隐变量**：简单的计数方法会失效。需要迭代方法，如期望最大化（EM）或采样技术（如 MCMC 章中介绍的 Gibbs 采样），以处理缺失值或隐变量带来的不确定性。
- **非共轭先验**：如果所选先验与似然不共轭，则后验分布可能没有闭合形式的解，这再次需要近似推断方法（MCMC，VI）。
- **连续变量**：对于连续变量，会使用不同的似然（例如高斯）和相应的共轭先验（例如高斯参数的 Normal-Wishart），它们涉及相似的原理但数学形式不同。

总之，贝叶斯参数学习提供了一种规范的方法，通过结合先验知识和观测数据来估计贝叶斯网络的参数。狄利克雷分布等共轭先验的使用，大大简化了离散网络在完整数据下的计算，使得更新高效，并提供了捕获参数不确定性的完整后验分布。这种不确定性是优于 MLE 等点估计的一个明显优势，并且可以传播到后续的推断任务中。

## 参考资料

- [Probabilistic Graphical Models: Principles and Techniques](https://mitpress.mit.edu/books/probabilistic-graphical-models) — Daphne Koller and Nir Friedman (2009)
  Publisher: The MIT Press; DOI: [10.7551/mitpress/8381.001.0001](https://doi.org/10.7551/mitpress/8381.001.0001)
  这本教科书全面介绍了贝叶斯网络，其中包括一章专门讨论使用狄利克雷先验的贝叶斯参数学习。
- [Pattern Recognition and Machine Learning](https://www.springer.com/book/9780387310732) — Christopher M. Bishop (2006)
  Publisher: Springer; DOI: [10.1007/bpa2480](https://doi.org/10.1007/bpa2480)
  这本书对贝叶斯推断、共轭先验（如狄利克雷分布）及其在概率模型参数估计中的应用进行了基础阐述。
- [Bayesian Networks and Decision Graphs](https://www.springer.com/book/9780387682822) — Finn V. Jensen and Thomas D. Nielsen (2007)
  Publisher: Springer; DOI: [10.1007/978-0-387-68282-2](https://doi.org/10.1007/978-0-387-68282-2)
  本书介绍了贝叶斯网络的参数学习方法，重点在于离散变量和狄利克雷先验相关的实际和理论基础。
- [CS228: Probabilistic Graphical Models, Lecture Materials](https://cs228.github.io/) — Daphne Koller and Stanford CS228 Course Staff (2024)
  Publisher: Stanford University
  官方课程材料涵盖了贝叶斯网络的贝叶斯参数学习，包括最大似然估计、狄利克雷先验和后验推断的讨论。
