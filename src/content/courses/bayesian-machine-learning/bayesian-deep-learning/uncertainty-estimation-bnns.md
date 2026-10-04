---
course: "bayesian-machine-learning"
chapter: "bayesian-deep-learning"
lesson: "uncertainty-estimation-bnns"
sourceId: 3628
sourceUrl: "https://apxml.com/zh/courses/bayesian-machine-learning/chapter-6-bayesian-deep-learning/uncertainty-estimation-bnns"
title: "BNN中的不确定性估计"
description: "使用BNN量化和解释偶然不确定性和认知不确定性。"
order: 6
plots: ["plots/3628-0.json"]
sourceHash: "53c1f6a32555baebde83406b74e129e4e1f3b9ccdc6c6fc6261cb4eceb7ec88a"
sourceCorrections: []
---

传统深度学习 (deep learning)模型为预测提供点估计，本身不提供对其置信度的衡量。这在需要了解预测可靠性的应用中可能造成问题。贝叶斯神经网络 (neural network)（BNN）通过学习模型参数 (parameter)（权重 (weight)和偏差，$w$）的分布而非单个点估计来处理这一问题。这种方法使BNN能够量化 (quantization)不确定性。这个后验分布，$p(w | \mathcal{D})$（其中 $\mathcal{D}$ 表示训练数据），是量化BNN预测不确定性的依据。

理解和量化不确定性使我们能够构建更值得信赖和信息更丰富的模型。在BNN中，不确定性并非单一种类，通常分为两种主要类型：

### 偶然不确定性

偶然不确定性（有时称为数据不确定性）捕捉数据生成过程中固有的噪声或随机性。它代表即使拥有无限数据也无法减少的不确定性，因为它源于观测中不可约减的变异性。

设想一个回归任务，其中对于完全相同的输入特征 $x$，可能有多个不同的输出值 $y$。这种变异性是偶然的。在BNN中，偶然不确定性通常通过似然函数，$p(y | x, w)$ 来建模。例如，回归中使用高斯似然：


$$
p(y | x, f_w(x)) = \mathcal{N}(y | f_w(x), \sigma^2)
$$


其中，$f_w(x)$ 是参数 (parameter)为 $w$ 的神经网络 (neural network)的输出，$\sigma^2$ 表示观测噪声方差。

- **同方差不确定性：** 假设噪声方差 $\sigma^2$ 对所有输入都保持不变。它可以被视为要学习的参数或预先固定。
- **异方差不确定性：** 允许噪声方差依赖于输入 $x$。BNN可以被设计为预测平均输出 $f_w(x)$ 和方差 $\sigma^2(x)$。当噪声水平在输入空间中变化时，这很有用。

### 认知不确定性

认知不确定性，也称为模型不确定性或知识不确定性，反映了我们对真实模型参数 (parameter)的未知。它捕捉了由于数据有限而无法充分约束后验分布 $p(w | \mathcal{D})$ 所产生的不确定性。随着我们收集更多相关数据，认知不确定性应该减少，因为后验分布会更集中于最优参数值。

在BNN中，认知不确定性直接源于我们拥有权重 (weight)分布 $p(w | \mathcal{D})$ 而非单一权重集。从后验分布中采样的不同可行权重配置，对于相同的输入 $x$ 会产生不同的预测。这些预测中的变动反映了模型对其所学功能的未知。

### 量化 (quantization)预测不确定性

为了得到新输入 $x^*$ 的预测，我们关注的是 *预测分布* $p(y^* | x^*, \mathcal{D})$。这需要对模型参数 (parameter)进行边缘化：


$$
p(y^* | x^*, \mathcal{D}) = \int p(y^* | x^*, w) p(w | \mathcal{D}) dw
$$


这个积分将预测 $p(y^* | x^*, w)$ 对所有可能的参数值进行加权平均，权重 (weight)是它们的后验概率 $p(w | \mathcal{D})$。这个预测分布的方差捕捉了 *总* 不确定性，涵盖了偶然和认知两种来源。

由于后验 $p(w | \mathcal{D})$ 通常难以计算，我们依赖于前面讨论的近似方法：

1. **MCMC 方法：** 诸如随机梯度HMC等技术从后验分布 $p(w | \mathcal{D})$ 中提供样本 $w^{(1)}, w^{(2)}, ..., w^{(S)}$。我们可以通过为每个样本生成预测并形成一个经验分布来近似预测分布：

   
   $$
   p(y^* | x^*, \mathcal{D}) \approx \frac{1}{S} \sum_{s=1}^S p(y^* | x^*, w^{(s)})
   $$
   

   对于具有高斯似然的回归，这意味着获取 $S$ 个预测均值 $f_{w^{(s)}}(x^*)$ 以及可能的方差 $\sigma^2_{w^{(s)}}(x^*)$。预测分布的均值可以通过 $\frac{1}{S} \sum_s f_{w^{(s)}}(x^*)$ 来估计，方差（总不确定性）可以从样本 $\{f_{w^{(s)}}(x^*)\}_{s=1}^S$ 的方差加上平均预测偶然方差来估计。
2. **变分推断 (VI)：** 诸如反向传播 (backpropagation)贝叶斯等VI方法学习近似后验 $q(w; \phi)$。预测分布近似为：

   
   $$
   p(y^* | x^*, \mathcal{D}) \approx \int p(y^* | x^*, w) q(w; \phi) dw
   $$
   

   该积分通常使用蒙特卡洛采样来近似：抽取样本 $w^{(s)} \sim q(w; \phi)$ 并对预测 $p(y^* | x^*, w^{(s)})$ 求平均，这与MCMC方法类似。
3. **蒙特卡洛 Dropout：** 一种计算成本较低且应用广泛的技术是，在权重层之前应用 dropout 来训练标准神经网络 (neural network)。在测试时，dropout 保持激活状态，对相同的输入 $x^*$ 执行多次前向传播（$S$ 次）。由于 dropout 的随机性，每次前向传播都会产生不同的输出。这组输出 $\{f_{w^{(s)}}(x^*)\}_{s=1}^S$（其中 $w^{(s)}$ 代表第 $s$ 次传播中的网络配置）可以被视为近似预测分布的样本。此方法近似于深度高斯过程中的贝叶斯推断。

### 解释和使用不确定性

近似预测分布的散布（例如，方差或四分位距）提供了模型置信度的一种衡量。高方差表明低置信度。

- **区分不确定性类型：** 虽然总预测方差结合了两种效应，但存在将偶然不确定性和认知不确定性分离的技术。例如，在回归中使用蒙特卡洛样本 $\{f_{w^{(s)}}(x^*), \sigma^2_{w^{(s)}}(x^*)\}_{s=1}^S$：
  - **认知不确定性：** 可以通过采样均值的方差来估计： $Var_w(f_w(x^*))$。
  - **偶然不确定性：** 可以通过预测方差的平均值来估计： $\mathbb{E}_w[\sigma^2_w(x^*)]$。
  - **总不确定性：** 近似为 $Var_w(f_w(x^*)) + \mathbb{E}_w[\sigma^2_w(x^*)]$。

了解不确定性的来源很有价值。高认知不确定性表明模型由于输入空间该区域缺乏数据而不确定，这提示了在何处收集更多数据可能有利。高偶然不确定性表明基于给定特征和噪声水平的可预测性存在固有限制。



![BNN预测分布示例（回归）](plots/3628-0.json)



> 1D回归问题中BNN的预测均值和总不确定性（阴影区域，例如，均值±2标准差）的示例。不确定性在训练数据点附近较低，而在远离数据的区域较高，这反映了认知不确定性的增加。

### 不确定性估计的应用

来自BNN的可靠不确定性估计在许多方面都很有用：

- **安全关键系统：** 在自动驾驶或医疗诊断等应用场景中，识别模型不确定的预测必不可少，这允许人工干预或采取替代措施。
- **主动学习：** 选择模型最不确定的新数据点（高认知不确定性）可以带来更高效的数据获取。
- **分布外检测：** BNN对与训练数据分布显著不同的输入通常表现出更高的不确定性。
- **强化学习 (reinforcement learning)：** 不确定性估计可以指导探索策略，鼓励代理在不确定性较高的状态或动作中进行探索。
- **模型诊断：** 分析认知不确定性的空间分布可以展现训练数据或模型架构中的局限性。

通过提供一种合理的方式来表达和量化 (quantization)模型 *未知* 的部分，BNN比标准深度学习 (deep learning)模型有显著的优势，使AI系统更值得信赖。

## 参考资料

- [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142) — Yarin Gal, Zoubin Ghahramani (2016)
  Journal: Proceedings of the 33rd International Conference on Machine Learning (ICML); Pages: 1050-1059; DOI: [10.48550/arXiv.1506.02142](https://doi.org/10.48550/arXiv.1506.02142)
  介绍了蒙特卡洛Dropout作为贝叶斯推断的一种近似方法，这是本节讨论的不确定性量化中的关键技术。
- [What Uncertainties Do We Need in Bayesian Deep Learning for Safe and Reliable AI?](https://arxiv.org/abs/1703.04977) — Alex Kendall and Yarin Gal (2017)
  Journal: NIPS 2017; Volume: 30; Pages: 5574-5584; DOI: [10.48550/arXiv.1703.04977](https://doi.org/10.48550/arXiv.1703.04977)
  定义并区分了深度学习中的偶然不确定性（aleatoric uncertainty）和认知不确定性（epistemic uncertainty），并展示了如何分离它们，这是本节的核心概念。
- [Weight Uncertainty in Neural Networks](https://arxiv.org/abs/1505.05424) — Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, and Daan Wierstra (2015)
  Journal: Proceedings of the 32nd International Conference on Machine Learning (ICML 2015); Volume: 37; Pages: 161-169; DOI: [10.48550/arXiv.1505.05424](https://doi.org/10.48550/arXiv.1505.05424)
  介绍了通过反向传播实现的贝叶斯（Bayes by Backprop），一种用于训练BNN的变分推断方法，本节将其作为后验推断的近似技术提及。
- [Bayesian Learning for Neural Networks](https://doi.org/10.1007/978-1-4612-0745-0) — Radford M. Neal (1996)
  Publisher: Springer-Verlag; DOI: [10.1007/978-1-4612-0745-0](https://doi.org/10.1007/978-1-4612-0745-0)
  一本关于神经网络贝叶斯方法的奠基性著作，包括MCMC技术，为估计后验分布提供了详细背景。
