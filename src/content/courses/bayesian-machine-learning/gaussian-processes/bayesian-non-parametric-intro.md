---
course: "bayesian-machine-learning"
chapter: "gaussian-processes"
lesson: "bayesian-non-parametric-intro"
sourceId: 3584
sourceUrl: "https://apxml.com/zh/courses/bayesian-machine-learning/chapter-4-gaussian-processes/bayesian-non-parametric-intro"
title: "贝叶斯非参数建模介绍"
description: "介绍贝叶斯非参数方法，其模型复杂度会随数据量增长。"
order: 1
plots: ["plots/3584-0.json"]
sourceHash: "c37710870d391aa9c277df824a2a1de0ba2f5ae046c565da33bf1407d984bc62"
sourceCorrections: []
---

多数机器学习 (machine learning)模型都属于*参数 (parameter)模型*的范畴。例如线性回归、逻辑回归，甚至是标准深度神经网络 (neural network)。这些模型假定一个特定的函数形式，并具有*固定*且有限数量的参数（如权重 (weight) $\beta$ 或网络权重 $W$）。我们使用数据来估计这些参数的值，通常通过优化（如最大似然估计）或贝叶斯推断（计算后验概率 $p(\theta | D)$）来完成。核心假定是，一旦这些参数被确定，它们就能掌握我们所需了解的输入与输出之间的一切关系，且这种关系受限于所选的模型结构。

然而，当生成数据的底层过程比我们固定的参数形式所允许的更复杂时，会发生什么？将数据强行纳入僵化的模型结构，可能导致欠拟合 (underfitting)或有偏差的结论。我们可能无法事先知晓“正确”的复杂度。这时，*非参数*方法就派上用场了。

‘非参数’这个术语可能有点误导性。它通常不意味着“没有”参数。相反，它指那些有效参数的数量并非*事先*固定不变，而是能随着更多数据的获得而增长和调整的模型。这些模型提供更大的灵活性，使结构复杂度本身可以从数据中推断出来。

在贝叶斯框架中，这意味着定义先验分布不仅针对有限的参数集，也针对更复杂、可能是无限维的对象。不再是针对 $\theta \in \mathbb{R}^d$ 的先验 $p(\theta)$，我们可以考虑针对函数、划分或测度的先验。这就是**贝叶斯非参数建模**的范围。

高斯过程，作为本章的重点，是贝叶斯非参数方法的核心要素，特别适用于回归和分类任务。不同于假定一个特定的参数函数如 $f(x) = \beta_0 + \beta_1 x_1 + ... + \beta_p x_p$ 并对 $\beta_i$ 系数设置先验，高斯过程*直接在所有可能函数* $f(x)$ 的空间上定义先验。它假定在任意有限输入点集 $x_1, ..., x_n$ 上的函数值 $f(x_1), f(x_2), ..., f(x_n)$ 遵循多元高斯分布。

这种以函数为中心的方式提供了非常大的灵活性。模型复杂度不受固定数量参数的限制；它会根据观测数据和先验中编码的性质进行调整（具体来说是协方差或核函数，我们稍后会讨论）。这种方法自然地纳入了不确定性量化 (quantization)：因为我们推断出的是函数的分布，所以我们得到的预测附带相关的置信区间，这反映了模型在哪些区域确定性更高或更低。

考虑一个简单的回归场景。像线性回归这样的参数模型可能会遇到困难，如果真实关系是非线性的。像高斯过程这样的贝叶斯非参数模型可以捕捉复杂模式，而无需我们事先指定非线性的确切形式。



![参数拟合与非参数拟合示例](plots/3584-0.json)



> 比较简单的线性参数拟合与更灵活的非参数拟合（此处所示）的示例。非参数方法能更好地捕捉数据趋势，并提供不确定性带，这些不确定性带通常在数据较少的区域会变宽。

本章将引导您了解高斯过程的数学基本原理和实际操作。我们将从正式定义高斯过程为函数的先验分布开始，分析协方差函数如何影响这些先验，并推导在回归情境下进行预测的方程。您将学习如何处理超参数 (hyperparameter)，并将此框架扩展到分类任务，以及使高斯过程在处理大型数据集时计算可行的技术。

## 参考资料

- [Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/) — Carl Edward Rasmussen and Christopher K. I. Williams (2006)
  Publisher: The MIT Press
  关于高斯过程的权威教材，详细阐述了其在机器学习中的理论和应用。
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/) — Christopher Bishop (2006)
  Publisher: Springer
  一本广泛使用的机器学习教材，其中包含关于高斯过程的专门章节，并区分了参数化模型和非参数化模型。
- [Bayesian Nonparametrics](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFGWEmfamonHnFCkSjiVCMIYyuQ2xMStK3rBuLCSx17cVvJXQa6kK5HvP2mcFF_T-nJGqAKCU22DvbbBrWSRFggzna113t4Js6iab6tgyg1glbGYUOwUaI613AOzoimrOgSB_t6VA_-D2u_Syt3xPuK7UFS2_RGq3aZ74HyHnwoi5AnpufUv5_etL55PQ==) — Jayanta K. Ghosh and R. V. Ramamoorthi (2003)
  Publisher: Springer Science & Business Media; Pages: 305; DOI: [10.1007/b97500](https://doi.org/10.1007/b97500)
  一本基础教材，介绍了贝叶斯非参数建模的理论基础，包括无限维对象上的先验分布等概念。
