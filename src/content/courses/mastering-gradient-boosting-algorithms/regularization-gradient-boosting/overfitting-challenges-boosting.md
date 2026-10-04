---
course: "mastering-gradient-boosting-algorithms"
chapter: "regularization-gradient-boosting"
lesson: "overfitting-challenges-boosting"
sourceId: 1956
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-3-regularization-gradient-boosting/overfitting-challenges-boosting"
title: "提升算法中的过拟合问题"
description: "理解提升模型为何易于过拟合。"
order: 1
plots: ["plots/1956-0.json"]
sourceHash: "39f25eb8269b9d79843a6098ac7cf9b718adbf09be2c4eedd77b36d23a296c27"
sourceCorrections: []
---

梯度提升算法按顺序构建模型，其中每个新的基本学习器（通常是决策树）试图修正之前学习器集成所犯的错误。这种迭代改进过程，侧重于当前预测的损失函数 (loss function)的*伪残差*或梯度，赋予了梯度提升算法强大的预测能力。然而，正是这种机制使其特别容易过拟合 (overfitting)。

主要问题源于算法不断努力最小化训练数据上的损失。在初始阶段，新树对主要模式进行建模，并显著减少预测误差。随着过程持续多轮迭代，剩余残差逐渐不仅代表了潜在的精细结构，还代表了训练样本中固有的随机噪声。如果基本学习器足够复杂（例如，深度决策树）且提升过程不中断，算法就会开始精细地拟合这些噪声。模型本质上开始记忆训练数据，包括其特殊性。

从偏差-方差角度来看，梯度提升算法通过构建一个复杂的加性函数来积极减少偏差。没有限制的话，这个过程可能导致极高的方差。模型变得对训练数据过度拟合，但未能推广到新的、未见过的数据点，因为其在后期学到的“模式”仅仅是噪声伪影。这表现为训练集和独立验证集上性能指标的典型分歧。随着提升轮次的增加，训练误差可能持续稳定下降，而验证误差达到最小值后开始上升，因为模型出现了过拟合。



![提升算法中典型的过拟合模式](plots/1956-0.json)



> 训练误差通常会随着迭代次数的增加而持续下降，而验证误差通常在开始时下降，但随后随着模型开始过拟合而上升。

基本学习器的复杂度扮演着重要角色。决策树，尤其是允许其生长得很深时，可以创建分割点，从而隔离极少量的训练实例，甚至可能是单个数据点。在拟合残差时，这类复杂树可以轻易找到完美解释这些少量实例相关噪声的分割点，从而导致高度特异性且不具备泛化能力的规则。

这种行为与随机森林（基于装袋法）等方法形成对比，随机森林通过平均多个独立训练的深层树的预测来帮助减少总体方差。提升算法的顺序特性（每棵树都依赖于之前的树）缺乏通过简单平均来减少方差的固有机制，反而需要明确的正则化 (regularization)来防止方差过度增加。

了解这些过拟合机制非常重要。认识到无约束的提升算法几乎不可避免地会使训练数据过拟合，这强调了本章其余部分讨论的正则化技术的重要性。这些方法提供了必要的控制，以在使用提升算法能力的同时，确保所得模型能很好地应对新的问题。

## 参考资料

- [Greedy Function Approximation: A Gradient Boosting Machine](http://www.jstor.org/stable/2699986) — Jerome H. Friedman (2001)
  Journal: The Annals of Statistics; Publisher: Institute of Mathematical Statistics; Volume: 29; Pages: 1189-1232; DOI: [10.2307/2699986](https://doi.org/10.2307/2699986)
  介绍了梯度提升机器算法，这是理解其迭代性质和过拟合倾向的基础。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  提供了梯度提升的统计学阐述，详细分析了偏差-方差权衡和导致过拟合的机制。
- [Ensemble Methods: Foundations and Algorithms](https://doi.org/10.1201/b12207) — Zhi-Hua Zhou (2012)
  Publisher: Chapman and Hall/CRC; Pages: 236; DOI: [10.1201/b12207](https://doi.org/10.1201/b12207)
  一本专门介绍集成学习的综合教材，提供了对提升算法行为、其优势以及未受适当控制时固有过拟合挑战的见解。
