# 提升（Boosting）原理介绍

来源：[原文](https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-1-ensemble-learning-and-boosting-foundations/introduction-to-boosting)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管装袋法（Bagging）通过平均独立模型来构建集成，但提升法（Boosting）采用的是更具协作性、更讲究顺序的方法。它不是让能力相当的模型进行一轮投票，而是构建一个专家团队，其中每个新成员都经过训练，以修正团队迄今为止所犯的错误。这个迭代过程是提升法能够生成高准确度模型的根本。

设想一群学生正在学习一门有难度的主题。第一个学生学习材料并进行一次练习测试，他们答对了一些问题，也答错了一些。老师随后给第二个学生相同的材料，但会着重指出第一个学生答错的问题。第二个学生会将精力集中在这些有难度的问题上。这个过程会持续下去，每个后续的学生都专注于剩余的薄弱环节。最终的“模型”是所有学生知识的结合，对于那些掌握了更难知识点的学生，会给予更多肯定。

这正是提升法背后的原理。该算法构建了一系列模型，通常是称为**弱学习器**的简单模型，链中的每个模型都经过训练，以修正其前一个模型的错误。

### 顺序学习过程

弱学习器是一种表现仅略好于随机猜测的模型。在提升法中，最常见的弱学习器是**决策树桩**，它是一种只有单次分裂的决策树。单独来看，决策树桩并不是非常强大。然而，通过以结构化、顺序化的方式结合数百或数千个决策树桩，提升法算法可以构建出高准确度且强大的最终模型。

该过程通常遵循以下步骤：

1. **初始化**：从一个初始模型开始，它可能简单到仅预测所有数据点的平均值。
2. **迭代**：对于指定次数的迭代：
   a. **训练弱学习器**：将一个新的弱学习器拟合到数据上，侧重于当前集成表现不佳的实例。
   b. **更新集成**：将新的弱学习器添加到集成中，根据其表现为其分配一个权重 (weight)。它帮助正确分类的实例在下一次迭代中会得到较少关注。
3. **组合**：最终预测是所有弱学习器预测的加权和。

下图说明了这个迭代流程。

> 每个弱学习器都在一个数据集版本上进行训练，其中被先前学习器错误分类的点被赋予更高的关注。最终模型是所有学习器预测的加权和。

### 模型如何“侧重”于错误

“侧重于错误”的方法是区分不同提升算法之处。从宏观层面来看，主要有两种方法：

- **通过重新加权实例**：
  这种方法（由AdaBoost等算法使用）增加了错误分类数据点的权重 (weight)。在下一次迭代中，弱学习器的训练目标会进行调整，使其更关注正确分类这些高权重点的任务。
- **通过拟合 (overfitting)残差**：
  这是梯度提升（Gradient Boosting）使用的方法。新的弱学习器不是重新加权原始数据，而是被训练来预测当前集成的*错误*，或者说**残差**。对于一个给定的数据点，如果模型预测为8，而真实值为10，则残差为2。下一个弱学习器将尝试预测这个2的值，从而直接修正错误。

### 形成最终预测

无论具体技术如何，最终预测并非仅由最后一个模型做出。相反，它是训练过程中所有弱学习器的加权组合。一个强学习器 $H(x)$，通过对所有弱学习器 $h_t(x)$ 的输出求和形成，每个输出都由一个权重 (weight) $\alpha_t$ 进行缩放。


$$
H(x) = \sum_{t=1}^{T} \alpha_t h_t(x)
$$


权重 $\alpha_t$ 通常反映了弱学习器 $h_t(x)$ 的表现；表现更好的学习器在最终结果中拥有更大的发言权。这种加权聚合将一系列简单、弱的模型转化为一个单一、强大的预测器。

这种顺序的、修正错误的过程是提升法的决定性特点。在下一节中，我们将研究AdaBoost，它是第一个实用且非常成功的提升算法，以此来了解这些思想的具体实现。

## 参考资料

- [A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting](https://www.sciencedirect.com/science/article/pii/S002200009791504X) — Yoav Freund and Robert E. Schapire (1997)
  Journal: Journal of Computer and System Sciences; Publisher: Elsevier; Volume: 55; Pages: 119-139; DOI: [10.1006/jcss.1997.1504](https://doi.org/10.1006/jcss.1997.1504)
  原始论文介绍了AdaBoost，一种在提升中实现重新加权的关键算法。
- [Greedy Function Approximation: A Gradient Boosting Machine](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-Function-Approximation-A-Gradient-Boosting-Machine/10.1214/aos/1013203451.full) — Jerome H. Friedman (2001)
  Journal: The Annals of Statistics; Publisher: Institute of Mathematical Statistics; Volume: 29; Pages: 1189-1232; DOI: [10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)
  这篇开创性论文介绍了梯度提升，解释了通过拟合残差进行修正的方法。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://www.springer.com/gp/book/9780387848570) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  统计学习领域的标准教科书，其中包含关于集成方法和提升算法的详细章节。

---

[上一节](02-Bagging%20%E4%B8%8E%20Boosting.md) · [下一节](04-AdaBoost%E7%AE%97%E6%B3%95%EF%BC%9A%E6%A2%AF%E5%BA%A6%E6%8F%90%E5%8D%87%E7%AE%97%E6%B3%95%E7%9A%84%E5%89%8D%E8%BA%AB.md)
