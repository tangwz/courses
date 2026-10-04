---
course: "mastering-gradient-boosting-algorithms"
chapter: "gradient-boosting-algorithm-depth"
lesson: "gbm-classification-loss-functions"
sourceId: 1938
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-2-gradient-boosting-algorithm-depth/gbm-classification-loss-functions"
title: "分类任务的常用损失函数"
description: "GBM中对数损失（二项/多项偏差）的分析。"
order: 4
plots: ["plots/1938-0.json"]
sourceHash: "94859a832517737f871b76d18df5ec0075e751efb8e7cb9222e9709ce6d8a60a"
sourceCorrections: []
---

回归任务通常使用均方误差或平均绝对误差等损失函数 (loss function)，分类问题则需要不同的度量方式，以量化 (quantization)预测的类别概率（或得分）与实际离散类别标签之间的差异。对于梯度提升机（GBM）而言，损失函数的选择非常重要，因为其负梯度决定了后续基学习器将要预测的“伪残差”。

下面我们来考察GBM框架内用于分类的最常用损失函数。

### 二元分类：对数损失（二项偏差）

对于二元分类问题，其中目标变量 $y$ 的取值为 $\{0, 1\}$，标准且最广泛使用的损失函数 (loss function)是**对数损失**，也称作**二项对数似然**或**二项偏差**。

它直接源于最大似然估计的原则。如果 $F(x)$ 是模型对输入 $x$ 预测的原始输出（logit 或对数几率），那么正类别（$y=1$）的概率通常通过逻辑（sigmoid）函数得到：


$$
p(x) = P(y=1 | x) = \frac{1}{1 + e^{-F(x)}}
$$


单个观测 $(x, y)$ 的对数损失定义为给定预测概率下真实标签似然的负对数：


$$
L(y, p(x)) = -[y \log(p(x)) + (1-y) \log(1 - p(x))]
$$


此损失函数会严厉惩罚那些置信度高但错误的预测。例如，如果真实标签是 $y=1$ 但预测概率 $p(x)$ 接近 0，则项 $-y \log(p(x)) = -\log(p(x))$ 将趋向于无穷大。反之，如果预测正确且置信度高（当 $y=1$ 时 $p(x)$ 接近 1），损失将趋近于 0。



![二元对数损失（真实标签 y=1）](plots/1938-0.json)



> 当真实标签为1时的二元对数损失。随着预测概率接近0（预测错误），损失会明显增加。

**梯度与伪残差：**
在GBM算法中，我们需要损失函数相对于模型在当前第 $m-1$ 次迭代的*原始输出* $F(x)$ 的负梯度，记为 $F_{m-1}(x)$。这个梯度指导着下一个树 $h_m(x)$ 的拟合。


$$
-\frac{\partial L(y, F(x))}{\partial F(x)} = -\frac{\partial L}{\partial p} \frac{\partial p}{\partial F(x)}
$$


使用链式法则，并已知 $\frac{\partial p}{\partial F(x)} = p(x)(1-p(x))$（sigmoid 函数的导数）且 $\frac{\partial L}{\partial p} = -\frac{y}{p} + \frac{1-y}{1-p} = \frac{p-y}{p(1-p)}$，我们得到：


$$
-\frac{\partial L(y, F(x))}{\partial F(x)} = - \left( \frac{p(x) - y}{p(x)(1-p(x))} \right) (p(x)(1-p(x))) = y - p(x)
$$


这个结果非常简洁。负梯度，它是第 $m$ 次迭代中观测 $i$ 的伪残差 $r_{im}$，简单来说，就是真实标签（0 或 1）与当前预测概率 $p_{i, m-1} = P(y_i=1 | x_i; F_{m-1})$ 之间的差异。


$$
r_{im} = y_i - p_{i, m-1}
$$


因此，在用对数损失进行二元分类训练的GBM中，每个新树 $h_m(x)$ 都是为了预测*当前概率估计*的残差。

### 多元分类：对数损失（多项偏差）

这个思想自然地扩展到多元分类问题，在其中目标变量 $y$ 可以属于 $K$ 个类别中的一个，即 $\{1, ..., K\}$。通常我们对真实标签使用独热编码，因此 $y_i$ 是一个向量 (vector)，如果观测 $i$ 属于类别 $k$，则 $y_{ik}=1$，否则 $y_{ik}=0$。

模型需要输出 $K$ 个分数或 logits，即 $F_1(x), ..., F_K(x)$。这些通过**softmax 函数**转换为概率 $p_k(x) = P(y=k | x)$：


$$
p_k(x) = \frac{e^{F_k(x)}}{\sum_{j=1}^K e^{F_j(x)}}
$$


所使用的损失函数 (loss function)是**多项对数损失**，也称作**多项偏差**或**交叉熵损失**：


$$
L(\{y_k\}, \{p_k(x)\}) = - \sum_{k=1}^K y_k \log(p_k(x))
$$


对于给定的观测，只有一个 $y_k$ 为 1，其他为 0，这可以简化为 $-\log(p_c(x))$，其中 $c$ 是真实类别索引。

**梯度与伪残差：**
类似于二元情况，我们计算损失函数相对于每个原始输出 $F_k(x)$ 的负梯度。由于softmax函数，计算稍微复杂一些，但结果依然简洁。对于特定的类别 $k$ 和观测 $i$，在第 $m-1$ 次迭代的负梯度为：


$$
r_{imk} = -\frac{\partial L(y_i, F_{m-1}(x_i))}{\partial F_{k, m-1}(x_i)} = y_{ik} - p_{ik, m-1}
$$


其中 $p_{ik, m-1}$ 是基于模型 $F_{m-1}$ 对观测 $i$ 类别 $k$ 的预测概率。

同样地，每个类别的伪残差是真实标签（独热编码中的 0 或 1）与该类别当前预测概率之间的差异。

在实际中，大多数GBM实现通过在每次提升迭代 $m$ 时构建 $K$ 个独立的回归树来处理多元分类问题。第 $k$ 个树使用伪残差 $r_{imk} = y_{ik} - p_{ik, m-1}$ 作为其目标变量进行训练。这些 $K$ 个树的输出随后被结合，以更新每个类别的总体模型分数 $F_k(x)$。

### 为何选择对数损失？

对数损失（或多项偏差）之所以优于指数损失（原始AdaBoost算法中使用）等替代方案，主要是因为它对异常值或标记 (token)错误的数据点不那么敏感。指数损失以指数方式惩罚错误，这意味着一个非常错误的预测可能主导梯度计算，并可能扭曲模型。对数损失为分类任务提供了更稳定的性能衡量，它直接关联信息论（交叉熵）和概率解释（对数似然）。其平滑的梯度也有助于GBM框架内的优化过程。

通过理解这些分类损失函数 (loss function)如何运作以及它们的梯度如何产生伪残差，我们能够更好地了解梯度提升机在分类问题中学习过程的核心机制。这种选择直接影响后续树尝试纠正的错误类型，从而塑造最终的预测模型。

## 参考资料

- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://link.springer.com/book/10.1007/978-0-387-84858-7) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  涵盖统计学习的理论基础，包括广义加性模型、梯度提升、逻辑回归和各种损失函数（如偏差/对数损失）的详细解释。
- [Greedy Function Approximation: A Gradient Boosting Machine](https://doi.org/10.1214/aos/1013203451) — Jerome H. Friedman (2001)
  Journal: The Annals of Statistics; Volume: 29; Pages: 1189-1232; DOI: [10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)
  介绍了梯度提升机算法，详细说明了各种损失函数（包括分类偏差）的负梯度如何构成后续树拟合的伪残差。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapters 3, 5
  提供了机器学习基础知识的全面介绍，清晰地解释了交叉熵损失（对数损失）及其与最大似然估计的联系，对分类问题很重要。
