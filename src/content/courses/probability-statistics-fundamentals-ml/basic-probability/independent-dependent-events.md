---
course: "probability-statistics-fundamentals-ml"
chapter: "basic-probability"
lesson: "independent-dependent-events"
sourceId: 2436
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-fundamentals-ml/chapter-3-basic-probability/independent-dependent-events"
title: "独立事件与关联事件"
description: "区分独立事件与关联事件，并把握其对概率计算的意义。"
order: 5
plots: []
sourceHash: "5754c558da3e17a37aadb12f327d2a0e9b10b6882bcf980d560f52513c3fb474"
sourceCorrections: []
---

条件概率 $P(A|B)$ 表示在事件 $B$ 已经发生的情况下，事件 $A$ 发生的概率。这一基本概念有助于描述事件如何相互关联。在事件关系的研究中，区分独立性与关联性是主要方面。这两个概念描述了某个事件的发生是否会影响另一个事件的概率。辨别这种区别对于分析数据和创建某些类型的机器学习 (machine learning)模型很有帮助。

### 独立性意味着什么？

如果一个事件的发生不影响另一个事件发生的概率，那么两个事件 $A$ 和 $B$ 被认为是**独立的**。设想抛掷一枚均匀硬币两次。第一次抛掷的结果（比如，得到正面）会改变第二次抛掷得到正面的概率吗？不会，概率仍然是 $1/2$。硬币没有记忆。

正式来说，我们可以通过两种主要方式来确定独立性：

1. **使用条件概率：** 如果知道 $B$ 发生不改变 $A$ 的概率，则事件 $A$ 和 $B$ 是独立的。

   
   $$
   P(A|B) = P(A)
   $$
   

   类似地，如果知道 $A$ 发生不改变 $B$ 的概率：

   
   $$
   P(B|A) = P(B)
   $$
   

   （第一个情况假设 $P(B) > 0$，第二个情况假设 $P(A) > 0$）。
2. **使用联合概率：** 如果两个事件同时发生的概率是它们各自概率的乘积，则事件 $A$ 和 $B$ 是独立的。

   
   $$
   P(A \cap B) = P(A) P(B)
   $$
   

   这通常是检验独立性最实用的方法。如果此等式成立，则事件是独立的。反之，它们是关联的。

#### 例子：抛掷硬币

让我们回顾一下两次抛掷硬币。
事件 A：第一次抛掷得到正面。$P(A) = 1/2$。
事件 B：第二次抛掷得到正面。$P(B) = 1/2$。

两次抛掷都得到正面的概率 $P(A \cap B)$ 是多少？所有可能的结果是 HH、HT、TH、TT。只有 HH 同时满足这两个事件。因此，$P(A \cap B) = 1/4$。

现在我们用公式来验证：
$P(A) P(B) = (1/2) \times (1/2) = 1/4$。

由于 $P(A \cap B) = P(A) P(B)$，这两个事件是独立的，这与我们的直觉相符。

### 关联性意味着什么？

如果一个事件的发生*确实*会影响另一个事件发生的概率，那么这些事件就是**关联的**。

设想从一副标准的52张扑克牌中抽两张牌，且*不放回*第一张牌。
事件 A：第一次抽到一张K。
事件 B：第二次抽到一张K。

这些事件是独立的吗？我们来考虑一下概率。
$P(A) = 4/52 = 1/13$ (52张牌中有4张K)。

那么，在第一次抽到一张K的情况下，第二次抽到一张K的概率是多少？这就是 $P(B|A)$。如果我们第一次抽到了一张K，现在只剩下3张K和51张牌。
$P(B|A) = 3/51 = 1/17$。

由于 $P(B|A) = 1/17$ 而 $P(A) = 1/13$（从直觉上讲，如果我们不知道第一张牌的结果，$P(B)$ 也会是 $1/13$），我们看到 $P(B|A) \neq P(B)$。知道第一次抽牌的结果改变了第二次抽牌的概率。所以，这些事件是**关联的**。

我们也可以通过联合概率的确定方式来确认这一点。我们从条件概率公式中得知 ($P(A \cap B) = P(B|A)P(A)$)：
$P(A \cap B) = P(\text{第一次K且第二次K}) = P(B|A)P(A) = (3/51) \times (4/52) = 12/2652 \approx 0.0045$

现在将其与它们各自概率的乘积进行比较：
$P(A)P(B) = (4/52) \times (4/52) = (1/13) \times (1/13) = 1/169 \approx 0.0059$

由于 $P(A \cap B) \neq P(A)P(B)$，这些事件被证实是关联的。不放回抽牌的行为使结果相关联。如果我们放回了第一张牌，这些事件就会是独立的。

### 独立性在机器学习 (machine learning)中的作用

独立性与关联性在机器学习的许多方面都很基本：

1. **特征工程：** 为模型选择特征时，了解特征是否关联很重要。高度关联的特征可能存在冗余，提供相似的信息。有时，组合关联特征或删除其中一个可以提升模型表现或降低复杂度。
   "2. **模型假设：** 有些模型对独立性有明确的假设。一个典型示例是**朴素贝叶斯分类器**。它的原理是假设在给定类别标签的情况下，所有输入特征彼此独立。这是一种“朴素”的假设，因为数据中的特征通常是关联的。然而，这种简化让计算变得简单得多，并且该模型在许多情况下（如文本分类）效果出乎意料的好。了解独立性有助于您把握此类模型的潜在假设和可能的局限性。"
2. **概率模型：** 在创建概率模型（如贝叶斯网络）时，变量之间的关联性被明确地描绘出来。独立性使得模型结构和计算得以简化。

总之，区分独立事件与关联事件使我们能够正确计算涉及多个事件的概率，并了解某些机器学习算法背后的假设。独立性简化了计算 ($P(A \cap B) = P(A)P(B)$)，而关联性则需要用到条件概率 ($P(A \cap B) = P(A|B)P(B) = P(B|A)P(A)$)。

## 参考资料

- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  全面涵盖统计学习方法，包括模型中的概率基础和假设。
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-and-machine-learning-2/) — Christopher M. Bishop (2006)
  Publisher: Springer; DOI: [10.1007/978-0-387-44958-6](https://doi.org/10.1007/978-0-387-44958-6)
  提供了机器学习的详细概率视角，重点关注贝叶斯方法和独立性在其中的作用。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://books.google.com/books?id=0Mh2EAAAQBAJ) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  提供机器学习模型的实践示例和解释，包括朴素贝叶斯分类器及其独立性假设。
